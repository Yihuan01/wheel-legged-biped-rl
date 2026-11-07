# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : control.py
# Purpose   : 执行层：动作裁剪 → 腿位置 PD / 轮速度环（实测加速度 D 项，α 低通）→ 逐 dof 力矩
# 限幅。语义与 wbr_flat4._compute_torques 逐位一致，且**每个物理子步重算一次**
# （Isaac 的 200 Hz 力矩更新节奏，轮 D 项依赖它）。
# ==============================================================================

import torch


class PDController:
    def __init__(self, contract):
        c = contract
        self.c = c
        self.names = list(c["dof_names"])
        self.adi = [self.names.index(n) for n in c["action_dof_names"]]
        self.wheel_dof = [i for i, n in enumerate(self.names) if "wheel" in n]
        self.knee_dof = [i for i, n in enumerate(self.names) if "knee" in n]
        # 动作序里的轮索引（四轮 action 序=[hip,hip,hip,hip,wheel,wheel]、两轮=[hip,knee,wheel,hip,knee,wheel]）
        self.wheel_act = [i for i, n in enumerate(c["action_dof_names"]) if "wheel" in n]
        self.default = torch.tensor(c["default_dof_pos"], dtype=torch.float32)
        self.kp = torch.tensor(c["kps"], dtype=torch.float32)
        self.kd = torch.tensor(c["kds"], dtype=torch.float32)
        self.lim = torch.tensor(c["torque_limits"], dtype=torch.float32)
        # 动作缩放按**动作名**构建（两轮 action 序轮/腿交错，不能写死前 4 后 2）
        self.scale_vec = torch.tensor(
            [c["wheel_action_scale"] if "wheel" in n else c["leg_action_scale"]
             for n in c["action_dof_names"]], dtype=torch.float32)
        self.clip_actions = float(c["clip_actions"])
        self.alpha = float(c["wheel_accel_filter_alpha"])
        self.accel_limit = float(c["wheel_target_accel_limit"])
        self.dt = float(c["dt_physics"])
        # 【S2 两轮】行走模板的膝偏移常数（off_knee 恒值）：左 + 右 −（flat.py additive 分支镜像语义）。
        # 四轮契约无此键 → 0 → 不影响四轮路径。jump_template 启用时由相位机每拍接管（见 set_phase_gains）。
        self.off_knee = float(c.get("off_knee_const", 0.0))
        self.kp0 = self.kp.clone()
        self.kd0 = self.kd.clone()
        # 【S3 P7T】脚本起立三接口：腿绝对目标覆盖 / 腿 P 倍率 / 轮力矩覆盖
        self.leg_override = None
        self.leg_boost = 1.0
        self.wheel_override = None
        self.wheel_override_l = None      # 左右差动（设置时优先于 wheel_override）
        self.wheel_override_r = None
        self.off_hip = float(c.get("off_hip_const", 0.0))    # 【S2b 跳跃】动态注入（回放式模板）
        self.knee_l = self.names.index("left_knee_joint") if "left_knee_joint" in self.names else None
        self.knee_r = self.names.index("right_knee_joint") if "right_knee_joint" in self.names else None
        self.hip_l = self.names.index("left_hip_joint") if "left_hip_joint" in self.names else None
        self.hip_r = self.names.index("right_hip_joint") if "right_hip_joint" in self.names else None
        self.reset()

    def set_contract(self, contract):
        """【S3 uni 链】运行时换契约：动作序/缩放/default/增益全部按新契约重建，
        轮加速度滤波状态保留（尺寸不变，避免换段瞬间的力矩跳变）。"""
        c = contract
        self.c = c
        self.names = list(c["dof_names"])
        self.adi = [self.names.index(n) for n in c["action_dof_names"]]
        self.wheel_dof = [i for i, n in enumerate(self.names) if "wheel" in n]
        self.knee_dof = [i for i, n in enumerate(self.names) if "knee" in n]
        self.wheel_act = [i for i, n in enumerate(c["action_dof_names"]) if "wheel" in n]
        self.default = torch.tensor(c["default_dof_pos"], dtype=torch.float32)
        self.kp = torch.tensor(c["kps"], dtype=torch.float32)
        self.kd = torch.tensor(c["kds"], dtype=torch.float32)
        self.lim = torch.tensor(c["torque_limits"], dtype=torch.float32)
        self.scale_vec = torch.tensor(
            [c["wheel_action_scale"] if "wheel" in n else c["leg_action_scale"]
             for n in c["action_dof_names"]], dtype=torch.float32)
        self.clip_actions = float(c["clip_actions"])
        self.alpha = float(c["wheel_accel_filter_alpha"])
        self.accel_limit = float(c["wheel_target_accel_limit"])
        self.dt = float(c["dt_physics"])
        self.off_knee = float(c.get("off_knee_const", 0.0))
        self.off_hip = float(c.get("off_hip_const", 0.0))
        self.kp0 = self.kp.clone()
        self.kd0 = self.kd.clone()
        self.knee_l = self.names.index("left_knee_joint") if "left_knee_joint" in self.names else None
        self.knee_r = self.names.index("right_knee_joint") if "right_knee_joint" in self.names else None
        self.hip_l = self.names.index("left_hip_joint") if "left_hip_joint" in self.names else None
        self.hip_r = self.names.index("right_hip_joint") if "right_hip_joint" in self.names else None
        self.leg_override = None
        self.leg_boost = 1.0
        self.wheel_override = None
        self.wheel_override_l = None
        self.wheel_override_r = None

    def reset(self):
        self.last_wheel = torch.zeros(len(self.wheel_dof), dtype=torch.float32)
        self.filt = torch.zeros(len(self.wheel_dof), dtype=torch.float32)
        self.wheel_target = torch.zeros(len(self.wheel_dof), dtype=torch.float32)

    def set_phase_gains(self, knee_damp_extra=0.0, knee_kp_extra=0.0):
        """跳跃相位专用膝增益增量（flat.py _compute_torques 的相位段）。"""
        self.kd = self.kd0.clone()
        self.kp = self.kp0.clone()
        if knee_damp_extra:
            for i in self.knee_dof:
                self.kd[i] = max(self.kd0[i] + knee_damp_extra, 0.0)
        if knee_kp_extra:
            for i in self.knee_dof:
                self.kp[i] = self.kp0[i] + knee_kp_extra

    def torques(self, action, dof_pos, dof_vel):
        c = self.c
        a = torch.clip(torch.as_tensor(action, dtype=torch.float32),
                       -self.clip_actions, self.clip_actions)
        dof_pos = torch.as_tensor(dof_pos, dtype=torch.float32)
        dof_vel = torch.as_tensor(dof_vel, dtype=torch.float32)

        scaled = torch.zeros(len(self.names), dtype=torch.float32)
        if self.leg_override is not None:
            # 【S3 P7T】脚本语义（flat.py 同款）：a=0、default=脚本角 ⇒ PD 目标=default。
            # 腿位 scaled 置 0，脚本角由调用方写 self.default 的腿列。
            scaled[self.adi[:4]] = 0.0
        else:
            scaled[self.adi] = a * self.scale_vec
        if self.off_knee != 0.0 and self.knee_l is not None:
            scaled[self.knee_l] += self.off_knee
            scaled[self.knee_r] -= self.off_knee
        if self.off_hip != 0.0 and self.hip_l is not None:
            scaled[self.hip_l] += self.off_hip
            scaled[self.hip_r] -= self.off_hip

        target = a[self.wheel_act].clone() * float(c["wheel_action_scale"])
        if self.accel_limit > 0.0:
            delta = torch.clamp(target - self.wheel_target, -self.accel_limit * self.dt,
                                self.accel_limit * self.dt)
            self.wheel_target = self.wheel_target + delta
        else:
            self.wheel_target = target
        w = self.wheel_dof
        scaled[w] = self.wheel_target

        wv = dof_vel[w]
        raw = (wv - self.last_wheel) / self.dt
        self.filt = self.alpha * self.filt + (1.0 - self.alpha) * raw

        tau = self.kp * (scaled + self.default - dof_pos) - self.kd * dof_vel
        tau[w] = self.kp[w] * (self.wheel_target - wv) - self.kd[w] * self.filt
        self.last_wheel = wv.clone()

        tau = torch.clip(tau, -self.lim, self.lim)
        if self.wheel_override_l is not None or self.wheel_override_r is not None:
            # 左右差动覆盖（roll 控制）；未设置的侧用 wheel_override 值
            base = self.wheel_override if self.wheel_override is not None else 0.0
            tau[self.wheel_dof[0]] = float(self.wheel_override_l if self.wheel_override_l is not None else base)
            tau[self.wheel_dof[1]] = float(self.wheel_override_r if self.wheel_override_r is not None else base)
        elif self.wheel_override is not None:
            # 【S3 P7T】抬升期轮恒力矩覆盖（两驱动轮同值）
            for i in self.wheel_dof:
                tau[i] = float(self.wheel_override)
        return tau
