# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : sim_loop.py
# Purpose   : MuJoCo 四轮运行核（一核两壳）：MjWbr 物理状态访问 + SimLoop 控制循环
# （指令限速 → 观测拼装 → 策略推理 → 腿 PD/轮速度环 200 Hz 力矩 → 终止判定
# → 遥测）。run_policy.py（固定指令批量）与 serve.py（控制台实时）共用，
# 物理语义与 run_policy 逐位一致（golden 对账把关）。
# ==============================================================================

import json
import os
import sys
import time

import mujoco
import numpy as np
import torch

from legged_gym.scripts.sim2sim_mujoco.deploy.obs_builder import ObsBuilder
from legged_gym.scripts.sim2sim_mujoco.deploy.control import PDController
from legged_gym.scripts.sim2sim_mujoco.deploy.policy import Policy
from legged_gym.scripts.sim2sim_mujoco.deploy.com_model import com_constants, com_body_offset

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(PKG, "../../.."))

# 指令限速（设计稿验收第 5 条的数值）：vx m/s²、h m/s。批量 CLI 模式关闭（与 run_policy 逐位一致）。
SLEW_VX = 0.8
SLEW_H = 0.15


def quat_R(q):
    """wxyz → 旋转矩阵（机体→世界）。"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class MjWbr:
    def __init__(self, mjcf, contract):
        self.m = mujoco.MjModel.from_xml_path(mjcf)
        self.d = mujoco.MjData(self.m)
        self.c = contract
        self.has_caster = bool(contract.get("with_4w", True))   # 【S2 两轮】v1 无从动轮
        self.names = list(contract["dof_names"])
        self.qadr = {}
        self.dadr = {}
        for n in self.names:
            j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
            assert j >= 0, n
            self.qadr[n] = int(self.m.jnt_qposadr[j])
            self.dadr[n] = int(self.m.jnt_dofadr[j])
        self.dof_idx = [self.dadr[n] for n in self.names]
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.geom_body = self.m.geom_bodyid
        self.feet = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_wheel_link"),
                     mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_wheel_link")]
        if self.has_caster:
            self.casters = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_knee_caster"),
                            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_knee_caster")]
        else:
            self.casters = []
        self.term_bodies = {self.base}
        for n in ("left_thigh_link", "right_thigh_link"):
            self.term_bodies.add(mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n))
        self.cc = com_constants("cpu")
        self.default = np.array(contract["default_dof_pos"], dtype=np.float64)
        self.dt = float(contract["dt_physics"])
        self.decimation = int(contract["decimation"])

    # ------------------------------------------------------------------ 状态
    def reset(self):
        d = self.d
        d.qpos[:] = 0.0
        d.qpos[0:3] = np.array(self.c["initial_root_pos"], dtype=np.float64)
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for n, v in zip(self.names, self.default):
            d.qpos[self.qadr[n]] = v
        d.qvel[:] = 0.0
        d.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.m, d)

    def dof_pos(self):
        return np.array([self.d.qpos[self.qadr[n]] for n in self.names])

    def dof_vel(self):
        return np.array([self.d.qvel[self.dadr[n]] for n in self.names])

    def base_state(self):
        d = self.d
        q = d.qpos[3:7].copy()
        R = quat_R(q)
        lin_world = d.qvel[0:3].copy()
        ang_body = d.qvel[3:6].copy()          # 自由关节角速度按机体表达（见 README 的坐标系核对）
        return R, q, R.T @ lin_world, ang_body

    def projected_gravity(self, R):
        return R.T @ np.array([0.0, 0.0, -1.0])

    # ------------------------------------------------------------------ 接触
    def _contact_forces(self):
        """逐刚体法向力（N）：只统计法向分量，用于触地与终止判定。"""
        out = {}
        d = self.d
        f = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            mujoco.mj_contactForce(self.m, d, i, f)
            b1 = int(self.geom_body[con.geom1])
            b2 = int(self.geom_body[con.geom2])
            out[b1] = out.get(b1, 0.0) + float(f[0])
            out[b2] = out.get(b2, 0.0) + float(f[0])
        return out

    def caster_contacts(self, forces=None):
        if not self.has_caster:
            return np.zeros(2, dtype=np.float32)          # 【S2 两轮】无从动轮，obs 不消费，占位 0
        f = forces if forces is not None else self._contact_forces()
        return np.array([1.0 if f.get(b, 0.0) > 0.1 else 0.0 for b in self.casters], dtype=np.float32)

    def tip_margin(self, dof_pos):
        if not self.has_caster:
            return 0.0                                    # 【S2 两轮】obs 无 tip 通道，占位 0
        xs = np.array([self.d.xpos[b, 0] for b in self.feet + self.casters])
        q = torch.tensor(dof_pos, dtype=torch.float32).unsqueeze(0)
        com = com_body_offset(self.cc, q[:, self.names.index("left_hip_joint")],
                              q[:, self.names.index("right_hip_joint")],
                              q[:, self.names.index("left_knee_joint")],
                              q[:, self.names.index("right_knee_joint")])
        com_x = float(self.d.qpos[0]) + float(com[0, 0])
        return float(min(xs.max() - com_x, com_x - xs.min()))

    def body_pitch_roll(self, pg):
        pitch = float(np.arcsin(np.clip(-pg[0], -1.0, 1.0)))
        roll = float(np.arcsin(np.clip(-pg[1], -1.0, 1.0)))
        return pitch, roll

    # ------------------------------------------------------------------ 步进
    def apply(self, tau):
        self.d.qfrc_applied[:] = 0.0
        for k, i in enumerate(self.dof_idx):
            self.d.qfrc_applied[i] = float(tau[k])
        mujoco.mj_step(self.m, self.d)


class SimLoop:
    """一个控制步 = 限速指令 → 观测 → actor → decimation × (PD/轮速环 → mj_step)。
    遥测与终止判定用**步首**状态（与旧 run_policy 逐位一致）。"""

    # 指令 7 维槽位：[0]=vx [1]=vy [2]=yaw [4]=jump [5]=h [6]=roll（obs 契约顺序）
    CMD_IDX = {"lin_vel_x": 0, "lin_vel_y": 1, "ang_vel_yaw": 2, "base_height": 5,
               "ang_roll": 6, "jump_height": 4}

    def __init__(self, mjcf, contract, ckpt=None, mode="policy"):
        self.c = contract
        self.mode = mode
        self.sim = MjWbr(mjcf, contract)
        self.mjcf_name = os.path.basename(mjcf)
        self.obs_b = ObsBuilder(contract)
        self.ctl = PDController(contract)
        self.pol = Policy(ckpt, contract["obs_dim"], contract["act_dim"]) if mode == "policy" and ckpt else None
        self.model_desc = self.pol.describe() if self.pol else None
        self.viewer = None
        self.follow_cam = True
        self._speed = 1.0
        self._t0 = None
        self._last_tick = None             # 【节拍】上一拍墙钟时刻（独立于 reset，见 tick()）
        self.stop_requested = False        # 【停止】桥置位，主循环下一拍即退出
        self.wheel_i = [i for i, n in enumerate(contract["dof_names"]) if "wheel" in n]
        self.slew = True                       # serve.py 打开（控制台平滑）；批量 CLI 关闭（与 run_policy 逐位一致）
        self.out_tsv = None
        self.cmds = np.zeros(7)
        self.cmd_target = np.zeros(7)
        self.reset_requested = False
        # 【S2b 跳跃】相位机实时移植（jump_template.py）：ph/off_knee/off_hip 由模板闭环生成；
        # 回放式驱动接口保留（set_off_knee/set_phase/set_jump_air 供 golden 对照）。
        self.jump = {"ph": 0, "off": 0.0, "off_hip": 0.0, "air_z": 0.0, "goal": 0.0, "latched": False}
        # 【S2c 趴下】switch 相位（外部脚本驱动，obs 尾块消费）
        self.p7t = None                      # 【S3】P7T 脚本起立（activate_p7t 激活）
        self.p7t_done = False
        self.term_pitch_ovr = None           # 【S3】姿态终止阈覆盖（uni 链 W3 段放宽给网救回窗口）
        self.term_roll_ovr = None
        # 【S4 敏感度】运行时扰动钩子（默认全 0/true＝与基准逐位一致）
        self.delay_obs = 0                   # 观测延迟（控制步）
        self.delay_act = 0                   # 动作延迟（控制步）
        self.lin_vel_mode = "true"           # true | noise | zero（base_lin_vel 的口径）
        self.lin_vel_noise = 0.02
        self._obs_hist = []                  # [(obs, ...)] 延迟缓冲
        self._act_hist = []
        self.action_ramp = None              # 交棒后动作渐入：(剩余拍数, 总拍数)，None=关闭
        self.action_ramp_from = None         # 渐入起点动作（None＝从 0 渐入）
        self.action_ramp_left = None         # 逐通道渐入剩余拍数
        self.action_ramp_total = None        # 逐通道渐入总拍数（0＝该通道不渐入）
        self.sw_state = {"ph": int(contract.get("switch_tail", {}).get("ph_start", 3)),
                         "task": int(contract.get("switch_tail", {}).get("task", 0)), "uep": 0.0}
        self.obs_b.sw_state = self.sw_state
        self.jt = None
        if contract.get("jump_template") and contract.get("with_4w") is False:
            from legged_gym.scripts.sim2sim_mujoco.deploy.jump_template import JumpTemplate
            self.jt = JumpTemplate(k_lim=tuple(contract["knee_limits"]), h_lim=tuple(contract["hip_limits"]),
                                   dt=float(contract["control_dt"]),
                                   ref_override=contract.get("jump_ref_override"))
        self.reset()

    # ------------------------------------------------------------------ 生命周期
    def reset(self):
        """回到部署冷启动：契约初值 + 零历史 + 清遥测。指令目标保留。"""
        self.sim.reset()
        self.obs_b.reset()
        self.ctl.reset()
        self.last_action = np.zeros(self.c["act_dim"], dtype=np.float32)
        self.caster_air = 0
        self.caster_air_ok = False      # 起立段置 True：从动轮离地是必然过程，不算终止
        self.rows = []
        self.steps_done = 0
        self.terminated = None
        self._t0 = None                        # 节律基准一并重置（否则复位后按旧基准狂奔）
        self._last_tick = None                 # tick() 首拍重新起锚，不受 steps_done 影响

    def request_reset(self):
        """请求复位（HTTP 线程只置 flag，主循环在 step 安全点执行——与 play02 的
        mode_shared['reset'] 同构，避免跨线程改物理/控制状态）。"""
        self.reset_requested = True

    def warmup(self, n=1):
        """起跑前零动作控制步（对齐 Isaac reset 协议；观测照常拼装，不记遥测不判终止）。"""
        for _ in range(int(n)):
            self._control(zero_action=True)
            self.steps_done += 1
            if not self.sync_viewer():
                return False
        return True

    # ------------------------------------------------------------------ 指令
    def set_cmd(self, key, value):
        """设置指令目标（key 为通道名）。目标只在 cmd_target，限速在步内生效。"""
        self.cmd_target[self.CMD_IDX[key]] = float(value)

    # ---------------- 跳跃回放接口（s2b_jump_run.py 驱动） ----------------
    def set_off_knee(self, v):
        """模板膝偏移注入（瞬时生效：control.torques 当拍读 control.off_knee）。"""
        self.ctl.off_knee = float(v)
        self.jump["off"] = float(v)

    def set_off_hip(self, v):
        self.ctl.off_hip = float(v)
        self.jump["off_hip"] = float(v)

    def set_phase(self, ph):
        self.jump["ph"] = int(ph)

    def set_jump_air(self, air_z, goal, latched):
        self.jump["air_z"] = float(air_z)
        self.jump["goal"] = float(goal)
        self.jump["latched"] = bool(latched)

    def _system_com_z(self, R, dp):
        """系统质心 z（base z ＋ 机体系质心偏移旋到世界）：flat.py _system_com_z 同语义。"""
        names = self.c["dof_names"]
        q = torch.tensor([[dp[names.index("left_hip_joint")], dp[names.index("right_hip_joint")],
                           dp[names.index("left_knee_joint")], dp[names.index("right_knee_joint")]]],
                         dtype=torch.float32)
        com = com_body_offset(self.sim.cc, q[:, 0], q[:, 1], q[:, 2], q[:, 3])
        return float(self.sim.d.qpos[2] + (R @ np.asarray(com[0].detach().numpy()))[2])

    def activate_p7t(self, params=None):
        """【S3】激活 P7T 脚本起立（四轮蹲姿起步）。done 后 runner 切两轮行走网。"""
        from legged_gym.scripts.sim2sim_mujoco.deploy.p7t_standup import P7TStandup
        self.p7t = P7TStandup(dt=float(self.c["control_dt"]), params=params)
        self.p7t_done = False

    def activate_p7t2(self, params=None):
        """激活平台式起立（轨迹前馈＋PD；四轮蹲姿起步）。接口与 P7T 同形。
        起立过程从动轮必然离地（四轮→两轮支撑转移），关闭 caster_air 终止。"""
        from legged_gym.scripts.sim2sim_mujoco.deploy.p7t2_platform_standup import P7T2PlatformStandup
        p = dict(params or {})
        # ① 静稳段保持**当前四轮站姿**（从契约取，别让控制器猜；2026-11-07 实测根因修复）
        names = self.c["dof_names"]
        p.setdefault("pose_4w", [float(self.c["default_dof_pos"][names.index(n)])
                                 for n in ("left_hip_joint", "left_knee_joint",
                                           "right_hip_joint", "right_knee_joint")])
        self.p7t = P7T2PlatformStandup(dt=float(self.c["control_dt"]), params=p)
        self.p7t_done = False
        self.caster_air_ok = True
        self.caster_air = 0
        # 这里**不能**预置 off_knee=-0.1：四轮契约的 off_knee_const 是 0，
        # 进 RISE 第一拍膝目标就跳 0.1 rad ⇒ 机体瞬间前倾 8.8°（实测），轮矩律随即前驱"纠正"，
        # 整机先前进 ~0.29 m 再回摆——"前进后晃"的根源就在这里。
        # 交棒偏置改在 `_enter("IDLE2W")` 里设（2W 契约本身 off_knee_const=-0.1），并在
        # action_ramp_from 里补偿掉，交棒瞬间目标仍然连续。
        self.ctl.off_knee = 0.0

    def set_switch_phase(self, ph, ph_t=None):
        """【S2c 趴下】外部相位脚本驱动（console 趴下时序语义）。"""
        self.sw_state["ph"] = int(ph)

    def jump_request(self, h):
        """装填一跳（评测口径：写 h_request＋is_armed）。"""
        if self.jt is not None:
            self.jt.request(float(h))
        self.jump["goal"] = float(h)

    def _jump_prog(self, R, dp):
        """离地进度：与 flat.py 同语义 (com_z − air_z)/goal × (ph>0) × air_latched。"""
        j = self.jump
        if j["ph"] == 0 or not j["latched"] or j["goal"] <= 1e-6:
            return 0.0
        names = self.c["dof_names"]
        q = torch.tensor([[dp[names.index("left_hip_joint")], dp[names.index("right_hip_joint")],
                           dp[names.index("left_knee_joint")], dp[names.index("right_knee_joint")]]],
                         dtype=torch.float32)
        com = com_body_offset(self.sim.cc, q[:, 0], q[:, 1], q[:, 2], q[:, 3])
        com_z = float(self.sim.d.qpos[2] + (R @ np.asarray(com[0].detach().numpy()))[2])
        return float(np.clip((com_z - j["air_z"]) / j["goal"], 0.0, 1.0))

    def snap_cmds(self):
        """初始指令一步到位（不经限速）：冷启动的 h/vx 必须落在训练分布内，
        限速只该用于运行中的滑条变化。"""
        self.cmds[:] = self.cmd_target

    def _slew(self):
        """vx/h 限速渐进，其余通道直通。slew=False 时整体直通（批量模式）。"""
        if not self.slew:
            self.cmds[:] = self.cmd_target
            return
        dt = float(self.c["control_dt"])
        for idx, rate in ((0, SLEW_VX), (5, SLEW_H)):
            dmax = rate * dt
            self.cmds[idx] += float(np.clip(self.cmd_target[idx] - self.cmds[idx], -dmax, dmax))
        for idx in (1, 2, 4, 6):
            self.cmds[idx] = self.cmd_target[idx]

    # ------------------------------------------------------------------ 一拍
    def _control(self, zero_action=False):
        """一个控制步的物理部分：状态捕获 → 观测 → 动作 → 力矩。"""
        sim = self.sim
        R, q, lin_b, ang_b = sim.base_state()
        pg = sim.projected_gravity(R)
        dp, dv = sim.dof_pos(), sim.dof_vel()
        forces = sim._contact_forces()
        caster = sim.caster_contacts(forces)
        tip = sim.tip_margin(dp)

        if self.jt is not None:
            fz = self.sim._contact_forces()
            roll_t, pitch_t = self.sim.body_pitch_roll(pg)
            dpd = {n: float(dp[i]) for i, n in enumerate(self.c["dof_names"])}
            com = self._system_com_z(R, dp)
            out = self.jt.update(z=float(self.sim.d.qpos[2]), vz=float(self.sim.d.qvel[2]),
                                 fz_l=fz.get(self.sim.feet[0], 0.0), fz_r=fz.get(self.sim.feet[1], 0.0),
                                 roll=roll_t, pitch=pitch_t, com_z=com, dof_pos=dpd)
            self.jump.update(ph=out["ph"], off=out["off_knee"], off_hip=out["off_hip"],
                             air_z=out["air_z"], goal=out["goal"], latched=out["latched"])
            self.ctl.off_knee = out["off_knee"]
            self.ctl.off_hip = out["off_hip"]
            self.ctl.set_phase_gains(out["knee_damp_extra"], out["knee_kp_extra"])
            self.cmds[4] = out["h_cmd_obs"]
        if self.p7t is not None and not self.p7t_done:
            # 【S3 P7T】脚本驱动：default=脚本角、腿 P×boost、轮恒力矩；act 强制 0
            z_now = float(self.sim.d.qpos[2])
            vz_now = float(self.sim.d.qvel[2])
            _pitch_t, _roll_t = self.sim.body_pitch_roll(pg)
            # 注意：MJ 自由关节 qvel[4] 与 pitch 反号（pitch = asin(-pg[0])）⇒ 取负
            o = self.p7t.step(z=z_now, body_pitch=float(_pitch_t), vz=vz_now,
                              pitch_rate=-float(ang_b[1]), roll=float(_roll_t),
                              vx=float(lin_b[0]))
            legs4 = [self.c["dof_names"].index(n) for n in
                     ("left_hip_joint", "left_knee_joint", "right_hip_joint", "right_knee_joint")]
            self.ctl.leg_override = o["default_leg"]
            for k, i in enumerate(legs4):
                self.ctl.default[i] = float(o["default_leg"][k])
            self.ctl.off_hip = float(o.get("off_hip", 0.0))
            self.ctl.leg_boost = float(o["boost"])
            # flat.py 语义：p_gains[leg_dof] = 名义 × boost（四条腿 dof 直写）
            for i in legs4:
                self.ctl.kp[i] = self.ctl.kp0[i] * o["boost"]
            if "wheel_torque_l" in o:      # 左右差动（roll 控制）
                self.ctl.wheel_override_l = o["wheel_torque_l"]
                self.ctl.wheel_override_r = o["wheel_torque_r"]
                self.ctl.wheel_override = o["wheel_torque_l"]   # 兼容：取左值
            else:
                self.ctl.wheel_override = o["wheel_torque"]
                self.ctl.wheel_override_l = None
                self.ctl.wheel_override_r = None
            self.last_action = np.zeros(self.c["act_dim"], dtype=np.float32)
            if o["done"]:
                self.p7t_done = True
                self.ctl.leg_override = None
                self.ctl.wheel_override = None
                self.ctl.wheel_override_l = None   # 差动残留会压住 W3 轮速环 → 后翻
                self.ctl.wheel_override_r = None
                self.ctl.leg_boost = 1.0
                self.ctl.off_hip = 0.0
                # default 沿用**起立脚本最后一拍的腿角**，不硬编码两轮站姿。
                # ⚠ 不能写死 [1.0,-2.1]：RISE 终点腿在 1.415/−2.113（W3 稳态姿），交棒瞬间
                # default 若跳到 1.0 ⇒ CoM 后移 8 cm ⇒ 机体立刻后仰落到从动轮上，W3 再也起不来。
                for k, i in enumerate(legs4):
                    self.ctl.default[i] = float(o["default_leg"][k])
        if self.c.get("switch_tail"):
            roll_s, pitch_s = self.sim.body_pitch_roll(pg)
            self.sw_state["uep"] = float(np.clip(pitch_s / 0.35, -1.0, 1.0))   # UEP 代理（俯仰/0.35 归一）
        if self.lin_vel_mode == "zero":
            lin_b = np.zeros(3)
        elif self.lin_vel_mode == "noise":
            lin_b = np.asarray(lin_b) + np.random.normal(0.0, self.lin_vel_noise, 3)
        j = self.jump
        prog = self._jump_prog(R, dp) if j["ph"] != 0 else 0.0
        obs = self.obs_b.build(lin_vel=lin_b, ang_vel=ang_b, gravity=pg, commands=self.cmds,
                               dof_pos=dp, dof_vel=dv, last_action=self.last_action,
                               caster_contacts=caster, tip_margin=tip, phase=j["ph"],
                               off_knee=j["off"], prog=prog)
        self.obs_b.push(dp, dv)

        if self.delay_obs > 0:               # 【S4】观测延迟（队列长度＝延迟步数）
            self._obs_hist.append(obs)
            if len(self._obs_hist) > self.delay_obs:
                obs = self._obs_hist.pop(0)

        if zero_action or self.pol is None:
            act = np.zeros(self.c["act_dim"], dtype=np.float32)
        else:
            act = self.pol.act(obs)[0].numpy()
            if self.action_ramp is not None:   # 交棒后动作渐入
                if self.action_ramp == "hold1s_then_ramp":
                    # 特殊模式：前 1s 完全零动作（腿 PD 锚住），之后 0.5s 线性渐入
                    if not hasattr(self, "_ramp_start") or self._ramp_start is None:
                        self._ramp_start = self.steps_done
                    off = self.steps_done - self._ramp_start
                    if off < 100:            # 1s 零动作
                        act = np.zeros_like(act)
                    else:
                        r = min((off - 100) / 50.0, 1.0)
                        act = act * r
                        if r >= 1.0:
                            self.action_ramp = None
                            self._ramp_start = None
                else:
                    left, total = self.action_ramp
                    r = 1.0 - left / total
                    _from = getattr(self, "action_ramp_from", None)
                    _left = getattr(self, "action_ramp_left", None)
                    if _from is not None and _left is not None:
                        # 从**起立末态对应的动作值**过渡到策略动作。
                        # · 腿：2 s 长渐入——直接 act*r 会让 r=0 时目标=default（1.0/−2.1），
                        # 腿瞬间后移 8 cm ⇒ CoM 后移 ⇒ 机体后仰落到从动轮上，W3 起不来；
                        # · 轮：0.3 s 短渐入——轮是平衡执行器，交棒瞬间保持 −32 rad/s（起立末态）
                        # 等于没有平衡，机体会无阻尼后仰；也不能直接清零（−2 m/s 硬刹 → 后仰 30°）。
                        _tot = np.asarray(self.action_ramp_total, dtype=np.float32)
                        _r = np.where(_tot > 0,
                                      np.clip(1.0 - _left / np.maximum(_tot, 1.0), 0.0, 1.0),
                                      1.0).astype(np.float32)
                        act = _from * (1.0 - _r) + act * _r
                        self.action_ramp_left = np.maximum(_left - 1, 0)
                    elif _from is not None:
                        act = _from * (1.0 - r) + act * r
                    else:
                        act = act * r
                    self.action_ramp = (left - 1, total) if left > 0 else None
                    if self.action_ramp is None:
                        self.action_ramp_from = None
                        self.action_ramp_left = None
                        self.action_ramp_total = None
        if self.delay_act > 0:               # 【S4】动作延迟
            self._act_hist.append(act)
            if len(self._act_hist) > self.delay_act:
                act = self._act_hist.pop(0)
        self.last_action = act

        tau = None
        for _ in range(sim.decimation):
            tau = self.ctl.torques(act, sim.dof_pos(), sim.dof_vel()).numpy()
            sim.apply(tau)
        return dict(lin_b=lin_b, ang_b=ang_b, pg=pg, dp=dp, dv=dv,
                    caster=caster, tip=tip, tau=tau, forces=forces)

    def step(self, record=True):
        """一个带遥测与终止判定的控制步。终止不抛异常，写 self.terminated 由调用方收尾。"""
        if self.reset_requested:              # 复位在 step 安全点执行（主循环线程）
            self.reset_requested = False
            self.reset()
            self.warmup(1)
        self._slew()
        snap = self._control()
        self.steps_done += 1

        pg, dv, caster, tip, tau = snap["pg"], snap["dv"], snap["caster"], snap["tip"], snap["tau"]
        t = len(self.rows)
        pitch, roll = self.sim.body_pitch_roll(pg)
        if record:
            row = dict(
                t=round(t * float(self.c["control_dt"]), 3),
                cmd_vx=float(self.cmd_target[0]), cmd_yaw=float(self.cmd_target[2]), cmd_h=float(self.cmd_target[5]),
                vx=float(snap["lin_b"][0]), vy=float(snap["lin_b"][1]), wz=float(snap["ang_b"][2]),
                x=float(self.sim.d.qpos[0]), y=float(self.sim.d.qpos[1]), z=float(self.sim.d.qpos[2]),
                roll=roll, pitch=pitch,
                caster_l=float(caster[0]), caster_r=float(caster[1]), tip=tip,
                wheel_l=float(dv[self.wheel_i[0]]), wheel_r=float(dv[self.wheel_i[1]]),
                tau_max=float(np.abs(tau).max()),
                # 【绘图页遥测】vz（竖直速度）/ cmd_roll（滚转指令）/ dbg_ph（跳跃相位 0-5）
                vz=float(self.sim.d.qvel[2]), cmd_roll=float(self.cmd_target[6]),
                dbg_ph=int(self.jump["ph"]),
            )
            # 按 Isaac 遥测**同列名**补全部 dof × 6 列
            # （pos_tgt/pos_act/vel_tgt/vel_act/tau_tgt/tau_act，Isaac 对所有 dof 都写 6 列）。
            # ⚠ 必须**逐 dof_names（8 个，含 caster）**而不是 action_dof_names（6 个）：
            # plot02 按 #joints 建面板，`left_knee_caster_joint` 名字里含 "knee" 会被归为
            # 膝类生成"左膝 速度/扭矩"面板（caster 非腿 → 取 vel_* 列）——缺列就显示
            # "无该列数据"（MJ 右髋区域两个空面板实为 caster 面板）。
            # 目标值复现：PD 目标 = default + scale·act + off_knee/off_hip（与 control.torques
            # 同式）；轮 dof 目标走 ctl.wheel_target（速度环）；caster 无执行器 ⇒ 目标=实际。
            _dp, _dv2 = self.sim.dof_pos(), self.sim.dof_vel()
            _act = np.asarray(self.last_action, dtype=float)
            _adi = self.ctl.adi
            _scale = self.ctl.scale_vec.numpy()
            _offk = float(self.ctl.off_knee)
            _offh = float(self.ctl.off_hip)
            _wtgt = {i: float(self.ctl.wheel_target[k])
                     for k, i in enumerate(self.ctl.wheel_dof)}
            _act_of = {self.ctl.adi[_j]: _j for _j in range(len(self.ctl.adi))}
            for _i, _n in enumerate(self.c["dof_names"]):
                _j = _act_of.get(_i)
                if "wheel" in _n:
                    _vt = _wtgt.get(_i, 0.0)
                elif _j is None:
                    _vt = float(_dp[_i])            # caster 等无执行器 dof：目标=实际
                else:
                    _vt = (float(self.ctl.default[_i]) + float(_scale[_j]) * float(_act[_j]))
                    if _offk and "knee" in _n:
                        _vt += _offk if "left" in _n else -_offk
                    if _offh and "hip" in _n:
                        _vt += _offh if "left" in _n else -_offh
                row[f"pos_tgt__{_n}"] = float(_dp[_i]) if ("wheel" in _n or _j is None) else _vt
                row[f"pos_act__{_n}"] = float(_dp[_i])
                row[f"vel_tgt__{_n}"] = _vt if "wheel" in _n else 0.0
                row[f"vel_act__{_n}"] = float(_dv2[_i])
                row[f"tau_tgt__{_n}"] = float(tau[_i])
                row[f"tau_act__{_n}"] = float(tau[_i])
            self.rows.append(row)

        # 终止判定用**步首**接触力（与旧 run_policy 逐位一致：遥测与终止同拍）
        forces = snap["forces"]
        base_f = forces.get(self.sim.base, 0.0)
        thigh_f = max(forces.get(b, 0.0) for b in self.sim.term_bodies if b != self.sim.base)
        if base_f > 1.0 or thigh_f > 1.0:
            self.terminated = f"contact(base={base_f:.1f},thigh={thigh_f:.1f})"
        elif abs(pitch) > (self.term_pitch_ovr or (1.0 if self.p7t is not None else 0.5)):
            # 【S3 P7T】蹲稳段机体后仰可达 ~46°（Isaac 同款），终止阈放宽；交棒后由 runner 恢复
            self.terminated = f"pitch({pitch:.2f})"
        elif abs(roll) > (self.term_roll_ovr or 0.6):
            self.terminated = f"roll({roll:.2f})"
        elif (self.sim.has_caster and self.c.get("caster_required", True)
              and not getattr(self, "caster_air_ok", False) and not caster.any()):
            self.caster_air += 1
            if self.caster_air > int(0.4 / float(self.c["control_dt"])):
                self.terminated = "caster_air"
        elif self.sim.has_caster:
            self.caster_air = 0

    # ------------------------------------------------------------------ viewer
    def attach_viewer(self, speed=1.0):
        import mujoco.viewer
        try:
            self.viewer = mujoco.viewer.launch_passive(self.sim.m, self.sim.d)
        except Exception as e:  # 无显示/无 GL 时给明确提示，不静默退化
            raise SystemExit(f"无法打开 MuJoCo 窗口（DISPLAY={os.environ.get('DISPLAY')}）：{e}")
        self.viewer.cam.distance = 1.7
        self.viewer.cam.elevation = -12.0
        self.viewer.cam.azimuth = 90.0
        self._speed = float(speed)
        self._t0 = time.time()
        print("[viewer] 已打开：拖动可改视角，空格暂停，Esc/关窗退出", flush=True)

    def sync_viewer(self):
        """每控制步一次：相机跟随 + sync + 墙钟节律（仅窗口模式；CLI 无窗口不节流跑满速）。
        返回 False = 窗口已关，调用方应收尾。"""
        v = self.viewer
        if v is None:
            return True
        if not v.is_running():
            return False
        if self.follow_cam:
            v.cam.lookat[:] = self.sim.d.qpos[0:3]
        v.sync()
        if self._t0 is None:
            self._t0 = time.time()
        target = (self.steps_done) * float(self.c["control_dt"]) / max(self._speed, 1e-3)
        dt_sleep = target - (time.time() - self._t0)
        if dt_sleep > 0:
            time.sleep(dt_sleep)
        return True

    def tick(self):
        """控制台节拍（serve.py 用）：无窗口时也按墙钟实时节律（否则滑条驱动的世界在 13 倍速狂奔）；
        有窗口时 sync_viewer 自带节律。返回 False = 窗口已关 / 收到停止请求。

        ⚠ 节律基准必须**独立于 reset()**：以前用 `steps_done × dt` 对 `_t0` 求差，而 reset()
        会把 `_t0` 置 None 但不清 steps_done ⇒ 下一拍重设基准后 target 巨大、dt_sleep 恒为负
        ⇒ 主循环空转（实测 CPU 856%、SIGINT 来不及处理、停止要 21 s）。
        现改为"基准 = now − steps_done×dt"，无论中途 reset 多少次，剩余等待量都只由步数差决定。"""
        if self.stop_requested:
            return False
        if self.viewer is None:
            # 每拍重算基准（O(1)），等价于"从本拍起还欠多少墙钟时间"
            elapsed_needed = float(self.c["control_dt"]) / max(self._speed, 1e-3)
            # 用上一拍的墙钟时刻做锚，避免累计漂移；首拍/复位后直接放行一拍
            now = time.time()
            if self._last_tick is None:
                self._last_tick = now
            else:
                dt_sleep = elapsed_needed - (now - self._last_tick)
                if dt_sleep > 0:
                    time.sleep(dt_sleep)
                self._last_tick = time.time()
        return self.sync_viewer()

    def close_viewer(self):
        """GLFW/NVIDIA 在解释器退出阶段段错误（窗口/模型已正常关过）：硬退避开析构。
        必须在遥测落盘与 stdout flush 之后调用。"""
        if self.viewer is None:
            return
        self.viewer.close()
        sys.stdout.flush()
        os._exit(0 if self.terminated is None else 1)

    # ------------------------------------------------------------------ 产物
    def summary(self, dur_target=None):
        rows = self.rows
        arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]} if rows else {}
        return {
            "mode": self.mode,
            "cmd": {"vx": float(self.cmd_target[0]), "vy": float(self.cmd_target[1]),
                    "yaw": float(self.cmd_target[2]), "h": float(self.cmd_target[5])},
            "dur_target": dur_target, "steps": len(rows), "terminated": self.terminated,
            "policy": self.model_desc,
            "mean": {k: float(arr[k].mean()) for k in ("vx", "vy", "wz", "z", "roll", "pitch") if k in arr},
            "p95": {k: float(np.percentile(np.abs(arr[k]), 95)) for k in ("vx", "vy", "wz", "roll", "pitch") if k in arr},
            "z_range": [float(arr["z"].min()), float(arr["z"].max())] if "z" in arr else None,
            "caster_contact_frac": [float(arr["caster_l"].mean()), float(arr["caster_r"].mean())] if "caster_l" in arr else None,
            "tip_min": float(arr["tip"].min()) if "tip" in arr else None,
            "travel": [float(arr["x"][-1]), float(arr["y"][-1])] if "x" in arr else None,
            "tau_max": float(arr["tau_max"].max()) if "tau_max" in arr else None,
        }

    def dump(self, out=None, summary=None):
        """遥测 TSV + summary.json 落盘，返回 run 目录。"""
        summary = summary if summary is not None else self.summary()
        out = out or os.path.join(PKG, "eval/runs",
                                  time.strftime("%Y%m%d-%H%M%S") + f"_{self.mode}_vx{self.cmd_target[0]}_h{self.cmd_target[5]}")
        os.makedirs(out, exist_ok=True)
        if self.rows:
            tsv = os.path.join(out, "telemetry.tsv")
            with open(tsv, "w", encoding="utf-8") as f:
                f.write("\t".join(self.rows[0].keys()) + "\n")
                for r in self.rows:
                    f.write("\t".join(str(r[k]) for k in r) + "\n")
            self.out_tsv = tsv
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        return out


def load_contract(path=None):
    contract = json.load(open(path or os.path.join(PKG, "assets/deploy_contract.json"), encoding="utf-8"))
    return contract


def resolve_model(contract, release=None, model=None):
    """release 目录（相对 REPO_ROOT 或绝对）+ 模型文件名 → ckpt 绝对路径。"""
    rel = release or contract["sources"]["release"]
    if not os.path.isabs(rel):
        rel = os.path.join(ROOT, rel)
    name = model or contract["sources"]["release_model"]
    return os.path.join(rel, name) if not os.path.isabs(name) else name
