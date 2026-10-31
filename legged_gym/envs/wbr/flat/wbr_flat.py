# ==============================================================================
# Date      : 2026-04-07
# Author    : SYH
# File      : wbr_flat.py
# Purpose   : 两轮 WBR 的环境本体：观测拼装、奖励、指令目标层（包络 / 压弯 / 限速）、
#             跳跃相位机与遥测快照。基类为 LeggedRobot，flat4 / switch / uni 均由它派生。
# ==============================================================================

import numpy as np
import torch
from isaacgym.torch_utils import *
from legged_gym.utils.math import *
from legged_gym.envs import LeggedRobot
from isaacgym import gymapi, gymtorch, gymutil
from .wbr_flat_config import WBRFlatCfg

# 转向指令对外一律「正数 = 右转」，而 base_ang_vel[:,2] 绕 +z（正 = 左转）⇒ 对齐时乘它。
YAW_SIGN = -1.0

# 三套高度口径，不要混读：
#   起飞脉冲 h_flight_pred = max(air_vz, 0)² / 2g  → 用**质心**
#   奖励 tracking_base_height / IK 几何高 / 蹲姿判定 → 用**机身**（腿伸量相对轮子）
#   验收真跳高 → 用**轮底离地**（轮心 z − 轮半径，双轮同离地窗内的峰值）
# ⚠ 机身只占 35.7% 质量、会被腿伸缩抬起来 ⇒ 机身读数必须与"确认起飞 n/N"并读，不能当离地高度。
COM_M = {"base": 2.5418, "thigh": 0.80815,
         "shank_l": 0.93138, "shank_r": 0.92691, "wheel": 0.55336}
COM_G = {"base": (4.4472e-05, 1.156e-04, 0.010416),
         "thigh_l": (0.15659, 3.2005e-04, 0.02025),
         "thigh_r": (0.15659, 1.9029e-04, 0.02025),
         "shank_l": (0.12156, 4.8719e-03, 1.996e-02),
         "shank_r": (0.12114, 5.1863e-03, 2.0146e-02),
         "wheel_l": (0.0, 0.0, 0.025381), "wheel_r": (0.0, 0.0, 0.025381)}
COM_TOTAL = (COM_M["base"] + 2 * COM_M["thigh"] + COM_M["shank_l"]
             + COM_M["shank_r"] + 2 * COM_M["wheel"])  # 7.12311 kg
COM_HIP_Y, COM_KNEE_O, COM_WHEEL_O = 0.0936, (0.2, 0.0, 0.0403), (0.18, 0.0, 0.0403)


def _com_rot_z(q):
    """[n] 的绕自身 +z 转角 → [n,3,3]。三行按 dim=1 堆叠（按 -1 堆会得到转置，踩过）。"""
    c, s = torch.cos(q), torch.sin(q)
    o, z = torch.ones_like(q), torch.zeros_like(q)
    return torch.stack([torch.stack([c, -s, z], -1),
                        torch.stack([s, c, z], -1),
                        torch.stack([z, z, o], -1)], 1)


def _com_hip_pre(side, dev):
    """髋关节固定 rpy（左 (−π/2,π/2,0)、右 (π/2,π/2,0)）；cos(±π/2)=0 ⇒ 直接写常量。"""
    sgn = 1.0 if side == "r" else -1.0
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -sgn], [0.0, sgn, 0.0]],
                      dtype=torch.float32, device=dev)
    ry = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
                      dtype=torch.float32, device=dev)
    return ry @ rx


def _com_ev(R, v):
    """[n,3,3] 作用到常量 [3] 或逐环境 [n,3]，返回 [n,3]。"""
    if v.dim() == 1:
        return torch.einsum("nij,j->ni", R, v.to(R.device))
    return torch.einsum("nij,nj->ni", R, v)


def com_constants(dev):
    """质心 FK 需要的常量件（每设备建一次）。单独成函数是为了能被零位标定脚本直接调用比对。"""
    mk = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)
    return {"base": mk(COM_G["base"]), "o_knee": mk(COM_KNEE_O),
            "hip_l": mk((0.0, COM_HIP_Y, 0.0)), "hip_r": mk((0.0, -COM_HIP_Y, 0.0)),
            "pre_l": _com_hip_pre("l", dev), "pre_r": _com_hip_pre("r", dev),
            "o_wheel_wh": mk([a + b for a, b in zip(COM_WHEEL_O, COM_G["wheel_l"])]),
            **{k: mk(COM_G[k]) for k in ("thigh_l", "thigh_r", "shank_l", "shank_r")}}


def com_body_offset(c, hip_l, hip_r, knee_l, knee_r):
    """机体系质心偏移 [n,3]；c 是 `_system_com_z` 缓存的常量件。两轮重心恰在自转轴上 ⇒ 与轮角无关。"""
    n = hip_l.shape[0]
    acc = c["base"].expand(n, 3).clone() * COM_M["base"]
    for s, qh, qk in (("l", hip_l, knee_l), ("r", hip_r, knee_r)):
        th = c["pre_" + s] @ _com_rot_z(qh)                 # 大腿系 → 机体系
        sh = th @ _com_rot_z(qk)                            # 小腿系 → 机体系
        p_th = c["hip_" + s] + _com_ev(th, c["thigh_" + s])
        p_sh = c["hip_" + s] + _com_ev(th, c["o_knee"]) + _com_ev(sh, c["shank_" + s])
        p_wh = c["hip_" + s] + _com_ev(th, c["o_knee"]) + _com_ev(sh, c["o_wheel_wh"])
        acc = acc + COM_M["thigh"] * p_th + COM_M["shank_" + s] * p_sh + COM_M["wheel"] * p_wh
    return acc / COM_TOTAL


def com_world_dz(quat, com_body):
    """机体系偏移投影到世界 z。**必须用完整四元数**（偏航进得来）：只取 roll/pitch 会把
    横向/前后偏移错投影（偏航 90° 时误差量级 110 mm）。
    quat 用 Isaac Gym 的 (x, y, z, w) 顺序。"""
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return ((2 * (qx * qz - qw * qy)) * com_body[:, 0]
            + (2 * (qy * qz + qw * qx)) * com_body[:, 1]
            + (1 - 2 * (qx * qx + qy * qy)) * com_body[:, 2])


class WBRFlatMob(LeggedRobot):
    # 继承 LeggedRobot，实现 WBR 的奖励函数与观测构造。
    # ⚠ 上游 `_prepare_reward_function` 只遍历 `rewards.scales` 的键并统一 ×dt ⇒ **函数写了
    #   但键不在 scales 里 = 整项静默不存在**；同名方法后定义的覆盖先定义的。
    # 奖励项索引：终止安全(no_fly)｜指令跟踪(tracking_*)｜姿态平顺(orientation, lin_vel_z,
    #   ang_vel_xy, wrong_way, no_moonwalk, *_action_rate)｜关节执行(z_adjust_leg, dof_*_limits,
    #   torque_limits, torques)｜跳跃(jump_lift, jump_*, hip_reach)。权重见 config 的 scales。

    def _reward_leg_action_rate(self):
        # 腿部沿用原始 action 空间中的平滑惩罚。
        action_delta = self.actions[:, self.leg_indices] - self.last_actions[:, self.leg_indices]
        # 【P6 S1】爆发段免罚：模板那一拍要的就是"动作目标一步跳 1 rad"，罚它＝向执行模板收税
        return torch.sum(torch.square(action_delta), dim=1) * (
            ~self._jump_blocked("leg_action_rate"))

    def _reward_wheel_action_rate(self):
        # 轮子按实际速度目标变化量惩罚，单位为 rad/s。
        target_delta = (
            self.actions[:, self.wheel_indices] - self.last_actions[:, self.wheel_indices]
        ) * self.cfg.control.wheel_action_scale
        # 腾空／够地两段免罚：那两拍轮子没有附着力可言，罚轮速跳变＝向腾空姿态收税，
        #   而这两拍正是 `jump_lift` 唯一的发放处（净和为负会被 only_positive_rewards 压成 0）。
        return torch.sum(torch.square(target_delta), dim=1) * (
            ~self._jump_blocked("wheel_action_rate"))

    def _jump_blocked(self, term):
        """本控制步这项奖励是否被跳跃相位关闭（True ⇒ 这一拍乘 0）。表在 `wbr_flat_config.py::rewards.jump.mask`。

        """
        vec = self._jump_mask_vec.get(term)
        if vec is None or not getattr(self.cfg.rewards.flags, "enable_jump", False):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return vec[self.ph]

    def _reward_torques(self):
        # 基类实现 ＋ 按相位掩蔽（爆发段允许力矩顶到限幅，见 mask.torques）
        return torch.sum(torch.square(self.torques), dim=1) * (~self._jump_blocked("torques"))

    def _reward_dof_acc(self):
        # 基类实现（(last_dof_vel − dof_vel)/dt，本 fork 没有 dof_acc 缓冲区）＋ 按相位掩蔽
        dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        return torch.sum(torch.square(dof_acc), dim=1) * (~self._jump_blocked("dof_acc"))

    # ==================== [高度/滚转角解算器] ====================

    def _body_pitch(self):
        """机体俯仰角 p（rad）。与 _compute_observations 里 projected_gravity 的取法一致。"""
        return torch.asin(torch.clamp(-self.projected_gravity[:, 0], -1.0, 1.0))

    def _body_roll(self):
        """机体滚转角 φ（rad）：pg_y = −sin φ ⇒ φ = asin(−pg_y)。⚠ 不能用 `atan2(−pg_y,−pg_x)`——站立时 pg_x≈0 会给出 ±π，把相位机的 `|roll|<0.35` 门永久挡住。

        """
        return torch.asin(torch.clamp(-self.projected_gravity[:, 1], -1.0, 1.0))

    def _ik_knee_for_leg(self, target_height, hip, knee_cur, knee_idx, sign):
        """单腿几何 IK：目标机身高度 → 该腿膝角。`H = r + Lt·cos(hip∓p) + Ls·cos(hip∓p+knee)`。
        cos 反解有内外折两支且常同时在限位内 ⇒ 取**离当前膝角最近**的一支（固定取 acos 主值会让高度微调时目标跳支）。
        """
        rw = self.cfg.rewards.wheel_radius
        lt = self.cfg.rewards.thigh_length
        ls = self.cfg.rewards.shank_length
        p = self._body_pitch()
        thigh = hip - sign * p
        cos_shank = (target_height - rw - lt * torch.cos(thigh)) / ls
        a = torch.acos(torch.clamp(cos_shank, -1.0, 1.0))
        # H = r + Lt·cos(thigh) + Ls·cos(thigh + knee) ⇒ knee = ±a − thigh
        k1 = a - thigh
        k2 = -a - thigh
        tgt = torch.where(torch.abs(k1 - knee_cur) <= torch.abs(k2 - knee_cur), k1, k2)
        lo = self.dof_pos_limits[knee_idx, 0]
        hi = self.dof_pos_limits[knee_idx, 1]
        return torch.clamp(tgt, lo, hi)

    def _reach_max_height(self, hip, knee_idx, sign):
        """给定实测髋角时该腿能撑到的最大高度（膝角受本腿限位约束）。

        """
        ls = self.cfg.rewards.shank_length
        rw = self.cfg.rewards.wheel_radius
        lt = self.cfg.rewards.thigh_length
        thigh = hip - sign * self._body_pitch()
        lo = thigh + self.dof_pos_limits[knee_idx, 0]
        hi = thigh + self.dof_pos_limits[knee_idx, 1]
        # max cos(膝绝对角)：区间含 0（小腿竖直）则取 1，否则取离 0 近的端点
        near_zero = torch.where((lo <= 0) & (hi >= 0), torch.zeros_like(lo),
                                torch.where(torch.abs(lo) < torch.abs(hi), lo, hi))
        return rw + lt * torch.cos(thigh) + ls * torch.cos(near_zero)

    def _compute_target_knee_angles(self):
        """统一膝角目标解算器：IK + 侧倾差动 + 跳跃相位偏移 + 硬限幅。返回 (左膝目标, 右膝目标)。
        ⚠ **两个通道符号相反**：侧倾差动同号加两腿，跳跃偏移镜像（左 +off、右 −off）。
        """
        # 1. 基础膝角：用各腿实测髋角与机体俯仰反解，并夹到各自限位
        hip_l = self.dof_pos[:, self.hip_l_idx]
        hip_r = self.dof_pos[:, self.hip_r_idx]
        h_ceil = torch.minimum(self._reach_max_height(hip_l, self.knee_l_idx, +1.0),
                               self._reach_max_height(hip_r, self.knee_r_idx, -1.0))
        h_cmd = torch.minimum(self.commands[:, 5], h_ceil)
        self.dbg["h_ik_in"][:] = h_cmd          # 真正喂给 IK 的高度（可能已被夹）
        self.dbg["h_ceiling"][:] = h_ceil       # 当前髋＋俯仰下的几何可达上限
        base_l = self._ik_knee_for_leg(h_cmd, hip_l, self.dof_pos[:, self.knee_l_idx],
                                       self.knee_l_idx, +1.0)
        base_r = self._ik_knee_for_leg(h_cmd, hip_r, self.dof_pos[:, self.knee_r_idx],
                                       self.knee_r_idx, -1.0)

        # 2. 侧倾参考＝指令 Roll。压弯偏置已由 `_update_turning_roll_targets` 合成进
        #    `commands[:,6]`，这里不能再加第二次。
        roll_cmd = self.commands[:, 6]

        # 3. 侧倾差动 = 前馈 + 反馈。前馈增益随髋膝工况现算 δ = λ_eff·tan(φ)/(g_L+g_R)，
        #    分母下限由 roll_gain_denom_floor 夹住，防膝目标打到限位。
        actual_roll = self._body_roll()
        ls = self.cfg.rewards.shank_length
        g_l = ls * torch.abs(torch.sin(hip_l + self.dof_pos[:, self.knee_l_idx]))
        g_r = ls * torch.abs(torch.sin(hip_r + self.dof_pos[:, self.knee_r_idx]))
        ff_gain = self.cfg.rewards.roll_lever_eff / torch.clamp(
            g_l + g_r, min=self.cfg.rewards.roll_gain_denom_floor)
        delta = (ff_gain * torch.tan(roll_cmd)
                 + self.cfg.rewards.K_roll_fb * (roll_cmd - actual_roll))

        # 4. 跳跃相位对称偏移（P6：模板由 _update_jump_phase 每控制步推进一次，这里只读 ⇒
        #    谁调用本函数、调用几次都不影响相位时钟）
        off_knee = torch.zeros_like(base_l)
        if getattr(self.cfg.rewards.flags, "enable_jump", False):
            off_knee = self._compute_jump_phase_offset()

        # 5. 合成 + 按各腿限位截断。⚠ **两个通道符号相反**：侧倾差动**同号**加两腿，
        #    跳跃偏移**镜像**（左 +off、右 −off）——反了就变成一腿伸一腿折。
        target_l = torch.clamp(base_l + delta + off_knee,
                               self.dof_pos_limits[self.knee_l_idx, 0],
                               self.dof_pos_limits[self.knee_l_idx, 1])
        target_r = torch.clamp(base_r + delta - off_knee,
                               self.dof_pos_limits[self.knee_r_idx, 0],
                               self.dof_pos_limits[self.knee_r_idx, 1])
        # 只读快照（不参与计算）：膝目标、侧倾差动、现算增益
        self.dbg["knee_tgt_l"][:] = target_l
        self.dbg["knee_tgt_r"][:] = target_r
        self.dbg["roll_delta"][:] = delta
        self.dbg["ff_gain"][:] = ff_gain
        return target_l, target_r

    def _compute_jump_phase_offset(self):
        """跳跃相位模板的当前偏移（rad，正值＝两腿同时伸长；调用处按镜像反号加到左右膝）。

        """
        return self.off_knee

    def _system_com_z(self):
        """系统质心的世界系高度 [n]（跳跃升高量唯一的弹体口径）。常量件首拍缓存，之后每拍只算 FK。"""
        if getattr(self, "_com_c", None) is None:
            self._com_c = com_constants(self.device)
        q = self.dof_pos
        cb = com_body_offset(self._com_c, q[:, self.hip_l_idx], q[:, self.hip_r_idx],
                             q[:, self.knee_l_idx], q[:, self.knee_r_idx])
        return self.root_states[:, 2] + com_world_dz(self.root_states[:, 3:7], cb)

    def _jump_2r_solve(self, dh):
        """腿竖直伸量目标 ΔL → 髋＋膝两自由度解，约束"轮子相对髋的水平偏移"。
        `anchor_x_com` 给常数（解算器口径，＝−质心x的 FK 口径）⇒ 锚在写死位置；
        给 `None` ⇒ 用触发帧冻结值（历史行为）。
        闭式解：`D = hypot(x, L*)`；`cos k = (D²−Lt²−Ls²)/(2·Lt·Ls)`；`θ = atan2(x,L*) − atan2(Ls·sin k, Lt+Ls·cos k)`。
        D 越界 ⇒ 顶到关节限位、`clamped` 置位（读数必须打标）。
        """
        j = self.cfg.rewards.jump
        lt, ls = self.cfg.rewards.thigh_length, self.cfg.rewards.shank_length
        h0, k0 = self.ref_hip, self.ref_k
        cos_h, sin_h = torch.cos(h0), torch.sin(h0)
        L_ref = lt * cos_h + ls * torch.cos(h0 + k0)
        L_star = L_ref + dh
        _axc = getattr(j, "anchor_x_com", None)
        _curve = getattr(j, "anchor_curve", None)
        if _curve is not None:
            # 平衡曲线锚：轮固定、系统质心落在轮正上方时，髋相对轮的水平偏移随腿长走
            #   （L 越小蹲越深、x_ref 越大；L=0.38 全直时 x_ref=0）。避免冻结常数 0.098 那种前倾蹲。
            Lc = torch.clamp(L_star, 0.10, 0.380)
            x_ref = (_curve[0] * Lc ** 3 + _curve[1] * Lc ** 2
                     + _curve[2] * Lc + _curve[3])
        elif _axc is None:
            x_ref = lt * sin_h + ls * torch.sin(h0 + k0)      # 触发帧冻结
        else:
            x_ref = torch.full_like(L_ref, float(_axc))       # 写死常数
        # 锚点绕轮心前转 anchor_tilt_deg 度，抵消蹬地时机身被往后甩的俯仰：
        #   x' = x·cosθ + L·sinθ，L' = L·cosθ − x·sinθ。模长不变 ⇒ 膝角不受影响，只有髋角前移。
        _tilt = float(getattr(j, "anchor_tilt_deg", 0.0) or 0.0)
        if abs(_tilt) > 1e-6:
            _ct, _st = np.cos(np.radians(_tilt)), np.sin(np.radians(_tilt))
            _x0 = x_ref
            x_ref = _x0 * _ct + L_star * _st
            L_star = L_star * _ct - _x0 * _st
        D = torch.sqrt(torch.clamp(L_star * L_star + x_ref * x_ref, min=1e-8))
        D_hi, D_lo = lt + ls, abs(lt - ls) + 1e-4
        D = torch.clamp(D, D_lo, D_hi)
        cos_k = torch.clamp((D * D - lt * lt - ls * ls) / (2.0 * lt * ls), -1.0, 1.0)
        k_star = -torch.acos(cos_k)                            # 左腿膝角为负的一支
        k_lim = self.dof_pos_limits[self.knee_l_idx]
        k_c = torch.clamp(k_star, k_lim[0], k_lim[1])
        A = lt + ls * torch.cos(k_c)
        B = ls * torch.sin(k_c)
        th_star = torch.atan2(x_ref, torch.clamp(L_star, min=1e-4)) - torch.atan2(B, A)
        h_lim = self.dof_pos_limits[self.hip_l_idx]
        h_c = torch.clamp(th_star, h_lim[0], h_lim[1])
        clamped = (k_star != k_c) | (th_star != h_c) | (D != torch.sqrt(
            torch.clamp(L_star * L_star + x_ref * x_ref, min=1e-8)))
        off_knee = torch.clamp(k_c - k0, -j.off_limit, j.off_limit)
        off_hip = h_c - h0
        return off_knee, off_hip, L_ref, L_star, clamped

    def _lerp_by_h(self, spec):
        """按本跳冻结的 `self.h_target` 逐 env 线性插值。
        `spec=(h_lo, h_hi, v_lo, v_hi)`：h≤h_lo 取 v_lo、h≥h_hi 取 v_hi、中间线性。None⇒返回 None
        （调用方回退到标量）。用于"预蹲目标/爆发目标随起跳指令动态变化"。"""
        if not spec:
            return None
        h_lo, h_hi, v_lo, v_hi = [float(x) for x in spec]
        if h_hi <= h_lo:
            return torch.full_like(self.h_target, v_hi)
        r = torch.clamp((self.h_target - h_lo) / (h_hi - h_lo), 0.0, 1.0)
        return v_lo + (v_hi - v_lo) * r

    def _update_jump_phase(self):
        """跳跃相位机：预蹲(1) → 爆发伸腿(2) → 腾空(3) → 够地(4) → 触地缓冲(5)，**事件驱动**。
        转移挂可观测事件（离地 / vz≤0 / 触地），`t_*_max` 只当保险丝；模板幅度用腿伸量增量 ΔL，按触发那一拍冻结的位姿精确反解。
        """
        if not getattr(self.cfg.rewards.flags, "enable_jump", False):
            return
        j = self.cfg.rewards.jump
        dt = self.dt
        z = self.root_states[:, 2]
        vz = self.root_states[:, 9]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        n_ground = contacts.sum(dim=1)
        grounded = n_ground == 2
        # 承重轮法向力之和 (N)；落地冲击相对 W_stand 算，不相对触发拍锁存值（那个会被策略抬高）。
        _sum_fz = torch.clamp(self.contact_forces[:, self.feet_indices, 2], min=0.0).sum(dim=1)
        self.W_stand = torch.where(self.W_stand <= 0.0, _sum_fz, self.W_stand)
        self.W_stand = torch.where(grounded & (self.ph == 0),
                                   0.98 * self.W_stand + 0.02 * _sum_fz,
                                   self.W_stand)          # 两轮都实 = 站得住（与 _reward_no_moonwalk/探针同口径）
        airborne = n_ground == 0           # 两轮都空 = 真的腾空（一輪着地不算，防蹭地刷分）
        # ---- 轮底离地（真跳高口径）：轮底 = 轮心世界 z − 轮半径，逐拍算 ----
        #   wheel_peak_min 在双轮同离地窗内锁存峰值；wheel_at_air 记离地首拍（零点自检）。
        #   须在本拍 refresh 过 _rb_state（见 post_physics_step）。
        _wb = self._rb_state[:, self.feet_indices, 2] - self.cfg.rewards.wheel_radius
        self.wheel_clear = _wb.min(dim=1).values
        _w_first = airborne & (~self.wheel_latched)
        self.wheel_at_air = torch.where(_w_first, self.wheel_clear, self.wheel_at_air)
        self.wheel_latched = self.wheel_latched | airborne
        self.wheel_peak_min = torch.where(airborne,
                                          torch.maximum(self.wheel_peak_min, self.wheel_clear),
                                          self.wheel_peak_min)
        roll = self._body_roll()
        pitch = self._body_pitch()
        upright = (torch.abs(roll) < j.upright_limit) & (torch.abs(pitch) < j.upright_limit)
        com_z = self._system_com_z()      # 弹体＝系统质心（不是机身；见文件头【高度口径】）
        h_cmd = torch.clamp(self.commands[:, 4], min=0.0)
        n_confirm = max(int(getattr(j, "confirm_steps", 2)), 1)

        # ---- 0. 接触连续计数：离地 / 落地都要连续 n_confirm 拍才认事件 ----
        # 下蹲时的低速抖空不能算离地，否则奖励白给。
        self.cnt_air = torch.where(airborne, self.cnt_air + 1,
                                        torch.zeros_like(self.cnt_air))
        self.cnt_gnd = torch.where(grounded, self.cnt_gnd + 1,
                                        torch.zeros_like(self.cnt_gnd))

        # ---- 1. 触发：IDLE + 已装填 + 双轮站好 + 直立 + 有跳跃指令 ----
        # 装填只在重采样时置位 ⇒ 一条指令最多跳一次；arm_delay 要求已站定够久。
        v_z = self.root_states[:, 9]
        ref_ok = (self.ph == 0) & grounded & upright
        a_ref = min(max(float(getattr(j, "settle_ref_tau", 0.20)), 1e-3), 1.0)
        self.z_stand_ref = torch.where(
            ref_ok, (1.0 - a_ref) * self.z_stand_ref + a_ref * z, self.z_stand_ref)
        # 未初始化过的环境（参考高还是 0）直接接当前高度，避免"永远不静定"的静默死法
        self.z_stand_ref = torch.where(
            self.z_stand_ref.abs() < 1e-6, z, self.z_stand_ref)
        # 静定判据：位置在带内还不够，竖直/水平速度与姿态角速度都要收住（取世界系分量）
        quiet_now = ref_ok & (v_z.abs() <= float(getattr(j, "settle_vz_max", 0.10))) \
            & ((z - self.z_stand_ref).abs() <= float(getattr(j, "settle_dz_max", 0.02))) \
            & (torch.norm(self.root_states[:, 7:9], dim=1)
               <= float(getattr(j, "settle_vxy_max", 0.10))) \
            & (torch.norm(self.root_states[:, 10:12], dim=1)
               <= float(getattr(j, "settle_wxy_max", 0.50)))
        self.cnt_quiet = torch.where(quiet_now, self.cnt_quiet + 1,
                                          torch.zeros_like(self.cnt_quiet))
        settled = (self.cnt_quiet >= max(int(getattr(j, "settle_steps", 10)), 0)) \
            if bool(getattr(j, "settle_gate", False)) else torch.ones_like(grounded)
        arm_steps = max(int(float(getattr(j, "arm_delay", 0.0)) / dt), 0)
        # 触发依据是内部 h_request：站立期 command_targets[:,4] 恒 0（不泄漏到观测），
        # 只有触发这一拍才写进去；request_armed 用与 trigger_active 相同的条件，记"本拍是否要求起跳"。
        trig = ((self.ph == 0) & self.is_armed & grounded & upright
                & (self.h_request > 0.0) & settled
                & ((self.episode_length_buf - self.arm_at) >= arm_steps))
        # 触发这一拍把"请求的跳跃高度"写进观测通道（站立期一直保持 0，避免泄漏）
        self.command_targets[trig, 4] = self.h_request[trig]
        self.ph[trig] = 1
        self.ph_t[trig] = 0.0
        self.ref_z[trig] = z[trig]
        self.z_cmd[trig] = com_z[trig]        # 指令语义的零点＝收到指令那一拍的**质心**高度
        self.h_cmd_apex[trig] = 0.0
        self.h_push_cmd[trig] = 0.0
        self.com_z_prev[trig] = com_z[trig]       # 差分基准接上当前值（不接会有 1 拍假尖峰）
        self.h_pred[trig] = 0.0
        self.h_flight_pred[trig] = 0.0
        self.ev_lift[trig] = 0.0
        self.ev_land_pay[trig] = 0.0
        self.ev_land_frac[trig] = 0.0
        self.tick_seg[trig] = 0.0
        self.tick_crouch_frac[trig] = 0.0
        self.crouch_credit[trig] = 0.0        # 预蹲记账每跳从零累计（诊断用；发钱已改逐拍）
        self.crouch_credit_step[trig] = 0.0   # 本拍预蹲得分（奖励函数的发放量）
        self.z_body_min[trig] = z[trig]      # 绝对蹲到位判定：事件内机身最低点从触发拍起算
        # ---- 事件量采集：时间轴与绝对质心位置，只写不读 ----
        #   t_cmd 恒为 0（触发拍＝时间零点）⇒ 不占缓冲。
        self.ev_t[trig] = 0.0
        self.z_min[trig] = com_z[trig]
        self.t_min[trig] = 0.0
        self.z_land[trig] = 0.0
        self.z_rec[trig] = 0.0
        self.t_lo[trig] = 0.0
        self.t_apex[trig] = 0.0
        self.t_land[trig] = 0.0
        self.t_rec[trig] = 0.0
        self.fz_land_peak[trig] = 0.0
        self.imp_land[trig] = 0.0
        self.gate_lift[trig] = 1.0
        self.gate_chain[trig] = 1.0
        self.ref_L[trig] = 0.0
        self.ref_clamped[trig] = False
        self.z_body_lift[trig] = z[trig]
        self.air_latched[trig] = False          # 离地首拍锁存：每跳重新计
        self.air_z[trig] = 0.0
        self.air_vz[trig] = 0.0
        self.air_body_z[trig] = 0.0
        self.air_body_vz[trig] = 0.0
        self.wheel_peak_min[trig] = 0.0        # 轮底离地峰值每跳重新计
        self.wheel_at_air[trig] = 0.0
        self.wheel_latched[trig] = False
        self.h_body_apex[trig] = 0.0
        self.h_body_lift[trig] = 0.0
        self.ev_flew[trig] = False
        # 本跳的"结算凭证"全部清零（S1 ③：一切跳跃奖励都必须有**确认起飞**才发，见 lift/takeoff）
        self.ev_takeoff[trig] = False
        self.ev_takeoff_ever[trig] = False     # "曾经确认起飞"凭证每跳重置（跨 reset 保留由 reset_idx 侧写 extras 完成）
        self.tick_crouch_ok[trig] = False
        self.v_launch[trig] = 0.0
        self.v_land[trig] = 0.0
        self.L_air_hold[trig] = 0.0
        self.off_knee[trig] = 0.0
        self.off_hip[trig] = 0.0
        self.dem_crouch[trig] = 0.0
        self.tgt_crouch[trig] = 0.0
        self.err_crouch_abs[trig] = 0.0
        self.cnt_settle[trig] = 0              # 预蹲停稳连拍每跳重置（否则上一跳的计数会带进来）
        self.hold_since[trig] = -1.0          # 软停止诊断计时每跳重置
        self.dh_hold[trig] = 0.0              # 软停止目标锁存每跳重置
        self.knee_hold[trig] = 0.0            # 软停止积分器锁存每跳重置
        self.hip_hold[trig] = 0.0
        self.ph5_L_entry[trig] = 0.0          # 【P6R52】相位5 ease-out 起点锁存每跳重置
        self.cnt_air[trig] = 0
        self.cnt_gnd[trig] = 0
        # 摘除装填：一条指令只跳一次。漏这行会"事件一结束立刻再触发"⇒ 跳跃窗吃掉 ~98% 步
        self.is_armed[trig] = False
        # 冻结本事件的基准髋/膝（左腿 local）：ΔL→膝角按**触发那一拍**的位姿精确解，不用当拍
        # 雅可比（那会正反馈发散，见 docstring）。
        self.ref_hip[trig] = self.dof_pos[trig, self.hip_l_idx]
        self.ref_hip_r[trig] = self.dof_pos[trig, self.hip_r_idx]
        # 指令冻结：一跳内的目标与分母都读冻结值，避免重采样边界切进事件中间。
        # 冻结用 h_request：触发那一拍 commands[:,4] 还是 0（下一拍平滑后才反映新目标）。
        self.h_target[trig] = self.h_request[trig]
        # ⚠ 必须在 h_target 冻结**之后**再清零 h_request（否则冻结读到 0 ⇒ 模板几乎不蹬、不起飞）
        self.h_request[trig] = 0.0
        self.ref_k[trig] = self.dof_pos[trig, self.knee_l_idx]
        self.ref_k_r[trig] = self.dof_pos[trig, self.knee_r_idx]

        # ---- 2. 当前相位要求的腿伸量变化 ΔL（m），按冻结姿态解成膝角偏移 ----
        #   cos(hip + k) = (L_ref + ΔL − Lt·cos hip) / Ls，自变量夹 ±1 ⇒ 几何极限处自然饱和；
        #   ±acos 两支取离当前膝角最近的一支。
        p = self.ph
        t = self.ph_t
        lt, ls = self.cfg.rewards.thigh_length, self.cfg.rewards.shank_length
        cos_h = torch.cos(self.ref_hip)
        L_ref = lt * cos_h + ls * torch.cos(self.ref_hip + self.ref_k)
        dh = torch.zeros_like(z)
        # 预蹲目标 = 绝对机身高度（值越小蹲得越深）。crouch_abs_by_h 按本跳冻结的 h_target
        # 逐 env 插值（0.10→0.21 m，0.15→0.20 m），None 时回退标量 crouch_abs_height。
        # 顶到限位时置 ref_clamped，读数要打标。
        _abs_h_s = self._lerp_by_h(getattr(j, "crouch_abs_by_h", None))
        _abs_h = _abs_h_s if _abs_h_s is not None else float(getattr(j, "crouch_abs_height", 0.0))
        # crouch_sag_ff 是可选静差前馈，默认 0；静差由下方 crouch_z_kp 的机身高度伺服负责。
        # 两者不可同开（同开会让 z_cmd 高于 _abs_h，到位判据永远差一个 ff）。
        _ff = float(getattr(j, "crouch_sag_ff", 0.0) or 0.0)
        crouch_cmd = (self.ref_z - (_abs_h + _ff)).clamp(min=0.0,
                                                      max=float(getattr(j, "crouch_depth_cap", 0.14)))
        # 机身口径的要求下蹲量：本相位内为常量（`ref_z` 是触发拍冻结值）⇒ 记账可反复引用。
        _crouch_cmd_abs = crouch_cmd
        # 爬升曲线用五次 smootherstep：r = 6s⁵ − 15s⁴ + 10s³，两端速度与加速度都为 0（C2）。
        # 爬升可在相位 1 结束前到位（crouch_ramp_end < 1），余下拍数用来等停稳。
        ramp_end = float(getattr(j, "crouch_ramp_end", 1.0))
        _s = torch.clamp(t / (j.t_crouch * ramp_end), max=1.0)
        ramp = _s * _s * _s * (_s * (_s * 6.0 - 15.0) + 10.0)
        # 预蹲主目标 = −cmd·五次曲线 + 机身高度伺服。
        # 只开环不够：模板的 dh 是腿长变化，机身高度还受俯仰影响 ⇒ P 反馈必须闭到机身高度上。
        _z_cmd = self.ref_z - crouch_cmd * ramp
        _kz = float(getattr(j, "crouch_z_kp", 0.0) or 0.0)
        dh = torch.where(p == 1, -crouch_cmd * ramp + _kz * (_z_cmd - z), dh)
        _band_tol = float(getattr(j, "crouch_hold_tol", 0.0) or 0.0)
        if (_abs_h_s is not None) and (_band_tol > 0.0):
            # 软停止带随机身目标动态：带 = [机身目标−tol, 机身目标+tol]（逐 env）。锁在带底 ⇒ 蹲底≈目标−tol。
            #   带随**指令目标**（含 `crouch_sag_ff`）一起平移，否则带底仍落在 0.188（旧缺陷）。
            _hmin = _abs_h_s + _ff - _band_tol
            _hmax = _abs_h_s + _ff + _band_tol
            _band_on = True
        else:
            _hmin = float(getattr(j, "crouch_hold_min", 0.0))
            _hmax = float(getattr(j, "crouch_hold_max", 0.0))
            _band_on = (_hmin > 0.0) and (_hmax > _hmin)
        # 软停止：进带后目标冻结为"首次进带那一拍的 dh"（dh_hold），蹲过头也保持该值。
        # ⚠ 冻结的是目标乘积整体而不是 ramp 系数，否则 ramp 仍在涨、蹲过头不降反升。
        if _band_on:
            _in_band = (p == 1) & (z <= _hmax) & (z >= _hmin)
            _over = (p == 1) & (z < _hmin)
            _fresh = _in_band & (self.hold_since < 0)
            self.hold_since = torch.where(_fresh, t, self.hold_since)
            self.hold_since = torch.where(p == 0, torch.full_like(self.hold_since, -1.0),
                                          self.hold_since)
            # 锁存值 = 带底对应的目标 + 首次进带那一拍的积分器现值（膝/髋）。
            # 必须锁积分器：只冻 dh 拦不住限速下积分器的继续爬。
            _band_cmd = torch.clamp(self.ref_z - _hmin, min=0.0,
                                    max=float(getattr(j, "crouch_depth_cap", 0.14)))
            self.dh_hold = torch.where(_fresh, -_band_cmd, self.dh_hold)
            self.knee_hold = torch.where(_fresh, self.off_knee, self.knee_hold)
            self.hip_hold = torch.where(_fresh, self.off_hip, self.hip_hold)
            _keep = _fresh | _over | ((self.hold_since >= 0) & (p == 1))
            dh = torch.where(_keep, self.dh_hold, dh)
        self.dem_crouch = torch.where(p == 1, crouch_cmd, self.dem_crouch)
        # "蹲到位"＝**机身高度进到绝对目标 ± 容差**（双向，蹲过头同样判不过），不是"下降量 ÷ 要求量"。
        self.z_body_min = torch.where(p == 1, torch.minimum(self.z_body_min, z),
                                            self.z_body_min)
        # 口径统一（预蹲一律看机身高度）：到位票与记账共用这个误差，ΔL 只当执行层接口。
        #   `err_crouch_abs` 带符号：正＝还没蹲够（机身偏高）、负＝蹲过头；单位 m。
        self.err_crouch_abs = torch.where(
            p == 1, z - _abs_h,
            self.err_crouch_abs)
        self.tick_crouch_abs_ok = torch.where(
            p == 1, (self.err_crouch_abs.abs() <= float(j.crouch_reach_tol)).float(),
            self.tick_crouch_abs_ok)
        # "此刻应该蹲到多深"＝终值 × 爬升比例；跟踪项判的是**跟没跟住这条曲线**。
        # 质心最低点（`z_body_min` 是机身口径，顶替不了）；严格 < ⇒ 只记首次到达那一拍。
        _low_now = (p >= 1) & (~self.ev_takeoff) & (com_z < self.z_min)
        self.z_min = torch.where(_low_now, com_z, self.z_min)
        self.t_min = torch.where(_low_now, self.ev_t, self.t_min)
        self.tgt_crouch = torch.where(p == 1, crouch_cmd * ramp, self.tgt_crouch)
        # 爆发段：目标钉在几何伸直极限之外（+ push_overshoot），让 PD 误差全程不为零、
        # 力矩一直饱和 ⇒ 行程换成冲量。关 push_to_limit 即退回"按行程到位"。
        dh_cap = (lt * cos_h + ls) - L_ref + j.push_overshoot
        dh_travel = crouch_cmd + j.stroke_gain * self.h_target   # 冻结值，见 trig 处
        dh_push = torch.minimum(dh_travel, dh_cap) if getattr(j, "push_to_limit", False) else dh_travel
        # 爆发段目标腿长硬上限：更屈膝，避免腿拉太直导致直立后仰。0/None 不限制。
        # burst_L_by_h 按本跳冻结的 h_target 逐 env 插值（0.10→310 mm，0.15→330 mm），
        # None 时回退标量 burst_L_max。
        _blm_s = self._lerp_by_h(getattr(j, "burst_L_by_h", None))
        if _blm_s is not None:
            dh_push = torch.minimum(dh_push, _blm_s - L_ref)
        else:
            _blm = float(getattr(j, "burst_L_max", 0.0) or 0.0)
            if _blm > 0.0:
                dh_push = torch.minimum(dh_push, torch.full_like(dh_push, _blm) - L_ref)
        # 爆发段内收腿：实际腿长到 burst_retract_L 后，目标换成 burst_retract_to，
        # 让腿自己收（PD 误差反向），但不切相位——相位转移仍按真离地 / 超时判定。
        _brl = float(getattr(j, "burst_retract_L", 0.0) or 0.0)
        _brt = float(getattr(j, "burst_retract_to", 0.0) or 0.0)
        if _brl > 0.0 and _brt > 0.0:
            _hip_p = self.dof_pos[:, self.hip_l_idx]
            _Lact_p = lt * torch.cos(_hip_p) + ls * torch.cos(_hip_p + self.dof_pos[:, self.knee_l_idx])
            _retract = (p == 2) & (_Lact_p >= _brl)
            dh_push = torch.where(_retract, torch.full_like(dh_push, _brt) - L_ref, dh_push)
        # `burst_template=False` ⇒ **爆发段模板静默**：这一段不给腿长目标也不给偏移，蹬的形状
        # 由策略自己在惩罚项约束下长出来。
        burst_template = bool(getattr(j, "burst_template", True))
        dh_push_v = dh_push if burst_template else torch.zeros_like(z)
        dh = torch.where(p == 2, dh_push_v, dh)
        # 相位 3（腾空）：air_hold_leg == 0 时取消收腿、维持离地那一拍的腿型；
        # 否则 dh = L_air_hold − L_ref。离地未确认时退回 −tuck_depth。
        _air_hold = float(getattr(j, "air_hold_leg", 0.0))
        _air_hold_L = float(getattr(j, "air_hold_L", 0.0) or 0.0)
        # air_hold_joint = 1：相位 3 把腿长钉在起飞瞬间的腿长 L_air_hold（lift 帧冻结），
        #   2R 解算出的关节角等于起飞那一帧 ⇒ 腿随机身一起走、PD 误差趋零、零刹车。
        # air_hold_L 走的是机身锚 L_ref，机身上升时 PD 会把脚按在机身下方，反作用力把机身压下去。
        # 0 = 用机身上的绝对腿长 air_hold_L 保持。
        _air_hold_joint = float(getattr(j, "air_hold_joint", 0.0) or 0.0)
        _air_follow = float(getattr(j, "air_hold_follow", 0.0) or 0.0)
        if _air_hold_L > 0.0:
            if _air_follow != 0.0 and float(self.L_air_hold.max()) > 1e-6:
                # 相位 3 腿长目标 = 起飞冻结腿长 + follow × 机身升高。
                #   follow > 0 腿随机身伸长；follow < 0 升得越高收得越短；0 = 用绝对口径。
                _dz_air = z - self.air_body_z
                _dh_air = (self.L_air_hold + _air_follow * _dz_air) - L_ref
            elif _air_hold_joint > 0.5 and float(self.L_air_hold.max()) > 1e-6:
                _dh_air = self.L_air_hold - L_ref
            else:
                # 相位3 绝对腿长目标（取自 `air_hold_L`）：整条 dh 链是相对触发基准的增量
                _dh_air = torch.full_like(z, _air_hold_L) - L_ref
        elif _air_hold <= 0.0:
            _dh_air = torch.where(
                self.L_air_hold > 1e-6,
                self.L_air_hold - L_ref,
                torch.full_like(z, -j.tuck_depth))
        else:
            _dh_air = torch.full_like(z, -_air_hold)
        dh = torch.where(p == 3, _dh_air, dh)
        # 相位 4（够地）：`land_hold_leg == 0` 时与相位 3 共用同一设定点 ⇒ 腾空→够地零行程
        # （三个相邻相位对同一条腿的目标必须互相衔接，否则矛盾）。
        _land_hold = float(getattr(j, "land_hold_leg", 0.0))
        _land_hold_L = float(getattr(j, "land_hold_L", 0.0) or 0.0)
        if _land_hold_L > 0.0:
            # 相位4 绝对腿长目标（= `land_hold_L`，够地好落地）
            _dh_land = torch.full_like(z, _land_hold_L) - L_ref
        elif _land_hold <= 0.0:
            _dh_land = torch.where(
                self.L_air_hold > 1e-6,
                self.L_air_hold - L_ref,  # 与相位 3 同一个设定点 ⇒ 零行程
                torch.full_like(z, j.land_extend))
        else:
            _dh_land = torch.full_like(z, j.land_extend)
        dh = torch.where(p == 4, _dh_land, dh)
        # 相位 5（缓冲）：常数设定点 = 触发拍的站立基准（dh = 0），靠冲击力矩惩罚与 PD 阻尼吸能。
        # absorb_abs_height > 0 时改瞄绝对机身高度。
        _abs5 = float(getattr(j, "absorb_abs_height", 0.0))
        _pe = float(getattr(j, "absorb_ease_p", 0.0) or 0.0)
        if _abs5 > 0.0:
            _depth5 = (self.ref_z - _abs5).clamp(
                min=0.0, max=float(getattr(j, "crouch_depth_cap", 0.14)))
            if _pe > 0.0 and float(j.t_absorb) > 0.0:
                # 相位 5 腿长目标 ease-out：L = L_settle + (L_entry − L_settle)(1 − s)^p，
                # s = 相位内时钟 / t_absorb，p = 2 ⇒ 起步最快、末端速度归零。
                # L_entry 在 4→5 交界锁存；未锁存（≤0）时退回 L_settle。
                _L_settle = L_ref - _depth5
                _L_entry = torch.where(self.ph5_L_entry > 1e-3, self.ph5_L_entry, _L_settle)
                _s5 = torch.clamp(t / float(j.t_absorb), max=1.0)
                _L5 = _L_settle + (_L_entry - _L_settle) * torch.pow(1.0 - _s5, _pe)
                dh = torch.where(p == 5, _L5 - L_ref, dh)
            else:
                dh = torch.where(p == 5, -_depth5, dh)
        else:
            dh = torch.where(p == 5, torch.zeros_like(z), dh)
        # ΔL 目标改由**两自由度**解（髋＋膝）给出、水平分量冻结 ⇒ 行程不再被"髋角冻结下
        # 的膝完全伸直"钉死。旧的一维 arccos 解法对照见 `_jump_2r_solve` 的注释。
        off_des, hip_des, L_ref, L_star, star_clamped = self._jump_2r_solve(dh)
        self.ref_L[:] = torch.where(p > 0, L_star, self.ref_L)
        self.ref_clamped[:] = star_clamped & (p > 0)

        # ---- 2a-bis. 空中腿长滑动窗：只记相位 3（腾空）的帧 ----
        #   记两条腿竖直伸量的 max（不是均值）：轮底净空 = 机身高 − max(两腿伸量) − 轮半径，
        #   用均值会让一条腿长把轮底吃掉而奖励照拿。窗长 leg_apex_win（默认 3 拍），
        #   整窗都没过长才拿满分。非相位 3 的帧记 −1，不参与 max。
        _Ll = lt * torch.cos(self.dof_pos[:, self.hip_l_idx]) + ls * torch.cos(
            self.dof_pos[:, self.hip_l_idx] + self.dof_pos[:, self.knee_l_idx])
        _Lr = lt * torch.cos(self.dof_pos[:, self.hip_r_idx]) + ls * torch.cos(
            self.dof_pos[:, self.hip_r_idx] + self.dof_pos[:, self.knee_r_idx])
        _Lmax_now = torch.maximum(_Ll, _Lr)
        self.L_max_hist[:, :-1] = self.L_max_hist[:, 1:].clone()
        self.L_max_hist[:, -1] = torch.where(p == 3, _Lmax_now,
                                             torch.full_like(_Lmax_now, -1.0))

        # ---- 2b. 执行校验：腿**实际**到没到达模板要求的伸量 ----
        #   L_act 用**实测髋＋膝**（不是冻结值）：髋被策略搬走也算没执行到位。
        hip_now = self.dof_pos[:, self.hip_l_idx]
        L_act = lt * torch.cos(hip_now) + ls * torch.cos(
            hip_now + self.dof_pos[:, self.knee_l_idx])
        self.err_reach = torch.where(p > 0, (L_act - (L_ref + dh)).abs(),
                                         torch.zeros_like(self.err_reach))
        # 带符号的"腿伸量实际变了多少"（相对触发拍）：预蹲段 dh 是负的、到位判据是 L_act−L_ref ≈ dh，
        # 这就是 S0 门槛①的读数口径。
        self.dbg["L_delta"][:] = torch.where(p > 0, L_act - L_ref,
                                                  torch.zeros_like(self.err_reach))

        # ---- 2f. 预蹲跟踪记账（奖励由 _reward_jump_crouch_track 逐拍发放）----
        #   记在相位机而不是奖励函数里：本函数每控制步恰好一次，奖励函数会被探针重复调用。
        #   口径 = 机身高度：_got_c 是实际下蹲量、_dem_c 是要求下蹲量。
        _got_c = _crouch_cmd_abs - self.err_crouch_abs          # 机身口径：实际已下蹲多少 (m)
        _got_c = _got_c.clamp(min=0.0).clamp(max=_crouch_cmd_abs)
        _dem_c = torch.clamp(_crouch_cmd_abs, min=1e-3)
        _frac_c = torch.clamp(_got_c / _dem_c, 0.0, 1.0)
        _tgt_c = torch.clamp(_crouch_cmd_abs * ramp, min=1e-3)
        # 给分用**高斯**：矩形窗在边界处梯度是断的——差一点拿满分、
        #   过一点拿 0，策略在那一侧收不到"再往回一点更好"的信号。σ = max(frac×要求, abs)，与脉冲项同构。
        _sig_c = torch.clamp(
            float(getattr(j, "crouch_sigma_frac", 0.0)) * _tgt_c,
            min=float(getattr(j, "crouch_sigma_abs", 0.008))).clamp(min=1.0e-4)
        _e_c = (_got_c - _tgt_c) / _sig_c
        _in_band_c = torch.exp(-0.5 * _e_c * _e_c)
        # 逐拍发：本拍得分单独存一份（`crouch_credit` 仍作累计记账，供诊断与增量对比）。
        self.crouch_credit_step = ((p == 1).float() * _frac_c * _in_band_c
                                   * self._jump_reward_upright().float())
        self.crouch_credit = torch.where(p == 1,
                                         self.crouch_credit + self.crouch_credit_step,
                                         self.crouch_credit)
        self.dbg["crouch_credit"][:] = self.crouch_credit


        # ---- 2c. 模板偏移限速：爆发相位**不限速**（要的就是目标远跑在实测前面＝持续 PD
        #   误差＝冲量），其余相位限速。`burst_rate_max`（rad/拍）0＝阶跃。
        rate = float(getattr(j, "off_rate_max", 0.0))
        brate = float(getattr(j, "burst_rate_max", 0.0))
        if rate > 0.0 or brate > 0.0:
            step = torch.clamp(off_des - self.off_knee, -rate, rate) if rate > 0.0 \
                else (off_des - self.off_knee)
            if brate > 0.0:
                step = torch.where(p == 2,
                                   torch.clamp(off_des - self.off_knee, -brate, brate), step)
                self.off_knee = self.off_knee + step
            else:
                self.off_knee = torch.where(p == 2, off_des, self.off_knee + step)
        else:
            self.off_knee = off_des
        # 软停止：冻结落在积分器上 —— 只冻 dh 的话，限速下的积分器仍会继续往深蹲爬。
        # 进带后把膝 / 髋积分器一并钳住。
        if _band_on:
            _hold_knee = torch.where(self.hold_since >= 0, self.knee_hold, self.off_knee)
            self.off_knee = torch.where((p == 1) & (self.hold_since >= 0),
                                        _hold_knee, self.off_knee)
            self.off_hip = torch.where((p == 1) & (self.hold_since >= 0),
                                       self.hip_hold, self.off_hip)
        # 爆发段模板静默：这两拍**不给膝/髋任何偏移**（既不打目标、也不举住预蹲位去挡
        # 策略），蹬的形状完全交给策略自己在惩罚项约束下长。
        if not burst_template:
            self.off_knee = torch.where(p == 2, torch.zeros_like(self.off_knee), self.off_knee)
        # 髋通道跟随两自由度解的目标。
        # 同一套限速：设定点切换要爬坡，否则一帧跳变＝膝反力矩把机身甩翻。
        hrate = float(getattr(j, "hip_rate_max", 0.25))
        hstep = torch.clamp(hip_des - self.off_hip, -hrate, hrate) if hrate > 0.0 \
            else hip_des - self.off_hip
        self.off_hip = self.off_hip + hstep
        if not burst_template:
            self.off_hip = torch.where(p == 2, torch.zeros_like(self.off_hip),
                                            self.off_hip)
        self.off_hip = torch.where(p > 0, self.off_hip, torch.zeros_like(self.off_hip))

        # ---- 2d. 髋通道：可选叠加的纯旋转补偿（把轮子搬回髋正下方），
        #   增益 0 时不写 `off_hip_comp`。
        gain = float(getattr(self.cfg.rewards.jump, "hip_comp_gain", 0.0))
        if gain != 0.0:
            lt, ls = self.cfg.rewards.thigh_length, self.cfg.rewards.shank_length
            h0, k0 = self.ref_hip, self.ref_k
            ka = k0 + self.off_knee
            L_a = lt * torch.cos(h0) + ls * torch.cos(h0 + ka)
            x_a = lt * torch.sin(h0) + ls * torch.sin(h0 + ka)
            x_0 = lt * torch.sin(h0) + ls * torch.sin(h0 + k0)
            R = torch.sqrt(L_a * L_a + x_a * x_a).clamp(min=1e-6)
            th = torch.asin(torch.clamp(x_0 / R, -1.0, 1.0)) - torch.atan2(x_a, L_a)
            room = (1.5708 - torch.abs(h0) - 0.05).clamp(min=0.0)   # 髋限位余量
            self.off_hip_comp = torch.where(self.ph > 0,
                                                 torch.clamp(th, -room, room),
                                                 torch.zeros_like(th))

        # ---- 2e. 离地首拍锁存：起飞窗口（相位 ≥ 2）内第一次两轮同空那一拍存下 z / vz ----
        #   lift 要连续 confirm_steps 拍才认，取样点晚 1~2 拍；不锁存会把脉冲分子系统性压低。
        _air_first = airborne & (p >= 2) & (~self.air_latched)
        _com_vz_now = (com_z - self.com_z_prev) / dt
        self.com_vz_step = _com_vz_now   # 逐拍质心竖直速度（相位2 push_lead 用；在 com_z_prev 更新前差分 ⇒ 本拍值）
        self.air_z = torch.where(_air_first, com_z, self.air_z)
        self.air_vz = torch.where(_air_first, _com_vz_now, self.air_vz)
        self.air_body_z = torch.where(_air_first, z, self.air_body_z)
        self.air_body_vz = torch.where(_air_first, vz, self.air_body_vz)
        self.air_latched = self.air_latched | _air_first

        # ---- 3. 事件转移（转移条件全部按**本拍进入时**的相位算，防一格里连跳两档）----
        p0 = p.clone()
        active = p0 > 0
        self.ph_t = torch.where(active, t + dt, t)
        # `ph_t` 每次跳转清零＝相位内时钟；`ev_t` 只在触发/复位清零 ⇒ 事件时刻取"边沿那一拍的 ev_t"。
        self.ev_t = torch.where(active, self.ev_t + dt, self.ev_t)
        tn = self.ph_t
        # 相位内时钟的快照：tn 是 self.ph_t 的别名，而 _jump_advance 会就地清零 ph_t
        # ⇒ 转移帧上 tn 变 0。快照留给相位 3→4 的顶点判据用（见下方 _apex_my）。
        tn_p3 = self.ph_t.clone()
        # 起飞确认：爆发**或腾空初段**里连续 n_confirm 拍两轮同空、且这一拍仍在上升。门要开到
        # 相位 3：小跳的离地常发生在 `t_push_max` 保险丝**之后**，只开相位 2 会把真跳判成没起飞。
        lift = ((p0 >= 2) & (p0 <= 3) & (~self.ev_takeoff)
                & (self.cnt_air >= n_confirm) & (vz >= float(j.takeoff_vz_min)))
        self.ev_takeoff = self.ev_takeoff | lift
        self.ev_takeoff_ever = self.ev_takeoff_ever | lift   # 锁存：reset 清 ev_takeoff 后探针仍能分"跳了但撞地"
        # 尺子零点与离场速度取**离地首拍锁存值**（`lift` 成立时 2e 必已锁存 ⇒ 不需要回退分支）
        self.z_body_lift = torch.where(lift, self.air_body_z, self.z_body_lift)
        self.v_launch = torch.where(lift, self.air_body_vz, self.v_launch)
        self.t_lo = torch.where(lift, self.ev_t, self.t_lo)
        # 在**离地确认那一拍**存实测腿伸量，供相位 3「维持离地腿型」当设定点。必须存
        # "离地那一拍的腿型"而不是触发基准（蹲着的）：两者差一个预蹲深度 40~50 mm。
        _hip_n = self.dof_pos[:, self.hip_l_idx]
        _knee_n = self.dof_pos[:, self.knee_l_idx]
        _hip_r = self.dof_pos[:, self.hip_r_idx]
        _knee_r = self.dof_pos[:, self.knee_r_idx]
        _lt, _ls = self.cfg.rewards.thigh_length, self.cfg.rewards.shank_length
        _L_now = 0.5 * (_lt * torch.cos(_hip_n) + _ls * torch.cos(_hip_n + _knee_n)
                         + _lt * torch.cos(_hip_r) + _ls * torch.cos(_hip_r + _knee_r))
        self.L_air_hold = torch.where(lift, _L_now, self.L_air_hold)
        self.ev_flew = self.ev_flew | (airborne & (p0 >= 2))       # 只作对照
        self.h_body_apex = torch.where(active, torch.maximum(self.h_body_apex, z - self.ref_z),
                                     self.h_body_apex)
        self.h_body_lift = torch.where(self.ev_takeoff,
                                         torch.maximum(self.h_body_lift,
                                                       z - self.z_body_lift),
                                         self.h_body_lift)
        # 质心口径的两项拆分：h_cmd_apex = 事件内 max(质心 − 触发拍质心)，
        # h_push_cmd = 离地首拍质心 − 触发拍质心。须在通道更新前判，否则 t_apex 会被刷到事件末尾。
        _apex_now = active & ((com_z - self.z_cmd) > self.h_cmd_apex)
        self.h_cmd_apex = torch.where(active,
                                         torch.maximum(self.h_cmd_apex, com_z - self.z_cmd),
                                         self.h_cmd_apex)
        self.t_apex = torch.where(_apex_now, self.ev_t, self.t_apex)
        self.h_push_cmd = torch.where(lift, self.air_z - self.z_cmd, self.h_push_cmd)
        # ① 预蹲：蹲到位 ∧ 停稳 ∧ 命令结束后保持够久，才交棒给爆发。
        #   到位 = 机身高度进绝对目标 ± crouch_reach_tol；停稳 = |vz| ≤ crouch_vz_tol；
        #   两者须连续 crouch_settle_steps 拍同时成立。兜底：t_crouch 保险丝、
        #   过深救助（比目标低超过 crouch_deep_tol 立刻蹬）。
        _tol_c = float(j.crouch_reach_tol)
        _t_ramp = float(j.t_crouch) * float(getattr(j, "crouch_ramp_end", 1.0))
        _ok_c = (p0 == 1) & (self.err_crouch_abs.abs() <= _tol_c)
        _still = _ok_c & (torch.abs(vz) <= float(getattr(j, "crouch_vz_tol", 0.02)))
        self.cnt_settle = torch.where(_still, self.cnt_settle + 1,
                                      torch.zeros_like(self.cnt_settle))
        _settled = ((p0 == 1)
                    & (tn >= _t_ramp + float(getattr(j, "crouch_hold_time", 0.0)))
                    & (self.cnt_settle >= int(getattr(j, "crouch_settle_steps", 5))))
        _deep = (p0 == 1) & (self.err_crouch_abs < -float(getattr(j, "crouch_deep_tol", 0.010)))
        _exit_c = _settled | _deep | ((p0 == 1) & (tn >= j.t_crouch))
        # 口径统一：出口这一拍按**机身高度**判到位（探针的"真蹲到位率"）。
        self.tick_crouch_ok = self.tick_crouch_ok | (_exit_c & _ok_c)
        self._jump_advance(_exit_c, 2)
        # ② 爆发 → 腾空：**只认真起飞**。保险丝到期仍未起飞也转腾空（让晚离地的小跳还有被确认的
        #    机会），但跳跃奖励全挂在 `ev_takeoff` 上 ⇒ 没离地的失败跳一分钱拿不到。
        self._jump_advance((p0 == 2) & self.ev_takeoff & airborne, 3)
        self._jump_advance((p0 == 2) & (~self.ev_takeoff) & (tn >= j.t_push_max), 3)
        # ③ 腾空 → 够地：到顶点（vz≤0）或空中保险丝到期
        self._jump_advance((p0 == 3) & (tn >= j.t_flight_min)
                           & ((vz <= 0.0) | (tn >= j.t_flight_max)), 4)
        # ④ 够地 → 缓冲：连续 n_confirm 拍回地才算落地，落地那一拍记冲击速度（S0 第四个门槛）
        touch = (p0 == 4) & (self.cnt_gnd >= n_confirm)
        self.v_land = torch.where(touch, torch.abs(vz), self.v_land)
        self.z_land = torch.where(touch, com_z, self.z_land)          # E5 触地·质心 (m)
        self.t_land = torch.where(touch, self.ev_t, self.t_land)
        self._jump_advance(touch, 5)
        self._jump_advance((p0 == 4) & (tn >= j.t_land_max), 5)
        # 4→5 交界锁存相位 4 出口腿长（相位 5 ease-out 的起点 L_entry），进相位 5 那一拍才写。
        #   ref_L 本拍已按相位4 目标更新（见上方 _jump_2r_solve 后的 self.ref_L 赋值）。
        _to5 = touch | ((p0 == 4) & (tn >= j.t_land_max))
        self.ph5_L_entry = torch.where(_to5, self.ref_L, self.ph5_L_entry)
        # 落地窗（缓冲相位内）累计：法向力峰值 (N) 与相对体重的冲量 ∫|ΣFz − W|dt (N·s)。
        _land_win = (self.ph == 5)
        self.fz_land_peak = torch.where(_land_win, torch.maximum(self.fz_land_peak, _sum_fz),
                                        self.fz_land_peak)
        self.imp_land = self.imp_land + _land_win.float() \
            * (_sum_fz - self.W_stand).abs() * dt
        # E6 恢复＝缓冲结束回常态那一拍；要在改相位之前取 ev_t/质心高
        _rec = (p0 == 5) & (tn >= j.t_absorb)
        self.z_rec = torch.where(_rec, com_z, self.z_rec)
        self.t_rec = torch.where(_rec, self.ev_t, self.t_rec)
        # 回待机时清掉观测里的跳跃指令（与 h_request 清零配合：站立期观测第12维恒0，不泄漏）
        self.command_targets[_rec, 4] = 0.0
        self._jump_advance(_rec, 0)               # 缓冲结束 → 回常态

        # ---- 3' 逐事件锁存完成票 + 链式封顶门：前序没达标的，后继相位正分最多 +5%（leak）----
        #   票一律用比值或几何误差，不用高斯误差分。`p` 是 self.ph 的视图、转移已就地改过它
        #   ⇒ 入口相位用 p0，当前相位用 p。
        # ⚠ 顶点判据必须用相位计时器快照 tn_p3，不能用 tn：tn 是 self.ph_t 的别名，
        #   _jump_advance 把 ph_t 就地清零后，转移帧上"本相位已过多久"恒为 0 ⇒ apex 永不成立。
        apex = (p0 == 3) & (tn_p3 >= j.t_flight_min) & ((vz <= 0.0) | (tn_p3 >= j.t_flight_max))
        leak = float(getattr(j, "chain_leak", 0.05))
        shape_tol = float(getattr(j, "shape_tol", 0.02))
        # g1 用比值（实蹲 ÷ 要求）而不是布尔：布尔门一旦全关，后续正分被压在 5% 封顶。
        # 分母用本档终值要求，不用爬升瞬时值。分子 / 分母同为机身高度口径。
        # 严格布尔 tick_crouch_ok 只给判据 / 探针用，不发钱。
        crouch_frac_now = torch.clamp(
            (_crouch_cmd_abs - self.err_crouch_abs) / torch.clamp(self.dem_crouch, min=1e-3),
            0.0, 1.0)
        self.tick_crouch_frac = torch.where(p0 == 1,
                                            torch.maximum(self.tick_crouch_frac, crouch_frac_now),
                                            self.tick_crouch_frac)
        # 差分基准每拍接上当前质心高（复位/触发那一拍也接，避免开局假尖峰）
        self.com_z_prev[:] = com_z
        # 离地那一拍结算：`h_flight_pred`（离地后上升的预测值，**脉冲的钱**，分子取**质心口径**）
        #   ＋ `h_pred`（接触段＋弹道，只作辅助诊断）。预测值≠实测 `h_flight`：兑现率随构型变化。
        h_flight_pred = torch.clamp(self.air_vz, min=0.0) ** 2 / (2.0 * 9.81)
        self.h_pred[:] = torch.where(
            lift, self.h_push_cmd + torch.clamp(self.air_vz, min=0.0) ** 2 / (2.0 * 9.81),
            self.h_pred)
        self.h_flight_pred[:] = torch.where(lift, h_flight_pred, self.h_flight_pred)
        self.ev_lift[:] = lift.float()
        ok1 = self.tick_crouch_ok                                 # 严格票（只给判据/探针读，不发钱）
        ok2 = lift & (h_flight_pred >= float(getattr(j, "launch_ok_frac", 0.5)) * self.h_target)
        ok3 = apex & (self.err_reach <= shape_tol)
        ok4 = touch & (self.err_reach <= shape_tol)
        self.tick_seg[:, 0] = torch.where(ok2, torch.ones_like(self.tick_seg[:, 0]), self.tick_seg[:, 0])
        self.tick_seg[:, 1] = torch.where(ok3, torch.ones_like(self.tick_seg[:, 1]), self.tick_seg[:, 1])
        self.tick_seg[:, 2] = torch.where(ok4, torch.ones_like(self.tick_seg[:, 2]), self.tick_seg[:, 2])
        g1 = leak + (1.0 - leak) * self.tick_crouch_frac
        g2 = leak + (1.0 - leak) * self.tick_seg[:, 0]
        g3 = leak + (1.0 - leak) * self.tick_seg[:, 1]
        g4 = leak + (1.0 - leak) * self.tick_seg[:, 2]
        chain = torch.ones_like(z)
        chain = torch.where(p == 2, g1, chain)
        chain = torch.where(p == 3, g1 * g2, chain)
        chain = torch.where(p == 4, g1 * g2 * g3, chain)
        chain = torch.where(p == 5, g1 * g2 * g3 * g4, chain)
        self.gate_chain[:] = chain
        self.gate_lift[:] = g1                                   # 离地脉冲专用门（只看预蹲票）
        self.gate12[:] = torch.where(p == 3, g1 * g2, torch.where(p >= 4, g1 * g2, torch.ones_like(g1)))
        # 相位 3/4/5 用 g1·g2（不含 g3），相位 2 用 g1，相位 0/1 用 1。
        #   专供 `_reward_jump_shape_track`，让它去掉 g3 梯度、保留 R57 的 g1·g2 结构。
        # g3 落进 dbg：它是"顶点腿长票"这一环，收工时看一眼是否在动 —— 恒定不动说明门又关上了。
        #   （`apex` 从未触发的 bug，见 :897 注释）。以后每轮收工都该看一眼它**是否在动**
        #   —— 恒定不动 = 门又关上了（静默失效只看输出数字发现不了）。
        self.dbg["g3"][:] = g3
        # ---- 顶点腿长奖的本拍乘子：相位 3→4 那一拍（顶点）一次结清 ----
        #   目标 = self.ref_L（模板自己算出的腿长目标，与相位 3/4 模板同源、逐拍连续）；
        #   测量 = 顶点前 leg_apex_win 拍的空中腿长滑动最大（两腿取 max = 绑定轮口径）；
        #   形式 = 双边高斯 exp(−0.5·((L_peak − L_target)/σ)²)；
        #   门 = 确认起飞 × 直立 × g1 × g2。不用 g3：g3 就是本项的布尔版，拿它当门会循环。
        _lsig = float(getattr(j, "leg_apex_sigma", 0.0) or 0.0)
        #   顶点掩码自己算一份、不复用共享的 apex：apex 依赖的 tn 会被就地清零（见上方说明），
        #   共享值不可靠；本项只按自己算的掩码发奖。
        _apex_my = ((p0 == 3) & (tn_p3 >= float(j.t_flight_min))
                    & ((vz <= 0.0) | (tn_p3 >= float(j.t_flight_max))))
        _Lpk = self.L_max_hist.max(dim=1).values
        if _lsig > 0.0:
            _z = (_Lpk - self.ref_L) / _lsig
            _sc = torch.exp(-0.5 * _z * _z)
        else:
            _sc = torch.zeros_like(_Lpk)
        _ok_leg = (_Lpk > 0.0) & (self.ref_L > 1e-6)   # 窗里没有相位3 帧 ⇒ 不发
        self.ev_apex_leg_pay[:] = (_apex_my.float() * self.ev_takeoff.float()
                                   * self._jump_reward_upright().float()
                                   * g1 * g2 * _sc * _ok_leg.float())
        self.dbg["apex_leg_err"][:] = torch.where(_apex_my, (_Lpk - self.ref_L) * 1000.0,
                                                  self.dbg["apex_leg_err"])
        self.dbg["apex_leg_score"][:] = torch.where(_apex_my, _sc,
                                                    self.dbg["apex_leg_score"])
        # ---- 真跳高（轮底离地）结果奖：同一顶点帧（相位 4 起点）一次性结清 ----
        #   奖励量 = wheel_peak_min（双轮同离地窗的轮底峰值，与判据同口径）；
        #   目标 = top_height_goal_by_h 随指令插值的可达轮底（None 时回退指令 h_target）；
        #   双边高斯，超调也扣。门 = 确认起飞 × 直立 × g1 × g2 × 真离过地。
        #   不等触地再发，避免把"赖地不走"算进奖励。
        _thsig = float(getattr(j, "top_height_sigma", 0.0) or 0.0)
        _goal_by_h = self._lerp_by_h(getattr(j, "top_height_goal_by_h", None))
        _goal_src = _goal_by_h if _goal_by_h is not None else self.h_target
        _goal_th = torch.clamp(_goal_src, min=float(getattr(j, "goal_floor", 0.005)))
        if _thsig > 0.0:
            _zth = (self.wheel_peak_min - _goal_th) / _thsig
            _sc_th = torch.exp(-0.5 * _zth * _zth)
        else:
            _sc_th = torch.zeros_like(_goal_th)
        _ok_wheel = (self.wheel_peak_min > 0.0)   # 没真离过地 ⇒ 不发
        self.ev_top_height_pay[:] = (_apex_my.float() * self.ev_takeoff.float()
                                     * self._jump_reward_upright().float()
                                     * g1 * g2 * _sc_th * _ok_wheel.float())
        self.dbg["top_height_err"][:] = torch.where(
            _apex_my, (self.wheel_peak_min - _goal_th) * 1000.0, self.dbg["top_height_err"])
        self.dbg["top_height_score"][:] = torch.where(
            _apex_my, _sc_th, self.dbg["top_height_score"])
        # ---- 落地结算的**本拍乘子**（只在触地那一拍算一次，存给奖励函数）----
        # 门＝确认起飞 × 直立 × 落地冲击 soft × g1 × 腿伸比值；前序票只取 g1 不取 g2。
        land_frac = torch.clamp(
            1.0 - self.err_reach / float(getattr(j, "land_leg_tol", 0.060)), 0.0, 1.0)
        # `land_frac` 每拍都算（"当前腿伸离够地设定点多远"的连续量），但 `ev_land_frac`
        # 只在触地那一拍存下来当判据读。
        self.ev_land_frac[:] = torch.where(touch, land_frac, self.ev_land_frac)
        # ⚠⚠ 必须乘 `touch` 当**指示位**，不能只靠 `torch.where(touch, ...)` 锁存：锁存版
        #   在事件走完后不清零，那一跳的钱会被每拍重复发到下一次触发。
        self.ev_land_pay[:] = touch.float() * (
            self.ev_takeoff.float() * self._jump_reward_upright().float()
            * torch.clamp(1.0 - self.v_land / float(j.vz_landing_soft), 0.0, 1.0)
            * (leak + (1.0 - leak) * g1)
            * (leak + (1.0 - leak) * land_frac))
        # 防"顶起式刷分"靠分子本身（`h_flight_pred` 不含接触段）。
        self.dbg["gate_chain"][:] = self.gate_chain
        self.dbg["h_pred"][:] = self.h_pred
        self.dbg["h_flight_pred"][:] = self.h_flight_pred
        self.dbg["z_body_min"][:] = self.z_body_min
        self.dbg["tick_crouch_abs_ok"][:] = self.tick_crouch_abs_ok
        # 离地首拍锁存的四条读数＋锁存旗标：探针可离线把「机身/质心 × 首拍/确认拍」四种口径一次算全
        self.dbg["air_body_z"][:] = self.air_body_z
        self.dbg["air_body_vz"][:] = self.air_body_vz
        self.dbg["air_z"][:] = self.air_z
        self.dbg["air_vz"][:] = self.air_vz
        self.dbg["air_latched"][:] = self.air_latched.float()
        # 站定成立率（指令侧门的曝光代价读这一列）
        self.dbg["cmd_settled"][:] = (
            self.cnt_quiet >= max(int(getattr(j, "settle_steps", 10)), 0)).float()
        self.dbg["ref_clamped"][:] = self.ref_clamped.float()   # 目标落进限位外 ⇒ 行程读数要打标

        # ---- 3' 跳跃回合的**连跳限流** ----
        # 事件结束后有限次重装填（`rearm_max`/`rearm_gap`）；只对跳跃回合子集开放。
        rearm_max = int(getattr(j, "rearm_max", 0))
        if rearm_max > 0:
            gap_steps = max(int(float(getattr(j, "rearm_gap", 0.0)) / dt), 0)
            can_again = (self.is_jump_round
                         & (self.rearm_cnt < rearm_max)
                         & (h_cmd > 0.0)
                         & grounded & upright
                         & (self.episode_length_buf >= self.rearm_at))
            self.is_armed = self.is_armed | can_again
            self.rearm_cnt = self.rearm_cnt + can_again.long()
            self.rearm_at = torch.where(
                can_again, self.episode_length_buf + gap_steps, self.rearm_at)

        # ---- 4. 对外状态：is_jumping = "在本跳的事件窗口内" ----
        # 不是"跳跃指令期间"——后者会让高度跟踪/髋可达/no_fly 连坐屏蔽整段重采样窗。
        self.is_jumping[:] = self.ph > 0
        self.dbg["off_knee"][:] = self.off_knee
        self.dbg["ph"][:] = self.ph.float()
        self.dbg["h_body_rise"][:] = self.h_body_apex
        self.dbg["h_cmd"][:] = self.commands[:, 4]   # 活指令
        self.dbg["h_target"][:] = self.h_target            # 触发拍冻结值
        # S0 四门槛的原始量（评测/探针直接读这些，不要再各自数一遍接触）
        self.dbg["ev_takeoff"][:] = self.ev_takeoff.float()
        self.dbg["tick_crouch_ok"][:] = self.tick_crouch_ok.float()
        self.dbg["err_reach"][:] = self.err_reach
        self.dbg["v_launch"][:] = self.v_launch
        self.dbg["h_body_lift"][:] = self.h_body_lift
        # 验收读数＝机身两条（`h_body_rise` 含预蹲深度、`h_body_lift`＝判据认的 `h_flight`）；
        #   质心两条留作对照。⚠ 必须与"确认起飞 n/N"并读（见文件头【高度口径】）。
        self.dbg["h_cmd_apex"][:] = self.h_cmd_apex
        self.dbg["h_push_cmd"][:] = self.h_push_cmd
        self.dbg["v_land"][:] = self.v_land
        # ---- 事件量导出：play02 按 dbg 键序自动多录这些列 ----
        self.dbg["ev_t"][:] = self.ev_t
        self.dbg["t_min"][:] = self.t_min
        self.dbg["t_lo"][:] = self.t_lo
        self.dbg["t_apex"][:] = self.t_apex
        self.dbg["t_land"][:] = self.t_land
        self.dbg["t_rec"][:] = self.t_rec
        self.dbg["z_min"][:] = self.z_min
        self.dbg["z_land"][:] = self.z_land
        self.dbg["z_rec"][:] = self.z_rec
        self.dbg["fz_land_peak"][:] = self.fz_land_peak
        self.dbg["imp_land"][:] = self.imp_land
        self.dbg["W_stand"][:] = self.W_stand
        # 轮底离地（真跳高口径）只读快照
        self.dbg["wheel_clear"][:] = self.wheel_clear
        self.dbg["wheel_peak_min"][:] = self.wheel_peak_min
        self.dbg["wheel_at_air"][:] = self.wheel_at_air

    def _jump_advance(self, mask, nxt):
        """相位跳转：置新相位并把计时器清零（mask 为本拍成立的事件）"""
        self.ph[mask] = nxt
        self.ph_t[mask] = 0.0

    def reset_jump_state(self, env_ids):
        """环境重置/复位时把跳跃状态机按回待机，防止旧回合的相位、基准高度串到新回合"""
        if len(env_ids) == 0:
            return
        self.ph[env_ids] = 0
        self.ph_t[env_ids] = 0.0
        self.hold_since[env_ids] = -1.0       # 软停止计时复位（防上一跳带内状态串进新回合）
        self.dh_hold[env_ids] = 0.0           # 软停止目标锁存复位
        self.knee_hold[env_ids] = 0.0
        self.hip_hold[env_ids] = 0.0
        self.ph5_L_entry[env_ids] = 0.0       # 【P6R52】相位5 ease-out 起点锁存复位
        self.ref_z[env_ids] = 0.0
        self.z_cmd[env_ids] = 0.0
        self.h_target[env_ids] = 0.0
        self.h_cmd_apex[env_ids] = 0.0
        self.h_push_cmd[env_ids] = 0.0
        self.com_z_prev[env_ids] = 0.0
        self.h_pred[env_ids] = 0.0
        self.h_flight_pred[env_ids] = 0.0
        self.ev_lift[env_ids] = 0.0
        self.ev_land_pay[env_ids] = 0.0
        self.ev_land_frac[env_ids] = 0.0
        self.ev_apex_leg_pay[env_ids] = 0.0
        self.ev_top_height_pay[env_ids] = 0.0
        self.L_max_hist[env_ids] = -1.0
        self.tick_seg[env_ids] = 0.0
        self.tick_crouch_frac[env_ids] = 0.0
        self.crouch_credit[env_ids] = 0.0
        self.crouch_credit_step[env_ids] = 0.0
        self.z_body_min[env_ids] = 0.0
        self.tick_crouch_abs_ok[env_ids] = 0.0
        self.gate_chain[env_ids] = 1.0
        self.gate_lift[env_ids] = 1.0
        self.ref_L[env_ids] = 0.0
        self.ref_clamped[env_ids] = False
        self.ref_hip[env_ids] = 0.0
        self.ref_hip_r[env_ids] = 0.0
        self.ref_k[env_ids] = 0.0
        self.ref_k_r[env_ids] = 0.0
        self.h_body_apex[env_ids] = 0.0
        self.ev_flew[env_ids] = False
        self.off_knee[env_ids] = 0.0
        self.off_hip[env_ids] = 0.0
        self.dem_crouch[env_ids] = 0.0
        self.tgt_crouch[env_ids] = 0.0
        self.err_crouch_abs[env_ids] = 0.0
        # 装填不能在这里无条件清掉：super().reset_idx() 里的 _resample_commands 已按本窗
        # 是否装了跳跃指令写好 is_armed，清了会抹掉结果。这里按本窗请求的高度恢复装填，
        # 并把 arm_delay 的起算点归零。
        self.arm_at[env_ids] = 0
        # 按"本窗请求的跳跃高度"恢复装填（`command_targets[:,4]` 站立期恒 0，不能用它判）
        self.is_armed[env_ids] = self.h_request[env_ids] > 0.0
        # 连跳限流的三件状态一起复位，否则旧回合的重装填计数会串到新回合。
        # ⚠ is_jump_round 不在这里清：_resample_commands 已在 super().reset_idx() 里抽好本窗
        # 归属，清了会把跳跃回合全当成行走回合。
        self.rearm_cnt[env_ids] = 0
        self.rearm_at[env_ids] = 0
        self.cnt_air[env_ids] = 0
        self.cnt_gnd[env_ids] = 0
        self.cnt_settle[env_ids] = 0      # 预蹲"到位且停稳"的连续拍数
        # 静定门状态：参考高取"复位那一刻的高度"起步（留 0 会让 |z−ref| 永远超限 ⇒ 曝光静默归零）
        self.z_stand_ref[env_ids] = self.root_states[env_ids, 2]
        # 计数同样播种成已站定：复位位形就是默认站立位；留 0 会让只在复位后重采样的调用方
        #   （play02/评测）永远被 `cmd_settle_gate` 挡死。
        self.cnt_quiet[env_ids] = max(int(getattr(self.cfg.rewards.jump, "settle_steps", 10)), 1)
        self.ev_takeoff[env_ids] = False
        self.ev_takeoff_ever[env_ids] = False
        self.tick_crouch_ok[env_ids] = False
        self.z_body_lift[env_ids] = 0.0
        self.air_latched[env_ids] = False
        self.air_z[env_ids] = 0.0
        self.air_vz[env_ids] = 0.0
        self.air_body_z[env_ids] = 0.0
        self.air_body_vz[env_ids] = 0.0
        self.wheel_peak_min[env_ids] = 0.0
        self.wheel_at_air[env_ids] = 0.0
        self.wheel_latched[env_ids] = False
        self.h_body_lift[env_ids] = 0.0
        self.v_launch[env_ids] = 0.0
        self.v_land[env_ids] = 0.0
        self.L_air_hold[env_ids] = 0.0
        for _k in ("ev_t", "z_min", "t_min", "z_land", "z_rec", "t_lo", "t_apex",
                   "t_land", "t_rec", "fz_land_peak", "imp_land"):
            getattr(self, _k)[env_ids] = 0.0
        self.err_reach[env_ids] = 0.0
        self.is_jumping[env_ids] = False

    def _reward_z_adjust_leg(self):
        """膝长跟踪惩罚：跟踪解算器输出的目标。相位 1~4 免罚——模板故意让目标跑在实测前面（那是冲量来源），罚这个差＝罚冲量本身。

        """
        if not getattr(self.cfg.rewards.flags, "enable_leg_adjust", False):
            return torch.zeros(self.num_envs, device=self.device)
        target_l, target_r = self._compute_target_knee_angles()
        err_l = torch.square(self.dof_pos[:, self.knee_l_idx] - target_l)
        err_r = torch.square(self.dof_pos[:, self.knee_r_idx] - target_r)
        # 【激励侧】按相位免罚（1~4）：本项罚的正是 `push_to_limit` 故意造出来的那个
        # 膝目标差，留着它＝训练里"蹬得越狠罚得越重"。
        return (err_l + err_r) * (~self._jump_blocked("z_adjust_leg"))

    def _reward_tracking_lin_vel(self):
        # 固定误差范围内线性下降，避免较大误差时仍靠高斯函数获得可观奖励。
        longitudinal_error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        longitudinal_reward = torch.clamp(
            1.0 - longitudinal_error / self.cfg.rewards.velocity_tracking_error_limit,
            min=0.0,
        )
        lateral_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        lateral_reward = torch.exp(-lateral_error / self.cfg.rewards.tracking_sigma)
        # 【激励侧】按相位掩蔽：预蹲/蹬地会把机体前后甩一下，这三项跟踪分在事件窗内不该计。
        # 掩蔽不等于免罚：这些拍两项都不给分，梯度就不站在"别蹲"那一边。
        return longitudinal_reward * lateral_reward * (~self._jump_blocked("tracking_lin_vel"))

    def _reward_tracking_ang_vel(self):
        # 对偏航误差使用线性奖励，避免较大的跟踪误差仍获得过多高斯奖励。
        # YAW_SIGN：对外指令口径是"正数=右转"，而 base_ang_vel 正数是左转 ⇒ 指令侧取负再比。
        yaw_error = torch.abs(YAW_SIGN * self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.clamp(
            1.0 - yaw_error / self.cfg.rewards.yaw_tracking_error_limit,
            min=0.0,
        ) * (~self._jump_blocked("tracking_ang_vel"))

    def _reward_tracking_wheel_lin_vel(self):
        # 轮速等效线速度必须乘 cos(俯仰) 才与 commands[:,0] 同口径：
        #   v = r_w·(ω_L − ω_R)/2 · cosθ。不乘则两个速度跟踪项互相矛盾，轮速项会给抬头发钱。
        wheel_lin_vel = 0.5 * self.cfg.rewards.wheel_radius * (
            self.dof_vel[:, self.wheel_l_idx] - self.dof_vel[:, self.wheel_r_idx]
        ) * torch.cos(self._body_pitch())
        tracking_error = torch.abs(self.commands[:, 0] - wheel_lin_vel)
        tracking_reward = torch.clamp(
            1.0 - tracking_error / self.cfg.rewards.velocity_tracking_error_limit,
            min=0.0,
        )
        wheel_contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        both_wheels_contact = torch.sum(wheel_contacts, dim=1) == 2
        return tracking_reward * both_wheels_contact * (~self._jump_blocked("tracking_wheel_lin_vel"))

    def _reward_wrong_way(self):
        cmd_vx = self.commands[:, 0]
        real_vx = self.base_lin_vel[:, 0]
        has_motion_command = torch.abs(cmd_vx) > self.cfg.rewards.wrong_way_command_threshold
        has_meaningful_motion = torch.abs(real_vx) > self.cfg.rewards.wrong_way_velocity_threshold
        opposite_direction = cmd_vx * real_vx < 0.0
        return (has_motion_command & has_meaningful_motion & opposite_direction).float()

    def _reward_no_moonwalk(self):
        """【P8W1b 2026-10-29 口径 v2】惩罚"两轮前后错位"（moonwalk 的直接定义）。

        旧口径（罚左右髋/膝**关节和**不对称）有两个结构性问题（量化见主文档 §3.4 与
        10-29 进展日志 12:35 条）：
          1. 容差 0.01 rad ≈ 左右轮底高差 3 mm ⇒ 横向 5° 坡（16 mm）就把容差吃满，
             容差比地形需求小 5~10 倍；
          2. 语义悖论：强行"机身水平 + 左右差 Δz"时，旧口径的最便宜合规姿势是
             **两轮前后错开 19~95 mm** —— 惩罚的最优应对就是它自己要禁止的 moonwalk。

        新口径：直接罚两轮在**机体系 x** 上的前后错位
            e   = x_left_wheel − x_right_wheel        （`_rb_state[:, feet_indices, 0]`）
            pen = relu(|e| / e_tol − 1)
                  + 0.05 · relu(|ė| / ė_tol − 1)     （错位变化率，防快速蹭步）
            e_tol = 25 mm（平地 p99 6.8 mm、台阶 p95 27 mm ⇒ 不误伤爬坎）
        权重 −20 对无量纲 pen 语义不变；相位掩蔽 [2,3,4] 与"指令 roll 豁免"全部保持。

        数据源：`_rb_state` 由 post_physics_step 开头 refresh（轮位置在腾空段也可用）。
        """
        # 两轮轮心位置（世界系）→ 机体系 x：用 base 的 yaw 四元数把世界系 xy 差转进机体系。
        #   `quat_rotate_inverse` 作用在"世界系向量"上，轮心差是世界系向量 ⇒ 正确口径。
        w_l = self._rb_state[:, self.feet_indices[0], 0:3]
        w_r = self._rb_state[:, self.feet_indices[1], 0:3]
        d_world = w_l - w_r                                   # (N,3)
        d_body = quat_rotate_inverse(self.base_quat, d_world)  # 机体系
        e = d_body[:, 0]                                      # 前后错位 [m]

        # 变化率：前后错位的快变（蹭步）另加小罚；用一阶差分（控制步长 self.dt）
        e_prev = getattr(self, "_moonwalk_e_prev", None)
        if e_prev is None or e_prev.shape != e.shape:
            e_dot = torch.zeros_like(e)
        else:
            e_dot = (e - e_prev) / self.dt
        self._moonwalk_e_prev = e.clone()

        cfg = self.cfg.rewards
        e_tol = float(getattr(cfg, "moonwalk_e_tol", 0.025))       # m
        ed_tol = float(getattr(cfg, "moonwalk_ed_tol", 0.30))      # m/s
        w_rate = float(getattr(cfg, "moonwalk_rate_weight", 0.05)) # 变化率项权重（无量纲）

        # 指令 roll 激活时豁免（压弯需要左右差动构型）
        roll_command_active = (
            torch.abs(self.commands[:, 6])
            > self.cfg.commands.roll_command_deadzone
        )
        pen = torch.relu(torch.abs(e) / e_tol - 1.0) \
            + w_rate * torch.relu(torch.abs(e_dot) / ed_tol - 1.0)

        # 腾空/够地段免罚（掩蔽表 [2,3,4]）+ roll 豁免；行走段（ph=0）口径如上。
        return pen * (~roll_command_active) * (~self._jump_blocked("no_moonwalk"))


    def _reward_ang_vel_xy(self):
        # 惩罚xy轴上的基座角速度，鼓励保持水平稳定
        ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

        # 腾空／够地两段免罚机身横向角速度：这两拍策略能动的只有腿与轮。
        return ang_vel_xy * (~self._jump_blocked("ang_vel_xy"))

    def _reward_orientation(self):
        # 姿态追踪：俯仰始终期望为 0（gravity_xy[:,0] == 0），滚转追踪 commands[:,6]。
        # 全程照罚，不在掩蔽表里。
        target_roll = self.commands[:, 6]
        target_gravity_y = -torch.sin(target_roll)

        # 计算 X 轴偏差 (俯仰误差)
        pitch_error = torch.square(self.projected_gravity[:, 0])

        # 计算 Y 轴偏差 (滚转误差)
        roll_error = self.cfg.rewards.roll_tracking_gain * torch.square(
            self.projected_gravity[:, 1] - target_gravity_y
        )

        # 汇总惩罚
        tilt_penalty = pitch_error + roll_error

        return tilt_penalty

    def _leg_extension(self):
        """沿机体轴的腿伸量 L（左右平均，不含俯仰），单位 m。

        口径与 `_reward_tracking_base_height` 完全一致：
            L = mean( lt·cos(hip) + ls·cos(hip+knee) )，左右两腿平均。
        只依赖关节角，**不读地形高度** ⇒ 在起伏地形上依然成立
        （这正是"体高"在轮腿平台上应有的定义：腿伸了多少，而不是 root_z 减地面）。
        """
        lt = self.cfg.rewards.thigh_length
        ls = self.cfg.rewards.shank_length
        hip_l, knee_l = self.dof_pos[:, self.hip_l_idx], self.dof_pos[:, self.knee_l_idx]
        hip_r, knee_r = self.dof_pos[:, self.hip_r_idx], self.dof_pos[:, self.knee_r_idx]
        return 0.5 * (
            lt * torch.cos(hip_l) + ls * torch.cos(hip_l + knee_l)
            + lt * torch.cos(hip_r) + ls * torch.cos(hip_r + knee_r)
        )

    def _reward_tracking_base_height(self):
        """体高跟踪：追踪 `commands[:,5]`，hinge 形式。
        被奖励的量是**沿机体轴的腿伸量** L（不含俯仰），不是世界系 root 离地高度——后者永远能被"抬头"买到。
        """
        leg_extension = self._leg_extension()
        height_target = self.commands[:, 5] - self.cfg.rewards.wheel_radius
        height_error = torch.abs(leg_extension - height_target)
        reward = torch.clamp(
            1.0 - height_error / self.cfg.rewards.height_tracking_error_limit, min=0.0)
        # 跳跃相位掩蔽（见表）：蹲/推/腾/够地这四段"故意偏离体高目标"是任务本身，罚它＝逼策略抗模板
        return (~self._jump_blocked("tracking_base_height")) * reward

    def _reward_base_height(self):
        # 【已停用】旧的高度平方惩罚（权重 0，上游整项丢弃）。保留实现做回退对照。
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        error_sq = torch.square(base_height - self.commands[:, 5])
        # 曾按 |cmd_vx| 加权（`base_height_speed_gain`）压"高速抬身"，随本项一起停用
        error_sq = error_sq * (1.0 + getattr(self.cfg.rewards, "base_height_speed_gain", 0.0)
                               * torch.abs(self.commands[:, 0]))
        return (~self.is_jumping) * error_sq

    def _reward_no_fly(self):
        # 1. 检查两个轮子是否都踩实了地面 (阈值 0.1 N 接触力)
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1

        # 2. 零容忍判定：只要贴地的轮子总数 小于 2（即 0 个或 1 个），就视为违规离地！
        is_flying = torch.sum(1.0 * contacts, dim=1) < 2

        # 3. 转化为浮点数作为惩罚基数 (离地就是 1.0，完全贴地就是 0.0)
        penalty = is_flying.float()

        # 4. 依然保留白名单：只有**允许离地的那几段**（爆发/腾空/够地）才免罚
        #    【P6 S1】旧写法是整个事件窗免罚 ⇒ 预蹲/缓冲两段"轮子蹭地"也跟着免了
        return (~self._jump_blocked("no_fly")) * penalty

    def _reward_lin_vel_z(self):
        # 1. 获取 Z 轴线速度平方
        z_vel_sq = torch.square(self.base_lin_vel[:, 2])
        rew = z_vel_sq.clone()
        # 2. 判定是否正在主动升降底盘
        #    口径可切（rewards.lin_vel_z_height_from_legs）：
        #    - True（新口径）：用**腿伸量**+轮半径。与 `_reward_tracking_base_height` 同源，
        #      语义正确：腿伸量偏离体高目标 ⇒ 正在主动升降底盘 ⇒ 竖直速度免罚。
        #      起伏地形上 root_z 会随地面波动，旧口径的掩码会随机开关，把惩罚变成噪声。
        #    - False（原口径）：`root_z − measured_heights`。
        #      平地下 measured_heights ≡ 0 ⇒ 退化成"root_z 与体高指令比大小"，掩码基本恒真。
        #    ⚠ 热启动 W12 权重时必须用原口径（False）：换口径等于改奖励分布，会让速度跟踪反向、训练崩。
        if getattr(self.cfg.rewards, "lin_vel_z_height_from_legs", False):
            current_height = self._leg_extension() + self.cfg.rewards.wheel_radius
        else:
            current_height = torch.mean(
                self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        target_height = self.commands[:, 5]
        is_adjusting_height = torch.abs(current_height - target_height) > 0.01  # 放宽到 1cm，防止在边缘反复横跳

        # 逻辑路由：升降底盘时免罚、掩蔽表允许竖直运动的那几段免罚，其余照常计分。
        mask_adjusting = is_adjusting_height & (~self._jump_blocked("lin_vel_z"))
        rew[mask_adjusting] = 0.0

        # 模式 B：跳跃相位表允许 vertical motion 的那几段**免罚**（掩蔽统一走 mask 表，
        #   避免这里和模式 A 各写一套口径）
        rew[self._jump_blocked("lin_vel_z")] = 0.0

        # 模式 C：平地滚动时 (自然保留 rew 为正的 z_vel_sq，配合 config 负系数变成扣分)

        return rew


    def _jump_reward_upright(self):
        """跳跃奖励用的姿态门（比**触发**门宽，见 config jump.upright_limit_reward 的说明）。"""
        j = self.cfg.rewards.jump
        roll, pitch = self._body_roll(), self._body_pitch()
        return ((torch.abs(roll) < j.upright_limit_reward)
                & (torch.abs(pitch) < j.upright_limit_reward))

    def _reward_jump_lift(self):
        """跳跃结果奖：只在**确认起飞那一拍**发一次，被奖励量＝预测总升高。
        分子是高斯带通 `exp(−(h−goal)²/2σ²)`，σ = max(frac×指令, abs) ⇒ 超调越多扣得越多。
        """
        if not getattr(self.cfg.rewards.flags, "enable_jump", False):
            return torch.zeros(self.num_envs, device=self.device)
        # 只在确认起飞那一拍一次结清：奖励量 = h_flight_pred = v²/2g，其中 v 是离地首拍
        # 锁存的质心竖直速度（不含接触段）。读的是相位机在 lift 那拍存的缓冲，与调用顺序无关。
        j = self.cfg.rewards.jump
        goal = torch.clamp(self.h_target, min=float(getattr(j, "goal_floor", 0.005)))
        # 分子是**高斯带通** `exp(−(h−goal)²/2σ²)`（有 0.5 口径，σ 是标准差）：自带上界 1，
        #   且超调越多扣得越多。σ = max(`pulse_sigma_frac`×指令, `pulse_sigma_abs`)。
        sig_frac = float(getattr(j, "pulse_sigma_frac", 0.0))
        sig_abs = float(getattr(j, "pulse_sigma_abs", 0.0))
        if sig_frac > 0.0 or sig_abs > 0.0:
            sigma = torch.clamp(sig_frac * goal, min=sig_abs).clamp(min=1e-3)
            h_flight_pred = self.h_flight_pred
            z = (h_flight_pred - goal) / sigma
            # 口径：单边高斯 exp(−0.5·clamp(z, max=0)²)，σ 固定（不随指令缩放，否则高档梯度太小）。
            #   够到 goal 即满分、超了不罚 —— jump_lift 只负责"确保起飞"，
            #   "越高越好"交给双边的 jump_top_height。
            score = torch.exp(-0.5 * torch.clamp(z, max=0.0) ** 2)
        else:
            score = torch.clamp(h_flight_pred / goal, max=1.0)   # 回退锚：P6R21 前的线性比例
        return (self.ev_lift * self._jump_reward_upright().float()
                * score * self.gate_lift)

    def _reward_jump_leg_apex(self):
        """【P6R57】顶点腿长奖：**相位3→4 那一拍（顶点）一次结清**。

        为什么要有这一项：跳跃奖励一直只买**质心**起飞 vz（`jump_lift` = `air_vz²/2g`），
        **从不为"轮子离地多高"付钱**，空中腿长也没人管 ⇒ 策略的最优解是**把腿伸着**
        （P6R56 实测：+12P 后腾空段腿 253→264 不收反伸，真跳高 55.7→33.1mm）。
        本项直接为"腿在顶点时到位"付钱，把主指标（轮底离地）变成优化目标。

        口径：目标是**模板自己的腿长目标** `ref_L`（不写死数字 ⇒ 与相位3/4 模板平滑衔接）；
        测量是顶点前 `leg_apex_win` 拍的空中腿长**滑动最大**（两腿取 max＝绑定轮口径）；
        分子是高斯 `exp(−0.5·((L−L_target)/σ)²)`。

        ⚠ 本函数**只读**相位机算好的 `ev_apex_leg_pay`（含高斯分与全部门控）：
        奖励函数会被探针重复调用，必须幂等，不能在这里做状态推进。
        """
        if not getattr(self.cfg.rewards.flags, "enable_jump", False):
            return torch.zeros(self.num_envs, device=self.device)
        return self.ev_apex_leg_pay

    def _reward_jump_top_height(self):
        """【P6R68】真跳高（轮底离地）结果奖：**顶点（相位4 起点）一次结清**。

        为什么要有这一项：`jump_lift` 只买**质心**起飞 vz（`v²/2g`）、`jump_leg_apex` 只买**腿长到 220**
        ——两者都不是判据指标（轮底离地）。实测 R64：质心抬升 93mm 但轮底只 69.4mm（顶点腿伸着）。
        本项**直接**用 `wheel_peak_min`（双轮同离地窗轮底峰值，与判据同口径）追指令 `h_target`。
        口径：目标＝`h_target`（指令）；双边高斯 `exp(−0.5·((wheel_peak_min−h_target)/σ)²)`，超调也扣。
        ⚠ 只读相位机算好的 `ev_top_height_pay`（含高斯分与全部门控）：奖励函数会被探针重复调用，必须幂等。
        """
        if not getattr(self.cfg.rewards.flags, "enable_jump", False):
            return torch.zeros(self.num_envs, device=self.device)
        return self.ev_top_height_pay

    def _reward_jump_shape_track(self):
        """空中/够地/缓冲三段的腿伸量设定点跟踪：到位就给分，不约束形状。必须留——空中不收腿、落地前不伸腿都会摔。

        """
        if not getattr(self.cfg.rewards.flags, "enable_jump", False):
            return torch.zeros(self.num_envs, device=self.device)
        j = self.cfg.rewards.jump
        in_shape = (self.ph == 3) | (self.ph == 4) | (self.ph == 5)
        lt, ls = self.cfg.rewards.thigh_length, self.cfg.rewards.shank_length
        hip_l, knee_l = self.dof_pos[:, self.hip_l_idx], self.dof_pos[:, self.knee_l_idx]
        hip_r, knee_r = self.dof_pos[:, self.hip_r_idx], self.dof_pos[:, self.knee_r_idx]
        L_act = 0.5 * (lt * torch.cos(hip_l) + ls * torch.cos(hip_l + knee_l)
                       + lt * torch.cos(hip_r) + ls * torch.cos(hip_r + knee_r))
        tol = float(getattr(j, "shape_tol", 0.02))
        val = torch.clamp(1.0 - (L_act - self.ref_L).abs() / tol, 0.0, 1.0)
        # 门用 self.gate12（= g1·g2，不含 g3）：相位 4/5 保留 tracking，但不注入 g3 梯度。
        w_all = float(getattr(self.cfg.rewards.scales, "jump_shape_track", 0.0))
        w45 = float(getattr(j, "jump_shape_track_ph45", w_all))
        _ratio45 = (w45 / w_all) if w_all != 0.0 else 1.0
        _scale = torch.where(self.ph == 3,
                             torch.ones_like(val),
                             torch.full_like(val, _ratio45))
        return (in_shape.float() * val * self._jump_reward_upright().float() * self.gate12 * _scale)

    def _reward_jump_push_lead(self):
        """爆发段的**微弱**向上引导：只奖励"**质心**在往上加速"这个方向（比值封顶 1.3），不规定腿怎么走。
        挂在直立门与链式门后（预蹲没做对就打折）。口径＝质心（与 P2 后的 `h_flight_pred`/`h_target` 一致）。
        """
        if not getattr(self.cfg.rewards.flags, "enable_jump", False):
            return torch.zeros(self.num_envs, device=self.device)
        j = self.cfg.rewards.jump
        goal = torch.clamp(self.h_target, min=float(getattr(j, "goal_floor", 0.005)))
        v_req = torch.sqrt(2.0 * 9.81 * goal)          # 指令＝离地后上升 ⇒ √(2g·h*) 就是所需离场速度
        # 分子也是高斯带通 `exp(−e²/2σ²)`（有 0.5 口径），σ_v = `push_sigma_frac`×v_req。
        #   σ=0 时退回 clamp(v_z / v_req, 0, 1)。
        _sig = float(getattr(j, "push_sigma_frac", 0.0))
        _vz_com = self.com_vz_step                # 质心口径：真实上冲动量，不含「伸腿甩机身」那份
        if _sig > 0.0:
            sg = _sig * v_req
            lead = torch.exp(-torch.square(_vz_com - v_req) / (2.0 * sg * sg))
        else:
            lead = torch.clamp(_vz_com / v_req, 0.0, 1.3)   # 上限 1.3：封顶到 1.0 会让 vz ≥ v_req 时梯度归零、策略提前撤回动作
        return ((self.ph == 2).float() * lead
                * self._jump_reward_upright().float() * self.gate_chain)

    def _reward_jump_crouch_track(self):
        """预蹲跟踪奖：相位 1 内**逐拍发放**，每拍＝权重×Δt×本条得分（见 `_update_jump_phase` 2f）。
        口径＝机身高度 vs 绝对目标，得分是高斯 `exp(−e²/2σ²)` ⇒ 偏离越多扣得越多但梯度不断。
        不挂 `ev_lift`：起飞不可达时那条闸会把预蹲的梯度整条掐死。
        """
        if not getattr(self.cfg.rewards.flags, "enable_jump", False):
            return torch.zeros(self.num_envs, device=self.device)
        return (self.ph == 1).float() * self.crouch_credit_step

    def _reward_jump_landing(self):
        """落地质量奖：触地那一拍**一次结清**。
        门＝确认起飞 × 直立门 × 落地冲击 soft × g1 × 腿伸比值；**前序票只取 g1**——落地钱只该问"落得稳不稳"。
        """
        if not getattr(self.cfg.rewards.flags, "enable_jump", False):
            return torch.zeros(self.num_envs, device=self.device)
        return self.ev_land_pay

    def post_physics_step(self):
        # viewer 灯光幂等重放（首帧生效；有 cfg.viewer.light_* 才动手）
        if not getattr(self, "_light_applied", False):
            self.apply_viewer_light()
            self._light_applied = True

        self.theta_right = self.dof_pos[:, self.knee_r_idx]
        self.theta_left = self.dof_pos[:, self.knee_l_idx]

        # 刷新刚体状态张量：`_update_jump_phase`（在父类 post_physics_step 的回调里推进）要读
        #   `_rb_state` 算轮底离地，必须在父类回调之前刷新，否则读到上一拍。
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        super().post_physics_step()

        # ==================== [核心修复：过滤轮子误差] ====================
        # 1. 计算当前的关节位置误差
        current_dof_err = self.dof_pos - self.default_dof_pos

        # 2. 剥夺轮子的绝对位置记忆 (将轮子通道的误差强行设为 0)
        current_dof_err[:, self.wheel_indices] = 0.0
        # ==================================================================

        # 使用滑动窗口更新 DoF 的历史信息 (用于 obs_buf 中的 161 维观测)
        self.dof_pos_error_history = torch.roll(self.dof_pos_error_history, shifts=-1, dims=1)
        # 3. 存入经过过滤的 current_dof_err，而不是原始的 dof_pos - default_dof_pos
        self.dof_pos_error_history[:, -1, :] = current_dof_err

        # 速度历史可以直接存，因为轮子的速度是有限大小的，不会无限膨胀，且网络需要知道轮速
        self.dof_vel_history = torch.roll(self.dof_vel_history, shifts=-1, dims=1)
        self.dof_vel_history[:, -1, :] = self.dof_vel

    def reset_idx(self, env_ids):
        # 环境重置时的处理，调用父类的重置逻辑和清空历史信息
        super().reset_idx(env_ids=env_ids)

        if len(env_ids) == 0:
            return

        # reset 清 ev_takeoff 之前，把"本事件内曾经确认起飞"与终止原因写进 extras，
        # 供探针区分"压根没跳"和"跳了但撞地"。只进诊断，不参与奖励。
        self.extras["episode"]["jump_takeoff_ever"] = \
            float(self.ev_takeoff_ever[env_ids].float().mean().item())
        self.extras["episode"]["jump_terminated_in_phase"] = \
            float(((self.ph[env_ids] > 0) & (self.ph[env_ids] < 5)).float().mean().item())

        # 重置时必须清空历史信息，防止旧数据干扰新回合
        if hasattr(self, "command_targets"):
            self.commands[env_ids] = self.command_targets[env_ids]
        self.reset_jump_state(env_ids)   # P6：相位机不能带着上一回合的相位/基准高度进新回合

        self.dof_pos_error_history[env_ids].zero_()
        self.dof_vel_history[env_ids].zero_()

    def _update_turning_roll_targets(self, env_ids=None):
        """roll 指令 ＝ 静态 roll 目标 ＋ 转向压弯前馈，整体夹回 `ranges.ang_roll`。

        """
        lo, hi = self.command_ranges["ang_roll"]
        targets = self.command_targets if env_ids is None else self.command_targets[env_ids]
        base = self.base_roll_targets if env_ids is None else self.base_roll_targets[env_ids]
        if getattr(self.cfg.rewards.flags, "enable_turning_tilt", False):
            coeff = getattr(self.cfg.commands, "turning_roll_coeff", -1.0 / 9.81)
            tilt = coeff * targets[:, 0] * (YAW_SIGN * targets[:, 2])
        else:
            tilt = torch.zeros_like(targets[:, 0])
        roll_targets = torch.clamp(base + tilt, lo, hi)
        _ids = slice(None) if env_ids is None else env_ids
        # 只读快照：① 原始侧倾指令 ② 压弯偏置 ③ 夹持后的侧倾目标 ⇒ 可直接查"目标 vs 指令"。
        self.dbg["roll_base"][_ids] = base
        self.dbg["roll_tilt"][_ids] = tilt
        self.dbg["roll_tgt"][_ids] = roll_targets
        if env_ids is None:
            self.command_targets[:, 6] = roll_targets
        else:
            self.command_targets[env_ids, 6] = roll_targets

        # ---- [目标可达包络：指令 → 目标这一步] ----
        # 包络只加在**目标层**、不缩 `commands.ranges` ⇒ 观测与跟随的都是目标。

    def _leg_height_ceiling(self, hip, knee_idx, sign):
        """当前髋角能撑起的最大**高度目标**，与 tracking_base_height 同口径（俯仰无关）：
        H = r_w + Lt·cos(髋) + Ls。解算器内部用的 _reach_max_height 是带俯仰的，
        这里必须用俯仰无关版本，否则又会造出"奖励说做不到、解算器说做得到"的口径分裂。"""
        lt = self.cfg.rewards.thigh_length
        ls = self.cfg.rewards.shank_length
        rw = self.cfg.rewards.wheel_radius
        return rw + lt * torch.cos(hip) + ls

    def _apply_command_envelope(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        env_cfg = getattr(self.cfg.commands, "envelope", None)
        if env_cfg is None:
            return
        expand = getattr(env_cfg, "expand", 1.0)
        v_tgt = self.command_targets[ids, 0]
        # 包络输入是**原始指令** `command_raw_yaw`，不是 `command_targets[:,2]`：
        #   ⇒ **包络永远从"指令"算，不从上一次的"目标"算**。
        w_tgt = self.command_raw_yaw[ids]
        # 带帽直线 ω ≤ expand·(b − k·|v|)：低速不削、满速端削
        cap = expand * (getattr(env_cfg, "yaw_taper_b", 2.14)
                        - getattr(env_cfg, "yaw_taper_k", 1.34) * torch.abs(v_tgt))
        cap = torch.clamp(cap, min=0.0)
        w_new = torch.sign(w_tgt) * torch.minimum(torch.abs(w_tgt), cap)
        # 诊断：记"本控制步 ω 需求被削过"与当时的 cap，供固定指令实测区分两种失败——目标层还在
        # 提不可达需求 vs 需求已收到能力内但策略没跟上。
        self.env_yaw_clipped[ids] = ((torch.abs(w_tgt) - torch.abs(w_new)) > 1e-4).float()
        self.env_yaw_cap[ids] = cap
        # 只读快照必须在写回之前取：w_tgt 是 command_targets 的视图，写回后就看不到原始指令了
        self.dbg["yaw_raw"][ids] = w_tgt
        self.dbg["yaw_tgt"][ids] = w_new
        self.command_targets[ids, 2] = w_new
        # 高度目标夹到"当前实测髋角下的几何可达上限"：跟踪奖励与膝角目标必须同口径，
        # 否则髋前伸后等于对做不到的目标硬罚，策略会改用俯仰补。
        if getattr(env_cfg, "height_target_use_reach_clamp", False):
            ceil_l = self._leg_height_ceiling(self.dof_pos[ids, self.hip_l_idx], self.knee_l_idx, +1.0)
            ceil_r = self._leg_height_ceiling(self.dof_pos[ids, self.hip_r_idx], self.knee_r_idx, -1.0)
            self.command_targets[ids, 5] = torch.minimum(
                self.command_targets[ids, 5], torch.minimum(ceil_l, ceil_r))

    def _hip_allow_ceiling(self, h_target, knee_idx, sign):
        """撑得起当前高度目标的最大髋角，**与跟踪奖励同口径（俯仰无关）**。
        返回**正**角度（没用 sign 参数），右髋 `dof_pos` 是负的 ⇒ 调用处必须按 |髋| 比较。
        """
        lt = self.cfg.rewards.thigh_length
        ls = self.cfg.rewards.shank_length
        rw = self.cfg.rewards.wheel_radius
        cos_t = (h_target - rw - ls) / lt
        return torch.acos(torch.clamp(cos_t, -1.0, 1.0))

    def _reward_hip_reach(self):
        """髋可达单边惩罚：只罚"髋前伸到撑不起当前高度目标"。跳跃期整项关闭。

        """
        h_tgt = self.commands[:, 5]
        margin = getattr(self.cfg.rewards, "hip_reach_margin", 0.02)
        allow_l = self._hip_allow_ceiling(h_tgt, self.knee_l_idx, +1.0) - margin
        allow_r = self._hip_allow_ceiling(h_tgt, self.knee_r_idx, -1.0) - margin
        # `_hip_allow_ceiling` 返回**正**角度（没用 sign 参数）而右髋 `dof_pos` 是负的 ⇒ 必须按
        #   **|髋|** 比较；旧写法 `relu(hip_r − allow_r)` 自变量恒为负，右髋从来没被罚过。
        over = torch.relu(torch.abs(self.dof_pos[:, self.hip_l_idx]) - allow_l) \
             + torch.relu(torch.abs(self.dof_pos[:, self.hip_r_idx]) - allow_r)
        # 诊断（P5 R7 ④）：先记"髋撑不起当前高度目标"的原始越限标记，**不受速度门控影响**，
        # 这样被门控放行的那部分也能在固定指令实测里数出来（否则分不清"没越限"和"越了但没罚"）。
        self.env_hip_infeasible[:] = (over > 0).float()
        # 只读快照：两腿的允许髋角（已减容差）与原始越限量 ⇒ 评测能直接算"还差多少度收不回来"
        self.dbg["hip_allow_l"][:] = allow_l
        self.dbg["hip_allow_r"][:] = allow_r
        self.dbg["hip_over"][:] = over
        gate = getattr(self.cfg.rewards, "hip_reach_speed_gate", 0.0)
        if gate > 0.0:
            over = over * (torch.abs(self.commands[:, 0]) <= gate)
        return (~self._jump_blocked("hip_reach")) * over

    def _apply_command_smoothing(self):
        if not getattr(self.cfg.commands, "smooth_commands", False):
            self.commands[:] = self.command_targets[:]
        else:
            max_delta = self.command_rate_limits * self.dt
            delta = self.command_targets - self.commands
            limited_delta = torch.minimum(torch.maximum(delta, -max_delta), max_delta)
            self.commands[:] += limited_delta

    def dbg_snapshot(self, env_idx=0):
        """把诊断量拷进 `self.dbg`（只读快照，不参与计算）。

        """
        keys = list(self.dbg.keys())
        extra = ["yaw_clipped", "yaw_cap"]          # 1 = 本控制步 ω 需求被包络削过 / 当时的 ω 上限
        flat = torch.stack([self.dbg[k] for k in keys]
                           + [self.env_yaw_clipped, self.env_yaw_cap])[:, env_idx]
        return dict(zip(keys + extra, flat.detach().cpu().tolist()))  # 一次 CPU 传输，别逐字段 .item()

    def _post_physics_step_callback(self):

        # 按照配置的时间间隔重新采样指令
        env_ids = (self.episode_length_buf
                   % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        if self.cfg.commands.heading_command:
            # 计算朝向误差并转化为偏航角速度指令
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            # YAW_SIGN：朝向误差为正＝目标在左（要左转得给机身**正**角速度），而对外口径是"正=右转"
            # ⇒ 乘 YAW_SIGN。当前 `heading_command=False` 是死代码，但留着不翻将来一开就会左右颠倒。
            self.command_targets[:, 2] = torch.clip(
                YAW_SIGN * 1.5 * wrap_to_pi(self.command_targets[:, 3] - heading),
                self.cfg.commands.ranges.ang_vel_yaw[0],
                self.cfg.commands.ranges.ang_vel_yaw[1],
            )

        # 【P5R7】顺序＝包络 → 压弯合成 → 平滑：包络必须在前，压弯前馈是 coeff·v·ω，先用未削的 ω
        # 合成侧倾再削 ω，机身就会按"没打算转那么多"的角度倾斜（旧顺序在 |v|>1.0 角点上虚给 15%）。
        self._apply_command_envelope()
        self._update_turning_roll_targets()
        self._apply_command_smoothing()
        # P6：跳跃相位机放在回调里推进（每控制步恰好一次），解算器只读它的当前偏移 ⇒ 探针/play/
        # 奖励函数多次调用解算器都不会快进时钟。放在平滑之后是因为触发要看 `commands[:,4]`。
        self._update_jump_phase()

        if self.cfg.terrain.measure_heights:    # 地形高度测量
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.cfg.domain_rand.push_interval == 0
        ):   # 定期施加随机外部扰动，增强鲁棒性
            self._push_robots()

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        # 生成目标速度指令。实际下发的 self.commands 会在每个控制步中平滑逼近 self.command_targets。
        self.command_targets[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.command_targets[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        # 设置速度指令的死区，防止过小的指令导致不稳定的动作频繁切换
        self.command_targets[env_ids, :2] *= (
            torch.norm(self.command_targets[env_ids, :2], dim=1) > 0.1
        ).unsqueeze(1)

        if self.cfg.commands.heading_command:  # 基于朝向
            self.command_targets[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:  # 基于角速度
            self.command_targets[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

            # 保留足够的零偏航样本，避免放开转向后挤占直线和站立训练。
            turning_mask = torch.rand(len(env_ids), device=self.device) < getattr(
                self.cfg.commands, "turning_command_probability", 0.5
            )
            yaw_deadzone = getattr(self.cfg.commands, "yaw_command_deadzone", 0.1)
            meaningful_yaw = torch.abs(self.command_targets[env_ids, 2]) > yaw_deadzone
            self.command_targets[env_ids, 2] *= turning_mask & meaningful_yaw

        # 记下**原始偏航指令**（未经包络）：`_apply_command_envelope` 从这里取输入 ⇒
        #   目标层是"指令的函数"而不是"上一拍目标的函数"。
        self.command_raw_yaw[env_ids] = self.command_targets[env_ids, 2]

        # 事件窗内（`ph > 0`）不覆盖跳跃通道：观测喂活指令、结算用触发拍冻结的 `h_target`，
        #   中途改写会让两者分叉。其余六个通道照常重采。
        _busy = self.ph[env_ids] > 0
        _h_prev = self.command_targets[env_ids, 4].clone()
        # 清除之前的跳跃/高度/roll 目标，再重新采样
        self.command_targets[env_ids, 4:7] = 0.0
        target_heights = torch_rand_float(
            self.command_ranges["base_height"][0],
            self.command_ranges["base_height"][1],
            (len(env_ids), 1),
            device=self.device,
        )

        self.command_targets[env_ids, 5] = target_heights.squeeze(1)

        # 2. 采样基础滚转角 (使用 torch_rand_float)
        base_rolls = torch_rand_float(
            self.command_ranges["ang_roll"][0],
            self.command_ranges["ang_roll"][1],
            (len(env_ids), 1),
            device=self.device,
        )
        roll_mask = torch.rand(len(env_ids), device=self.device) < getattr(
            self.cfg.commands, "roll_command_probability", 1.0
        )
        meaningful_roll = (
            torch.abs(base_rolls.squeeze(1))
            > self.cfg.commands.roll_command_deadzone
        )
        base_rolls *= (roll_mask & meaningful_roll).unsqueeze(1)
        self.base_roll_targets[env_ids] = base_rolls.squeeze(1)

        # ---- [跳跃指令采样] ----
        # 放在 v/ω/roll 三个指令定下来之后（门要看这三条：带转向/带侧倾不跳）。
        cmd = self.cfg.commands
        l, r = self.command_ranges["jump_height"]
        jump_heights = torch.rand(len(env_ids), 1, device=self.device) * (r - l) + l
        # **跳跃回合子集每次重采样重新抽**，不固定到某个环境（永久划分会让那批环境永远只
        #   练站立跳跃）。跳跃回合发：vx=ω=roll=0 ＋ 高度取 `jump_stand_height` ＋ 必发跳跃指令。
        jr = torch.rand(len(env_ids), device=self.device) < getattr(cmd, "jump_stand_ratio", 0.0)
        if bool(jr.any()):
            self.command_targets[env_ids[jr], 0] = 0.0    # vx
            self.command_targets[env_ids[jr], 2] = 0.0    # yaw
            base_rolls[jr] = 0.0
            self.base_roll_targets[env_ids[jr]] = 0.0
            sh_lo, sh_hi = getattr(cmd, "jump_stand_height", self.command_ranges["base_height"])
            self.command_targets[env_ids[jr, None], 5] = (
                torch.rand(int(jr.sum()), 1, device=self.device) * (sh_hi - sh_lo) + sh_lo)
        want = (torch.rand(len(env_ids), device=self.device) < cmd.jump_command_probability) | jr
        quiet = (
            (torch.abs(self.command_targets[env_ids, 0]) <= cmd.jump_vx_gate)
            & (torch.abs(self.command_targets[env_ids, 2]) <= cmd.jump_yaw_gate)
            & (torch.abs(base_rolls.squeeze(1)) <= cmd.jump_roll_gate)
            # 指令侧站定门·高度：体高指令必须落在 [jump_cmd_h_min, jump_cmd_h_max]。
            # ⚠ 实测侧（quiet_now 那一项）刻意不加绝对带：预蹲会把 z 压到带外，
            #   加了会导致跳一次之后永久不再发指令。
            & (self.command_targets[env_ids, 5] >= cmd.jump_cmd_h_min)
            & (self.command_targets[env_ids, 5] <= cmd.jump_cmd_h_max)
        )
        # 指令侧站定门：要求机器真站住（quiet_now 连续 settle_steps 拍）。只管发不发指令，不管触发。
        # ⚠ 评测 / 回放只在复位后那一拍重采样 ⇒ 靠 reset_jump_state 播种 cnt_quiet。
        if bool(getattr(self.cfg.rewards.jump, "cmd_settle_gate", False)):
            quiet = quiet & (
                self.cnt_quiet[env_ids]
                >= max(int(getattr(self.cfg.rewards.jump, "settle_steps", 10)), 1))
        jump_heights[~(want & quiet)] = 0.0
        # 跳跃指令不整窗写进观测：只把请求高度存进 h_request，command_targets[:,4] 保持 0，
        # 触发那一拍相位机才写进去。_busy（事件窗内）不覆盖 h_request，避免打断正在进行的一跳。
        self.h_request[env_ids] = torch.where(
            _busy, self.h_request[env_ids], jump_heights.squeeze(1))
        self.command_targets[env_ids, 4] = torch.where(
            _busy, _h_prev, torch.zeros_like(jump_heights.squeeze(1)))
        # 装填：一条指令最多跳一次，落地缓冲结束后不回摆，等下一次重采样再装填
        # 装填由 h_request 驱动（command_targets[:,4] 在站立期恒 0）
        self.is_armed[env_ids] = (~_busy) & (self.h_request[env_ids] > 0.0)
        # 记下装填时刻，arm_delay 从这里起算
        self.arm_at[env_ids] = self.episode_length_buf[env_ids]
        # 【P6 S2c】记下本窗哪些环境属于跳跃回合子集（连跳只对它们开放），
        # 并把连跳计数清零。`is_jump_round` 必须与触发段读到的 `jr` 完全一致 ⇒ 同一个掩码。
        self.is_jump_round[env_ids] = jr
        self.rearm_cnt[env_ids] = 0

        # 最终 roll 目标 = 基础随机 roll + 基于最新 yaw 目标的转向压弯偏置
        # （顺序与 _update_commands 里一致：包络先于压弯合成，前馈才用削过的 ω）
        self._apply_command_envelope(env_ids)
        self._update_turning_roll_targets(env_ids)

    def _get_noise_scale_vec(self, cfg):
        """Build noise scales for WBR's 33 + 120 observation layout (161 dim total)."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = cfg.noise.add_noise
        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level

        # 基础观测 33 维
        # 0:3  - base_lin_vel
        noise_vec[0:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        # 3:6  - base_ang_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # 6:9  - projected_gravity
        noise_vec[6:9] = noise_scales.gravity * noise_level
        # 观测噪声：0:9 速度/重力、15:21 dof_pos、21:27 dof_vel 加噪声；9:15 指令、27:33 动作不加。
        dof_pos_noise = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[15:21] = dof_pos_noise
        # 21:27 - dof_vel (6 joints)
        dof_vel_noise = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[21:27] = dof_vel_noise
        if cfg.env.enable_joint_state_history:
            history_size = cfg.env.joint_state_history_length * self.num_actions  # 10 * 6 = 60 per pos/vel
            history_pos_start = 33
            history_vel_start = history_pos_start + history_size  # 33 + 60 = 93

            # 位置历史噪声 (33:93)
            history_pos_noise = noise_vec[history_pos_start:history_vel_start].view(
                cfg.env.joint_state_history_length, self.num_actions
            )
            history_pos_noise[:, self.leg_indices] = (
                dof_pos_noise * self.obs_scales.joint_state_history
            )
            # 轮子位置历史不加噪声（已在观测中清零）
            history_pos_noise[:, self.wheel_indices] = 0.0

            # 速度历史噪声 (93:153)
            noise_vec[history_vel_start:history_vel_start + history_size] = (
                dof_vel_noise * self.obs_scales.joint_state_history
            )

        return noise_vec

    def compute_observations(self):
        # 轮子位置误差**强制清零**：[极度重要] `dof_pos_error_history` 存的必须是这个清零后的
        # `dof_err`（不是原始 `dof_pos`），否则网络仍能感知轮子绝对位置。
        dof_err = self.dof_pos - self.default_dof_pos
        # 将轮子的位置误差强行设为 0（剥夺网络对轮子绝对位置的感知）
        dof_err[:, self.wheel_indices] = 0.0

        # ⚠ 极度重要：别处更新 `dof_pos_error_history` 时必须存这个**清零后的 `dof_err`**。
        command_obs = self.commands[:, :3] * self.commands_scale
        command_obs[:, 2] = YAW_SIGN * self.commands[:, 2] * self.obs_scales.yaw_command
        # 体高指令单独归一化到 ±1（按指令范围中心化后除以半宽）。
        h_lo, h_hi = self.cfg.commands.ranges.base_height
        h_mid, h_half = 0.5 * (h_lo + h_hi), max(0.5 * (h_hi - h_lo), 1.0e-6)
        auxiliary_command_obs = self.commands[:, 5:7] * self.obs_scales.dof_pos
        auxiliary_command_obs[:, 0] = (self.commands[:, 5] - h_mid) / h_half
        auxiliary_command_obs[:, 1] = (
            self.commands[:, 6] * self.obs_scales.roll_command
        )

        # 跳跃指令归一化到 0~1（按量程 `j_hi`）：沿用 height_measurements=5.0 时观测只有
        #   0.05~0.10、与"不发跳"(0) 差不到 0.1 ⇒ 网络分不出该不该跳而模板已经在动膝。
        j_hi = max(self.cfg.commands.ranges.jump_height[1], 1.0e-6)

        # 构建基础观测 (33维)
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,           # 3
            self.base_ang_vel * self.obs_scales.ang_vel,           # 3
            self.projected_gravity,                                # 3
            command_obs,                                           # 3
            (self.commands[:, 4] / j_hi).unsqueeze(-1), # 1 (跳跃指令，归一化到 0~1)
            auxiliary_command_obs,                                # 2 (高度与滚转指令)
            dof_err * self.obs_scales.dof_pos,                     # 6 [已修复：轮子误差永远为0]
            self.dof_vel * self.obs_scales.dof_vel,                # 6 (速度不需要清零，网络必须知道轮速)
            self.actions                                           # 6
        ), dim=-1) # Total = 33

        # 拼接关节历史信息
        if self.cfg.env.enable_joint_state_history:
            self.obs_buf = torch.cat((
                self.obs_buf,
                self.dof_pos_error_history.view(self.num_envs, -1) * self.obs_scales.dof_pos * self.obs_scales.joint_state_history,
                self.dof_vel_history.view(self.num_envs, -1) * self.obs_scales.dof_vel * self.obs_scales.joint_state_history,
            ), dim=-1)

        # 拼接地形高度
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0,)
                * self.obs_scales.height_measurements
            )
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # 跳跃相位块 8 维追加在**最末尾**(153:161)：旧 153 列一列不挪 ⇒ 热启动只需补 8 个零列。
        #   内容＝相位 one-hot 6 ＋ 模板膝偏移 1 ＋ 升高进度 1。不按 `enable_jump` 关闭。
        j_jump = self.cfg.rewards.jump
        phase_one_hot = torch.stack(
            [(self.ph == k).float() for k in range(6)], dim=1)
        off_col = (self.off_knee / max(float(j_jump.off_limit), 1.0e-6)).unsqueeze(1)
        goal_col = torch.clamp(self.h_target, min=float(j_jump.goal_floor))
        # "跳到哪儿了"＝离地后**质心**上升 ÷ `h_target`，零点＝离地首拍质心高；没离地给 0（含接触段会把
        #   "顶高"读成进度）。与奖励分子同口径（质心）。
        prog_col = (torch.clamp(
            (self._system_com_z() - self.air_z) / goal_col, 0.0, 1.0)
            * (self.ph > 0).float() * self.air_latched.float()).unsqueeze(1)
        self.obs_buf = torch.cat((self.obs_buf, phase_one_hot, off_col, prog_col), dim=-1)

        # 注入噪声增强鲁棒性
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _compute_torques(self, actions):
        actions_scaled = torch.zeros_like(actions)
        # actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.leg_indices] = actions[:, self.leg_indices] * self.cfg.control.leg_action_scale

        desired_wheel_targets = (
            actions[:, self.wheel_indices] * self.cfg.control.wheel_action_scale
        )
        wheel_target_accel_limit = getattr(
            self.cfg.control, "wheel_target_accel_limit", 0.0
        )
        if wheel_target_accel_limit > 0.0:
            max_target_delta = wheel_target_accel_limit * self.sim_params.dt
            target_delta = torch.clamp(
                desired_wheel_targets - self.wheel_target_vel,
                -max_target_delta,
                max_target_delta,
            )
            self.wheel_target_vel += target_delta
        else:
            self.wheel_target_vel[:] = desired_wheel_targets
        actions_scaled[:, self.wheel_indices] = self.wheel_target_vel

        # 跳跃模板的**物理授权**：相位偏移加进真正下发给 PD 的膝角目标，不只是奖励目标。
        #   `action_authority` 是交接旋钮（1.0＝模板驱动腿、0.0＝全交给策略）。
        if getattr(self.cfg.rewards.flags, "enable_jump", False):
            authority = getattr(self.cfg.rewards.jump, "action_authority", 0.0)
            if authority > 0.0:
                off = authority * self.off_knee
                # `authority_mode="override"`：指定相位里膝的物理目标**直接等于**模板绝对角，
                #   策略的腿动作项整列丢掉；覆盖段之外仍走加性叠加。
                ovm = torch.zeros_like(off, dtype=torch.bool)
                hip_done = False
                if str(getattr(self.cfg.rewards.jump, "authority_mode", "additive")) == "override":
                    for k in tuple(getattr(self.cfg.rewards.jump, "override_phases", (1,))):
                        ovm |= (self.ph == int(k))
                    # air_override_hip：只对髋额外强制这几个相位（膝仍 additive），
                    # 把腾空相位 3 的髋钉到 2R 目标，防止策略把髋压回导致机身后仰。
                    ovm_hip = ovm.clone()
                    for k in tuple(getattr(self.cfg.rewards.jump, "air_override_hip", ()) or ()):
                        ovm_hip |= (self.ph == int(k))
                    tgt_l = torch.clamp(self.ref_k + off,
                                        self.dof_pos_limits[self.knee_l_idx, 0],
                                        self.dof_pos_limits[self.knee_l_idx, 1])
                    tgt_r = torch.clamp(self.ref_k_r - off,
                                        self.dof_pos_limits[self.knee_r_idx, 0],
                                        self.dof_pos_limits[self.knee_r_idx, 1])
                    actions_scaled[:, self.knee_l_idx] = torch.where(
                        ovm, tgt_l - self.default_dof_pos[:, self.knee_l_idx],
                        actions_scaled[:, self.knee_l_idx] + off)
                    actions_scaled[:, self.knee_r_idx] = torch.where(
                        ovm, tgt_r - self.default_dof_pos[:, self.knee_r_idx],
                        actions_scaled[:, self.knee_r_idx] - off)
                    if bool(getattr(self.cfg.rewards.jump, "override_covers_hip", True)):
                        hlim_l = self.dof_pos_limits[self.hip_l_idx]
                        hlim_r = self.dof_pos_limits[self.hip_r_idx]
                        h_off = authority * self.off_hip
                        tgt_hl = torch.clamp(self.ref_hip + h_off, hlim_l[0], hlim_l[1])
                        # 右髋用实测冻结值、不用左髋镜像（同 `ref_k_r` 那条理由：进覆盖段会硬跳）
                        tgt_hr = torch.clamp(self.ref_hip_r - h_off, hlim_r[0], hlim_r[1])
                        actions_scaled[:, self.hip_l_idx] = torch.where(
                            ovm_hip, tgt_hl - self.default_dof_pos[:, self.hip_l_idx],
                            actions_scaled[:, self.hip_l_idx] + h_off)
                        actions_scaled[:, self.hip_r_idx] = torch.where(
                            ovm_hip, tgt_hr - self.default_dof_pos[:, self.hip_r_idx],
                            actions_scaled[:, self.hip_r_idx] - h_off)
                        hip_done = True
                else:
                    actions_scaled[:, self.knee_l_idx] += off  # 镜像：左 + 右 −，见上面的 P6 符号说明
                    actions_scaled[:, self.knee_r_idx] -= off
                # 髋通道（additive 或 override 未覆盖到髋时）：两自由度解出的髋偏移（竖直行程来源）
                #   ＋ 可选的旧补偿（默认增益 0）＋ 俯仰反馈（**必须按相位门住**）。
                if not hip_done:
                    hoff = authority * self.off_hip
                    hcg = float(getattr(self.cfg.rewards.jump, "hip_comp_gain", 0.0))
                    if hcg != 0.0:
                        hoff = hoff + hcg * authority * self.off_hip_comp
                else:
                    hoff = torch.zeros_like(off)
                hpg = float(getattr(self.cfg.rewards.jump, "hip_pitch_gain", 0.0))
                if hpg != 0.0:
                    hoff = hoff + hpg * authority * self._body_pitch() * (self.ph > 0).float()
                if not hip_done:
                    actions_scaled[:, self.hip_l_idx] += hoff
                    actions_scaled[:, self.hip_r_idx] -= hoff
                else:
                    actions_scaled[:, self.hip_l_idx] += hoff  # 覆盖段只再叠俯仰反馈（hoff 已含 0 髋偏移）
                    actions_scaled[:, self.hip_r_idx] -= hoff

        # 跳跃相位专用膝阻尼：给相位 1 / 3 / 4 / 5 的环境加，行走 / 高度 / 侧倾不受影响。
        #   压住腾空与落地前那记快速伸腿；相位 2（爆发）不加，避免吃掉爆发扭矩。
        #   注意 d_gains[i] 是按自由度的标量而补偿要按环境 ⇒ 膝这两列做成 [n] 向量。
        dext = getattr(self.cfg.rewards.jump, "knee_damp_extra", 0.0)
        d_col = [self.d_gains[i] for i in range(self.num_dof)]
        if getattr(self.cfg.rewards.flags, "enable_jump", False):
            # knee_damp_extra 支持 5 维数组 index0..4 = 相位 1..5 的膝阻尼增量
            # （标量则只作用于相位 1/3/4/5）。
            #   按环境取对应相位增量；D 加 `clamp(min=0)` 防负阻尼（负阻尼＝往关节注能量、会炸）。
            if isinstance(dext, (list, tuple)):
                _dv = torch.tensor((list(dext) + [0.0] * 5)[:5], device=self.device)
                _de = _dv[(self.ph - 1).clamp(min=0, max=4).long()] * (self.ph > 0).float()
            else:
                _de = float(dext) * ((self.ph == 1) | (self.ph == 3) | (self.ph == 4) | (self.ph == 5)).float()
            if float(torch.abs(_de).max()) > 0.0:
                d_col[self.knee_l_idx] = torch.clamp(self.d_gains[self.knee_l_idx] + _de, min=0.0)
                d_col[self.knee_r_idx] = torch.clamp(self.d_gains[self.knee_r_idx] + _de, min=0.0)

        # 爆发相位（相位 2）专用膝 P 增益：加大位置刚度 ⇒ 同一段行程给出更大的力矩冲量。
        #   只加相位 2：相位 1 预蹲是 override 物理强制，加大 P 会让蹲过头；髋不是瓶颈，不动。
        #   注意 p_gains[i] 是按自由度的标量而补偿要按环境 ⇒ 膝这两列做成 [n] 向量。
        kp_ex = getattr(self.cfg.rewards.jump, "knee_kp_extra", 0.0)
        p_col = [self.p_gains[i] for i in range(self.num_dof)]
        if getattr(self.cfg.rewards.flags, "enable_jump", False):
            # knee_kp_extra 支持 5 维数组 index0..4 = 相位 1..5 的膝 P 增益增量（标量只作用于相位 2）。
            if isinstance(kp_ex, (list, tuple)):
                _kv = torch.tensor((list(kp_ex) + [0.0] * 5)[:5], device=self.device)
                _ke = _kv[(self.ph - 1).clamp(min=0, max=4).long()] * (self.ph > 0).float()
            else:
                _ke = float(kp_ex) * (self.ph == 2).float()
            if float(torch.abs(_ke).max()) > 0.0:
                p_col[self.knee_l_idx] = torch.clamp(self.p_gains[self.knee_l_idx] + _ke, min=0.0)
                p_col[self.knee_r_idx] = torch.clamp(self.p_gains[self.knee_r_idx] + _ke, min=0.0)

        torques = torch.zeros_like(self.dof_vel)
        # 新增一个容器，专门存还没被 clip（截断）的原始力矩
        raw_torques = torch.zeros_like(self.dof_vel)

        # D 项按物理子步计算，并对接触造成的轮速尖峰进行低通滤波。
        current_wheel_vel = self.dof_vel[:, self.wheel_indices]
        raw_wheel_acceleration = (
            current_wheel_vel - self.last_wheel_control_vel
        ) / self.sim_params.dt
        filter_alpha = self.cfg.control.wheel_acceleration_filter_alpha
        self.filtered_wheel_acceleration.mul_(filter_alpha).add_(
            raw_wheel_acceleration, alpha=1.0 - filter_alpha
        )
        wheel_acceleration = torch.zeros_like(self.dof_vel)
        wheel_acceleration[:, self.wheel_indices] = self.filtered_wheel_acceleration

        for i in range(self.num_dof):
            if i in self.wheel_indices:
                # ➡️ [轮子：V 控制 (引入 D 分量)]
                # 速度误差: 目标速度 - 当前速度
                vel_error = actions_scaled[:, i] - self.dof_vel[:, i]
                # 完整 PID 公式 (针对速度): Kp * (ΔV) - Kd * A
                calc_torque = (
                    self.p_gains[i] * vel_error
                    - self.d_gains[i] * wheel_acceleration[:, i]
                )
                raw_torques[:, i] = calc_torque
            else:
                # ➡️ [腿部：PD 位置控制]
                calc_torque = (
                    p_col[i] * (actions_scaled[:, i] + self.default_dof_pos[:, i] - self.dof_pos[:, i])
                    - d_col[i] * self.dof_vel[:, i]        # 跳跃相位可加阻尼（见上面 d_col 的定义）
                )
                raw_torques[:, i] = calc_torque

        torques = raw_torques.clone()
        self.last_wheel_control_vel[:] = current_wheel_vel

        # ==================== [安全裁剪] ====================
        torques_cliped = torch.clip(torques, -self.torque_limits, self.torque_limits)

        # 💡 [终极自动追踪]：直接导出 0 号机器人【全部关节】的一维数组 (长度为 6)
        self.debug_actions_scaled = actions_scaled[0].detach().clone()
        self.debug_dof_pos = self.dof_pos[0].detach().clone()
        self.debug_dof_vel = self.dof_vel[0].detach().clone()
        self.debug_raw_tau = raw_torques[0].detach().clone()
        self.debug_final_tau = torques_cliped[0].detach().clone()

        return torques_cliped

    def check_termination(self):
        # 跌倒判定：base_link 或腿部触地即重置
        # 使用我们在 config 中定义的 termination_contact_indices
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # 超时重置
        self.reset_buf |= self.time_out_buf

    def _init_buffers(self):

        joints = list(self.cfg.init_state.default_joint_angles.keys())
        self.knee_r_idx = joints.index("right_knee_joint")
        self.hip_r_idx = joints.index("right_hip_joint")
        self.knee_l_idx = joints.index("left_knee_joint")
        self.hip_l_idx = joints.index("left_hip_joint")

        self.wheel_l_idx = joints.index("left_wheel_joint")
        self.wheel_r_idx = joints.index("right_wheel_joint")
        self.wheel_indices = torch.tensor([self.wheel_l_idx, self.wheel_r_idx], dtype=torch.long, device=self.device)
        self.leg_indices = torch.tensor(
            [self.hip_l_idx, self.knee_l_idx, self.hip_r_idx, self.knee_r_idx],
            dtype=torch.long,
            device=self.device,
        )

        # 初始化 WBR 专用的数据缓冲区
        super()._init_buffers()
        # 预计算"贴地高度着色网格"模板（仅 viewer 用）
        self._init_terrain_grid()


        # ---- 轮底离地量（真跳高口径）：acquire 刚体状态张量，每控制步 refresh ----
        #   wheel_clear   = 左右轮底（轮心世界 z − 轮半径）的较小者，逐拍
        #   wheel_peak_min = 双轮同离地窗内 wheel_clear 的峰值（本跳锁存）
        #   wheel_at_air  = 离地首拍的轮底（零点自检，应 ≈0）
        _rb = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._rb_state = gymtorch.wrap_tensor(_rb).view(self.num_envs, self.num_bodies, 13)
        self.wheel_clear = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.wheel_peak_min = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.wheel_at_air = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.wheel_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        rate_limits = self.cfg.commands.rate_limits
        self.command_targets = torch.zeros_like(self.commands)
        # 偏航通道的**原始指令**（未经包络）：包络必须从它算，不能从被削过的 targets 自反馈，
        # 否则"指令→目标被削多少"在诊断上不可见、且 v 回落时已削的 ω 目标放不回来。
        self.command_raw_yaw = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # ---- 目标层拦截诊断（P5 R7 ④，含义见 _apply_command_envelope / _reward_hip_reach）----
        # 都是"本控制步"的瞬时标记，不按 episode 累积（wbr.py 没有 _reset_idx 覆写，累积会在环境重置后串味）
        self.env_yaw_clipped = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.env_yaw_cap = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.env_hip_infeasible = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # ---- 目标层只读快照 ----
        # 在**算完的当拍**存一份（纯读出、不参与计算）。
        self.dbg = {k: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                    for k in ("yaw_raw", "yaw_tgt", "roll_base", "roll_tilt", "roll_tgt",
                              "knee_tgt_l", "knee_tgt_r", "roll_delta", "ff_gain",
                              "h_ik_in", "h_ceiling", "hip_allow_l", "hip_allow_r", "hip_over",
                              "h_cmd", "h_target", "off_knee", "ph", "h_body_rise",
                              "ev_takeoff", "tick_crouch_ok", "err_reach",
                              "v_launch", "h_body_lift", "v_land",
                              "L_delta", "h_cmd_apex", "h_push_cmd",
                              "gate_chain", "h_pred", "h_flight_pred", "ref_clamped",
                              "z_body_min", "tick_crouch_abs_ok",
                              "air_z", "air_vz", "air_body_z", "air_body_vz",
                              "air_latched", "cmd_settled",
                              "ev_t", "t_min", "t_lo", "t_apex", "t_land", "t_rec",
                              "z_min", "z_land", "z_rec",
                              "fz_land_peak", "imp_land", "W_stand"
                              , "crouch_credit",
                              "apex_leg_err", "apex_leg_score", "g3",
                              "top_height_err", "top_height_score",
                              "wheel_clear", "wheel_peak_min", "wheel_at_air")}  # jump_* 全组需 flags.enable_jump=True，否则是陈旧零
        self.base_roll_targets = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.command_rate_limits = torch.tensor(
            [
                rate_limits.lin_vel_x,
                rate_limits.lin_vel_y,
                rate_limits.ang_vel_yaw,
                rate_limits.heading,
                rate_limits.jump_height,
                rate_limits.base_height,
                rate_limits.ang_roll,
            ],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # ---- A. 相位时钟与装填（跳不跳、什么时候能跳） ----
        self.ph = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )   # 0 待机 1 预蹲 2 爆发伸腿 3 腾空收膝 4 落地前伸腿 5 缓冲
        self.ph_t = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 当前相位已经走了多少秒（只当保险丝，转移主要挂事件）
        self.is_armed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )   # 装填标志：一次指令最多跳一次（S2c 起，跳跃回合环境可按 rearm_max 有限连跳）
        # ---- 跳跃回合的连跳限流（三件状态）----
        self.is_jump_round = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )   # 本指令窗是否被抽进"跳跃回合子集"（`_resample_commands` 里按 jump_stand_ratio 重抽）
        #   连跳**只对这批环境开放**——行走环境即使拿到跳跃指令也仍一窗一次，免得又把行走策略
        #   搅扰（S1 的病）。
        self.rearm_cnt = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )   # 本指令窗已额外重装填几次（重采样时清零）
        self.rearm_at = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )   # 最早可以再次装填的 episode 步号（= 上次装填步号 + rearm_gap/dt）
        self.arm_at = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )   # 【P6R35】本窗**装填那一刻**的 episode 步号。`arm_delay` 改从此刻起算
            #   ⇒ 高度指令先落地、等 2s 才允许触发（原语义判 episode_length_buf 是**回合**时间，
            #   回合中段重采样时早已过 200 拍 ⇒ 指令一下发当拍就能跳，高度还没追上）。
        self.h_request = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 【P6R36】本窗"请求的跳跃高度"(m)：站立期不进观测（command_targets[:,4] 保持0，防泄漏）；
            #   触发那一拍才写入 command_targets[:,4] 并冻结 h_target，跳完清零
        self.cnt_settle = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )   # 【P6R78】相位1内"到位 ∧ 停稳"的**连续**拍数（相位机出口判据；旧 cnt_band 更名）

        # ---- B. 触发拍冻结的基准（模板据它反解出目标位形） ----
        self.ref_z = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 触发瞬间的机体高度 = 升高量的基准（不是 default，策略当时站多高就按多高算）
        self.ref_hip = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 触发瞬间冻结的左髋角 (rad)：ΔL→膝角按这一拍的位姿精确解，杜绝雅可比正反馈
        self.ref_hip_r = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 触发瞬间冻结的**右**髋角 (rad)：覆盖段写右髋绝对目标要用它，用镜像会硬拽（同 `ref_k_r`）。
        self.ref_k = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 同上，左膝角 (rad)
        self.ref_k_r = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 触发瞬间冻结的**右**膝角 (rad)。只给膝存右值是 `authority_mode="override"` 需要的：
        #     覆盖授权要写出"右膝的绝对目标"，而 default 是严格镜像（−2.1/+2.1）**不代表实测
        #     镜像**（策略两膝可差 0.05 rad 量级），拿 `−ref_k` 顶替会在进覆盖段那拍造成目标跳变。
        self.ref_L = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 本拍模板要的**腿竖直伸量**目标 (m) ⇒ 形状跟踪项与执行校验的共同分母
        self.ref_clamped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )   # 目标落进关节限位外（＝旧"夹持区"的等价旗标）⇒ 这一拍的行程读数要打标
        self.off_hip_comp = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 旧"把轮子搬回髋正下方"的纯旋转补偿偏移 (rad)，只在 hip_comp_gain≠0 时才有值

        # ---- C. 指令量与高度读数（要求多少 / 实际多少） ----
        self.z_cmd = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 触发那一拍冻结的**系统质心**高度 (m) ⇒ 跳跃指令的零点（作者 2026-10-14 定）
        self.h_target = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 触发那一拍冻结的**跳跃指令** (m) ⇒ 一跳之内的目标与分母全读它；窗外留旧值无害
        self.h_cmd_apex = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 事件内 max(质心 − z_cmd) = 总升高 (m) ⇒ 与指令 c_h^cmd 同量纲，判据认它
        self.h_push_cmd = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 确认起飞那一拍的质心相对 z_cmd 的升高 (m) = 接触段那一份
        self.com_z_prev = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 上一拍的系统质心高度 (m) ⇒ 差分出自心竖直速度（复位/触发时接上当前值，防假尖峰）
        self.com_vz_step = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 逐拍质心竖直速度 (m/s) = (com_z − com_z_prev)/dt ⇒ **push_lead 的分子（质心口径）**
        self.h_pred = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 离地那一拍的**预测总升高** (m) = h_push_cmd + max(air_vz,0)²/2g（质心）⇒ 脉冲的被奖励量
        self.h_flight_pred = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 离地那一拍的**离地后上升预测值**(m) = max(air_vz,0)²/2g（**质心**）⇒ 离地脉冲的被奖励量
        self.h_body_apex = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 本次事件达到的最大升高量 (m)（机身口径，留作对照）
        self.h_body_lift = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 离地之后的最大再升高 (m) ⇒ 与 h_body_apex(从触发算) 分开，后者含预蹲深度
        self.z_body_lift = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 离地那一拍的机体高度 ⇒ 升高量从它起算（不含预蹲的下探）
        # ---- 离地首拍锁存（见 `_update_jump_phase` 2e）：尺子零点与脉冲分子的取样点 ----
        self.air_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )   # 本跳是否已锁存过离地首拍（触发/复位时清零）
        self.air_z = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 离地首拍的**系统质心**高度 (m) ⇒ `h_push_cmd` 的被减数
        self.air_vz = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 离地首拍的质心竖直速度 (m/s) ⇒ 只进 `h_pred`（质心口径的辅助诊断）
        self.air_body_z = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 离地首拍的**机身**高度 (m) ⇒ 实测 `h_flight`(=`h_body_lift`) 的零点
        self.air_body_vz = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 离地首拍的机身竖直速度 (m/s) ⇒ 仅诊断（`v_launch`）；**脉冲分子已改用质心 `air_vz`**
        self.z_body_min = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 事件内**机身最低点** (m) ⇒ 绝对蹲到位判定的被测量

        # ---- C2. 事件量采集：只写不读 ⇒ 策略行为逐位不变 ----
        #   时刻 s、位置 m、力 N、冲量 N·s。
        self.ev_t = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 本事件走过的时间 (s)＝各事件时刻的公共时基
        self.z_min = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 起飞前质心最低点 (m) ⇒ h_crouch = z_cmd − z_min，h_push = z_lo − z_min
        self.t_min = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 到达质心最低点那一拍的 ev_t (s)
        self.z_land = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 首次重新接触地面那一拍的质心高度 (m)
        self.z_rec = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 缓冲结束回常态那一拍的质心高度 (m)
        self.t_lo = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 确认起飞那一拍的 ev_t (s)＝t_cmd_to_lo
        self.t_apex = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 质心峰值那一拍的 ev_t (s) ⇒ t_lo_to_apex = t_apex − t_lo
        self.t_land = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 触地时刻 (s) ⇒ t_flight = t_land − t_lo
        self.t_rec = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 恢复时刻 (s) ⇒ t_land_to_rec = t_rec − t_land
        self.fz_land_peak = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 落地窗内承重轮法向力峰值 (N)
        self.imp_land = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 落地窗内 ∫|ΣFz − W_stand| dt (N·s)
        self.W_stand = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 站定体重参照 (N)＝相位 0 静置 ΣFz 的 EMA ⇒ 落地冲击的零点

        # ---- D. 事件凭证与结算（离地、落地、冲击） ----
        self.ev_lift = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 本控制步是否"确认起飞"那一拍（只在那一拍为 1）⇒ 离地奖一次结清的发放沿
        self.ev_land_pay = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 落地结算的**本拍乘子**（只在触地那一拍非零）：确认起飞 × 直立门 ×
        #   落地冲击 soft × 预蹲比值门 g1 × 腿伸比值门 ⇒ `_reward_jump_landing` 一次结清的钱。
        self.ev_land_frac = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 触地那一拍的腿伸比值 `clamp(1 − 腿伸误差/land_leg_tol, 0, 1)` ⇒ 只给判据/探针读
        # 顶点腿长奖：一次性指示位 + 空中腿长滑动窗
        self.ev_apex_leg_pay = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 顶点（相位3→4 那一拍）的腿长奖**本拍乘子**：高斯分 × 门 ⇒ `_reward_jump_leg_apex`
        #   一次结清的钱。非相位3 的帧记 −1（不参与 max）⇒ 窗里只算腾空段。
        # 真跳高（轮底离地）结果奖：同一顶点帧一次性结清，被奖励量 = wheel_peak_min
        self.ev_top_height_pay = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 顶点帧的轮底离地奖**本拍乘子**：双边高斯分 × 门 ⇒ `_reward_jump_top_height` 一次结清的钱
        self.L_max_hist = torch.full(
            (self.num_envs, max(1, int(getattr(self.cfg.rewards.jump, "leg_apex_win", 3)))),
            -1.0, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.tick_seg = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False
        )   # 各段完成票：0 起跳达标 / 1 空中收腿达标 / 2 够地达标（预蹲票沿用 tick_crouch_ok）
        self.ev_flew = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )   # 本次事件是否真的两轮同时离过地（奖励的第一道门，防"蹲起蹭分"）
        self.ev_takeoff = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )   # 本跳是否**确认起飞**（连续 confirm_steps 拍双空且 vz≥门限）⇒ 一切跳跃奖励的总门
        self.ev_takeoff_ever = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )   # 【P6R33】"本事件内**曾经**确认起飞"锁存：reset 清 `ev_takeoff` 时它已被
            #   reset_idx 写进 extras（探针可分"没跳过"与"跳了但撞地"）；不参与任何奖励发放
        self.v_launch = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 确认起飞那一拍的 v_z (m/s) ⇒ S0 门槛③
        self.v_land = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 确认落地那一拍的 |v_z| (m/s) ⇒ S0 门槛④（落地冲击）

        # ---- E. 门与完成票（钱发给谁） ----
        self.gate_chain = torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 当前相位该乘的正奖励门＝前序票的 leak 门之积（**只乘正项，负项不碰**）
        self.gate_lift = torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 离地脉冲专用门＝只看预蹲票（脉冲自己就是"起跳达标"的裁决，不能要求它先达到自己）
        # 相位 4/5 腿伸跟踪专用门 = g1·g2（不含 g3）。g3 是顶点的布尔票，
        # 把它做门会让"顶点腿长到位"的梯度注入相位 4/5、带偏策略。
        self.gate12 = torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.tick_crouch_frac = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 预蹲完成比例（事件内最大值，0~1）⇒ 链式门 g1 的连续口径
        self.crouch_credit = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 预蹲跟踪的逐拍累计记账（≤ 相位 1 拍数）；诊断口径，发钱不走它
        self.crouch_credit_step = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 预蹲跟踪的**本拍**得分（0~1）⇒ `_reward_jump_crouch_track` 的发放量；触发/复位清零
        self.tick_crouch_abs_ok = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 机身高度是否进到 目标 ± crouch_reach_tol（双向）⇒ 1＝真蹲到位置了
        self.tick_crouch_ok = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )   # 本跳的预蹲是否真到位（机身高度到达目标 ±容差）⇒ S0 门槛①

        # ---- F. 模板偏移与执行误差（每拍往目标上加多少、差多少） ----
        self.off_knee = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 本控制步加到膝目标上的偏移 (rad，正 = 两腿同时伸长，右腿按镜像取负)
        self.off_hip = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 髋补偿角 (rad)：把轮子搬回髋正下方，消掉"只动膝"带来的前后飘
        self.dem_crouch = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 本档要求的预蹲深度 (m，沿机体轴的腿伸量变化)，＝预蹲跟踪项的分母
        self.err_crouch_abs = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 预蹲机身高度误差 (m)：z − crouch_abs_height，正＝还没蹲够；判据与记账的口径
        self.hold_since = torch.full(
            (self.num_envs,), -1.0, dtype=torch.float, device=self.device, requires_grad=False
        )   # 【P6R33 软停止】首次进预蹲高度带那一拍的相位内时钟；−1＝未进带（诊断用）
        self.dh_hold = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 【P6R33 软停止】进带锁存的预蹲目标 (m)：首次进带那一拍的 dh 整体冻结
        self.knee_hold = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 【P6R33 软停止】进带锁存的膝积分器值 (rad)：冻结 2c 的 ±rate 爬坡
        self.hip_hold = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 【P6R33 软停止】进带锁存的髋积分器值 (rad)
        self.tgt_crouch = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 相位 1 每拍"模板此刻要求的下探深度"(m，爬升后的瞬时值)；跟踪项用它判带内
        self.err_reach = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # |实测腿伸量 − 模板要求腿伸量| (m) ⇒ "策略有没有真的执行模板"
        self.ph5_L_entry = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 【P6R52】相位5 ease-out 起点＝4→5 交界锁存的相位4出口腿长 (m)；0＝未锁存(退回常数阶跃)

        # ---- G. 静定／离地／落地的采样状态（滑窗计数） ----
        # 静定门的两个状态。参考高**必须独立锁存**（只在待机＋双轮接地＋直立时用滑窗更新）。
        self.z_stand_ref = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # 站立参考高 (m)：待机且踩实直立时的 EMA
        self.cnt_quiet = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )   # 连续满足静定条件的拍数（装填后还要等它够 settle_steps 才准触发）
        self.cnt_air = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )   # 连续两轮同空的拍数（离地确认用）
        self.cnt_gnd = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )   # 连续双轮回地的拍数（落地确认用）
        self.L_air_hold = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )   # **确认起飞那一拍**的实测腿竖直伸量 (m) ⇒ 相位 3「维持离地腿型」的
        #   设定点取**实测值不是模板值**："维持离地腿型"的语义就是"别动"。

        # ---- H. 任务级旗标（行走侧公用，不属跳跃相位） ----
        self.is_jumping = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )

        # ---- I. 掩蔽表：项名 → 相位号向量（相位索引一次成向量，无循环）----
        self._jump_mask_vec = {}
        m_cfg = getattr(self.cfg.rewards.jump, "mask", None)
        if m_cfg is not None:
            for name in [k for k in dir(m_cfg) if not k.startswith("_")]:
                phases = getattr(m_cfg, name)
                if not isinstance(phases, (list, tuple)):
                    continue
                vec = torch.zeros(6, dtype=torch.bool, device=self.device)
                if len(phases):
                    vec[torch.as_tensor(list(phases), dtype=torch.long,
                                        device=self.device)] = True
                self._jump_mask_vec[name] = vec

        # 关节历史缓冲区 (Env 数量, 历史步长, 动作数量)
        self.dof_pos_error_history = torch.zeros(self.num_envs, self.cfg.env.joint_state_history_length, self.num_actions,
                                               dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_history = torch.zeros(self.num_envs, self.cfg.env.joint_state_history_length, self.num_actions,
                                         dtype=torch.float, device=self.device, requires_grad=False)
        self.wheel_target_vel = torch.zeros(
            self.num_envs, len(self.wheel_indices), dtype=torch.float,
            device=self.device, requires_grad=False
        )
        self.last_wheel_control_vel = self.dof_vel[:, self.wheel_indices].clone()
        self.filtered_wheel_acceleration = torch.zeros_like(self.last_wheel_control_vel)

    def _reset_dofs(self, env_ids):
        """重置被选中环境的 DOF：左右腿各乘一个 0.95~1.05 的随机因子（左右同步，避免引入构型不对称）。

        """
        pair_scales = torch_rand_float(
            0.95,
            1.05,
            (len(env_ids), 2),
            device=self.device,
        )
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_pos[env_ids, self.hip_l_idx] *= pair_scales[:, 0]
        self.dof_pos[env_ids, self.hip_r_idx] *= pair_scales[:, 0]
        self.dof_pos[env_ids, self.knee_l_idx] *= pair_scales[:, 1]
        self.dof_pos[env_ids, self.knee_r_idx] *= pair_scales[:, 1]
        self.dof_vel[env_ids] = 0.
        self.wheel_target_vel[env_ids] = 0.0
        self.last_wheel_control_vel[env_ids] = 0.0
        self.filtered_wheel_acceleration[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
