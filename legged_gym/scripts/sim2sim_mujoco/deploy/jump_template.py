# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : jump_template.py
# Purpose   : 两轮跳跃相位机的**实时部署移植**（单环境 numpy 版）：wbr_flat.py
#             `_update_jump_phase`（801 行）＋`_jump_2r_solve` 的物理路径。奖励/诊断记账
#             段裁掉，相位事件、各相腿长目标、2 自由度反解、限速、相位增益增量全保留。
#             参数表内嵌 wbr_flat_config.WBRFlatCfg.rewards.jump 现值（2026-11-05 冻结），
#             改训练侧任一参数都要同步这里并重跑 golden。
# ==============================================================================

import numpy as np

# ---- wbr_flat_config.rewards.jump 现值（冻结拷贝） ----
J = dict(
    t_crouch=0.75, t_push_max=0.20, crouch_reach_tol=0.006, takeoff_vz_min=0.50,
    confirm_steps=2, off_rate_max=0.25, burst_template=True, hip_rate_max=0.25,
    t_flight_min=0.03, t_flight_max=0.35, t_land_max=0.25, t_absorb=0.20,
    stroke_gain=1.2, push_to_limit=True, push_overshoot=0.067,
    burst_L_by_h=(0.05, 0.15, 0.29, 0.38), burst_retract_L=0.25, burst_retract_to=0.18,
    burst_rate_max=0.3,
    air_hold_joint=0, air_hold_follow=0.0, air_hold_L=0.20, land_hold_L=0.20,
    tuck_depth=0.05, air_hold_leg=0.0, land_hold_leg=0.0,
    absorb_abs_height=0.22, absorb_ease_p=2.0, land_extend=0.06, off_limit=1.2,
    upright_limit=0.35,
    knee_damp_extra=[1.2, 0.0, 1.2, 1.2, 1.2], knee_kp_extra=[0.0, 12.0, 0.0, 0.0, 0.0],
    crouch_ramp_end=0.60, crouch_z_kp=6.0, crouch_settle_steps=5, crouch_vz_tol=0.02,
    crouch_hold_time=0.10, crouch_deep_tol=0.010,
    crouch_abs_by_h=(0.05, 0.15, 0.22, 0.19),
    crouch_hold_tol=0.0, crouch_hold_min=0.0, crouch_hold_max=0.225,
    crouch_depth_cap=0.14, settle_ref_tau=0.20, settle_vz_max=0.10,
    settle_dz_max=0.02, settle_vxy_max=0.10, settle_wxy_max=0.50,
    settle_steps=10, settle_gate=True, arm_delay=0.0,
    anchor_curve=[-4.083, 2.560, -0.564, 0.070], anchor_x_com=None,
    anchor_tilt_deg=0.0,
)
LT, LS = 0.20, 0.18          # thigh/shank（rewards.thigh_length/shank_length）


def _lerp_by_h(spec, h_target):
    """(h_lo,h_hi,v_lo,v_hi) 随本跳冻结的 h_target 线性插值。"""
    if spec is None:
        return None
    h_lo, h_hi, v_lo, v_hi = spec
    r = np.clip((h_target - h_lo) / (h_hi - h_lo), 0.0, 1.0)
    return v_lo + r * (v_hi - v_lo)


class JumpTemplate:
    """相位机状态＋每控制步 update()。单环境；SimLoop 每拍喂传感器，取回 (ph, off_knee, off_hip)。"""

    def __init__(self, k_lim, h_lim, dt, ref_override=None):
        self.k_lim = k_lim                     # 膝限位 (lower, upper)，如 (-2.7053, -0.1)
        self.h_lim = h_lim                     # 髋限位
        self.dt = dt
        # 【S2b 精调】触发拍站立基准冻结（None=用当拍实际角，训练侧原语义）。
        #   站立平衡点冗余 ⇒ 两引擎触发角微差经 2R 反解全程平移 off 序列（顶点差 23mm 主因）；
        #   冻结为 golden 触发拍角后绝对膝目标与 Isaac 逐拍同形。键：hip/knee（弧度，左腿）。
        self.ref_override = dict(ref_override) if ref_override else None
        self.reset(z=0.28)

    def reset(self, z):
        j = J
        self.ph = 0
        self.ph_t = 0.0
        self.ev_t = 0.0
        self.cnt_air = 0
        self.cnt_gnd = 0
        self.cnt_quiet = max(int(j["settle_steps"]), 1)   # reset_jump_state 播种（回放/评测口径）
        self.cnt_settle = 0
        self.z_stand_ref = z
        self.off_knee = 0.0
        self.off_hip = 0.0
        self.ref_hip = 0.0
        self.ref_k = 0.0
        self.ref_z = z
        self.z_cmd = 0.0
        self.h_target = 0.0
        self.h_request = 0.0
        self.is_armed = False
        self.arm_at = 0
        self.air_latched = False
        self.air_z = 0.0
        self.air_vz = 0.0
        self.L_air_hold = 0.0
        self.ph5_L_entry = 0.0
        self.hold_since = -1.0
        self.dh_hold = 0.0
        self.knee_hold = 0.0
        self.hip_hold = 0.0
        self.z_body_min = z
        self.err_crouch_abs = 0.0
        self.ev_takeoff = False
        self.com_z_prev = z
        self.h_cmd_obs = 0.0                   # 观测里的跳跃指令通道（触发拍写入、回待机清零）

    def request(self, h):
        """装填（评测/回放口径：直接写 h_request＋is_armed）。"""
        self.h_request = float(h)
        self.is_armed = self.h_request > 0.0
        self.arm_at = 0

    # ------------------------------------------------------------------ 每控制步
    def update(self, z, vz, fz_l, fz_r, roll, pitch, com_z, dof_pos):
        """返回 (ph, off_knee, off_hip, air_z, goal, latched)。dof_pos 为 dict(名→角)。"""
        j = J
        dt = self.dt
        contacts = (fz_l > 0.1, fz_r > 0.1)
        n_ground = sum(contacts)
        grounded = n_ground == 2
        airborne = n_ground == 0
        upright = (abs(roll) < j["upright_limit"]) & (abs(pitch) < j["upright_limit"])

        # 接触连续计数
        self.cnt_air = self.cnt_air + 1 if airborne else 0
        self.cnt_gnd = self.cnt_gnd + 1 if grounded else 0

        p = self.ph
        t = self.ph_t

        # ---- 触发（IDLE＋装填＋站好＋直立＋有请求＋静定）----
        ref_ok = (p == 0) and grounded and upright
        a_ref = min(max(j["settle_ref_tau"], 1e-3), 1.0)
        if ref_ok:
            self.z_stand_ref = (1 - a_ref) * self.z_stand_ref + a_ref * z
        if abs(self.z_stand_ref) < 1e-6:
            self.z_stand_ref = z
        quiet_now = (ref_ok & (abs(vz) <= j["settle_vz_max"])
                     & (abs(z - self.z_stand_ref) <= j["settle_dz_max"]))
        self.cnt_quiet = self.cnt_quiet + 1 if quiet_now else 0
        settled = (self.cnt_quiet >= max(int(j["settle_steps"]), 0)) if j["settle_gate"] else True
        arm_steps = max(int(j["arm_delay"] / dt), 0)
        trig = ((p == 0) & self.is_armed & grounded & upright
                & (self.h_request > 0.0) & settled & (arm_steps >= 0))
        if trig:
            self.command_jump = self.h_request       # 观测指令通道（触发拍写入）
            self.ph = 1
            self.ph_t = 0.0
            self.ref_z = z
            self.z_cmd = com_z
            self.air_latched = False
            self.air_z = 0.0
            self.air_vz = 0.0
            self.L_air_hold = 0.0
            self.ph5_L_entry = 0.0
            self.off_knee = 0.0
            self.off_hip = 0.0
            self.cnt_air = 0
            self.cnt_gnd = 0
            self.hold_since = -1.0
            self.z_body_min = z
            self.err_crouch_abs = 0.0
            self.ev_takeoff = False
            self.is_armed = False                    # 一条指令只跳一次
            if self.ref_override:
                self.ref_hip = float(self.ref_override.get("hip", dof_pos["left_hip_joint"]))
                self.ref_k = float(self.ref_override.get("knee", dof_pos["left_knee_joint"]))
            else:
                self.ref_hip = dof_pos["left_hip_joint"]
                self.ref_k = dof_pos["left_knee_joint"]
            self.h_target = self.h_request
            self.h_request = 0.0
            p, t = 1, 0.0

        # ---- 各相腿长目标 dh（相对触发基准的竖直伸量增量）----
        cos_h = np.cos(self.ref_hip)
        L_ref = LT * cos_h + LS * np.cos(self.ref_hip + self.ref_k)
        dh = 0.0
        abs_h = _lerp_by_h(j["crouch_abs_by_h"], self.h_target)
        if abs_h is None:
            abs_h = 0.22
        if p == 1:
            crouch_cmd = float(np.clip(self.ref_z - abs_h, 0.0, j["crouch_depth_cap"]))
            s = min(t / (j["t_crouch"] * j["crouch_ramp_end"]), 1.0)
            ramp = s * s * s * (s * (s * 6.0 - 15.0) + 10.0)
            z_cmd = self.ref_z - crouch_cmd * ramp
            dh = -crouch_cmd * ramp + j["crouch_z_kp"] * (z_cmd - z)
            self.err_crouch_abs = z - abs_h
        elif p == 2:
            crouch_cmd = float(np.clip(self.ref_z - abs_h, 0.0, j["crouch_depth_cap"]))
            dh_cap = (LT * cos_h + LS) - L_ref + j["push_overshoot"]
            dh_travel = crouch_cmd + j["stroke_gain"] * self.h_target
            dh = min(dh_travel, dh_cap)
            blm = _lerp_by_h(j["burst_L_by_h"], self.h_target)
            if blm is not None:
                dh = min(dh, blm - L_ref)
            hip_p = dof_pos["left_hip_joint"]
            L_act_p = LT * np.cos(hip_p) + LS * np.cos(hip_p + dof_pos["left_knee_joint"])
            if L_act_p >= j["burst_retract_L"]:       # 爆发段内收腿（不切相位）
                dh = j["burst_retract_to"] - L_ref
        elif p == 3:
            # air_hold_L=0.20（绝对腿长）；air_hold_follow=0、air_hold_joint=0
            if self.L_air_hold > 1e-6:
                dh = self.L_air_hold - L_ref          # 起飞确认后：钉在离地腿型
            else:
                dh = -j["tuck_depth"]
        elif p == 4:
            dh = (self.L_air_hold - L_ref) if self.L_air_hold > 1e-6 else j["land_extend"]
        elif p == 5:
            depth5 = float(np.clip(self.ref_z - j["absorb_abs_height"], 0.0, j["crouch_depth_cap"]))
            L_settle = L_ref - depth5
            L_entry = self.ph5_L_entry if self.ph5_L_entry > 1e-3 else L_settle
            s5 = min(t / j["t_absorb"], 1.0)
            L5 = L_settle + (L_entry - L_settle) * (1.0 - s5) ** j["absorb_ease_p"]
            dh = L5 - L_ref

        # ---- 2R 反解（anchor_curve 平衡曲线锚，无 tilt）----
        h0, k0 = self.ref_hip, self.ref_k
        L_star = L_ref + dh
        Lc = np.clip(L_star, 0.10, 0.380)
        c = j["anchor_curve"]
        x_ref = c[0] * Lc ** 3 + c[1] * Lc ** 2 + c[2] * Lc + c[3]
        D = np.sqrt(max(L_star * L_star + x_ref * x_ref, 1e-8))
        D_hi, D_lo = LT + LS, abs(LT - LS) + 1e-4
        D = np.clip(D, D_lo, D_hi)
        cos_k = np.clip((D * D - LT * LT - LS * LS) / (2.0 * LT * LS), -1.0, 1.0)
        k_star = -np.arccos(cos_k)
        k_c = float(np.clip(k_star, self.k_lim[0], self.k_lim[1]))
        A = LT + LS * np.cos(k_c)
        B = LS * np.sin(k_c)
        th_star = np.arctan2(x_ref, max(L_star, 1e-4)) - np.arctan2(B, A)
        h_c = float(np.clip(th_star, self.h_lim[0], self.h_lim[1]))
        off_des = float(np.clip(k_c - k0, -j["off_limit"], j["off_limit"]))
        hip_des = h_c - h0
        self.L_star = L_star                    # 供相位 4→5 的 L_entry 锁存

        # ---- 模板偏移限速（爆发 0.3、其余 0.25；髋 0.25）----
        rate, brate, hrate = j["off_rate_max"], j["burst_rate_max"], j["hip_rate_max"]
        if p == 2:
            self.off_knee += np.clip(off_des - self.off_knee, -brate, brate)
        else:
            self.off_knee += np.clip(off_des - self.off_knee, -rate, rate)
        self.off_hip += np.clip(hip_des - self.off_hip, -hrate, hrate)
        if p == 0:
            self.off_hip = 0.0                        # ph0 髋清零（膝保留：行走限位偏移 −0.1 靠它）

        # ---- 离地首拍锁存（相位 ≥2 两轮同空）----
        _air_first = airborne and (p >= 2) and (not self.air_latched)
        if _air_first:
            self.air_z = com_z
            self.air_vz = (com_z - self.com_z_prev) / dt
            self.air_latched = True

        # ---- 相位内时钟 ----
        if p > 0:
            self.ph_t += dt
            self.ev_t += dt
        tn = self.ph_t
        p0 = p

        # ---- 起飞确认：爆发/腾空初段连续 2 拍同空且在上升 ----
        if (2 <= p0 <= 3) and (not self.ev_takeoff) and self.cnt_air >= j["confirm_steps"] \
                and vz >= j["takeoff_vz_min"]:
            self.ev_takeoff = True
            _hip_n, _knee_n = dof_pos["left_hip_joint"], dof_pos["left_knee_joint"]
            _hip_r, _knee_r = dof_pos["right_hip_joint"], dof_pos["right_knee_joint"]
            self.L_air_hold = 0.5 * (LT * np.cos(_hip_n) + LS * np.cos(_hip_n + _knee_n)
                                     + LT * np.cos(_hip_r) + LS * np.cos(_hip_r + _knee_r))

        # ---- 事件转移（全部按本拍进入时相位 p0 判）----
        def advance(mask, nxt):
            if mask:
                self.ph = nxt
                self.ph_t = 0.0
                return True
            return False
        if p0 == 1:
            tol_c = j["crouch_reach_tol"]
            _t_ramp = j["t_crouch"] * j["crouch_ramp_end"]
            _ok_c = abs(self.err_crouch_abs) <= tol_c
            _still = _ok_c and (abs(vz) <= j["crouch_vz_tol"])
            self.cnt_settle = self.cnt_settle + 1 if _still else 0
            _settled = (tn >= _t_ramp + j["crouch_hold_time"]) and (self.cnt_settle >= j["crouch_settle_steps"])
            _deep = self.err_crouch_abs < -j["crouch_deep_tol"]
            advance(_settled or _deep or (tn >= j["t_crouch"]), 2)
        elif p0 == 2:
            if (self.ev_takeoff and airborne) or (not self.ev_takeoff and tn >= j["t_push_max"]):
                advance(True, 3)
        elif p0 == 3:
            if tn >= j["t_flight_min"] and (vz <= 0.0 or tn >= j["t_flight_max"]):
                advance(True, 4)
        elif p0 == 4:
            touch = self.cnt_gnd >= j["confirm_steps"]
            _to5 = touch or (tn >= j["t_land_max"])
            if _to5:
                self.ph5_L_entry = self.L_star   # 4→5 交界锁存模板腿长（ease-out 起点）
                advance(True, 5)
        elif p0 == 5:
            if tn >= j["t_absorb"]:
                self.h_cmd_obs = 0.0                  # 回待机清观测指令
                advance(True, 0)
        # 触发拍写观测指令（在 advance 之后仍保持：事件窗内 command_targets[4]=h_request）
        if self.ph > 0 and trig:
            self.h_cmd_obs = self.command_jump
        elif self.ph == 0:
            self.h_cmd_obs = 0.0

        # ---- 相位增益增量（膝阻尼 ph1/3/4/5 +1.2；膝 P ph2 +12）----
        dext = j["knee_damp_extra"][max(self.ph - 1, 0)] if self.ph > 0 else 0.0
        kext = j["knee_kp_extra"][max(self.ph - 1, 0)] if self.ph > 0 else 0.0

        self.com_z_prev = com_z
        return dict(ph=self.ph, off_knee=self.off_knee, off_hip=self.off_hip,
                    air_z=self.air_z, goal=self.h_target, latched=self.air_latched,
                    h_cmd_obs=self.h_cmd_obs, knee_damp_extra=dext, knee_kp_extra=kext)

