# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : p7t_standup.py
# Purpose   : P7T 传统起立脚本链的 MJ 实时移植（play02_core MODE_TRAD_RISE inline 分支
#             逐参数对照）：① 4W 静稳 0.6s → ② smoothstep 折蹲 0.5s → ②' 蹲稳 0.4s →
#             ③ smoothstep 伸腿 0.5s（腿 P×boost）→ ④ 轮恒力矩 C 抬升（门限
#             body_pitch<0.26 rad 且 z>0.20，超时保险 1.2s）→ ⑤ 交 W3。
#             腿序 [hip_l, knee_l, hip_r, knee_r]；act 恒 0（腿目标＝脚本绝对角）。
# ==============================================================================

import numpy as np

# 姿态（腿序 [hip_l, knee_l, hip_r, knee_r]）
DFLT_4W = np.array([0.9833, -2.5589, -0.9833, 2.5589])
CROUCH = np.array([1.5391, -2.2722, -1.5391, 2.2722])
DFLT_2W = np.array([1.0, -2.1, -1.0, 2.1])

# 时序/参数（wbr_flat4 系 trad_* 现值；console 趴下/起立参数同源）
# 【S3 交棒精调】抬升段新增闭环（默认关＝与 Isaac 脚本逐字一致）：
#   vz_goal>0 ⇒ C_eff = clamp(k_vz·(vz_goal − vz), 0, torque)，机体匀速抬升不冲高；
#   pitch_damp≠0 ⇒ 伸腿/抬升段髋前馈 off_hip = −pitch_damp·pitch_rate（俯仰阻尼）。
#   交棒判据可加姿态闸：|pitch|≤pitch_tol 且 |pitch_rate|≤rate_tol（0/0＝不启用）。
P = dict(t_hold=0.6, t_crouch=0.5, t_chold=0.4, t_ext=0.5, t_lift=1.2,
         torque=8.0, boost=2.0, pitch_handover=-0.26, z_min=0.20,
         vz_goal=0.0, k_vz=0.0, kp_pitch=0.0, kd_pitch=0.0, c_rate=0.0,
         kp_offhip=0.0, kd_offhip=0.0, off_fwd=0.0, off_rate=0.5,
         pitch_damp=0.0, pitch_tol=0.0, rate_tol=0.0)


def _smooth(a):
    a = np.clip(a, 0.0, 1.0)
    return a * a * (3.0 - 2.0 * a)


class P7TStandup:
    """五相脚本状态机。step() 返回当拍指令：
    default_leg[4]（腿 PD 绝对目标）、wheel_torque（两轮恒力矩，None=不覆盖）、
    boost（腿 P 倍率）、done（交 W3）。"""

    def __init__(self, dt, params=None):
        self.dt = float(dt)
        self.p = dict(P)
        if params:
            self.p.update(params)
        self.t = 0.0
        self.done = False
        self._boost_on = False
        self._c = 0.0
        self._off = 0.0

    def step(self, z, body_pitch, vz=0.0, pitch_rate=0.0):
        """推进一拍（内部 t += dt）并返回当拍指令。"""
        p = self.p
        out = {"default_leg": DFLT_4W.copy(), "wheel_torque": None, "boost": 1.0,
               "done": False, "off_hip": 0.0}
        if self.done:
            return out
        self.t += self.dt
        t = self.t
        t0, t1 = p["t_hold"], p["t_hold"] + p["t_crouch"]
        t2 = t1 + p["t_chold"]
        t3 = t2 + p["t_ext"]
        if t < t0:                                              # ① 4W 静稳
            pass
        elif t < t1:                                            # ② 折蹲
            al = _smooth((t - t0) / p["t_crouch"])
            out["default_leg"] = (1 - al) * DFLT_4W + al * CROUCH
        elif t < t2:                                            # ②' 蹲稳
            out["default_leg"] = CROUCH.copy()
        elif t < t3:                                            # ③ 伸腿（腿 P×boost）
            out["boost"] = p["boost"]
            al = _smooth((t - t2) / p["t_ext"])
            out["default_leg"] = (1 - al) * CROUCH + al * DFLT_2W
        else:                                                   # ④ 抬升（恒力矩 / vz 闭环）
            out["boost"] = p["boost"]
            out["default_leg"] = DFLT_2W.copy()
            c = float(p["torque"])
            off_hip = -float(p["pitch_damp"]) * float(pitch_rate)
            if p["off_fwd"] != 0.0:
                # 【S3】准静态髋前倾斜坡：抬升段不驱动轮，off_hip 以 off_rate 匀速走到 off_fwd，
                #   让机体在"腿推、轮自由滚"的状态下慢慢转到直立；pitch 到位即交棒（避免冲过平衡点）。
                step = p["off_rate"] * self.dt
                tgt = float(p["off_fwd"])
                if abs(tgt - self._off) <= step:
                    self._off = tgt
                else:
                    self._off += np.sign(tgt - self._off) * step
                off_hip = self._off
                c = 0.0
            elif p["kp_offhip"] != 0.0:
                # 【S3】抬升段改用髋俯仰闭环（不驱动轮）：off_hip 是左右髋的世界同步偏置，
                #   不产生前进速度与滚动失稳；极性由 kp_offhip 符号决定，正负都可扫。
                off_hip = (p["kp_offhip"] * float(body_pitch)
                           + p["kd_offhip"] * float(pitch_rate))
                off_hip = float(np.clip(off_hip, -0.5, 0.5))
                c = 0.0
            elif p["kp_pitch"] > 0.0:
                # 【S3】抬升段俯仰闭环：关节力矩对俯仰的响应是 θ̇ ≈ −a·C ⇒ C = kp·θ + kd·θ̇
                # （θ>0＝后仰；C 上限 torque，速率限幅 c_rate N·m/s 抑制阶跃冲击）
                c = p["kp_pitch"] * max(0.0, float(body_pitch)) + p["kd_pitch"] * float(pitch_rate)
                c = float(np.clip(c, 0.0, p["torque"]))
                if p["c_rate"] > 0.0:
                    lim = p["c_rate"] * self.dt
                    c = float(np.clip(c, self._c - lim, self._c + lim))
                self._c = c
            elif p["vz_goal"] > 0.0 and p["k_vz"] > 0.0:
                c = float(np.clip(p["k_vz"] * (p["vz_goal"] - vz), 0.0, p["torque"]))
            out["wheel_torque"] = c
            out["off_hip"] = off_hip
            pb = -body_pitch
            z_handover = float(p.get("z_handover", 0.0))
            pose_ok = True
            if p["pitch_tol"] > 0.0:
                pose_ok = (abs(body_pitch) <= p["pitch_tol"]
                           and abs(pitch_rate) <= (p["rate_tol"] or 1e9))
            if ((pb > p["pitch_handover"] and (z > p["z_min"] or z_handover > 0.0)
                 and pose_ok)
                    or (z_handover > 0.0 and z >= z_handover and pose_ok)
                    or t >= t3 + p["t_lift"]):
                # 【S3 精调】z_handover：抬升段机体冲高前**提前交棒**（C=8 实测冲 0.38，抢在 0.24）
                out["done"] = True                              # ⑤ 交 W3
        return out
