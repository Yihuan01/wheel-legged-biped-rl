# ==============================================================================
# Date      : 2026-11-06
# Author    : SYH
# File      : p7t2_platform_standup.py
# Purpose   : P7T2 平台式起立（代号 P7T2，2026-11-06 由 P9 更名；作者定）：MuJoCo 先调通、参数回灌 Isaac 验证。
#             按《平台资料 02_控制算法.md》§8（论文第 8 章 / wbr_stand_action.c）移植：
#               俯仰轨迹 θ_d(t) 三次多项式（−48°→0，T=0.5 s）
#               + 逆动力学前馈 u_ff(t)（WIP 方程消去 ẍ）
#               + PD 修正（误差=实际−目标，kp/kd 为正）
#             与现 P7T（伸腿甩尾＋恒力矩 C=8 阶跃）的本质区别：力矩是**连续解析轨迹**，
#             对 MuJoCo 硬接触（solref 0.01）不产生弹射——这正是 P7T 在 MJ 翻车的根源。
#
#   控制率（论文 8.4）：
#     u = u_ff/2 + kp·(θ − θ_d) + kd·(θ̇ − θ̇_d)      ← 单轮力矩（u_ff 是总力矩取一半）
#     u_ff = [−r·cθ·sθ·θ̇²·l²·m² + A₁·r·sθ·g·l·m − θ̈·l²·m·r·(A₁ − m·cθ²)]
#            / (cθ·l·m + A₁·r)
#
#   参数（按我们构型换算，平台原值在括号内）：
#     r = 0.08（同）；m = 6.1163（车+腿，平台 6.0208）；l = 0.0962（蹲姿轮心→CoM，平台 0.10354）；
#     A₁ = 7.8403（= m + 2mw + 2Iw/r²，mw=0.5534、Iw=0.001975；平台 7.7268）
#     θ₀ = −0.8377 rad（−47.997°）；T = 0.5 s；kp=5、kd=0.1（平台常量初值，MuJoCo 里重扫）
#
#   符号约定（待 MuJoCo 实测核对）：θ = 机体俯仰，起立从后仰 −48° 转到 0。
#     论文口径 θ<0＝后仰；我们的 body_pitch 由 projected_gravity 给出（asin(−pg[0])），
#     P7T 里 pb=−body_pitch>0 为后仰 ⇒ θ = −pb = body_pitch。上线前用蹲姿实测核对一次。
# ==============================================================================

import numpy as np

# ---- 平台常量（我们构型换算值；平台原值见文件头）----
R_WHEEL = 0.08
M_BODY = 6.1163          # 车+腿（整机 7.2231 − 两轮 2×0.5534）
L_PEND = 0.0962          # 蹲姿轮心→CoM（0.14 m 高度）
A1 = 7.8403              # M_BODY + 2·mw + 2·Iw/r²
G = 9.81

# ---- 轨迹参数（论文 §8.2）----
THETA0 = -0.8377         # rad，起始俯仰（后仰 48°；由 θ1=1.5391、θ2=−2.2722 推出）
T_DURATION = 0.5         # s

# ---- PD 增益（平台常量初值；MuJoCo 里重扫）----
KP = 5.0
KD = 0.1

# 起始腿姿（平台标定值＝P7T CROUCH 同款；h=0.14 m）
POSE_CROUCH = np.array([1.5391, -2.2722, -1.5391, 2.2722])   # [hip_l, knee_l, hip_r, knee_r]
POSE_2W = np.array([1.0, -2.1, -1.0, 2.1])


def _smooth(a):
    a = np.clip(a, 0.0, 1.0)
    return a * a * (3.0 - 2.0 * a)


def feedforward(theta, dtheta, ddtheta):
    """逆动力学前馈：给定俯仰角/角速度/角加速度，返回**总**轮力矩（左右合计）。
    论文式（8.3.1）。θ 接近 ±90° 时分母趋零——起立轨迹不会到那里，但 clip 防炸。"""
    c, s = np.cos(theta), np.sin(theta)
    denom = c * L_PEND * M_BODY + A1 * R_WHEEL
    if abs(denom) < 1e-6:
        return 0.0
    num = (-R_WHEEL * c * s * dtheta ** 2 * L_PEND ** 2 * M_BODY ** 2
           + A1 * R_WHEEL * s * G * L_PEND * M_BODY
           - ddtheta * L_PEND ** 2 * M_BODY * R_WHEEL * (A1 - M_BODY * c * c))
    return num / denom


class P7T2PlatformStandup:
    """P7T2 起立控制器：蹲姿准备（沿用 P7T 时序）＋ 平台式轨迹抬升。

    step() 返回当拍指令（与 P7TStandup 同形，可直接换进 SimLoop/UniChain）：
      default_leg  [4] 腿 PD 绝对目标（hip_l, knee_l, hip_r, knee_r）
      wheel_torque float 单轮力矩（None=不覆盖，交给策略/环）
      boost        float 腿 P 倍率
      off_hip      float 髋前馈（本方案不用，恒 0）
      done         bool  抬升完成（θ 轨迹走完）→ 交两轮
      diag         dict  诊断（θ/θ_d/误差/u_ff/u_pd，供扫描记账）
    """

    def __init__(self, dt, params=None):
        self.dt = float(dt)
        p = dict(t_hold=0.6, t_crouch=0.5, t_chold=0.4,   # t_hold 2.0：C1 循环时 4W 网需把趴姿撑回标准站姿
                 kp=KP, kd=KD, T=3.0, theta0=THETA0,
                 pitch_tol=0.10,          # 交棒姿态闸（±5.7°；甩接须真正收敛，W3 才接得住）
                 throw_pitch=0.55,       # 段2 启动阈：|pitch|<0.55 rad(31.5°) 上轮力矩（腿伸自然平衡点 26.3°）
                 rate_tol=0.5,            # 交棒角速度闸
                 t_lift_max=1.5)         # 抬升超时保险
        if params:
            p.update(params)
        self.p = p
        self.t = 0.0
        self.t_lift = 0.0
        self.done = False
        self._c = 0.0

    # ---- 轨迹与导数（论文 8.2.1/8.2.3）----
    def _traj(self, t):
        T = self.p["T"]
        tau = min(max(t / T, 0.0), 1.0)
        th0 = self.p["theta0"]
        th = th0 * (1 - 3 * tau ** 2 + 2 * tau ** 3)
        dth = th0 * (6 * tau / T) * (tau - 1)
        ddth = th0 * (6 / (T * T)) * (2 * t - T)
        return th, dth, ddth

    def step(self, z, body_pitch, pitch_rate, vz=0.0, roll=0.0):
        """推进一拍。body_pitch：MJ 口径（asin(−pg[0])，后仰为正）；
        论文 θ<0 为后仰 ⇒ θ = −body_pitch。pitch_rate 同取负。"""
        p = self.p
        out = {"default_leg": POSE_CROUCH.copy(), "wheel_torque": None, "boost": 1.0,
               "off_hip": 0.0, "done": False, "diag": {}}
        if self.done:
            return out
        self.t += self.dt
        t = self.t
        t0, t1 = p["t_hold"], p["t_hold"] + p["t_crouch"]
        t2 = t1 + p["t_chold"]

        if t < t0:                                              # ① 4W 静稳
            return out
        if t < t1:                                              # ② 折蹲（P7T 同款 smoothstep）
            al = _smooth((t - t0) / p["t_crouch"])
            out["default_leg"] = (1 - al) * np.array([0.9833, -2.5589, -0.9833, 2.5589]) + al * POSE_CROUCH
            return out
        if t < t2:                                              # ③ 蹲稳
            out["default_leg"] = POSE_CROUCH.copy()
            return out

        # ④ 两段式抬升【2026-11-06 第五轮定案（对照实验后）】
        #   对照实验（/tmp/p9_legonly）：纯腿慢伸 wt=0 → 机体**准静态**从 46° 前倾到 26.3° 平衡
        #   （轮全程着地、零打滑、零离地）；2W 姿态下 CoM 仍在轮轴后 26°，腿伸完就停在 26°。
        #   ⇒ 段1＝腿慢伸（wt=0，准静态）；段2＝pitch<30° 后轮力矩 PD 甩过平衡点
        #     （2W 姿态下单轮法向 35.4 N ⇒ μN·r=2.83 N·m，比蹲姿 1.67 大 70%，摩擦余量足）。
        self.t_lift += self.dt
        al = _smooth(self.t_lift / p["T"])
        # 段2（甩/接）期间腿从全伸回收到 al=0.85：降 CoM、留 W3 接手时的腿部动态余地
        #   （实测腿全伸 al=1.0 时机体 z 0.34 偏高，W3（平衡 z 0.25）接手即崩）
        al_eff = al if abs(float(body_pitch)) > p.get("throw_pitch", 0.55) else min(al, 0.85)
        out["default_leg"] = (1 - al_eff) * POSE_CROUCH + al_eff * POSE_2W
        out["boost"] = 2.0                            # 腿 P×boost（同 P7T）
        if abs(float(body_pitch)) > p.get("throw_pitch", 0.55):
            # 段1：准静态腿伸，轮不动（wt=0）
            out["wheel_torque"] = 0.0
            return out
        theta = -float(body_pitch)                    # MJ pitch(后仰+) → 论文 θ(后仰−)
        drate = -float(pitch_rate)
        th_d, dth_d, ddth_d = self._traj(self.t_lift)
        u_ff = feedforward(th_d, dth_d, ddth_d)
        err = theta - th_d
        derr = drate - dth_d
        # 【符号实测校正 2026-11-06】MuJoCo 轮力矩正方向＝机体**后仰**（恒力矩实验：C=+3 → pitch 46°→79°，
        #   C=−3 → 前翻过零），与论文平台（正力矩＝前翻）相反 ⇒ u 整体取负。
        #   （u_ff 本身为负＝前翻，在我们符号里天然同向；PD 项不取负会阻碍起立——起步 err>0 输出正力矩。）
        # 【重力补偿 2026-11-06】平台方案前提＝起始**前倾**（CoM 在轮轴前，重力矩帮助前倒，前馈只托住）。
        #   我们机器人 knee 单向限位 ⇒ 前倾蹲姿物理不存在，唯一蹲姿是**后仰 46°**（CoM 在轮轴后），
        #   重力矩 ~4.2 N·m 与前翻方向相反，纯前馈 1.43 N·m 推不动 ⇒ 叠加重力补偿项：
        #   u_grav = m·g·l·sin(pitch)/2（单轮，前翻方向）。总力矩仍随轨迹连续变化，无阶跃 ⇒ 不弹射。
        #   力矩 = P7T 恒力矩 C（前翻方向=负）×衰减 + 平台 PD（俯仰稳定）。
        #   C 在 |pitch|>30° 时全额，30°→0 线性衰减到 0（后半段 PD 接管，防过冲）。
        # 段2：目标 pitch→0（后仰归零），MJ 符号下前翻=负力矩 ⇒ u = −kp·pitch − kd·rate + u_ff 镜像。
        # 实测标定：C=−3 → 前翻 ✓。clamp ±2.5（2W 姿态 μN·r=2.83，留余量）。
        # 过零前后用不同增益：前段大 kp 甩（对抗 26° 重力矩），过零后 kd 主导（吸掉角速度）
        # 【甩+接两拍 2026-11-06】轨迹跟随式 PD 跟不上（腿慢伸限制了机体速度，轨迹走完
        #   机体才 25°，之后位置误差驱动的 PD 甩尾过冲、角速度 8 rad/s 失控）。
        #   改为能量法两拍：pitch>0（后仰）恒力矩 −1.5 甩（<2.83 极限）；
        #   过零（pitch≤0.10 rad）切纯阻尼 kd·rate 吸收角速度（kp=0 防再次加速）。
        # 甩力矩 = 重力补偿（随 pitch 衰减）+ 小常数 c：pitch→0 时力矩→c，
        #   角速度自然减速（防过零大角速度失控——恒力矩版实测过零 rate 8 rad/s 失控）。
        grav = M_BODY * G * L_PEND * abs(np.sin(float(body_pitch))) / 2.0
        c = float(p.get("throw_c", 0.2))
        if float(body_pitch) > 0.05:
            u = -(grav + c)
        else:
            u = -float(p.get("kd2", 1.4)) * float(pitch_rate) - c * float(np.sign(body_pitch))
        u = float(np.clip(u, -2.8, 2.8))   # 2W 姿态摩擦极限 2.83 N·m/轮
        # 【roll 差动】甩的过程 roll 无约束会累积（实测 pitch 13° 时 roll 1.32 翻）。
        #   roll>0（左倾）⇒ 左轮加力右轮减力（把机身滚回来）。真机同款可实现（IMU roll）。
        k_roll = float(p.get("k_roll", 2.0))
        out["wheel_torque_l"] = float(np.clip(u + k_roll * float(roll), -3.5, 3.5))
        out["wheel_torque_r"] = float(np.clip(u - k_roll * float(roll), -3.5, 3.5))
        out["wheel_torque"] = u
        out["diag"] = dict(u=round(u, 4), hold=True)

        # ⑤ 交棒判据（甩+接完成后）：连续 band_steps 拍 |pitch|<pitch_tol 且 |rate|<rate_tol
        self._band = getattr(self, "_band", 0)
        lifted = self.t_lift >= p["T"]
        # 交棒带在**前倾侧**（pitch<0）：W3 对前倾的平衡（后退行走）是强项；
        #   后仰侧交棒实测 W3 进入"后仰 38° 倒退"极限环拉不回（2026-11-06 实测）。
        in_band = (abs(float(body_pitch)) <= p["pitch_tol"]
                   and abs(float(pitch_rate)) <= p["rate_tol"]
                   and float(body_pitch) <= 0.02)
        if lifted and in_band:
            self._band += 1
        else:
            self._band = 0
        if self._band >= p.get("band_steps", 100) or self.t_lift >= p["T"] + p["t_lift_max"]:
            out["done"] = True
        return out
