# ==============================================================================
# Date      : 2026-11-06
# Author    : SYH
# File      : p7t2_platform_standup.py
# Purpose   : P7T2 平台式起立（MuJoCo 先调通、参数回灌 Isaac 验证）。
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
# 两轮站姿：**W3 策略稳态腿姿**（start=2w 收敛值 hip 1.415 / knee −2.113，
#   pitch −1.2°、z 0.247、caster 离地）。旧值 (1.0,−2.1) 是 `_init_2w_pose` 的起步姿，CoM 在轮轴后
#   8 cm ⇒ 平衡俯仰 +26°，俯仰 PD 在 pitch>0 时恒输出负力矩 ⇒ 机体永远向后加速（实测 −6.4 m 漂移）。
#   换成稳态姿后 CoM 正落在轮轴上方，pitch→0 时力矩→0，可静止交棒。
POSE_2W = np.array([1.415, -2.113, -1.415, 2.113])
# 四轮站姿（= 四轮契约 default_dof_pos 的腿四项）。① 静稳段必须**保持它**：
#   ⚠ t<t_hold 段不能直接回 POSE_CROUCH：进 RISE 第一拍腿就跳到深蹲姿，
#   机体被撑到后仰 45.8°（caster 撑地），甩段起点离平衡点更远 ⇒ 甩不起来。
POSE_4W = np.array([0.9833, -2.5589, -0.9833, 2.5589])


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
                 # 新轮矩律（+kv·vx 刹车）把机体收到 pitch≈0、vx≈−0.1，
                 #   交棒窗口＝抬升走完 + 俯仰/角速度/线速度三项同时进带（连续 0.1 s）。
                 pitch_tol=0.35,          # 交棒姿态闸（±20°）
                 rate_tol=1.0,            # 交棒角速度闸
                 vx_tol=0.6,              # 交棒线速度闸（|vx|）
                 band_steps=10,           # 交棒带连续拍数（0.1 s）
                 peak_min=0.10,           # 必须先甩到过 5.7°（防 4W 段误判；律提前介入后峰值只有 ~14°，取小值）
                 throw_pitch=0.55,        # 段2 启动阈：|pitch|<0.55 rad(31.5°) 上轮力矩（腿伸自然平衡点 26.3°）
                 t_lift_max=1.5)         # 抬升超时保险
        if params:
            p.update(params)
        self.p = p
        self.pose_4w = np.asarray(p.get("pose_4w", POSE_4W), dtype=float).reshape(4)
        self.pose_crouch = np.asarray(p.get("crouch") or POSE_CROUCH, dtype=float).reshape(4)
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

    def _apply_wheel_law(self, out, body_pitch, pitch_rate, vx, p, roll=0.0, bias=0.0):
        """轮矩律（唯一）：u = −(kp·pitch + kd·rate) + kv·vx，镜像对 l=u、r=−u。

        u<0＝后向驱动＝机体前翻；+kv·vx 是主动前倾让重力帮刹车（符号反了会发散）。
        折蹲/蹲稳段也调用（提前介入，抑制"腿把机体推到后仰 40°+再一次性翻"造成的大摆动）。
        """
        # 增益钳位（摆动抑制）：MuJoCo 单轮力矩上限 ±2.8 N·m（=μgN·r），
        #   增益再大只会更早饱和、把轮速打到 100+ rad/s 狂奔（实测 kv 14/18 直接发散）。
        #   控制台与 Isaac 共用一组参数（kv 14），这里钳到 MuJoCo 的安全上限。
        #   kd 还有个**下限**：控制台与 Isaac 共用参数（kd 3），但 MuJoCo 在 kd<4 时阻尼不够
        #   ⇒ kv 10 下直接前冲发散（实测跑出 +4.5 m）。故钳到 [4, 6]。
        kp = float(np.clip(float(p.get("kp_pitch", 14.0)), 8.0, 16.0))
        kd = float(np.clip(float(p.get("kd_pitch", 4.0)), 4.0, 6.0))
        kv = float(np.clip(float(p.get("kv_vx", 10.0)), 4.0, 10.0))
        u = -(kp * float(body_pitch) + kd * float(pitch_rate))
        # bias：折蹲前馈（抵消"腿折叠把机体往后推"），在 LIFT 段线性淡出到 0
        u += kv * float(vx) + float(bias)
        u = float(np.clip(u, -2.8, 2.8))   # 2W 姿态摩擦极限 2.83 N·m/轮
        k_roll = float(p.get("k_roll", 0.0))
        _ps = float(p.get("pair_sign", 1.0))
        out["wheel_torque_l"] = float(np.clip(_ps * (u + k_roll * float(roll)), -3.5, 3.5))
        out["wheel_torque_r"] = float(np.clip(_ps * (-u + k_roll * float(roll)), -3.5, 3.5))
        out["wheel_torque"] = _ps * u
        out["diag"] = dict(u=round(u, 4), hold=True)
        return out

    def step(self, z, body_pitch, pitch_rate, vz=0.0, roll=0.0, vx=0.0):
        """推进一拍。body_pitch：MJ 口径（asin(−pg[0])，后仰为正）；
        论文 θ<0 为后仰 ⇒ θ = −body_pitch。pitch_rate 同取负。"""
        p = self.p
        out = {"default_leg": self.pose_4w.copy(), "wheel_torque": None, "boost": 1.0,
               "off_hip": 0.0, "done": False, "diag": {}}
        if self.done:
            return out
        self.t += self.dt
        t = self.t
        # 摆动幅度从**进入脚本第一拍**就记（律已提前到折蹲段介入，摆峰出现在
        #   LIFT 之前；只在 LIFT 段记会读到负值 ⇒ peak_min 永不过 ⇒ 交棒退化成保险丝）。
        #   取**绝对值**：去掉起立起点的前倾阶跃后，机体多数是向前摆（pitch<0），带符号的
        #   峰值会一直是负的。
        self._swing = max(getattr(self, "_swing", 0.0), abs(float(body_pitch)))
        t0, t1 = p["t_hold"], p["t_hold"] + p["t_crouch"]
        t2 = t1 + p["t_chold"]

        # 轮矩律从 **折蹲起点** 就生效（甩段若零轮矩，伸腿动量会把机体甩到后仰 49°、
        #   之后一次性甩接 ⇒ 前冲过冲大）：
        #   等腿伸完才甩，机体已被腿推到后仰 40~49° 并被 caster 撑住，之后只能一次性翻过来
        #   ⇒ 前冲过冲大、来回晃。提前介入后峰值降低、来回摆动明显变小（Isaac 同改）。
        t_law = t0 + float(p.get("law_delay", 0.0))
        if t < t0:                                              # ① 4W 静稳（保持四轮站姿，不预蹲）
            return out
        if t < t1:                                              # ② 折蹲（P7T 同款 smoothstep）
            al = _smooth((t - t0) / p["t_crouch"])
            out["default_leg"] = (1 - al) * self.pose_4w + al * self.pose_crouch
            if t >= t_law:
                self._apply_wheel_law(out, body_pitch, pitch_rate, vx, p, roll,
                                      bias=float(p.get("u_bias", 0.4)))
            return out
        if t < t2:                                              # ③ 蹲稳
            out["default_leg"] = self.pose_crouch.copy()
            if t >= t_law:
                self._apply_wheel_law(out, body_pitch, pitch_rate, vx, p, roll,
                                      bias=float(p.get("u_bias", 0.4)))
            return out

        # ④ 抬升（轮矩律）
        #   实测（/tmp/opencode/mj_rise_law2.py + mj_chain_probe.py，MuJoCo 逐拍记账）：
        #     · 4W→蹲→2W 腿慢伸期间 caster 全程撑地，纯腿动作只会把机体越推越后仰
        #       （hip 0.98→1.42 时俯仰 0°→49°）⇒ **腿不能起立，只有轮矩能**；
        #     · 二轮站姿必须用**W3 策略稳态姿** (1.415, −2.113)：实测该姿 CoM 正落在轮轴上方
        #       （rel 0.1 mm）、pitch −1.2°、caster 离地；旧值 (1.0,−2.1) CoM 在轮轴后 8 cm，
        #       俯仰 PD 在 pitch>0 时恒输出负力矩 ⇒ 机体永远向后加速（实测漂 −6.4 m）；
        #     · u<0＝后向驱动＝机体前翻；u>0＝前驱＝后仰。
        #   轮矩律（唯一，无冲量段）：u = −(kp·pitch + kd·rate) **+ kv·vx**
        #     · pitch>0（后仰）时 PD 项把 u 打到 −2.8 饱和 ⇒ 天然就是"甩"（无需单独冲量段，
        #       实测 kick_c 1.8~2.4 对结果无影响：PD 早饱和）；
        #     · **+ kv·vx**（vx<0 时 u<0）＝主动前倾，让重力帮刹车。符号反了会发散：
        #       u −= kv·vx 是"直接前驱刹车"⇒ 机体被刹得后仰 ⇒ 再加速（实测 −10 m 漂移）。
        #   标定值 kp 8 / kd 2.5 / kv 6：俯仰 49°→+0.4°、机身最低 z 0.16、交棒 vx −0.09、
        #   位移 −1.14 m（起立必须先把 caster 抬过支撑，滚轮必然后移；方向与 Isaac 前移相反）。
        self.t_lift += self.dt
        al = _smooth(self.t_lift / p["T"])
        out["default_leg"] = (1 - al) * self.pose_crouch + al * POSE_2W
        out["boost"] = 2.0                            # 腿 P×boost（同 P7T）
        # 折蹲前馈在 LIFT 段线性淡出（T 走完归零），避免交棒时残留常值力矩
        _bias = float(p.get("u_bias", 0.4)) * max(0.0, 1.0 - self.t_lift / max(float(p["T"]), 1e-6))
        self._apply_wheel_law(out, body_pitch, pitch_rate, vx, p, roll, bias=_bias)

        # ⑤ 交棒判据：抬升走完 + 甩到过 peak_min + 连续 band_steps 拍 |pitch|<pitch_tol、|rate|<rate_tol
        self._band = getattr(self, "_band", 0)
        lifted = self.t_lift >= p["T"]
        in_band = (self._swing >= float(p["peak_min"])
                   and abs(float(body_pitch)) <= p["pitch_tol"]
                   and abs(float(pitch_rate)) <= p["rate_tol"]
                   and abs(float(vx)) <= p["vx_tol"])
        if lifted and in_band:
            self._band += 1
        else:
            self._band = 0
        if self._band >= p.get("band_steps", 10) or self.t_lift >= p["T"] + p["t_lift_max"]:
            out["done"] = True
        return out
