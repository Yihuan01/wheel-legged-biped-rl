# ==============================================================================
# Date      : 2026-10-28
# Author    : SYH
# File      : rise_script.py
# Purpose   : 起立脚本：RISE 相位内细分为 HOLD → CROUCH → CROUCH_HOLD → EXTEND → LIFT → DONE
# 六个子段。腿目标走 smoothstep 轨迹；轮力矩在 lift_profile='p7t2' 时为
# 俯仰 PD 律（u = −(kp·pitch + kd·rate) + kv·vx，镜像对施加），'const' 为恒力矩 C。
# ==============================================================================

import torch

# 子段编号（**不是** uni_ph；uni_ph 仍是 7=RISE 不变，这里只是 RISE 内的细分，走遥测 rise_sub）
SUB_HOLD = 0        # ① 4W 低姿态静稳
SUB_CROUCH = 1      # ② smoothstep 折蹲
SUB_CROUCH_HOLD = 2 # ③ 蹲姿稳住
SUB_EXTEND = 3      # ④ smoothstep 伸腿到 2W
SUB_LIFT = 4        # ⑤ 抬升（轮力矩律见 lift_profile）
SUB_DONE = 5        # 完成（已交接，等待上层切两轮）

SUB_NAMES = {SUB_HOLD: "HOLD", SUB_CROUCH: "CROUCH", SUB_CROUCH_HOLD: "CROUCH_HOLD",
             SUB_EXTEND: "EXTEND", SUB_LIFT: "LIFT", SUB_DONE: "DONE"}


def default_cfg():
    """起立脚本默认参数。lift_profile='p7t2'（默认）＝俯仰 PD 律；'const'＝恒力矩 C 旧律。

    ⚠ 平台软件限幅 8 N·m（`docs/04-平台资料/02_控制算法.md`）：`torque` 超过 8 在真机
      会被裁掉，属未验证行为。
    ⚠ `z_handover_min` 保持 None：真机没有机体高度测量，交接判据只能用俯仰。
    """
    return {
        "t_hold": 0.6,          # ① 4W 静稳
        "t_crouch": 0.5,        # ② 折蹲时长
        "t_crouch_hold": 0.4,   # ③ 蹲姿稳住
        "t_extend": 0.5,        # ④ 伸腿时长
        "t_lift_max": 1.2,      # ⑤ 抬升超时保险丝
        "torque": 8.0,          # ⑤ 恒定轮力矩（N·m；= 平台软件限幅上限）
        "boost": 2.0,           # ④⑤ 腿 P 增益倍率（压自旋）
        "advance": "pitch",     # 'timer' | 'pitch'
        "pitch_handover": -0.26,  # pb=−body_pitch 越此值即交接（rad，−0.26 ≈ −15°）
        "z_handover_min": None,   # 交接附加高度门；None ⇒ 纯俯仰（真机可测量口径）
        "hcmd": 0.28,           # 交接后两轮体高指令 = R81 原生站高（用 0.22 会矮 70 mm，
                                # 跳跃预蹲目标反而高于站高、蹲量被夹到 0）
        "hcmd_hold": 3.0,       # 交接后 hcmd 保持窗（秒）
        # ---- p7t2 轮矩律（默认）：u = −(kp·pitch + kd·rate) + kv·vx，镜像对施加 tau_L = p、tau_R = −p ----
        # ⚠ 符号（Isaac 实测标定）：p_roll>0 ⇒ 整机前移（caster 撑地、俯仰不动）；
        # p_roll<0 ⇒ 机体前翻。pitch 正=后仰，故 pitch>0 必须给负力矩；
        # PD 在俯仰 ≥30° 时自然饱和成"甩"，无需单独冲量段。
        "lift_profile": "p7t2",   # 默认：俯仰 PD 律（镜像对）。
        "kp_pitch": 12.0,         # 俯仰 P（N·m/rad）
        "kd_pitch": 3.0,          # 俯仰 D（N·m/(rad/s)）
        "kv_vx": 14.0,            # 线速度回馈：vx<0（后溜）⇒ 力矩更负＝主动前倾让重力帮刹车。
                                  # ⚠ 上限 ~20：再大俯仰摆动 ±35°；过小（6）后溜 >0.6 m。
        "vx_tol": 0.6,            # 交棒线速度闸（m/s）
        "t_lift_max_p7t2": 1.2,   # p7t2 保险丝（s；const 仍用 t_lift_max）
        "k_roll": 0.0,            # 滚转差动增益（默认关：差动＝偏航力矩，MJ 实测甩段 yaw 振荡）
        "extend_retract": 1.0,    # 伸腿腿长上限（1.0=全伸）
        "extend_boost": 1.5,      # 伸腿腿 P 倍率
        "band_pitch_tol": 0.10,   # 交棒姿态带 ±5.7°（实测 0.35 会在 16° 就交棒，没真站直）
        "band_rate_tol": 0.5,     # 交棒角速度闸（rad/s；0.5≈29°/s）
        "band_dwell": 0.10,       # 连续驻留时长（s）
        "band_front_max": None,   # 交棒带侧向限制（None=不限；旧值 0.02 只准前倾侧）
        "law_from_sub": 1,        # 轮矩律从哪个子段开始生效（1=CROUCH / 2=CROUCH_HOLD / 3=EXTEND / 4=LIFT）
        "peak_min": 0.25,         # 必须先甩到过 14.3°（防 4W 段 pitch≈0 误判；链路测试实测
                                  # 浅蹲时峰值仅 24.7°，0.5 会一直走保险丝）
        # 起立终点腿姿＝**W3 策略稳态腿姿**（hip 1.415 / knee −2.113；仅 p7t2 剖面生效）。
        # ⚠ 旧值 (1.0,−2.1) 的 CoM 在轮轴后 8 cm ⇒ 平衡俯仰 +26°，俯仰 PD 在该姿下
        # 永远停在 +26° 或持续后溜（MuJoCo 逐拍实测漂 −6.4 m）。
        "pose_2w": [1.415, -2.113, -1.415, 2.113],
    }


class RiseScript:
    """向量化 P7T 脚本链。状态：sub / t_sub / boosted，全部按 env 并行。

    用法（env 或控制台每控制步调一次）：
        rs = RiseScript(n, device, cfg, crouch_pose, pose_2w)
        out = rs.step(pitch=env._body_pitch(), z=env.root_states[:,2], dt=env.dt)
        out["pose"]         # [n,4] 腿目标关节角（顺序同 env.leg_dof）
        out["wheel_torque"] # [n,2] 两驱动轮力矩（其余段为 0）
        out["gain_mult"]    # [n]   腿 P 增益倍率（1.0 或 boost）
        out["sub"]          # [n]   当前子段
        out["done"]         # [n]   bool，本拍完成交接
    """

    def __init__(self, num_envs, device, cfg=None, crouch_pose=None, pose_2w=None, pose_4w=None):
        self.n = int(num_envs)
        self.device = device
        self.cfg = dict(default_cfg())
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if v is not None})
        # 蹲姿 [4]（hip/knee 镜像）与 4W/2W 默认腿姿 [4]
        self.crouch = self._vec(crouch_pose, [1.5391, -2.2722, -1.5391, 2.2722])
        self.pose_2w = self._vec(pose_2w, [1.0, -2.1, -1.0, 2.1])
        # p7t2 剖面：目标腿姿用 cfg 的 W3 稳态姿覆盖调用方传入值
        if str(self.cfg.get("lift_profile", "const")) == "p7t2" and self.cfg.get("pose_2w") is not None:
            self.pose_2w = self._vec(self.cfg["pose_2w"], None)
        self.pose_4w = self._vec(pose_4w, None)   # None ⇒ 由调用方传（env.default_dof_pos 起点）
        self.reset(torch.arange(self.n, dtype=torch.long, device=device))

    def _vec(self, v, default):
        if v is None:
            if default is None:
                return None
            v = default
        return torch.as_tensor(v, dtype=torch.float32, device=self.device).reshape(-1)

    # ------------------------------------------------------------------ 生命周期
    def reset(self, env_ids):
        if not hasattr(self, "sub"):
            self.sub = torch.zeros(self.n, dtype=torch.long, device=self.device)
            self.t_sub = torch.zeros(self.n, device=self.device)
            self.boosted = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.sub[env_ids] = SUB_HOLD
        self.t_sub[env_ids] = 0.0
        self.boosted[env_ids] = False
        self._band_t = torch.zeros(self.n, device=self.device)   # 交棒带连续驻留计时
        self._peak = torch.full((self.n,), -9.0, device=self.device)   # 起立段俯仰峰值

    # ------------------------------------------------------------------ 单步
    def step(self, pitch, z, dt, pose_4w=None, rate=None, roll=None, vx=None):
        """rate：d(pitch)/dt（[n]；Isaac 侧 = −base_ang_vel[:,1]，与 MJ `−ang_b[1]` 同源）。
        roll：机体滚转 asin(−pg_y)（[n]，与 MJ 同式）。vx：基座前向速度（m/s）。
        三者仅 lift_profile='p7t2' 消费，缺省按 0 处理（const 路径与旧调用逐位兼容）。"""
        c = self.cfg
        n, dev = self.n, self.device
        pitch = pitch.to(dev).reshape(-1)
        z = z.to(dev).reshape(-1)
        rate = torch.zeros_like(pitch) if rate is None else rate.to(dev).reshape(-1)
        roll = torch.zeros_like(pitch) if roll is None else roll.to(dev).reshape(-1)
        vx = torch.zeros_like(pitch) if vx is None else vx.to(dev).reshape(-1)
        p4 = self._vec(pose_4w, None) if pose_4w is not None else self.pose_4w
        if p4 is None:
            raise ValueError("RiseScript.step 需要 pose_4w（4W 默认腿姿）")

        def smooth(a):
            a = torch.clamp(a, 0.0, 1.0)
            return a * a * (3.0 - 2.0 * a)

        # ---- 1) 先推进子段（驻留时长 t_after = t_sub + dt），再产目标 ----
        # 顺序不能反：保证返回的 sub / 目标姿态 / 轮力矩属于同一拍。
        s = self.sub
        t_after = self.t_sub + dt
        nxt = s.clone()
        # 新子段的驻留时长：转移时**带过冲量**（t_after − 边界），不归零 —— 否则每段都吃一个 dt，
        # 4 段累计漂移 0.03~0.04s（老控制台用绝对时间不漂，这里等价于绝对时间语义）。
        ntime = t_after.clone()
        adv = torch.zeros(n, dtype=torch.bool, device=dev)

        a1 = (s == SUB_HOLD) & (t_after >= c["t_hold"])
        nxt = torch.where(a1, torch.full_like(nxt, SUB_CROUCH), nxt)
        ntime = torch.where(a1, t_after - float(c["t_hold"]), ntime); adv = adv | a1
        a2 = (s == SUB_CROUCH) & (t_after >= c["t_crouch"])
        nxt = torch.where(a2, torch.full_like(nxt, SUB_CROUCH_HOLD), nxt)
        ntime = torch.where(a2, t_after - float(c["t_crouch"]), ntime); adv = adv | a2
        a3 = (s == SUB_CROUCH_HOLD) & (t_after >= c["t_crouch_hold"])
        nxt = torch.where(a3, torch.full_like(nxt, SUB_EXTEND), nxt)
        ntime = torch.where(a3, t_after - float(c["t_crouch_hold"]), ntime); adv = adv | a3
        a4 = (s == SUB_EXTEND) & (t_after >= c["t_extend"])
        nxt = torch.where(a4, torch.full_like(nxt, SUB_LIFT), nxt)
        ntime = torch.where(a4, t_after - float(c["t_extend"]), ntime); adv = adv | a4
        # ⑤→DONE：p7t2 用「姿态/角速度/线速度 三闸驻留」；const 用旧 pitch 事件
        # ⚠ 旧 pb 事件（pitch 越 +15° 即交）会在机体仍带 −2 m/s 后溜时交棒，W3 接不住
        # （MuJoCo 同现象）⇒ 改与 MJ 同步的三闸驻留（姿态/角速度/线速度）。
        _peak = torch.maximum(getattr(self, "_peak", torch.full_like(pitch, -9.0)), pitch)
        self._peak = _peak
        if str(c.get("lift_profile", "const")) == "p7t2":
            in_band = ((_peak >= float(c.get("peak_min", 0.25)))
                       & (torch.abs(pitch) <= float(c.get("band_pitch_tol", 0.35)))
                       & (torch.abs(rate) <= float(c.get("band_rate_tol", 1.0)))
                       & (torch.abs(vx) <= float(c.get("vx_tol", 0.6))))
            fm = c.get("band_front_max", None)
            if fm is not None:
                in_band = in_band & (pitch <= float(fm))
            self._band_t = torch.where(in_band, self._band_t + dt, torch.zeros_like(self._band_t))
            lift_done = (self._band_t >= float(c.get("band_dwell", 0.10))) | \
                        (t_after >= float(c.get("t_lift_max_p7t2", c["t_lift_max"])))
        elif str(c["advance"]) == "pitch":
            # const 剖面仍用原版 pb 事件（摆动中交棒，R81 空中接——验证过的行为）。
            pb = -pitch
            ph_ok = pb > float(c["pitch_handover"])
            zh = c.get("z_handover_min", None)
            if zh is not None:
                ph_ok = ph_ok & (z > float(zh))
            lift_done = ph_ok | (t_after >= c["t_lift_max"])
        else:
            lift_done = (t_after >= c["t_lift_max"])
        a5 = (s == SUB_LIFT) & lift_done
        nxt = torch.where(a5, torch.full_like(nxt, SUB_DONE), nxt)
        ntime = torch.where(a5, torch.zeros_like(ntime), ntime); adv = adv | a5

        self.sub = nxt
        self.t_sub = torch.where(adv, ntime, t_after)
        self.boosted = self.boosted | (self.sub >= SUB_EXTEND)
        done = a5

        # ---- 2) 按**新** sub 产目标（进段的这一拍 tt=带过冲量，smoothstep 连续）----
        s = self.sub
        tt = self.t_sub

        pose = p4.unsqueeze(0).repeat(n, 1)
        wt = torch.zeros(n, 2, device=dev)
        gm = torch.ones(n, device=dev)

        # ① 静稳：4W 姿态（默认即 p4，无需覆盖）
        # ② 折蹲：4W → 蹲姿
        m = (s == SUB_CROUCH)
        al = smooth(tt / max(c["t_crouch"], 1e-6))
        crouch_tgt = (1.0 - al).unsqueeze(1) * p4.unsqueeze(0) + al.unsqueeze(1) * self.crouch.unsqueeze(0)
        pose = torch.where(m.unsqueeze(1), crouch_tgt, pose)
        # ③ 蹲稳：蹲姿
        m = (s == SUB_CROUCH_HOLD)
        pose = torch.where(m.unsqueeze(1), self.crouch.unsqueeze(0).expand(n, -1), pose)
        # ④ 伸腿：蹲姿 → 2W（×retract 收腿），腿 P ×boost
        m = (s == SUB_EXTEND)
        al = smooth(tt / max(c["t_extend"], 1e-6)) * float(c.get("extend_retract", 0.85))
        ext_tgt = (1.0 - al).unsqueeze(1) * self.crouch.unsqueeze(0) + al.unsqueeze(1) * self.pose_2w.unsqueeze(0)
        pose = torch.where(m.unsqueeze(1), ext_tgt, pose)
        gm = torch.where(m, torch.full_like(gm, float(c.get("extend_boost", c["boost"]))), gm)
        # ⑤ 抬升：2W 姿态（×retract 收腿，与 EXTEND 末连续）+ 力矩剖面
        m = (s == SUB_LIFT)
        _lift_tgt = (float(c.get("extend_retract", 0.85)) * self.pose_2w.unsqueeze(0)
                     + (1.0 - float(c.get("extend_retract", 0.85))) * self.crouch.unsqueeze(0))
        pose = torch.where(m.unsqueeze(1), _lift_tgt.expand(n, -1), pose)
        gm = torch.where(m, torch.full_like(gm, float(c["boost"])), gm)
        if str(c.get("lift_profile", "const")) == "p7t2":
            # ⚠ 符号（Isaac 实测标定）：u = −(kp·pitch + kd·rate) + kv·vx
            # · pitch 正 = 后仰（与 MuJoCo 同口径）；
            # · p_roll>0 = 前驱 ⇒ caster 撑地、俯仰不动、整机前移；
            # · p_roll<0 = 机体前翻 ⇒ 起立必须取负号；pitch≥30° 时 PD 自然饱和成"甩"；
            # · kv·vx 项：后溜（vx<0）时力矩更负＝主动前倾让重力帮刹车（符号反了会发散）。
            p_roll = (-(float(c.get("kp_pitch", 8.0)) * pitch
                        + float(c.get("kd_pitch", 2.5)) * rate)
                      + float(c.get("kv_vx", 10.0)) * vx)
            p_roll = torch.clamp(p_roll, -float(c["torque"]), float(c["torque"]))
            kr = float(c.get("k_roll", 0.0))
            if kr != 0.0:
                # 【滚转差动】差动＝偏航力矩（MJ 实测甩段 yaw 振荡即差动增益），保持小值。
                wt = torch.stack([p_roll - kr * roll, -p_roll + kr * roll], dim=1)
            else:
                wt = torch.stack([p_roll, -p_roll], dim=1)
            # 轮矩律从 **EXTEND** 就开始生效（若等 LIFT 才介入，伸腿动量已把机体推到
            # 后仰 49°、caster 撑地，之后一次性甩接 ⇒ 前冲过冲 −30°、回摆 +11°）。
            _from = int(c.get("law_from_sub", 3))
            mwt = (s >= _from) & (s <= SUB_LIFT)
            wt = torch.where(mwt.unsqueeze(1), wt, torch.zeros_like(wt))
        else:
            wt = torch.where(m.unsqueeze(1), torch.full_like(wt, float(c["torque"])), wt)
        # 完成段：保持 2W 姿态；p7t2 保留线速度回馈（防交棒后溜），const 零轮力矩
        m = (s == SUB_DONE)
        pose = torch.where(m.unsqueeze(1), self.pose_2w.unsqueeze(0).expand(n, -1), pose)
        if str(c.get("lift_profile", "const")) == "p7t2":
            _p = torch.clamp(float(c.get("kv_vx", 10.0)) * vx,
                             -float(c["torque"]), float(c["torque"]))   # 仅速度回馈（俯仰已交上层）
            _done_wt = torch.stack([_p, -_p], dim=1)
            wt = torch.where(m.unsqueeze(1), _done_wt, wt)

        return {
            "pose": pose,
            "wheel_torque": wt,
            "gain_mult": gm,
            "sub": self.sub.clone(),
            "done": done,
            "boosted": self.boosted.clone(),
        }
