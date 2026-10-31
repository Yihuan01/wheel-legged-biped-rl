# ==============================================================================
# Date      : 2026-10-28
# Author    : SYH
# File      : rise_script.py
# Purpose   : 起立脚本：RISE 相位内细分为 HOLD → CROUCH → CROUCH_HOLD → EXTEND → LIFT → DONE
#             六个子段，用 smoothstep 轨迹与恒力矩抬升把机身撑到两轮站姿。
# ==============================================================================

import torch

# 子段编号（**不是** uni_ph；uni_ph 仍是 7=RISE 不变，这里只是 RISE 内的细分，走遥测 rise_sub）
SUB_HOLD = 0        # ① 4W 低姿态静稳
SUB_CROUCH = 1      # ② smoothstep 折蹲
SUB_CROUCH_HOLD = 2 # ③ 蹲姿稳住
SUB_EXTEND = 3      # ④ smoothstep 伸腿到 2W
SUB_LIFT = 4        # ⑤ 恒力矩抬升
SUB_DONE = 5        # 完成（已交接，等待上层切两轮）

SUB_NAMES = {SUB_HOLD: "HOLD", SUB_CROUCH: "CROUCH", SUB_CROUCH_HOLD: "CROUCH_HOLD",
             SUB_EXTEND: "EXTEND", SUB_LIFT: "LIFT", SUB_DONE: "DONE"}


def default_cfg():
    """P7T 默认参数（＝控制台场景 `LQR起立R81跳跃` 里那组，harness 验证值）。

    ⚠ 2026-10-27 两次上机前修正：
      - `torque` 9.0 → 8.0：平台软件限幅 8 N·m（`docs/04-平台资料/02_控制算法.md`），
        C=9 在真机会被裁到 8，属未验证行为；链上以 8 重验，抬升轨迹与交接点须重新记账。
      - `z_handover_min` 0.20 → None：真机没有机体高度测量，交接判据只能用俯仰（纯 `pb > −0.26`）。
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
                                #   跳跃预蹲目标反而高于站高、蹲量被夹到 0）
        "hcmd_hold": 3.0,       # 交接后 hcmd 保持窗（秒）
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

    # ------------------------------------------------------------------ 单步
    def step(self, pitch, z, dt, pose_4w=None):
        c = self.cfg
        n, dev = self.n, self.device
        pitch = pitch.to(dev).reshape(-1)
        z = z.to(dev).reshape(-1)
        p4 = self._vec(pose_4w, None) if pose_4w is not None else self.pose_4w
        if p4 is None:
            raise ValueError("RiseScript.step 需要 pose_4w（4W 默认腿姿）")

        def smooth(a):
            a = torch.clamp(a, 0.0, 1.0)
            return a * a * (3.0 - 2.0 * a)

        # ---- 1) 先推进子段（驻留时长 t_after = t_sub + dt），再产目标 ----
        #   顺序不能反：保证返回的 sub / 目标姿态 / 轮力矩属于同一拍。
        s = self.sub
        t_after = self.t_sub + dt
        nxt = s.clone()
        # 新子段的驻留时长：转移时**带过冲量**（t_after − 边界），不归零 —— 否则每段都吃一个 dt，
        #   4 段累计漂移 0.03~0.04s（老控制台用绝对时间不漂，这里等价于绝对时间语义）。
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
        # ⑤→DONE：pitch 事件（p_g 越 −15°）优先，定时保险丝兜底
        #   口径照 harness：pb = −body_pitch（正=后仰）；pb > pitch_handover(默认 −0.26 rad ≈ −15°) 即交接。
        if str(c["advance"]) == "pitch":
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
        # ④ 伸腿：蹲姿 → 2W，腿 P ×boost
        m = (s == SUB_EXTEND)
        al = smooth(tt / max(c["t_extend"], 1e-6))
        ext_tgt = (1.0 - al).unsqueeze(1) * self.crouch.unsqueeze(0) + al.unsqueeze(1) * self.pose_2w.unsqueeze(0)
        pose = torch.where(m.unsqueeze(1), ext_tgt, pose)
        gm = torch.where(m, torch.full_like(gm, float(c["boost"])), gm)
        # ⑤ 抬升：2W 姿态 + 恒力矩
        m = (s == SUB_LIFT)
        pose = torch.where(m.unsqueeze(1), self.pose_2w.unsqueeze(0).expand(n, -1), pose)
        gm = torch.where(m, torch.full_like(gm, float(c["boost"])), gm)
        wt = torch.where(m.unsqueeze(1), torch.full_like(wt, float(c["torque"])), wt)
        # 完成段：保持 2W 姿态、零轮力矩（上层已切两轮）
        m = (s == SUB_DONE)
        pose = torch.where(m.unsqueeze(1), self.pose_2w.unsqueeze(0).expand(n, -1), pose)

        return {
            "pose": pose,
            "wheel_torque": wt,
            "gain_mult": gm,
            "sub": self.sub.clone(),
            "done": done,
            "boosted": self.boosted.clone(),
        }
