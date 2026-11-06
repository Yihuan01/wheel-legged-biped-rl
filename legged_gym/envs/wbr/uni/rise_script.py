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
        # ---- 【P7T2 回灌 2026-11-06】甩+接剖面（MuJoCo deploy/p7t2_platform_standup.py 第五轮终版）----
        #   lift_profile='p7t2' 时 SUB_LIFT 不再恒力矩 C：后仰侧重力补偿随 pitch 衰减（防过零角速度
        #   失控——MJ 恒力矩版 rate 8 rad/s 失控教训），过零后切纯阻尼吸角速度（kp=0 防再加速）；
        #   交棒判据换「前倾侧进带驻留 1s」（后仰侧交棒 W3 进 38° 倒退极限环——MJ 实测）。
        "lift_profile": "p7t2",    # **默认 p7t2（2026-11-06 泄漏修复后定）**：恒力矩 C＋pitch 事件交棒，
                                  #   控制台链验收 3/3。'p7t2'=C 淡出甩＋阻尼接＋带内驻留（v6）：
                                  #   立即接管语境（check_chain_script，R81 马上接手）ALL PASS 且
                                  #   ③ z 0.283 全站起；但控制台「交棒→停稳门→切网」窗口 0/3 塌
                                  #   （DONE 阻尼保持也压不住前栽）——等门窗口与带内停稳交棒不相容，
                                  #   翻默认需先改链路时序（交棒即切网），见当日进展日志。
        "fade_pitch": 0.17,       # C 平滑淡出区间（rad）：>0.17 全额 C（穿从动轮拖地钉死区），→0 纯接
        "grav_c": 0.4,            # 带回偏置 c·sign(pitch)（±0.4 N·m，pitch≈0 时防漂出带）
        "kd2": 3.0,               # 阻尼增益（全程叠加；v5 扫描定值）
        "t_lift_max_p7t2": 1.2,   # p7t2 保险丝（v6 定值：1.2s 交棒 vx −0.39 后向溜＝停稳门兼容方向，
                                  #   R81 接手 z 0.283 全站起；const 仍用 t_lift_max=1.2）
        "k_roll": 0.0,            # 滚转差动增益（默认关；MJ 用 2.5，Isaac 链 10-28 无差动通过）
        "band_pitch_tol": 0.10,   # 交棒姿态带 ±5.7°（甩接须真正收敛 W3 才接得住）
        "band_rate_tol": 0.5,     # 交棒角速度闸（rad/s；Isaac 姿态响应慢可能要放宽——待链测记账）
        "band_dwell": 1.0,        # 连续驻留时长（s；band 按时间计，跨 dt 可移植）
        "band_front_max": 0.02,   # 交棒带只在前倾侧（pitch≤0.02）
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
        self._band_t = torch.zeros(self.n, device=self.device)   # 【P7T2】交棒带连续驻留计时

    # ------------------------------------------------------------------ 单步
    def step(self, pitch, z, dt, pose_4w=None, rate=None, roll=None):
        """rate：d(pitch)/dt（[n]；Isaac 侧 = −base_ang_vel[:,1]，与 MJ `−ang_b[1]` 同源）。
        roll：机体滚转 asin(−pg_y)（[n]，与 MJ 同式）。两者仅 lift_profile='p7t2' 消费，
        缺省按 0 处理（const 路径与旧调用逐位兼容）。"""
        c = self.cfg
        n, dev = self.n, self.device
        pitch = pitch.to(dev).reshape(-1)
        z = z.to(dev).reshape(-1)
        rate = torch.zeros_like(pitch) if rate is None else rate.to(dev).reshape(-1)
        roll = torch.zeros_like(pitch) if roll is None else roll.to(dev).reshape(-1)
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
            if str(c.get("lift_profile", "const")) == "p7t2":
                # 【P7T2 交棒判据】前倾侧进带驻留：|pitch|≤±5.7° ∧ |rate|≤0.5 ∧ pitch≤前倾侧
                #   连续 band_dwell 秒 → DONE；t_lift_max 保险丝兜底。带计时只累计 LIFT 段。
                in_band = ((pitch.abs() <= float(c["band_pitch_tol"]))
                           & (rate.abs() <= float(c["band_rate_tol"]))
                           & (pitch <= float(c["band_front_max"])))
                self._band_t = torch.where((s == SUB_LIFT) & in_band,
                                           self._band_t + dt, torch.zeros_like(self._band_t))
                _fuse = float(c.get("t_lift_max_p7t2", c["t_lift_max"]))
                lift_done = ((self._band_t >= float(c["band_dwell"])) & (s == SUB_LIFT)) \
                    | (t_after >= _fuse)
            else:
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
        # ⑤ 抬升：2W 姿态 + 力矩剖面（const=恒力矩 C；p7t2=重力补偿衰减+过零纯阻尼）
        m = (s == SUB_LIFT)
        pose = torch.where(m.unsqueeze(1), self.pose_2w.unsqueeze(0).expand(n, -1), pose)
        gm = torch.where(m, torch.full_like(gm, float(c["boost"])), gm)
        if str(c.get("lift_profile", "const")) == "p7t2":
            # 【Isaac 适配 v4（2026-11-06 链测四枪定案）】
            #   实测：① Isaac 的「甩」由 EXTEND 腿推完成——LIFT 入口机体已在 ±100~300°/s 的
            #     前摆/回弹中，MJ 的准静态慢伸入口不存在；
            #   ② 后仰 ~25° 姿态从动轮拖地把俯仰钉死：2.2/2.3/2.55 N·m 三档力矩纹丝不动
            #     （MJ 的 grav+c 温柔甩、净前倾 0.4 N·m 在 Isaac 没有可行窗口）；
            #   ③ v3 的 bang-bang（后仰 C=8 ↔ 过零阻尼，分支不连续 ~10 N·m）在 crossover
            #     附近抖振：pitch 1.8°↔18.6° 来回弹，驻留计不满、保险丝交棒（rate −101°/s）。
            #   ⇒ v4：**C 平滑淡出**——wt = C·clip(pitch/fade_pitch,0,1) + kd2·rate + c·sign(pitch)，
            #     高后仰全额 C（顶开从动轮完成甩），pitch→0 平滑衰减到纯「阻尼接＋带回」，
            #     消除分支不连续；交棒判据=前倾侧进带驻留 band_dwell 秒。限幅 ±torque（=8）。
            fade = torch.clamp(pitch / max(float(c["fade_pitch"]), 1e-6), 0.0, 1.0)
            damp = float(c["kd2"]) * rate + float(c["grav_c"]) * torch.sign(pitch)
            wt_base = float(c["torque"]) * fade + damp
            wt_base = torch.clamp(wt_base, -float(c["torque"]), float(c["torque"]))
            kr = float(c.get("k_roll", 0.0))
            if kr != 0.0:
                # 【滚转差动】MJ: u_l = u + k·roll；Isaac 力矩符号反 ⇒ 差动项同反。
                #   roll>0（左倾）⇒ 左轮减力右轮加力（物理上把机身滚回来，与 MJ 同效）。
                _wt_p = torch.stack([wt_base - kr * roll, wt_base + kr * roll], dim=1)
            else:
                _wt_p = wt_base.unsqueeze(1).expand(n, 2)
            # ⚠ 只在 LIFT 帧生效，**其余帧必须清零**——2026-11-06 控制台链 0/3 塌的根因即此处
            #   误写 `torch.where(m, wt, wt)`（恒等）导致剖面力矩泄漏进 HOLD/CROUCH/EXTEND。
            wt = torch.where(m.unsqueeze(1), _wt_p, torch.zeros_like(wt))
        else:
            wt = torch.where(m.unsqueeze(1), torch.full_like(wt, float(c["torque"])), wt)
        # 完成段：保持 2W 姿态、零轮力矩（上层已切两轮）
        m = (s == SUB_DONE)
        pose = torch.where(m.unsqueeze(1), self.pose_2w.unsqueeze(0).expand(n, -1), pose)
        if str(c.get("lift_profile", "const")) == "p7t2":
            _done_wt = torch.clamp(float(c["kd2"]) * rate + float(c["grav_c"]) * torch.sign(pitch),
                                   -float(c["torque"]), float(c["torque"]))
            wt = torch.where(m.unsqueeze(1), _done_wt.unsqueeze(1).expand(n, 2), wt)

        return {
            "pose": pose,
            "wheel_torque": wt,
            "gain_mult": gm,
            "sub": self.sub.clone(),
            "done": done,
            "boosted": self.boosted.clone(),
        }
