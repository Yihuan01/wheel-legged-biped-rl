# ==============================================================================
# Date      : 2026-10-26
# Author    : SYH
# File      : fsm.py
# Purpose   : 相位状态机：由外部请求码（起立 / 趴下 / 跳跃）与事件推进相位跳转，含超时保护。
# ==============================================================================

import torch

from .phase import (
    PH_IDLE2W, PH_HOLD4W, PH_RISE, PH_HANDOVER, PH_LOWER, PH_TILT, PH_ABSORB,
    PH_JUMP1, PH_JUMP5, JUMP_PHASES, NUM_PHASES,
)

# 外部请求码（供部署 FSM / 回放 / 测试注入；训练时可全 0 由事件推进）
REQ_NONE = 0
REQ_RISE = 1   # 四轮 → 起立
REQ_LOWER = 2  # 两轮 → 趴下
REQ_JUMP = 3   # 两轮 → 跳跃（阶段1 预留）


class PhaseFSM:
    """统一模式状态机（向量化，num_envs 并行）。

    输入（每控制步一次）：
        phase : [n] long   当前相位（phase.py 枚举）
        ph_t  : [n] float  当前相位已持续时长（s）
        req   : [n] long   外部请求码（REQ_*）
        cond  : dict[str, Tensor]  可测条件，见 `_compute_events`
        dt    : float      控制步长

    输出：
        new_phase, new_ph_t, info(dict)
    """

    def __init__(self, cfg):
        # cfg: 一个带属性的简单对象（或 dict 包装）；缺失用默认值
        def g(k, d):
            return getattr(cfg, k, d) if cfg is not None else d
        self.t_rise = float(g("t_rise", 0.60))          # RISE 过渡时长（保险丝/脚本驱动时长）
        self.t_handover = float(g("t_handover", 0.50))  # HANDOVER 时长
        self.t_lower = float(g("t_lower", 0.50))        # LOWER 时长
        self.t_align = float(g("t_align", 0.30))        # HOLD4W / IDLE2W 对齐驻留（可 0）
        self.upright_pitch_tol = float(g("upright_pitch_tol", 0.15))
        self.settled_v_tol = float(g("settled_v_tol", 0.12))
        self.settled_w_tol = float(g("settled_w_tol", 0.10))
        self.ev_confirm_steps = int(g("ev_confirm_steps", 3))

    # ------------------------------------------------------------------ 事件
    def _compute_events(self, phase, cond):
        """从可测条件算本拍事件（起立到位 / 趴下到位 / 从动轮触地）。全部只用 cond 里的量。"""
        pitch = cond["pitch"]                       # [n] rad
        n_wheel = cond["n_wheel"]                   # [n] 0..2 驱动轮触地数
        n_caster = cond["n_caster"]                 # [n] 0..2 从动轮触地数
        speed = cond["speed"]                       # [n] m/s 水平速度
        yawrate = cond.get("yawrate")               # [n] rad/s 或 None

        # 起立到位（两轮站稳）：俯仰小 ∧ 双驱动轮触地 ∧ 双从动轮离地 ∧ 低速
        upright = (
            (torch.abs(pitch) < self.upright_pitch_tol)
            & (n_wheel == 2) & (n_caster == 0)
            & (speed < self.settled_v_tol)
        )
        # 趴下到位（四轮静稳）：双驱动轮 + 双从动轮全触地 ∧ 俯仰小 ∧ 低速 ∧ 偏航角速度小
        settled4 = (
            (n_wheel == 2) & (n_caster == 2)
            & (torch.abs(pitch) < self.upright_pitch_tol)
            & (speed < self.settled_v_tol)
        )
        if yawrate is not None:
            settled4 = settled4 & (torch.abs(yawrate) < self.settled_w_tol)
        # 从动轮触地（任一）
        caster_touch = n_caster >= 1
        return upright, settled4, caster_touch

    # ------------------------------------------------------------------ 主入口
    def update(self, phase, ph_t, req, cond, dt):
        n = phase.shape[0]
        dev = phase.device
        upright, settled4, caster_touch = self._compute_events(phase, cond)

        new_phase = phase.clone()
        new_ph_t = ph_t + dt
        # 事件新鲜标志（仅"转入该稳定相"那一拍为 True，供一次性结算/交接用）
        fresh_upright = torch.zeros(n, dtype=torch.bool, device=dev)
        fresh_settled4 = torch.zeros(n, dtype=torch.bool, device=dev)
        # 触发被守卫拒绝（供诊断：例如"跳跃中请求趴下"被挡）
        rejected = torch.zeros(n, dtype=torch.bool, device=dev)

        def go(mask, nxt):
            nonlocal new_phase, new_ph_t
            new_phase = torch.where(mask, torch.full_like(new_phase, int(nxt)), new_phase)
            new_ph_t = torch.where(mask, torch.zeros_like(new_ph_t), new_ph_t)

        in_jump = torch.zeros(n, dtype=torch.bool, device=dev)
        for jp in JUMP_PHASES:
            in_jump |= (phase == jp)

        # ── 四轮静稳 → 起立 ──（仅 HOLD4W，且驻留够 t_align；跳跃/过渡态不可达）
        to_rise = (phase == PH_HOLD4W) & (req == REQ_RISE) & (ph_t >= self.t_align)
        # 起立转起 → 交接（脚本/时长驱动）
        to_handover = (phase == PH_RISE) & (ph_t >= self.t_rise)
        # 交接 → 两轮（起立到位事件确认；这里用"新鲜 upright"作单拍触发，确认计数由 env 侧累计后传入更稳，
        #   本阶段简化为事件当拍触发）
        to_idle2w = (phase == PH_HANDOVER) & upright
        # ── 两轮 → 趴下 ──（仅 IDLE2W；跳跃相中请求被守卫拒绝）
        to_lower = (phase == PH_IDLE2W) & (req == REQ_LOWER) & (ph_t >= self.t_align)
        # 降高 → 探地（时长驱动）
        to_tilt = (phase == PH_LOWER) & (ph_t >= self.t_lower)
        # 探地 → 吸能（从动轮触地）
        to_absorb = (phase == PH_TILT) & caster_touch
        # 吸能 → 四轮（四轮到位事件）
        to_hold4w = (phase == PH_ABSORB) & settled4
        # ── 两轮 → 跳跃（只进入 JUMP1，后续时序由跳跃相位机推进）──
        to_jump1 = (phase == PH_IDLE2W) & (req == REQ_JUMP)

        # 守卫拒绝诊断：在跳跃相 / 过渡相里请求 rise/lower 被挡
        rejected = rejected | (in_jump & ((req == REQ_LOWER) | (req == REQ_RISE)))
        # 跳跃请求只允许在两轮 idle（PH_IDLE2W）发出；起立 / 趴下 / 过渡相一律拒绝。
        rejected = rejected | ((req == REQ_JUMP) & (phase != PH_IDLE2W))
        # 起立只允许从 HOLD4W 发、趴下只允许从 IDLE2W 发（其余相位拒绝，便于诊断）
        rejected = rejected | ((req == REQ_RISE) & (phase != PH_HOLD4W))
        rejected = rejected | ((req == REQ_LOWER) & (phase != PH_IDLE2W))

        # 应用转移
        go(to_rise, PH_RISE)
        go(to_handover, PH_HANDOVER)
        fresh_upright = fresh_upright | to_idle2w
        go(to_idle2w, PH_IDLE2W)
        go(to_lower, PH_LOWER)
        go(to_tilt, PH_TILT)
        go(to_absorb, PH_ABSORB)
        fresh_settled4 = fresh_settled4 | to_hold4w
        go(to_hold4w, PH_HOLD4W)
        go(to_jump1, PH_JUMP1)

        # 请求是否在本拍被"接受"（触发了对应转移）——供 env 侧清锁存用（未接受则保持，防丢请求）
        consumed = to_rise | to_lower | to_jump1
        info = {
            "fresh_upright": fresh_upright,
            "fresh_settled4": fresh_settled4,
            "rejected_request": rejected,
            "in_jump": in_jump,
            "consumed": consumed,
        }
        return new_phase, new_ph_t, info

    # ------------------------------------------------------------------ 阶段1 跳跃相占位推进
    def advance_jump_stub(self, phase, ph_t, dt, jump_phase_durs):
        """阶段1：跳跃相没有真实时序。若被手动置入 JUMP1..5，用一个简单定时把它们
        推到 JUMP5 后回 IDLE2W，仅保证 FSM 不卡死、结构可测。真实跳跃时序在阶段2 替换。"""
        new_phase = phase.clone()
        new_ph_t = ph_t + dt
        for jp in JUMP_PHASES:
            dur = float(jump_phase_durs.get(jp, 0.15))
            done = (phase == jp) & (ph_t >= dur)
            nxt = PH_JUMP5 if jp == PH_JUMP5 else jp + 1
            target = PH_IDLE2W if jp == PH_JUMP5 else nxt
            new_phase = torch.where(done, torch.full_like(new_phase, int(target)), new_phase)
            new_ph_t = torch.where(done, torch.zeros_like(new_ph_t), new_ph_t)
        return new_phase, new_ph_t
