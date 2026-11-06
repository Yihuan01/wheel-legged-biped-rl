#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Date      : 2026-10-26
# Author    : SYH
# File      : demo_cycle.py
# Purpose   : 四策略连跑自检：四轮 → 起立 → 两轮站定 → 趴下 → 四轮，逐段验证切换可用。
#             退出码 0 表示全通过。
# ==============================================================================

"""统一多模式程序「四策略实跑」验证（wbr_uni）。

一套 wbr_uni env 依次跑四个网络，忠实复刻 wbr_console 的 run_switch 时序，证明
「四轮/起立/趴下/两轮(W12) 全可用、可切换」：
  四轮(F3e 164) → 起立(P7S 173) → 两轮(W12 161, 站 5s) → 趴下(P7L 173) → 四轮(F3e)。

用法：python -m legged_gym.envs.wbr.uni.demo_cycle
退出码 0 = 全 PASS。
"""
import sys
import math

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.scripts.wbr_console.deps.play02_core import load_actor_critic
from legged_gym.envs.wbr.switch.wbr_switch_config import (
    SP_HOLD4W, SP_RISE, SP_HANDOVER, SP_HOLD2W, SP_LOWER, SP_TILT, SP_ABSORB,
)

CKPT = {
    'fourw': "releases/P7F3e_yaw1_dr15/model_11800.pt",
    'standup': "releases/P7S1b_standup/model_1300.pt",
    'w12': "releases/W12_noise10_2900_randwalk/model_2900.pt",
    'lie': "releases/P7L_liedown/model_300.pt",
}
W12_ACT_PERM = [0, 1, 3, 4, 2, 5]   # W12 序 → wbr4 序


def main():
    sys.argv = ["demo", "--task", "wbr_uni", "--headless", "--num_envs=3",
                "--sim_device", "cuda:0", "--pipeline", "gpu"]
    args = get_args()
    env, _ = task_registry.make_env(name=args.task, args=args)
    dev, n = env.device, env.num_envs
    ids = torch.arange(n, dtype=torch.long, device=dev)
    sw = env.cfg.rewards.switch

    p_4w = load_actor_critic(CKPT['fourw'], 164, dev)
    p_stand = load_actor_critic(CKPT['standup'], 173, dev)
    p_w12 = load_actor_critic(CKPT['w12'], 161, dev)
    p_lie = load_actor_critic(CKPT['lie'], 173, dev)

    dflt_4w = env.default_dof_pos.clone()
    dflt_2w = dflt_4w.clone()
    dflt_2w[:, env.leg_dof] = torch.tensor([1.0, -2.1, -1.0, 2.1], device=dev)

    def set_default(d):
        env.default_dof_pos.copy_(d)

    def set_cmd(vx=0.0, yaw=0.0, h=0.19, roll=0.0):
        env.command_targets[:, 0] = vx; env.command_targets[:, 1] = 0.0
        env.command_targets[:, 2] = yaw; env.command_raw_yaw[:] = yaw
        env.command_targets[:, 3] = 0.0; env.command_targets[:, 5] = h
        env.base_roll_targets[:] = roll

    def act_of(policy, obs):
        with torch.no_grad():
            return policy.act_inference(obs)

    def metrics():
        p = float(env._body_pitch()[0]) * 180 / math.pi
        z = float(env.root_states[0, 2])
        g = int((env.contact_forces[0, env.feet_indices, 2] > 0.1).sum())
        c = int((~env._caster_contacts()[0]).all())     # 两从动轮都离地=1
        return p, z, g, c

    def step(mode, policy, ph, h, dflt):
        set_default(dflt); set_cmd(h=h)
        env.playback_set_mode(ids, ph, task=1, ph_t=0.0)
        obs = env.build_obs_for_mode(mode)
        a = act_of(policy, obs)
        if mode == 'TWOW':
            a = a[:, W12_ACT_PERM]
        env.step(a)

    dt = env.dt
    fails = []

    def judge(name, p, z, g, c, want):
        ok = abs(p) < 40
        if want == 'fourw':
            ok = ok and g == 2 and c == 0
        elif want == 'twow':
            ok = ok and g == 2 and c == 1 and z > 0.24
        print(f"  [{ 'PASS' if ok else 'FAIL' }] {name:22s} pitch={p:+5.1f}° z={z:.3f} "
              f"驱动轮触地={g}/2 从动轮离地={c}")
        if not ok:
            fails.append(name)

    print("=== 四策略实跑（wbr_uni，一套 env 承载四网络）===")
    env.playback_reset()

    # ① 四轮：F3e 站 2s
    for _ in range(int(2.0 / dt)):
        step('FOURW', p_4w, SP_HOLD4W, 0.19, dflt_4w)
    judge("① 四轮(F3e)", *metrics(), want='fourw')

    # ② 起立：P7S 按 run_switch 时序 HOLD4W→RISE→HANDOVER→HOLD2W
    t = 0.0
    while t < sw.t_align + sw.t_rise + sw.t_handover + 0.5:
        ph = SP_HOLD4W if t < sw.t_align else (
             SP_RISE if t < sw.t_align + sw.t_rise else (
             SP_HANDOVER if t < sw.t_align + sw.t_rise + sw.t_handover else SP_HOLD2W))
        step('RISE', p_stand, ph, 0.19, dflt_4w)
        t += dt
    judge("② 起立(P7S)→两轮", *metrics(), want='twow')

    # ③ 两轮：W12 站 5s（两轮模式站高 0.28）
    for _ in range(int(5.0 / dt)):
        step('TWOW', p_w12, SP_HOLD2W, 0.28, dflt_2w)
    judge("③ 两轮(W12) 站5s", *metrics(), want='twow')

    # ④ 趴下：P7L 按 run_switch 时序 HOLD2W→LOWER→TILT→ABSORB→HOLD4W
    t = 0.0
    while t < sw.t_align + sw.t_lower + sw.t_tilt + sw.t_absorb + 0.5:
        ph = SP_HOLD2W if t < sw.t_align else (
             SP_LOWER if t < sw.t_align + sw.t_lower else (
             SP_TILT if t < sw.t_align + sw.t_lower + sw.t_tilt else (
             SP_ABSORB if t < sw.t_align + sw.t_lower + sw.t_tilt + sw.t_absorb else SP_HOLD4W)))
        step('LOWER', p_lie, ph, 0.19, dflt_4w)
        t += dt
    judge("④ 趴下(P7L)→四轮", *metrics(), want='fourw')

    # ⑤ 回到四轮：F3e 再站 1.5s
    for _ in range(int(1.5 / dt)):
        step('FOURW', p_4w, SP_HOLD4W, 0.19, dflt_4w)
    judge("⑤ 四轮(F3e)", *metrics(), want='fourw')

    print(f"\n==> {'ALL PASS：四轮/起立/趴下/两轮(W12) 全可用、可切换' if not fails else 'FAIL: ' + ', '.join(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
