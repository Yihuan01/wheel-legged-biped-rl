#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7T 相位脚本起立的全链集成测试：四轮(F3e) → 脚本起立(P7T) → 两轮(R81) → 趴下(P7L) → 四轮。

证明 env 内 RiseScript 路径可完成"起立→交两轮"，且与 R81/P7L 串联无碍（headless，无控制台）。
用法：python -m legged_gym.envs.wbr.uni.demo_chain（退出码 0 = PASS）
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
    'r81': "releases/P6R81_goalv2/model_4500.pt",
    'lie': "releases/P7L_liedown/model_300.pt",
}
W12_ACT_PERM = [0, 1, 3, 4, 2, 5]


def main():
    sys.argv = ["chain", "--task", "wbr_uni", "--headless", "--num_envs=1",
                "--sim_device", "cuda:0", "--pipeline", "gpu"]
    args = get_args()
    env, _ = task_registry.make_env(name=args.task, args=args)
    dev, n = env.device, env.num_envs
    ids = torch.arange(n, dtype=torch.long, device=dev)
    sw = env.cfg.rewards.switch
    dt = env.dt

    p_4w = load_actor_critic(CKPT['fourw'], 164, dev)
    p_r81 = load_actor_critic(CKPT['r81'], 161, dev)
    p_lie = load_actor_critic(CKPT['lie'], 173, dev)

    dflt_4w = env.default_dof_pos.clone()
    dflt_2w = dflt_4w.clone()
    dflt_2w[:, env.leg_dof] = torch.tensor([1.0, -2.1, -1.0, 2.1], device=dev)

    def set_cmd(h=0.19, vx=0.0, yaw=0.0):
        env.command_targets[:, 0] = vx; env.command_targets[:, 1] = 0.0
        env.command_targets[:, 2] = yaw; env.command_raw_yaw[:] = yaw
        env.command_targets[:, 3] = 0.0; env.command_targets[:, 5] = h
        env.base_roll_targets[:] = 0.0

    def metrics():
        p = float(env._body_pitch()[0]) * 180 / math.pi
        z = float(env.root_states[0, 2])
        g = int((env.contact_forces[0, env.feet_indices, 2] > 0.1).sum())
        c = int((~env._caster_contacts()[0]).all())
        return p, z, g, c

    def act_of(policy, obs):
        with torch.no_grad():
            return policy.act_inference(obs)

    fails = []

    def judge(name, p, z, g, c, want):
        ok = abs(p) < 45
        if want == 'fourw':
            ok = ok and g == 2 and c == 0
        elif want == 'twow':
            ok = ok and g == 2 and c == 1 and z > 0.23
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:20s} pitch={p:+5.1f}° z={z:.3f} 轮触={g}/2 从动离={c}")
        if not ok:
            fails.append(name)

    print("=== 全链：四轮 → 脚本起立(P7T) → 两轮(R81) → 趴下(P7L) → 四轮 ===")
    env.playback_reset()

    # ① 四轮站 1.5s
    for _ in range(int(1.5 / dt)):
        env.default_dof_pos.copy_(dflt_4w); set_cmd(h=0.19)
        env.playback_set_mode(ids, SP_HOLD4W, task=1, ph_t=0.0)
        env.step(act_of(p_4w, env.build_obs_for_mode('FOURW')))
    judge("① 四轮(F3e)", *metrics(), want='fourw')

    # ② 脚本起立：RISE 相位，RiseScript 推进到 DONE
    env.default_dof_pos.copy_(dflt_4w); set_cmd(h=0.19)
    env.playback_set_mode(ids, SP_RISE, task=1, ph_t=0.0)
    env.set_rise_p_nominal()
    env.init_rise_script(cfg={"advance": "pitch", "torque": 8.0, "boost": 2.0,
                              "z_handover_min": None})
    done = False
    t = 0.0
    while t < 5.0:
        out = env.rise_script_step()
        env.step(torch.zeros(n, 6, device=dev))
        if bool(out["done"].any()):
            done = True
            break
        t += dt
    print(f"  脚本起立 done={done} t={t:.2f}s  (交接前 pb={float(-env._body_pitch()[0])*57.3:+.1f}° z={float(env.root_states[0,2]):.3f})")
    env.rise_script_clear()
    if not done:
        fails.append("脚本起立未到 DONE")

    # ③ 交两轮：R81 站 3s（hcmd 0.22 → 0.28）
    set_cmd(h=0.22)
    env.playback_set_mode(ids, SP_HOLD2W, task=1, ph_t=0.0)
    env.default_dof_pos.copy_(dflt_2w)
    for i in range(int(3.0 / dt)):
        if i == int(1.0 / dt):
            set_cmd(h=0.28)
        env.step(act_of(p_r81, env.build_obs_for_mode('TWOW'))[:, W12_ACT_PERM])
    judge("③ 两轮(R81) 站3s", *metrics(), want='twow')

    # ④ 趴下：P7L 定时推进
    t = 0.0
    while t < sw.t_align + sw.t_lower + sw.t_tilt + sw.t_absorb + 0.5:
        ph = SP_HOLD2W if t < sw.t_align else (
             SP_LOWER if t < sw.t_align + sw.t_lower else (
             SP_TILT if t < sw.t_align + sw.t_lower + sw.t_tilt else (
             SP_ABSORB if t < sw.t_align + sw.t_lower + sw.t_tilt + sw.t_absorb else SP_HOLD4W)))
        env.default_dof_pos.copy_(dflt_4w); set_cmd(h=0.19)
        env.playback_set_mode(ids, ph, task=0, ph_t=t)
        env.step(act_of(p_lie, env.build_obs_for_mode('LOWER')))
        t += dt
    judge("④ 趴下(P7L)", *metrics(), want='fourw')

    print(f"\n==> {'ALL PASS：脚本起立全链可用' if not fails else 'FAIL: ' + ', '.join(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
