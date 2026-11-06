# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : s2b_jump_run.py
# Purpose   : S2b 两轮跳跃（MuJoCo）：**回放式模板**——触发后按 golden 真值轨迹
#             （isaac_golden_two/jump10 的 ph/off_knee 序列，对齐触发拍）驱动腿偏移，
#             其余（obs/策略/PD/物理）全部实时闭环。跳跃相位机（flat.py _update_jump_phase
#             ~1080 行）的实时移植登记续作；本脚本先验证物理侧对模板的响应与 Isaac 一致。
#             用法：python -m legged_gym.scripts.sim2sim_mujoco.eval.s2b_jump_run
# ==============================================================================

import json
import os

import numpy as np
import torch

from legged_gym.scripts.sim2sim_mujoco.deploy.com_model import com_body_offset
from legged_gym.scripts.sim2sim_mujoco.sim_loop import PKG, SimLoop, load_contract, resolve_model

MJCF = os.path.join(PKG, "assets/mjcf/wbr_v1.xml")
GOLDEN = os.path.join(PKG, "eval/golden/isaac_golden_two.npz")
OUT = os.path.join(PKG, "../../..", "eval_runs/2026-11-05_S2b_跳跃起立趴下")
DUR = 12.0


def main():
    contract = load_contract(os.path.join(PKG, "assets/deploy_contract_two.json"))
    ckpt = resolve_model(contract)
    gold = np.load(GOLDEN, allow_pickle=True)

    ph_g = gold["jump10/ph"][:, 0].astype(int)
    off_g = gold["jump10/off_knee"][:, 0].astype(float)
    offh_g = gold["jump10/off_hip"][:, 0].astype(float)
    air_z_g = gold["jump10/air_z"][:, 0].astype(float)
    goal_g = gold["jump10/h_target"][:, 0].astype(float)
    jcmd_g = gold["jump10/commands"][:, 4].astype(float)
    trig = int(np.argmax(ph_g == 1))          # 触发拍（0-based）
    end = int(np.max(np.nonzero(ph_g > 0)))   # 事件结束拍
    print(f"golden jump10: trig {trig} end {end} | ph 分布 {dict((k, int((ph_g==k).sum())) for k in sorted(set(ph_g)))}")

    loop = SimLoop(MJCF, contract, ckpt=ckpt, mode="policy")
    loop.slew = False
    loop.set_cmd("lin_vel_x", 0.0)
    loop.set_cmd("base_height", 0.28)
    loop.snap_cmds()
    loop.warmup(2)

    dt = float(contract["control_dt"])
    n = int(DUR / dt)
    seq_off, seq_ph, seq_prog0 = [], [], []
    mj_trig = None
    apex_z = 0.0
    air_frames = 0
    for i in range(n):
        t = i * dt
        if mj_trig is None and i >= trig:
            # 触发：与 golden 同拍装填（站立 settled 后）。
            #   ⚠ 2026-11-06 修复：原版这里有两个连续相同的 if，第一个把 mj_trig 置位后
            #   第二个（真正调用 jump_request 的）永假 ⇒ 装填从未执行、MJ 从未起跳
            #   （旧 jump_replay_summary.json 的 jump_height 0.0006 即此 bug 的产物）。
            mj_trig = i
            print(f"[trig] t={t:.2f}s z={loop.sim.d.qpos[2]:.4f}")
            loop.jump_request(0.10)          # 装填一跳（相位机闭环接管）
        loop.step(record=True)
        z_now = float(loop.sim.d.qpos[2])
        apex_z = max(apex_z, z_now)
        if loop.rows and loop.rows[-1]["caster_l"] + loop.rows[-1]["caster_r"] == 0:
            air_frames += 0                  # 两轮无 caster 列语义；离地用 z 或轮接触另算
        if loop.terminated:
            print("⛔ 终止:", loop.terminated)
            break

    rows = loop.rows
    z_arr = np.array([r["z"] for r in rows])
    # 事件窗指标
    if mj_trig is not None:
        t0 = mj_trig * dt
        win = [(r_t * dt - t0, r) for r_t, r in zip(range(len(rows)), rows) if r["t"] >= 0 and r_t >= mj_trig]
        win_z = np.array([r["z"] for _, r in win])
        summary = {
            "mj": {
                "z_stand": round(float(z_arr[max(0, mj_trig - 10):mj_trig].mean()), 4),
                "z_min_crouch": round(float(win_z.min()), 4),
                "z_apex": round(float(win_z.max()), 4),
                "jump_height": round(float(win_z.max() - win_z.min()), 4),
                "term": loop.terminated,
            },
            "isaac": {
                "z_stand": round(float(gold["jump10/root"][trig - 10:trig, 2].mean()), 4),
                "z_min_crouch": round(float(gold["jump10/root"][trig:, 2].min()), 4),
                "z_apex": round(float(gold["jump10/root"][trig:, 2].max()), 4),
                "jump_height": round(float(gold["jump10/root"][trig:, 2].max()
                                           - gold["jump10/root"][trig:, 2].min()), 4),
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=1))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "jump_replay_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary if mj_trig is not None else {"error": "no trig"}, f, ensure_ascii=False, indent=1)
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
