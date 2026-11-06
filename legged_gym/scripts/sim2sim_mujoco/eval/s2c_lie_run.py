# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : s2c_lie_run.py
# Purpose   : S2c 趴下（MuJoCo）：**A 口径**（off=0，作者裁决）——两轮站姿起点，
#             外部相位脚本（console 趴下时序：HOLD2W 0.3 → LOWER 0.5 → TILT 0.3 →
#             ABSORB 0.4 → HOLD4W）驱动 switch 尾块，策略（P7L@300，173 obs）输出动作。
#             与 isaac_golden_lie.npz 同协议对照（z/pitch 时序）。
# ==============================================================================

import json
import os

import numpy as np

from legged_gym.scripts.sim2sim_mujoco.sim_loop import PKG, SimLoop, load_contract, resolve_model

MJCF = os.path.join(PKG, "assets/mjcf/wbr_v2.xml")
GOLDEN = os.path.join(PKG, "eval/golden/isaac_golden_lie.npz")
OUT = os.path.join(PKG, "../../..", "eval_runs/2026-11-05_S2c_趴下")
POSE_2W = {"left_hip_joint": 1.0, "left_knee_joint": -2.1,
           "right_hip_joint": -1.0, "right_knee_joint": 2.1}
Z_START = 0.2697 + 0.005
DUR = 3.0


def main():
    contract = load_contract(os.path.join(PKG, "assets/deploy_contract_lie.json"))
    ckpt = resolve_model(contract)
    loop = SimLoop(MJCF, contract, ckpt=ckpt, mode="policy")
    loop.slew = False
    loop.set_cmd("base_height", 0.19)
    loop.snap_cmds()

    # 起点：两轮站姿（pose_2w）＋ z2w+5mm 净空（与 golden 同）
    loop.reset()
    names = contract["dof_names"]
    for n, v in POSE_2W.items():
        loop.sim.d.qpos[loop.sim.qadr[n]] = v
    loop.sim.d.qpos[2] = Z_START
    loop.sim.d.qvel[:] = 0.0
    import mujoco
    mujoco.mj_forward(loop.sim.m, loop.sim.d)

    gold = np.load(GOLDEN, allow_pickle=True)
    g_root = gold["lie/root"]
    g_ph = gold["lie/ph"][:, 0]

    dt = float(contract["control_dt"])
    n = int(DUR / dt)
    seq = []
    for i in range(n):
        t = i * dt
        # 外部相位脚本（与 golden 同款 console 时序）
        if t < 0.30:
            ph, pt = 3, t
        elif t < 0.80:
            ph, pt = 4, t - 0.30
        elif t < 1.10:
            ph, pt = 5, t - 0.80
        elif t < 1.50:
            ph, pt = 6, t - 1.10
        else:
            ph, pt = 0, t - 1.50
        loop.set_switch_phase(ph)
        loop.step(record=True)
        seq.append((t, float(loop.sim.d.qpos[2]), int(ph)))

    rows = loop.rows
    summary = {"ph_seq_mj": [p for _, p, _ in seq][::10],
               "mj": {"z_start": round(seq[0][1], 4),
                      "z_end": round(seq[-1][1], 4),
                      "z_min": round(min(s[1] for s in seq), 4),
                      "pitch_end_deg": None,
                      "term": loop.terminated},
               "isaac": {"z_start": round(float(g_root[0, 2]), 4),
                         "z_end": round(float(g_root[-1, 2]), 4),
                         "z_min": round(float(g_root[:, 2].min()), 4)}}
    if rows:
        summary["mj"]["pitch_end_deg"] = round(float(np.degrees(rows[-1]["pitch"])), 2)

    # 逐 0.1 s 对照表
    table = []
    for i in range(0, int(DUR / 0.1)):
        t = round(i * 0.1, 2)
        gi = min(int(t / 0.01), len(g_root) - 1)
        mj_z = next((zz for tt, zz, _ in seq if abs(tt - t) < 1e-6), None)
        table.append({"t": t, "isaac_z": round(float(g_root[gi, 2]), 4),
                      "mj_z": round(mj_z, 4) if mj_z is not None else None,
                      "isaac_ph": int(g_ph[gi]), "mj_ph": next(p for tt, _, p in seq if abs(tt - t) < 1e-6)})
    summary["table_01s"] = table
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "lie_compare.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    for r in table[:16]:
        print(f"t={r['t']:.1f} isaac z {r['isaac_z']:.4f} ph{r['isaac_ph']} | mj z {r['mj_z']:.4f} ph{r['mj_ph']}")
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
