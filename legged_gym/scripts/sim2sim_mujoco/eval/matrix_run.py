# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : matrix_run.py
# Purpose   : S1 命令矩阵 MuJoCo 侧：与 isaac_golden_matrix.npz 同场景对照（vx 全幅/负向/
#             大 yaw/组合高度），外加 push 推扰能力项（仅 MJ 侧，无 Isaac 对照——注入协议不同）。
#             用法：python -m legged_gym.scripts.sim2sim_mujoco.eval.matrix_run
# ==============================================================================

import json
import os

import numpy as np

from legged_gym.scripts.sim2sim_mujoco.sim_loop import PKG, SimLoop, load_contract, resolve_model

MJCF = os.path.join(PKG, "assets/mjcf/wbr_v2.xml")
GOLDEN = os.path.join(PKG, "eval/golden/isaac_golden.npz".replace("isaac_golden.npz", "isaac_golden_matrix.npz"))
DUR = 12.0
STEADY = 2.0

SCENARIOS = [
    ("m10", -1.0, 0.0, 0.21), ("m06", -0.6, 0.0, 0.21),
    ("v09", 0.9, 0.0, 0.21), ("v10", 1.0, 0.0, 0.21),
    ("yn05", 0.3, -0.5, 0.19), ("yn10", 0.3, -1.0, 0.19), ("y10", 0.3, 1.0, 0.19),
    ("c05h245", 0.5, 0.0, 0.245), ("c05h175", 0.5, 0.0, 0.175),
]
PUSHES = [  # 仅 MJ 能力项：t=6s 一次性速度注入（世界系，等价瞬时冲量）
    ("push_vy", 0.3, 0.0, 0.21, (0.0, 0.5, 0.0)),    # 侧向 +0.5 m/s
    ("push_vx", 0.3, 0.0, 0.21, (-0.5, 0.0, 0.0)),   # 迎面减速 −0.5 m/s
]


def steady_stats(loop):
    rows = loop.rows[-int(STEADY / float(loop.c["control_dt"])):]
    return {"vx": float(np.mean([r["vx"] for r in rows])),
            "z": float(np.mean([r["z"] for r in rows])),
            "pitch_deg": float(np.degrees(np.mean([r["pitch"] for r in rows]))),
            "caster": float(np.mean([min(r["caster_l"], 1.0) for r in rows]))}


def main():
    contract = load_contract()
    ckpt = resolve_model(contract)
    gold = np.load(GOLDEN, allow_pickle=True)
    out_dir = os.path.join(PKG, "../../..", "eval_runs/2026-11-05_S1_命令矩阵")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for name, vx, yaw, h in SCENARIOS:
        loop = SimLoop(MJCF, contract, ckpt=ckpt, mode="policy")
        loop.slew = False
        loop.set_cmd("lin_vel_x", vx)
        loop.set_cmd("ang_vel_yaw", yaw)
        loop.set_cmd("base_height", h)
        loop.snap_cmds()
        loop.warmup(2)
        for _ in range(int(DUR / float(contract["control_dt"]))):
            loop.step(record=True)
            if loop.terminated:
                break
        st = steady_stats(loop)
        g = gold[f"{name}/root"]
        isaac_v = float(np.linalg.norm(g[len(g) // 2:, 7:10], axis=1).mean())
        isaac_z = float(g[len(g) // 2:, 2].mean())
        rec = {"scenario": name, "cmd": [vx, yaw, h],
               "isaac_v": round(isaac_v, 4), "mj_v": round(abs(st["vx"]), 4),
               "dv_pct": round((abs(st["vx"]) - isaac_v) / isaac_v * 100, 1) if isaac_v > 1e-3 else None,
               "isaac_z": round(isaac_z, 4), "mj_z": round(st["z"], 4),
               "dz_mm": round((st["z"] - isaac_z) * 1000, 1),
               "pitch_deg": round(st["pitch_deg"], 2), "caster": round(st["caster"], 2),
               "terminated": loop.terminated}
        results.append(rec)
        print(f"[{name:>8}] isaac {rec['isaac_v']:.3f} mj {rec['mj_v']:.3f} "
              f"({rec['dv_pct']:+.1f}%) dz {rec['dz_mm']:+.1f}mm term {loop.terminated}", flush=True)

    for name, vx, yaw, h, kick in PUSHES:
        loop = SimLoop(MJCF, contract, ckpt=ckpt, mode="policy")
        loop.slew = False
        loop.set_cmd("lin_vel_x", vx)
        loop.set_cmd("base_height", h)
        loop.snap_cmds()
        loop.warmup(2)
        n = int(DUR / float(contract["control_dt"]))
        kick_at = int(6.0 / float(contract["control_dt"]))
        rec_v = []
        for i in range(n):
            if i == kick_at:
                loop.sim.d.qvel[0:3] += np.asarray(kick)   # 世界系线速度阶跃＝瞬时冲量
            loop.step(record=True)
            if loop.terminated:
                break
            if len(loop.rows):
                rec_v.append((i * float(contract["control_dt"]), loop.rows[-1]["vx"]))
        st = steady_stats(loop)
        # 恢复时间：kick 后 vx 重新进入 cmd±10% 带并保持到结束
        t_rec = None
        band = vx * 0.9
        tail_ok = 0
        for t, v in reversed(rec_v):
            if t <= 6.0:
                break
            if abs(v - vx) <= 0.1 * vx:
                tail_ok += 1
                t_rec = t
            else:
                break
        rec = {"scenario": name, "cmd": [vx, yaw, h], "kick": list(kick),
               "vx_pre": round(next(v for t, v in rec_v if t <= 6.0 - 0.1), 3),
               "vx_1s_after": round(next((v for t, v in rec_v if t >= 7.0), float("nan")), 3),
               "recover_s": round(7.0 - t_rec, 2) if (t_rec is not None and tail_ok >= int(1.0 / float(contract["control_dt"]))) else None,
               "terminated": loop.terminated}
        results.append(rec)
        print(f"[{name:>8}] kick@6s vx_1s {rec['vx_1s_after']} 恢复 {rec['recover_s']}s term {loop.terminated}", flush=True)

    with open(os.path.join(out_dir, "matrix_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    main()
