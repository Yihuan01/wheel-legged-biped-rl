# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : s3_p7t_lift.py
# Purpose   : S3 交棒精调——P7T 抬升段单变量扫描（vz 闭环 / 俯仰阻尼 / 恒力矩档位），
#             只跑「四轮行驶 1 s → 减速 0.4 s → P7T 五相」，输出交棒瞬间状态与抬升段包络。
#             判据（交棒瞬间）：|pitch| ≤ 15°、|pitch_rate| ≤ 1.0 rad/s、|vz| ≤ 0.3 m/s、
#             z ∈ [0.24, 0.29]（贴两轮平衡点 0.27）。
# Usage     : python -m legged_gym.scripts.sim2sim_mujoco.eval.s3_p7t_lift [--sweep]
# ==============================================================================

import argparse
import json
import os

import numpy as np

from legged_gym.scripts.sim2sim_mujoco.sim_loop import PKG, SimLoop, load_contract, resolve_model

MJCF = os.path.join(PKG, "assets/mjcf/wbr_v2.xml")
OUT = os.path.join(PKG, "../../..", "eval_runs/2026-11-05_S3_五槽uni整链")


def run_one(params, pre_s=1.0, brake_s=0.4, lift_max_s=4.0, verbose=False):
    c4 = load_contract(os.path.join(PKG, "assets/deploy_contract.json"))
    ck = resolve_model(c4)
    loop = SimLoop(MJCF, c4, ckpt=ck, mode="policy")
    loop.slew = False
    loop.set_cmd("lin_vel_x", 0.3)
    loop.set_cmd("base_height", 0.21)
    loop.snap_cmds()

    dt = float(c4["control_dt"])
    rows = []

    def step(n, tag):
        for _ in range(int(n)):
            loop.step(record=True)
            d = loop.sim.d
            R, q, lin, ang = loop.sim.base_state()
            pg = loop.sim.projected_gravity(R)
            pitch, roll = loop.sim.body_pitch_roll(pg)
            rows.append(dict(t=round(len(rows) * dt, 3), tag=tag, z=float(d.qpos[2]),
                             vz=float(d.qvel[2]), pitch=float(pitch),
                             pitch_rate=-float(ang[1]), roll=float(roll),
                             c=float(loop.p7t.p["torque"]) if tag == "LIFT" else 0.0))
            if loop.terminated:
                return False
        return True

    step(pre_s / dt, "DRIVE")
    loop.set_cmd("lin_vel_x", 0.0)
    step(brake_s / dt, "BRAKE")
    z_pre = float(loop.sim.d.qpos[2])
    loop.activate_p7t(params=params)
    t_lift0 = len(rows) * dt
    n = int(lift_max_s / dt)
    hand = None
    for _ in range(n):
        loop.step(record=True)
        d = loop.sim.d
        R, q, lin, ang = loop.sim.base_state()
        pg = loop.sim.projected_gravity(R)
        pitch, roll = loop.sim.body_pitch_roll(pg)
        rows.append(dict(t=round(len(rows) * dt, 3), tag="LIFT", z=float(d.qpos[2]),
                         vz=float(d.qvel[2]), pitch=float(pitch), pitch_rate=-float(ang[1]),
                         roll=float(roll), c=0.0))
        if loop.p7t_done and hand is None:
            hand = dict(t=round(len(rows) * dt, 3), z=float(d.qpos[2]),
                        vz=float(d.qvel[2]), pitch_deg=float(np.degrees(pitch)),
                        pitch_rate=-float(ang[1]), roll_deg=float(np.degrees(roll)))
            break
        if loop.terminated:
            hand = dict(t=round(len(rows) * dt, 3), z=float(d.qpos[2]),
                        vz=float(d.qvel[2]), pitch_deg=float(np.degrees(pitch)),
                        pitch_rate=-float(ang[1]), roll_deg=float(np.degrees(roll)),
                        term=loop.terminated)
            break

    lift = [r for r in rows if r["tag"] == "LIFT"]
    env = dict(
        z_max=max(r["z"] for r in lift),
        pitch_max_deg=float(np.degrees(max(abs(r["pitch"]) for r in lift))),
        rate_max=float(max(abs(r["pitch_rate"]) for r in lift)),
        vz_max=float(max(r["vz"] for r in lift)),
        vz_min=float(min(r["vz"] for r in lift)),
    )
    res = dict(params=params, z_pre=round(z_pre, 4), handover=hand, lift_span=env,
               lift_s=round(t_lift0 and (len(rows) * dt - t_lift0), 3),
               term=loop.terminated)
    res["pass"] = bool(hand and "term" not in hand
                       and abs(hand["pitch_deg"]) <= 15.0
                       and abs(hand["pitch_rate"]) <= 1.0
                       and abs(hand["vz"]) <= 0.3
                       and 0.235 <= hand["z"] <= 0.295)
    if verbose:
        print(json.dumps(res, ensure_ascii=False))
    return res, rows


GATE = dict(pitch_tol=0.15, rate_tol=1.0, z_handover=0.24, torque=4.0)
SWEEP = [
    ("w2_02", dict(GATE, kp_pitch=2.0, kd_pitch=0.2)),
    ("w4_05", dict(GATE, kp_pitch=4.0, kd_pitch=0.5)),
    ("w6_10", dict(GATE, kp_pitch=6.0, kd_pitch=1.0)),
    ("w8_10", dict(GATE, kp_pitch=8.0, kd_pitch=1.0)),
    ("w8_20", dict(GATE, kp_pitch=8.0, kd_pitch=2.0)),
    ("w6_10_c2", dict(GATE, kp_pitch=6.0, kd_pitch=1.0, torque=2.0)),
    ("w4_05_c2", dict(GATE, kp_pitch=4.0, kd_pitch=0.5, torque=2.0)),
    ("w6_10_r20", dict(GATE, kp_pitch=6.0, kd_pitch=1.0, c_rate=20.0)),
    ("w6_10_r10", dict(GATE, kp_pitch=6.0, kd_pitch=1.0, c_rate=10.0)),
    ("w10_20_r10", dict(GATE, kp_pitch=10.0, kd_pitch=2.0, c_rate=10.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--param", default="")     # "k=v,k=v"
    ap.add_argument("--out", default="p7t_lift_sweep.json")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.param:
        params = {}
        for kv in a.param.split(","):
            k, v = kv.split("=")
            params[k] = float(v)
        res, rows = run_one(params, verbose=True)
        with open(os.path.join(OUT, a.out), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        return

    out = []
    for name, p in SWEEP:
        res, _ = run_one(p)
        res["name"] = name
        print(f"{name:10s} pass={res['pass']!s:5s} "
              f"t={res['handover']['t'] if res['handover'] else None} "
              f"z={res['handover']['z'] if res['handover'] else None} "
              f"pitch={res['handover']['pitch_deg'] if res['handover'] else None} "
              f"rate={res['handover']['pitch_rate'] if res['handover'] else None} "
              f"vz={res['handover']['vz'] if res['handover'] else None} "
              f"zmax={res['lift_span']['z_max']} term={res['term']}", flush=True)
        out.append(res)
    with open(os.path.join(OUT, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[saved]", os.path.join(OUT, a.out))


if __name__ == "__main__":
    main()
