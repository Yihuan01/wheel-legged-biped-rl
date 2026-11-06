# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : s4_sensitivity.py
# Purpose   : S4 敏感度扫描（MuJoCo）——四轮行走 / 两轮行走两个基准槽位上做单变量扰动：
#             地面摩擦、轮摩擦、质量、腿 PD、观测/动作延迟、base_lin_vel 口径、质心偏移。
#             判据（相对基准）：|Δvx| ≤ 10% ∧ |Δz| ≤ 5 mm ∧ 无新增终止 ⇒ 不敏感；
#             超阈 ⇒ 敏感，登记并给出裕度建议（不改策略，处置回训练侧）。
# Usage     : python -m legged_gym.scripts.sim2sim_mujoco.eval.s4_sensitivity
# ==============================================================================

import json
import os

import numpy as np

from legged_gym.scripts.sim2sim_mujoco.sim_loop import PKG, SimLoop, load_contract, resolve_model

MJ = os.path.join(PKG, "assets/mjcf")
OUT = os.path.join(PKG, "../../..", "eval_runs/2026-11-05_S4_敏感度")
DUR = 8.0

SLOTS = {
    "four": dict(contract="deploy_contract.json", mjcf="wbr_v2.xml", h=0.19, vx=0.3),
    "two": dict(contract="deploy_contract_two.json", mjcf="wbr_v1.xml", h=0.275, vx=0.3),
}


def _scale_mass(m, factor, skip_world=True):
    m.body_mass[:] *= factor
    m.body_inertia[:] *= factor
    if skip_world:
        m.body_mass[0] = m.body_mass[0]          # world 质量本来就是 0，乘了也无影响


def _scale_friction(m, factor, wheels_only=False, wheel_bodies=()):
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:                # floor（world 上的 geom）
            if not wheels_only:
                m.geom_friction[g, 0] *= factor
        elif wheels_only and m.geom_bodyid[g] in wheel_bodies:
            m.geom_friction[g, 0] *= factor
        elif not wheels_only:
            m.geom_friction[g, 0] *= factor


def run(slot, perturb=None):
    cfg = SLOTS[slot]
    c = load_contract(os.path.join(PKG, "assets", cfg["contract"]))
    loop = SimLoop(os.path.join(MJ, cfg["mjcf"]), c, ckpt=resolve_model(c), mode="policy")
    loop.slew = False
    loop.set_cmd("lin_vel_x", cfg["vx"])
    loop.set_cmd("base_height", cfg["h"])
    loop.snap_cmds()
    p = dict(perturb or {})

    import mujoco
    m = loop.sim.m
    wheel_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                    for n in ("left_wheel_link", "right_wheel_link")}
    if "floor_mu" in p:
        _scale_friction(m, p["floor_mu"], wheels_only=False)
    if "wheel_mu" in p:
        _scale_friction(m, p["wheel_mu"], wheels_only=True, wheel_bodies=wheel_bodies)
    if "mass" in p:
        _scale_mass(m, p["mass"])
    if "kp" in p:
        loop.ctl.kp = loop.ctl.kp0 * p["kp"]
        loop.ctl.kd = loop.ctl.kd0 * p.get("kd", 1.0)
    if "delay_obs" in p:
        loop.delay_obs = int(p["delay_obs"])
    if "delay_act" in p:
        loop.delay_act = int(p["delay_act"])
    if "lin_vel" in p:
        loop.lin_vel_mode = p["lin_vel"]
    if "com" in p:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        m.body_ipos[bid] = m.body_ipos[bid] + np.array(p["com"], dtype=np.float64)

    n = int(DUR / float(c["control_dt"]))
    for _ in range(n):
        loop.step()
        if loop.terminated:
            break
    rows = loop.rows
    tail = rows[-200:] if len(rows) >= 200 else rows
    res = {
        "frames": len(rows),
        "term": loop.terminated,
        "vx": round(float(np.mean([r["vx"] for r in tail])), 4) if tail else None,
        "z": round(float(np.mean([r["z"] for r in tail])), 4) if tail else None,
        "pitch_max": round(float(max(abs(r["pitch"]) for r in rows)), 3) if rows else None,
        "roll_max": round(float(max(abs(r["roll"]) for r in rows)), 3) if rows else None,
    }
    return res


CASES = [
    ("baseline", {}),
    ("floor_mu_x0.6", {"floor_mu": 0.6}),
    ("floor_mu_x1.4", {"floor_mu": 1.4}),
    ("wheel_mu_x0.8", {"wheel_mu": 0.8}),
    ("wheel_mu_x1.25", {"wheel_mu": 1.25}),
    ("mass_+10%", {"mass": 1.10}),
    ("mass_+20%", {"mass": 1.20}),
    ("kp_x0.85", {"kp": 0.85}),
    ("kp_x1.15", {"kp": 1.15}),
    ("kd_x1.2", {"kp": 1.0, "kd": 1.2}),
    ("obs_delay_1", {"delay_obs": 1}),
    ("obs_delay_2", {"delay_obs": 2}),
    ("act_delay_1", {"delay_act": 1}),
    ("act_delay_2", {"delay_act": 2}),
    ("lin_vel_noise", {"lin_vel": "noise"}),
    ("lin_vel_zero", {"lin_vel": "zero"}),
    ("com_x+1cm", {"com": [0.01, 0.0, 0.0]}),
    ("com_x-1cm", {"com": [-0.01, 0.0, 0.0]}),
    ("com_z+1cm", {"com": [0.0, 0.0, 0.01]}),
]


def judge(base, r):
    if r["term"]:
        return "❌ 终止"
    dv = abs((r["vx"] or 0) - (base["vx"] or 0)) / max(abs(base["vx"] or 1e-6), 1e-6)
    dz = abs((r["z"] or 0) - (base["z"] or 0))
    if dv <= 0.10 and dz <= 0.005:
        return "✅"
    return f"⚠ vx {dv * 100:.1f}% / z {dz * 1000:.1f} mm"


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for slot in SLOTS:
        base = run(slot, {})
        rows = []
        print(f"\n== {slot} baseline vx={base['vx']} z={base['z']} term={base['term']}")
        for name, p in CASES:
            r = base if name == "baseline" else run(slot, p)
            j = "—" if name == "baseline" else judge(base, r)
            print(f"  {name:16s} vx={r['vx']} z={r['z']} term={str(r['term']):22s} {j}", flush=True)
            rows.append({"case": name, "perturb": p, **r, "judge": j})
        out[slot] = {"baseline": base, "rows": rows}
    with open(os.path.join(OUT, "s4_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[saved]", os.path.join(OUT, "s4_results.json"))


if __name__ == "__main__":
    main()
