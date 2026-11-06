# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : s1b_sweep.py
# Purpose   : S1b 速度 gap 归因：单变量扫描 MuJoCo 接触参数（轮 μ / 从动轮 μ / solref /
#             capsule↔cylinder），与 Isaac golden 参考值对照，判据「vx 差 ≤10% 或证明是
#             策略敏感」。扫参只改运行时 MJCF 副本（assets/mjcf/wbr_v2.xml 冻结不动）。
#             用法：python -m legged_gym.scripts.sim2sim_mujoco.eval.s1b_sweep [--quick]
# ==============================================================================

import argparse
import copy
import json
import os
import time
import xml.etree.ElementTree as ET

import numpy as np

from legged_gym.scripts.sim2sim_mujoco.sim_loop import PKG, SimLoop, load_contract, resolve_model

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MJCF = os.path.join(PKG, "assets/mjcf/wbr_v2.xml")
# Isaac 参考值（eval/golden/isaac_golden.npz 稳态 |v|，后 2 s）——由 dump_isaac_golden CPU 管线采集
ISAAC_REF = {"walk03": 0.3039, "walk06": 0.6057}
SCENARIOS = [("walk03", 0.3), ("walk06", 0.6)]

DUR = 12.0          # s，仿真时长（前 2 s 沉降，稳态窗取后 2 s）
STEADY = 2.0        # s，稳态窗


# ------------------------------------------------------------------ MJCF 变体
def _wheel_geoms(root):
    """两个轮 body 下的碰撞 geom（group=3 cylinder）。"""
    out = []
    for body in root.iter("body"):
        if body.get("name", "") in ("left_wheel_link", "right_wheel_link"):
            for g in body.findall("geom"):
                if g.get("type") == "cylinder":
                    out.append(g)
    return out


def _caster_geoms(root):
    out = []
    for body in root.iter("body"):
        if body.get("name", "").endswith("knee_caster"):
            for g in body.findall("geom"):
                if g.get("group") == "3":
                    out.append(g)
    return out


def make_variant(name):
    """返回 (mjcf_path, note)。在 /tmp/s1b/ 下生成改参 XML 副本；None = 原版。"""
    if name == "baseline":
        return BASE_MJCF, "原版（冻结件）"
    tree = ET.parse(BASE_MJCF)
    root = tree.getroot()
    note = name
    wheels = _wheel_geoms(root)
    casters = _caster_geoms(root)
    assert len(wheels) == 2 and len(casters) == 2, (len(wheels), len(casters))

    if name.startswith("wheelmu"):
        mu = float(name.split("_")[1])
        for g in wheels:
            g.set("friction", f"{mu} 0.005 0.0001")
        note = f"轮 μ={mu}（默认 1.0）"
    elif name.startswith("castmu"):
        mu = float(name.split("_")[1])
        for g in casters:
            g.set("friction", f"{mu} 0.005 0.0001")
        note = f"从动轮 μ={mu}（默认 0.02）"
    elif name.startswith("solref"):
        tc = name.split("_")[1]
        for g in wheels + casters + [root.find(".//geom[@name='floor']")]:
            g.set("solref", f"{tc} 1")
        note = f"solref timeconst={tc}s（默认 0.02，floor+轮+从动轮同设规避组合规则）"
    elif name == "capsule":
        for g in wheels:
            g.set("type", "capsule")     # size 语义相同：radius 半长（Isaac replace_cylinder_with_capsule=True 侧）
        note = "轮 cylinder→capsule（对齐 Isaac replace_cylinder_with_capsule=True）"
    else:
        raise ValueError(name)

    out_dir = "/tmp/s1b"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.xml")
    tree.write(path, encoding="utf-8")
    return path, note


# ------------------------------------------------------------------ 单次运行
def run_one(sim, contract, ckpt, vx_cmd, h_cmd):
    loop = SimLoop(sim, contract, ckpt=ckpt, mode="policy")
    loop.slew = False
    loop.set_cmd("lin_vel_x", vx_cmd)
    loop.set_cmd("base_height", h_cmd)
    loop.snap_cmds()
    loop.warmup(2)
    n = int(round(DUR / float(contract["control_dt"])))
    for _ in range(n):
        loop.step(record=True)
        if loop.terminated:
            break
    rows = loop.rows
    tail = rows[-int(STEADY / float(contract["control_dt"])):] if len(rows) > 50 else rows
    vx = float(np.mean([r["vx"] for r in tail])) if tail else float("nan")
    wz = float(np.mean([abs(r["wz"]) for r in tail])) if tail else float("nan")
    z = float(np.mean([r["z"] for r in tail])) if tail else float("nan")
    pitch = float(np.degrees(np.mean([r["pitch"] for r in tail]))) if tail else float("nan")
    caster = float(np.mean([min(r["caster_l"], 1.0) for r in tail])) if tail else float("nan")
    wl = [r["wheel_l"] for r in tail] if tail else [0.0]
    wr = [r["wheel_r"] for r in tail] if tail else [0.0]
    wspeed = float(np.mean([abs(a) + abs(b) for a, b in zip(wl, wr)])) / 2.0 if tail else float("nan")
    eff_r = vx / wspeed if wspeed and wspeed > 1e-6 and np.isfinite(vx) else float("nan")
    return {"vx": vx, "wz_abs": wz, "z": z, "pitch_deg": pitch, "caster_rate": caster,
            "wheel_speed": wspeed, "eff_radius": eff_r, "terminated": loop.terminated,
            "steps": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="减半时长（冒烟）")
    a = ap.parse_args()
    global DUR
    if a.quick:
        DUR = 6.0

    contract = load_contract()
    ckpt = resolve_model(contract)
    variants = ["baseline", "wheelmu_0.6", "wheelmu_1.5", "wheelmu_2.5",
                "castmu_0.005", "castmu_0.08",
                "solref_0.005", "solref_0.01", "solref_0.04", "capsule"]

    out_dir = os.path.join(PKG, "../../..", "eval_runs/2026-11-05_S1b_速度gap归因")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for vname in variants:
        mjcf, note = make_variant(vname)
        for sc, vcmd in SCENARIOS:
            t0 = time.time()
            m = run_one(mjcf, contract, ckpt, vcmd, 0.21)
            isaac = ISAAC_REF[sc]
            dv = m["vx"] - isaac
            dvp = dv / isaac * 100.0
            rec = {"variant": vname, "note": note, "scenario": sc, "cmd_vx": vcmd,
                   "isaac_vx": isaac, "mj_vx": round(m["vx"], 4), "dv": round(dv, 4),
                   "dv_pct": round(dvp, 1), "pass10": bool(abs(dvp) <= 10.0),
                   "z": round(m["z"], 4), "pitch_deg": round(m["pitch_deg"], 2),
                   "caster_rate": round(m["caster_rate"], 2),
                   "wheel_speed": round(m["wheel_speed"], 1),
                   "eff_radius": round(m["eff_radius"], 4) if np.isfinite(m["eff_radius"]) else None,
                   "terminated": m["terminated"], "wall_s": round(time.time() - t0, 1)}
            results.append(rec)
            print(f"[{vname:>14} {sc}] vx {rec['mj_vx']:.3f} vs isaac {isaac:.3f} "
                  f"({rec['dv_pct']:+.1f}%) {'PASS' if rec['pass10'] else 'fail'} "
                  f"caster {rec['caster_rate']:.2f} z {rec['z']:.4f} term {m['terminated']}", flush=True)

    with open(os.path.join(out_dir, "s1b_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    # TSV 便于看
    keys = ["variant", "scenario", "cmd_vx", "isaac_vx", "mj_vx", "dv", "dv_pct", "pass10",
            "z", "pitch_deg", "caster_rate", "wheel_speed", "eff_radius", "terminated", "wall_s"]
    with open(os.path.join(out_dir, "s1b_results.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for r in results:
            f.write("\t".join(str(r.get(k)) for k in keys) + "\n")
    print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    main()
