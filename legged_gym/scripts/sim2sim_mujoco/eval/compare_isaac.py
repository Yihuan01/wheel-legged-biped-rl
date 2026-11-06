# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : compare_isaac.py
# Purpose   : 同协议对照：Isaac golden（录制的 root/dof 状态）与 MuJoCo 遥测 TSV 计算同一套指标，
#             输出逐场景对照表（vx / z / pitch / roll / 偏航率），并给出偏差与判定。
# ==============================================================================

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def R_of(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def isaac_stats(npz, sc, tail=0.0):
    root = npz[f"{sc}/root"]
    if tail > 0:
        root = root[-int(round(tail / 0.01)):]
    vx, vy, wz, z, pitch, roll = [], [], [], [], [], []
    for r in root:
        # Isaac 的 root quat 是 (x,y,z,w)，本文件的 R_of 吃 (w,x,y,z) —— 必须换序
        R = R_of(r[[6, 3, 4, 5]])
        vb = R.T @ r[7:10]
        wb = R.T @ r[10:13]
        vx.append(vb[0]); vy.append(vb[1]); wz.append(wb[2]); z.append(r[2])
        pitch.append(np.arcsin(np.clip(R[2, 0], -1, 1)))
        roll.append(np.arcsin(np.clip(R[2, 1], -1, 1)))
    return dict(vx=np.array(vx), vy=np.array(vy), wz=np.array(wz), z=np.array(z),
                pitch=np.array(pitch), roll=np.array(roll))


def mj_stats(path, tail=0.0):
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(head, line.rstrip("\n").split("\t"))) for line in f if line.strip()]
    out = {}
    for k in ("vx", "vy", "wz", "z", "pitch", "roll", "tip", "caster_l", "caster_r", "tau_max"):
        if k in rows[0]:
            out[k] = np.array([float(r[k]) for r in rows])
    if tail > 0:
        n = int(round(tail / 0.01))
        out = {k: v[-n:] for k, v in out.items()}
    return out


def summarize(d):
    return {
        "vx": float(np.mean(d["vx"])),
        "z": float(np.mean(d["z"])),
        "pitch_deg": float(np.degrees(np.mean(d["pitch"]))),
        "roll_deg": float(np.degrees(np.mean(d["roll"]))),
        "|vx|_p95": float(np.percentile(np.abs(d["vx"]), 95)),
        "|wz|_p95": float(np.percentile(np.abs(d["wz"]), 95)),
        "|pitch|_p95_deg": float(np.degrees(np.percentile(np.abs(d["pitch"]), 95))),
        "caster": float(np.mean(np.minimum(d["caster_l"], d["caster_r"]))) if "caster_l" in d else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=os.path.join(PKG, "eval/golden/isaac_golden.npz"))
    ap.add_argument("--runs", nargs="+", required=True,
                    help="MuJoCo run 目录列表（含 telemetry.tsv），顺序与 --scenarios 对应")
    ap.add_argument("--scenarios", nargs="+", default=["stand", "walk03", "height023"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--tail", type=float, default=2.0, help="只比最后 N 秒（稳态）")
    a = ap.parse_args()

    npz = np.load(a.golden, allow_pickle=True)
    table = []
    ok = True
    for sc, run in zip(a.scenarios, a.runs):
        isa = summarize(isaac_stats(npz, sc, a.tail))
        mimo = summarize(mj_stats(os.path.join(run, "telemetry.tsv"), a.tail))
        dvx = mimo["vx"] - isa["vx"]
        dz = mimo["z"] - isa["z"]
        dpx = mimo["pitch_deg"] - isa["pitch_deg"]
        passed = abs(dvx) <= max(0.03, 0.1 * abs(isa["vx"])) and abs(dz) <= 0.015 and abs(dpx) <= 3.0
        ok = ok and passed
        table.append(dict(scenario=sc, isaac=isa, mujoco=mimo,
                          diff={"vx": dvx, "z": dz, "pitch_deg": dpx}, passed=passed))
        print(f"[{'PASS' if passed else 'FAIL'}] {sc}")
        print(f"    Isaac : vx={isa['vx']:+.4f} z={isa['z']:.4f} pitch={isa['pitch_deg']:+.2f}° "
              f"roll={isa['roll_deg']:+.2f}° |wz|p95={isa['|wz|_p95']:.4f}")
        print(f"    MuJoCo: vx={mimo['vx']:+.4f} z={mimo['z']:.4f} pitch={mimo['pitch_deg']:+.2f}° "
              f"roll={mimo['roll_deg']:+.2f}° |wz|p95={mimo['|wz|_p95']:.4f} "
              f"caster={mimo['caster']}")
        print(f"    Δ     : vx={dvx:+.4f} z={dz:+.4f} pitch={dpx:+.2f}°")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"table": table, "passed": ok}, f, ensure_ascii=False, indent=1)
        print(f"[{'ALL PASS' if ok else 'FAIL'}] {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
