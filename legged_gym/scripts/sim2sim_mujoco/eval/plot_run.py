# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : plot_run.py
# Purpose   : 把一次 MuJoCo 运行的遥测画成四联图（速度/高度/姿态/触地），可选叠加 Isaac golden
#             同场景曲线，用于肉眼比对；输出 PNG（Agg 后端，不需要显示器）。
# ==============================================================================

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(head, ln.rstrip("\n").split("\t"))) for ln in f if ln.strip()]
    return {k: np.array([float(r[k]) for r in rows]) for k in head}


def read_isaac(npz, sc):
    root = npz[f"{sc}/root"]
    out = {"t": np.arange(len(root)) * 0.01, "z": root[:, 2]}
    vx, pitch, roll, wz = [], [], [], []
    for r in root:
        R = R_of(r[[6, 3, 4, 5]])
        vx.append((R.T @ r[7:10])[0])
        wz.append((R.T @ r[10:13])[2])
        pitch.append(np.arcsin(np.clip(R[2, 0], -1, 1)))
        roll.append(np.arcsin(np.clip(R[2, 1], -1, 1)))
    out["vx"] = np.array(vx)
    out["wz"] = np.array(wz)
    out["pitch"] = np.degrees(pitch)
    out["roll"] = np.degrees(roll)
    out["caster"] = np.min(npz[f"{sc}/caster_contacts"], axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="MuJoCo run 目录（含 telemetry.tsv）")
    ap.add_argument("--latest", action="store_true", help="不传 --run：取 eval/runs 下最新一份带遥测的 run（控制台导出图用）")
    ap.add_argument("--golden", default=os.path.join(PKG, "eval/golden/isaac_golden.npz"))
    ap.add_argument("--scenario", default=None, help="Isaac golden 场景名，给了就叠加对照")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not a.run:
        runs_dir = os.path.join(PKG, "eval/runs")
        cands = sorted((d for d in os.listdir(runs_dir)
                        if os.path.isfile(os.path.join(runs_dir, d, "telemetry.tsv"))),
                       key=lambda d: os.path.getmtime(os.path.join(runs_dir, d, "telemetry.tsv")))
        if not cands:
            raise SystemExit("eval/runs 下没有带 telemetry.tsv 的 run（先跑一次并开录制）")
        a.run = os.path.join(runs_dir, cands[-1])

    mj = read_tsv(os.path.join(a.run, "telemetry.tsv"))
    isa = read_isaac(np.load(a.golden, allow_pickle=True), a.scenario) if a.scenario else None
    t = mj["t"]

    fig, ax = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True,
                           gridspec_kw={"height_ratios": [1.1, 1, 1]})
    style_mj = dict(lw=1.6, label="MuJoCo")
    style_isa = dict(lw=1.2, ls="--", alpha=0.85, label="Isaac")

    ax[0].plot(t, mj["vx"], color="#1f77b4", **style_mj)
    ax[0].axhline(float(mj["cmd_vx"][0]), color="#444", lw=0.9, ls=":", label="cmd")
    if isa is not None:
        ax[0].plot(isa["t"], isa["vx"], color="#ff7f0e", **style_isa)
    ax[0].set_ylabel("vx (m/s)")
    ax[0].legend(loc="lower right", ncol=3)
    ax[0].grid(alpha=0.25)

    ax[1].plot(t, mj["z"], color="#1f77b4", **style_mj)
    ax[1].axhline(float(mj["cmd_h"][0]), color="#444", lw=0.9, ls=":", label="cmd h")
    if isa is not None:
        ax[1].plot(isa["t"], isa["z"], color="#ff7f0e", **style_isa)
    ax[1].set_ylabel("base z (m)")
    ax[1].legend(loc="lower right")
    ax[1].grid(alpha=0.25)

    ax[2].plot(t, np.degrees(mj["pitch"]), color="#2ca02c", lw=1.5, label="pitch (MJ)")
    ax[2].plot(t, np.degrees(mj["roll"]), color="#9467bd", lw=1.0, label="roll (MJ)")
    if isa is not None:
        ax[2].plot(isa["t"], isa["pitch"], color="#2ca02c", lw=1.0, ls="--", alpha=0.8,
                   label="pitch (Isaac)")
    ax[2].set_ylabel("deg")
    ax[2].set_xlabel("t (s)")
    ax[2].legend(loc="lower right", ncol=3)
    ax[2].grid(alpha=0.25)

    title = a.title or os.path.basename(a.run.rstrip("/"))
    fig.suptitle(f"{title}   caster contact: L {mj['caster_l'].mean()*100:.0f}% / "
                 f"R {mj['caster_r'].mean()*100:.0f}%", fontsize=11)
    fig.tight_layout()
    out = a.out or os.path.join(a.run, "plot.png")
    fig.savefig(out, dpi=130)
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
