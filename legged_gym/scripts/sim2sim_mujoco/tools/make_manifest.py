# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : make_manifest.py
# Purpose   : S5 冻结交棒——把部署包的全部输入钉成一份清单（sha256）：
#             两份 MJCF、四份 deploy_contract、四份交付权重、运行核代码、对账读数。
#             真机/换机交接时只要清单里每条 sha 都对得上，就是同一套部署语义。
# Usage     : python -m legged_gym.scripts.sim2sim_mujoco.tools.make_manifest
# ==============================================================================

import hashlib
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(PKG, "../../.."))

CONTRACTS = ("deploy_contract.json", "deploy_contract_two.json",
             "deploy_contract_two_v2.json", "deploy_contract_lie.json")
MJCFS = ("wbr_v2.xml", "wbr_v1.xml")
WEIGHTS = (
    ("四轮行驶 164", "releases/P7F3e_yaw1_dr15", "model_11800.pt"),
    ("两轮行走 161", "releases/P8W7_push01_fric05", "model_2800.pt"),
    ("两轮跳跃 161", "releases/P8J3d_wheelfollow_lite", "model_5650.pt"),
    ("趴下 173", "releases/P7L_liedown", "model_300.pt"),
)
CODE = ("deploy/obs_builder.py", "deploy/control.py", "deploy/policy.py", "deploy/com_model.py",
        "deploy/jump_template.py", "deploy/p7t_standup.py", "deploy/p7t2_platform_standup.py",
        "deploy/uni_chain.py",
        "sim_loop.py", "serve.py", "mj_bridge.py", "run_policy.py")
GOLDEN = ("parity.json", "parity_two.json", "isaac_static.json", "isaac_static_two.json")
TERRAIN = ("mjcf/wbr_v2_demo.xml", "terrain/wbr_v2_demo.png", "terrain/wbr_v2_demo.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(label, rel):
    p = os.path.join(ROOT, rel)
    ok = os.path.isfile(p)
    return {"label": label, "path": rel, "exists": ok, "sha256": sha256(p)[:16] if ok else None}


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    man = {
        "name": "WBR Sim2Sim 部署包（MuJoCo）",
        "version": "SIM2SIM-V1.2",
        "created": "2026-11-05",
        "updated": "2026-11-06（P7T2 剖面回灌启用＋防泄漏修复；p7t2 平台式起立入运行核）",
        "git": git_sha(),
        "engine": "MuJoCo（CPU）；Isaac Gym 只在采集 golden 时用 CPU 管线",
        "contracts": [entry(os.path.basename(c), f"legged_gym/scripts/sim2sim_mujoco/assets/{c}")
                      for c in CONTRACTS],
        "mjcf": [entry(n, f"legged_gym/scripts/sim2sim_mujoco/assets/mjcf/{n}") for n in MJCFS],
        "weights": [entry(f"{lab} · {os.path.basename(d)}/{m}", f"{d}/{m}")
                    for lab, d, m in WEIGHTS],
        "code": [entry(c, f"legged_gym/scripts/sim2sim_mujoco/{c}") for c in CODE],
        "parity": [entry(g, f"legged_gym/scripts/sim2sim_mujoco/eval/golden/{g}") for g in GOLDEN],
        "terrain": [entry(os.path.basename(t), f"legged_gym/scripts/sim2sim_mujoco/assets/{t}") for t in TERRAIN],
        "notes": [
            "dof 顺序一律按名字取（Isaac 与 MuJoCo 不同）",
            "动作/观测 6 维顺序＝策略训练侧顺序（flat：[hip_l,knee_l,wheel_l,hip_r,knee_r,wheel_r]）",
            "两轮 off_knee=-0.1、off_limit=1.2 在契约内；四轮/趴下无此通道",
            "solref 0.01 已写死在 MJCF（build_mjcf.SOLREF_TC）",
            "P7T 起立在 MuJoCo 未收口：五槽链走两轮起步（uni --start 2w）",
        ],
    }
    out = os.path.join(PKG, "assets/DEPLOY_MANIFEST.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    miss = [x["path"] for k in ("contracts", "mjcf", "weights", "code", "parity", "terrain")
            for x in man[k] if not x["exists"]]
    print(f"[saved] {out}")
    print(f"  契约 {len(man['contracts'])} · MJCF {len(man['mjcf'])} · 权重 {len(man['weights'])} "
          f"· 代码 {len(man['code'])} · 对账 {len(man['parity'])} · 地形 {len(man['terrain'])}")
    if miss:
        print("  ⚠ 缺失：" + ", ".join(miss))
    return 0 if not miss else 1


if __name__ == "__main__":
    raise SystemExit(main())
