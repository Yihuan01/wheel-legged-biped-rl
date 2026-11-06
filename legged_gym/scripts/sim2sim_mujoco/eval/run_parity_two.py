# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : run_parity_two.py
# Purpose   : S2 两轮链路对账：用 Isaac golden_two 逐帧回放，比对观测（分块）/动作/力矩。
#             与 tools/run_parity.py 同方法论；两轮差异：161 obs 无四轮尾块、v1 无从动轮
#             （golden 的 caster/tip 为占位 0，不参与判定）、场景表从 npz keys 取。
#             用法：python -m legged_gym.scripts.sim2sim_mujoco.eval.run_parity_two
# ==============================================================================

import argparse
import json
import os

import numpy as np
import torch

from legged_gym.scripts.sim2sim_mujoco.deploy.policy import Policy
from legged_gym.scripts.sim2sim_mujoco.deploy.obs_builder import ObsBuilder
from legged_gym.scripts.sim2sim_mujoco.deploy.control import PDController

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(PKG, "../../.."))

GATES = {"obs": 1e-4, "act": 1e-5, "tau": 1e-4}
BLOCKS = (("base", 0, 33), ("history", 33, 153), ("phase", 153, 161))


def quat_rot_inv(q, v):
    """Isaac 的 quat_rotate_inverse：q=(x,y,z,w)，把世界向量转到机体系。"""
    q = torch.as_tensor(q, dtype=torch.float32)
    v = torch.as_tensor(v, dtype=torch.float32)
    w = q[3]
    qv = q[:3]
    a = v * (2.0 * w * w - 1.0)
    b = torch.cross(qv, v, dim=0) * w * 2.0
    c = qv * torch.dot(qv, v) * 2.0
    return a - b + c


def replay(contract, policy, npz, sc):
    obs_b = ObsBuilder(contract)
    h0 = torch.tensor(npz[f"{sc}/hist0_pos"], dtype=torch.float32)
    v0 = torch.tensor(npz[f"{sc}/hist0_vel"], dtype=torch.float32)
    # Isaac 落盘的 history 是"推入 state_0 之后"的 ⇒ 右移一行重建第 0 帧初值（第 0 行被挤掉的未知项）
    obs_b.hist_pos = torch.cat([torch.zeros(1, 6), h0[:-1]], dim=0)
    obs_b.hist_vel = torch.cat([torch.zeros(1, 6), v0[:-1]], dim=0)
    ctl = PDController(contract)
    ctl.last_wheel = torch.tensor(npz[f"{sc}/ctrl0_last"], dtype=torch.float32)
    ctl.filt = torch.tensor(npz[f"{sc}/ctrl0_filt"], dtype=torch.float32)
    ctl.wheel_target = torch.tensor(npz[f"{sc}/ctrl0_target"], dtype=torch.float32)

    root = npz[f"{sc}/root"]
    dof_pos = npz[f"{sc}/dof_pos"]
    dof_vel = npz[f"{sc}/dof_vel"]
    cmds = npz[f"{sc}/commands"]
    last_act = npz[f"{sc}/last_action"]
    obs_ref = npz[f"{sc}/obs"]
    act_ref = npz[f"{sc}/act"]
    tau_ref = npz[f"{sc}/tau"]
    call_pos = npz[f"{sc}/call_dof_pos"]
    call_vel = npz[f"{sc}/call_vel"] if f"{sc}/call_vel" in npz else npz[f"{sc}/call_dof_vel"]
    ph = npz[f"{sc}/ph"][:, 0]
    assert int(np.abs(ph).max()) == 0, "两轮行走 golden 的相位应恒为 0"

    n = root.shape[0]
    # 过渡场景（h020/h035）在 t=100 切高度指令：Isaac 的 obs 是上一拍末口径，切换点 ±2 帧
    # 的 obs/act 存在固有的一拍错位——排除出统计（tau 回放用 act_ref 不受影响，不排）。
    skip = set(range(100, 103)) if sc in ("h020", "h035") else set()
    rec = {k: [] for k in ("obs", "act", "act_on_ref", "tau")}
    block = {name: [] for name, _, _ in BLOCKS}

    for t in range(n):
        q = root[t, 3:7]
        lin_b = quat_rot_inv(q, root[t, 7:10])
        ang_b = quat_rot_inv(q, root[t, 10:13])
        grav = quat_rot_inv(q, torch.tensor([0.0, 0.0, -1.0]))

        obs = obs_b.build(
            lin_vel=lin_b, ang_vel=ang_b, gravity=grav, commands=cmds[t],
            dof_pos=dof_pos[t], dof_vel=dof_vel[t], last_action=last_act[t],
            caster_contacts=np.zeros(2, dtype=np.float32), tip_margin=0.0,
            phase=int(ph[t]),
            off_knee=float(npz[f"{sc}/off_knee"][t, 0]), prog=0.0)
        obs_b.push(dof_pos[t], dof_vel[t])

        act = policy.act(obs)[0]
        act_on_ref = policy.act(torch.tensor(obs_ref[t]))[0]

        tau = ctl.torques(act_ref[t], call_pos[t][0], call_vel[t][0])
        ctl.torques(act_ref[t], call_pos[t][1], call_vel[t][1])   # 推进控制器内部状态

        if t > 0 and t not in skip:
            rec["obs"].append(float(np.abs(obs.numpy() - obs_ref[t]).max()))
            for name, lo, hi in BLOCKS:
                block[name].append(float(np.abs(obs.numpy()[lo:hi] - obs_ref[t][lo:hi]).max()))
            rec["act"].append(float(np.abs(act.numpy() - act_ref[t]).max()))
        rec["act_on_ref"].append(float(np.abs(act_on_ref.numpy() - act_ref[t]).max()))
        rec["tau"].append(float(np.abs(tau.numpy() - tau_ref[t]).max()))

    out = {k: {"max": float(np.max(v)), "mean": float(np.mean(v))} for k, v in rec.items()}
    out["blocks"] = {k: float(np.max(v)) for k, v in block.items()}
    out["frames"] = int(n)
    out["passed"] = {"obs": out["obs"]["max"] < GATES["obs"],
                     "act": out["act"]["max"] < GATES["act"],
                     "tau": out["tau"]["max"] < GATES["tau"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default=os.path.join(PKG, "assets/deploy_contract_two.json"))
    ap.add_argument("--golden", default=os.path.join(PKG, "eval/golden/isaac_golden_two.npz"))
    ap.add_argument("--out", default=os.path.join(PKG, "eval/golden/parity_two.json"))
    a = ap.parse_args()

    contract = json.load(open(a.contract, encoding="utf-8"))
    ckpt = os.path.join(ROOT, contract["sources"]["release"], contract["sources"]["release_model"])
    policy = Policy(ckpt, contract["obs_dim"], contract["act_dim"])

    npz = np.load(a.golden, allow_pickle=True)
    scenes = sorted({k.split("/")[0] for k in npz.files if "/" in k})

    result = {"policy": policy.describe(), "gates": GATES, "scenarios": {}}
    ok = True
    for sc in scenes:
        r = replay(contract, policy, npz, sc)
        result["scenarios"][sc] = r
        ok = ok and all(r["passed"].values())
        print(f"[{'PASS' if all(r['passed'].values()) else 'FAIL'}] {sc}: "
              f"obs={r['obs']['max']:.2e} (base={r['blocks']['base']:.1e} "
              f"hist={r['blocks']['history']:.1e} phase={r['blocks']['phase']:.1e}) "
              f"act={r['act']['max']:.2e} (on_ref={r['act_on_ref']['max']:.2e}) "
              f"tau={r['tau']['max']:.2e}")
        if not all(r["passed"].values()):
            print("     failed:", [k for k, v in r["passed"].items() if not v])
    result["passed"] = ok

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"[{'ALL PASS' if ok else 'FAIL'}] {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
