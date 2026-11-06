# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : run_parity.py
# Purpose   : 链路对账：用 Isaac 录的 golden 逐帧回放，比较观测（分块）、策略动作、
#             力矩（逐物理子步）、倾翻裕度与从动轮接触。全程不跑物理——只验证部署链。
# ==============================================================================

import argparse
import json
import os

import numpy as np
import torch

from legged_gym.scripts.sim2sim_mujoco.deploy.policy import Policy
from legged_gym.scripts.sim2sim_mujoco.deploy.obs_builder import ObsBuilder, BLOCKS
from legged_gym.scripts.sim2sim_mujoco.deploy.control import PDController
from legged_gym.scripts.sim2sim_mujoco.deploy.com_model import com_constants, com_body_offset

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(PKG, "../../.."))

GATES = {"obs": 1e-4, "act": 1e-5, "tau": 1e-4, "tip": 1e-6}


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


def tip_margin(rb_state, feet, casters, root_x, dof_pos, contract, cc):
    xs = np.concatenate([rb_state[feet, 0], rb_state[casters, 0]])
    names = contract["dof_names"]
    q = torch.tensor(dof_pos, dtype=torch.float32).unsqueeze(0)
    com = com_body_offset(cc, q[:, names.index("left_hip_joint")],
                          q[:, names.index("right_hip_joint")],
                          q[:, names.index("left_knee_joint")],
                          q[:, names.index("right_knee_joint")])
    com_x = float(root_x) + float(com[0, 0])
    return float(min(xs.max() - com_x, com_x - xs.min()))


def replay(contract, policy, npz, sc, isaac_meta):
    obs_b = ObsBuilder(contract)
    # reset() 内部是 reset_idx + 一步 step：第 0 帧的 obs 用的是"推入 state_0 之前"的历史，
    # 落盘只能拿到推入之后的历史 ⇒ 右移一行得到第 0 帧的初值（第 0 行是被挤掉的未知项），
    # 第 0 帧不参与 obs/act 比对；再 push 一次后历史与落盘逐位一致。
    has_hist0 = f"{sc}/hist0_pos" in npz
    if has_hist0:
        h0 = torch.tensor(npz[f"{sc}/hist0_pos"], dtype=torch.float32)
        v0 = torch.tensor(npz[f"{sc}/hist0_vel"], dtype=torch.float32)
        obs_b.hist_pos = torch.cat([torch.zeros(1, 6), h0[:-1]], dim=0)
        obs_b.hist_vel = torch.cat([torch.zeros(1, 6), v0[:-1]], dim=0)
    ctl = PDController(contract)
    if f"{sc}/ctrl0_last" in npz:
        ctl.last_wheel = torch.tensor(npz[f"{sc}/ctrl0_last"], dtype=torch.float32)
        ctl.filt = torch.tensor(npz[f"{sc}/ctrl0_filt"], dtype=torch.float32)
        ctl.wheel_target = torch.tensor(npz[f"{sc}/ctrl0_target"], dtype=torch.float32)
    names = contract["dof_names"]
    feet = list(isaac_meta["feet_indices"])
    casters = list(isaac_meta["caster_body_indices"])
    cc = com_constants("cpu")

    root = npz[f"{sc}/root"]
    dof_pos = npz[f"{sc}/dof_pos"]
    dof_vel = npz[f"{sc}/dof_vel"]
    cmds = npz[f"{sc}/commands"]
    last_act = npz[f"{sc}/last_action"]
    obs_ref = npz[f"{sc}/obs"]
    act_ref = npz[f"{sc}/act"]
    tau_ref = npz[f"{sc}/tau"]
    call_pos = npz[f"{sc}/call_dof_pos"]
    call_vel = npz[f"{sc}/call_dof_vel"]
    rb = npz[f"{sc}/rb_state"]
    caster_force = npz[f"{sc}/caster_force"]
    caster_ref = npz[f"{sc}/caster_contacts"]
    tip_ref = npz[f"{sc}/tip_margin"][:, 0]
    ph = npz[f"{sc}/ph"][:, 0]

    n = root.shape[0]
    assert int(np.abs(ph).max()) == 0, "四轮 golden 的相位应恒为 0（enable_jump=False）"
    rec = {k: [] for k in ("obs", "act", "act_on_ref", "tau", "tip", "caster")}
    block = {name: [] for name, _, _ in BLOCKS}

    for t in range(n):
        q = root[t, 3:7]
        lin_b = quat_rot_inv(q, root[t, 7:10])
        ang_b = quat_rot_inv(q, root[t, 10:13])
        grav = quat_rot_inv(q, torch.tensor([0.0, 0.0, -1.0]))

        contact = (caster_force[t][:, 2] > 0.1).astype(np.float32)
        tip = tip_margin(rb[t], feet, casters, root[t, 0], dof_pos[t], contract, cc)

        obs = obs_b.build(
            lin_vel=lin_b, ang_vel=ang_b, gravity=grav, commands=cmds[t],
            dof_pos=dof_pos[t], dof_vel=dof_vel[t], last_action=last_act[t],
            caster_contacts=contact, tip_margin=tip, phase=int(ph[t]),
            off_knee=float(npz[f"{sc}/off_knee"][t, 0]), prog=0.0)
        obs_b.push(dof_pos[t], dof_vel[t])

        act = policy.act(obs)[0]
        act_on_ref = policy.act(torch.tensor(obs_ref[t]))[0]

        tau = ctl.torques(act_ref[t], call_pos[t][0], call_vel[t][0])
        # 第二个子步只推进控制器内部状态（last_wheel / 加速度低通），用录制的中间态
        ctl.torques(act_ref[t], call_pos[t][1], call_vel[t][1])

        if t > 0 or not has_hist0:
            rec["obs"].append(float(np.abs(obs.numpy() - obs_ref[t]).max()))
            for name, lo, hi in BLOCKS:
                block[name].append(float(np.abs(obs.numpy()[lo:hi] - obs_ref[t][lo:hi]).max()))
            rec["act"].append(float(np.abs(act.numpy() - act_ref[t]).max()))
        rec["act_on_ref"].append(float(np.abs(act_on_ref.numpy() - act_ref[t]).max()))
        rec["tau"].append(float(np.abs(tau.numpy() - tau_ref[t]).max()))
        rec["tip"].append(float(abs(tip - tip_ref[t])))
        rec["caster"].append(float(np.abs(contact - caster_ref[t]).max()))

    out = {k: {"max": float(np.max(v)), "mean": float(np.mean(v))} for k, v in rec.items()}
    out["blocks"] = {k: float(np.max(v)) for k, v in block.items()}
    out["frames"] = int(n)
    out["passed"] = {
        "obs": out["obs"]["max"] < GATES["obs"],
        "act": out["act"]["max"] < GATES["act"],
        "tau": out["tau"]["max"] < GATES["tau"],
        "tip": out["tip"]["max"] < GATES["tip"],
        "caster": out["caster"]["max"] == 0.0,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default=os.path.join(PKG, "assets/deploy_contract.json"))
    ap.add_argument("--golden", default=os.path.join(PKG, "eval/golden/isaac_golden.npz"))
    ap.add_argument("--out", default=os.path.join(PKG, "eval/golden/parity.json"))
    a = ap.parse_args()

    contract = json.load(open(a.contract, encoding="utf-8"))
    ckpt = os.path.join(ROOT, contract["sources"]["release"], contract["sources"]["release_model"])
    policy = Policy(ckpt, contract["obs_dim"], contract["act_dim"])

    npz = np.load(a.golden, allow_pickle=True)
    meta = json.loads(str(npz["meta"]))
    isa = json.load(open(os.path.join(PKG, "eval/golden/isaac_static.json"), encoding="utf-8"))

    result = {"policy": policy.describe(), "gates": GATES, "scenarios": {}}
    ok = True
    for sc in meta["scenarios"]:
        r = replay(contract, policy, npz, sc, isa)
        result["scenarios"][sc] = r
        ok = ok and all(r["passed"].values())
        print(f"[{'PASS' if all(r['passed'].values()) else 'FAIL'}] {sc}: "
              f"obs={r['obs']['max']:.2e} (base={r['blocks']['base']:.1e} "
              f"hist={r['blocks']['history']:.1e} jump={r['blocks']['jump']:.1e} "
              f"4w={r['blocks']['four_wheel']:.1e}) "
              f"act={r['act']['max']:.2e} tau={r['tau']['max']:.2e} "
              f"tip={r['tip']['max']:.2e} caster={r['caster']['max']:.0f}")
        if not all(r["passed"].values()):
            print("     failed:", [k for k, v in r["passed"].items() if not v])
    result["passed"] = ok

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"[{'ALL PASS' if ok else 'FAIL'}] {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
