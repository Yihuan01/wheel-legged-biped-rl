# ==============================================================================
# Date      : 2026-11-06
# Author    : SYH
# File      : s2b_jump_tune.py
# Purpose   : S2b 跳跃顶点 23mm 精调：验证「触发拍 ref 冻结为 golden 站立角」假设。
#             站立平衡点冗余 ⇒ MJ 触发拍关节角与 Isaac 的微差经 2R 反解全程平移 off
#             序列（off = k_c(L*) − ref_k，PD 目标 = default + act + off）。对照四组：
#             baseline（实际角）/ frozen（golden hip+knee）/ hip-only / knee-only。
#             用法：python -m legged_gym.scripts.sim2sim_mujoco.eval.s2b_jump_tune
# ==============================================================================
import json
import os

import numpy as np

from legged_gym.scripts.sim2sim_mujoco.sim_loop import PKG, SimLoop, load_contract, resolve_model

MJCF = os.path.join(PKG, "assets/mjcf/wbr_v1.xml")
GOLDEN = os.path.join(PKG, "eval/golden/isaac_golden_two.npz")
OUT = os.path.join(PKG, "../../..", "eval_runs/2026-11-06_S2b_跳跃顶点精调")
DUR = 6.0
H = 0.10
TRIG = 199                      # golden 触发拍（2.0 s 站立 settled 后）
N_REPEAT = 2

GOLD_APEX = 0.4005
GOLD_CROUCH = 0.1994


def run_once(ref_override, repeat, shadow=False, trig=None):
    trig_frame_cfg = trig
    contract = load_contract(os.path.join(PKG, "assets/deploy_contract_two.json"))
    ckpt = resolve_model(contract)
    loop = SimLoop(MJCF, contract, ckpt=ckpt, mode="policy")
    loop.slew = False
    if ref_override is not None:
        loop.jt.ref_override = dict(ref_override)
    if shadow:
        # 【ph1 影子】蹲相期间 off_knee/off_hip 直接取 golden 序列（相位机自身积分被同步），
        #   爆发段（ph2）起恢复模板闭环——用于分离「蹲姿伺服残差」与「被控对象差」。
        gold_np = np.load(GOLDEN, allow_pickle=True)
        off_g = gold_np["jump10/off_knee"][:, 0].astype(float)
        offh_g = gold_np["jump10/off_hip"][:, 0].astype(float)
        orig_update = loop.jt.update
        state = {"k": 0}
        def shadow_update(z, vz, fz_l, fz_r, roll, pitch, com_z, dof_pos, _o=orig_update, _s=state,
                          _og=off_g, _oh=offh_g):
            out = _o(z, vz, fz_l, fz_r, roll, pitch, com_z, dof_pos)
            if out["ph"] == 1:
                _s["k"] += 1                      # 第 k 个 ph1 帧 ↔ golden 第 TRIG+k-1 帧
                idx = TRIG + _s["k"] - 1
                if idx < len(_og):
                    out["off_knee"] = float(_og[idx])
                    out["off_hip"] = float(_oh[idx])
                    loop.jt.off_knee = out["off_knee"]
                    loop.jt.off_hip = out["off_hip"]
            return out
        loop.jt.update = shadow_update
    loop.set_cmd("lin_vel_x", 0.0)
    loop.set_cmd("base_height", 0.28)
    loop.snap_cmds()
    loop.warmup(2)
    dt = float(contract["control_dt"])
    names = contract["dof_names"]
    i_hip, i_knee = names.index("left_hip_joint"), names.index("left_knee_joint")

    n = int(DUR / dt)
    trig_frame = None
    nonlocal_trig = trig_frame_cfg if trig_frame_cfg is not None else TRIG
    stand_z = None
    trig_ang = None
    crouch_min = None
    apex = 0.0
    term = None
    for i in range(n):
        if trig_frame is None and i >= nonlocal_trig:
            loop.jump_request(H)
            trig_frame = i
            stand_z = float(loop.sim.d.qpos[2])
            q = loop.sim.d.qpos
            trig_ang = (float(q[7 + i_hip]), float(q[7 + i_knee]))   # qpos: 7 root + dof
        loop.step(record=True)
        z = float(loop.sim.d.qpos[2])
        if trig_frame is not None:
            ph = loop.jump["ph"]
            if ph in (1, 2):
                crouch_min = z if crouch_min is None else min(crouch_min, z)
            apex = max(apex, z)
        if loop.terminated:
            term = loop.terminated
            break
    return dict(repeat=repeat, ref=ref_override, stand_z=round(stand_z, 4),
                trig_hip=round(trig_ang[0], 4), trig_knee=round(trig_ang[1], 4),
                z_crouch=round(crouch_min, 4) if crouch_min else None,
                z_apex=round(apex, 4), apex_gap_mm=round((apex - GOLD_APEX) * 1000, 1),
                term=str(term))


def main():
    gold = np.load(GOLDEN, allow_pickle=True)
    dof = gold["jump10/dof_pos"]
    g_hip, g_knee = float(dof[TRIG, 0]), float(dof[TRIG, 1])   # [L_hip, L_knee, L_wheel, ...]
    print(f"golden 触发拍角: hip={g_hip:.4f} knee={g_knee:.4f} | apex {GOLD_APEX} crouch {GOLD_CROUCH}")

    variants = [
        ("baseline", None, False),
        ("shadow_ph1", None, True),
        ("frozen", {"hip": g_hip, "knee": g_knee}, False),
    ]
    results = []
    for name, ov, sh in variants:
        for r in range(N_REPEAT if name != "shadow_ph1" else 1):
            row = run_once(ov, r, shadow=sh)
            row["variant"] = name
            results.append(row)
            print(f"[{name} #{r}] stand {row['stand_z']} trig(h,k)=({row['trig_hip']},{row['trig_knee']}) "
                  f"crouch {row['z_crouch']} apex {row['z_apex']} gap {row['apex_gap_mm']}mm term {row['term']}",
                  flush=True)

    # 触发时刻敏感度：只改装填拍（站立时长），其余全同 ⇒ 顶点对 history 的敏感度
    for tf in (100, 300):
        row = run_once(None, tf, trig=tf)
        row["variant"] = f"trig{tf}"
        results.append(row)
        print(f"[trig{tf}] stand {row['stand_z']} trig(h,k)=({row['trig_hip']},{row['trig_knee']}) "
              f"crouch {row['z_crouch']} apex {row['z_apex']} gap {row['apex_gap_mm']}mm term {row['term']}",
              flush=True)

    os.makedirs(OUT, exist_ok=True)
    summary = {"golden": {"hip": g_hip, "knee": g_knee, "apex": GOLD_APEX, "crouch": GOLD_CROUCH},
               "results": results}
    with open(os.path.join(OUT, "jump_tune_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("saved:", os.path.join(OUT, "jump_tune_summary.json"))


if __name__ == "__main__":
    main()
