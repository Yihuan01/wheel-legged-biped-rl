# ==============================================================================
# Date      : 2026-11-06
# Author    : SYH
# File      : s2b_jump_trace.py
# Purpose   : S2b 跳跃顶点差定位：MJ 闭环一跳逐拍对齐 golden jump10（触发拍对齐），
#             找出发散帧。打印 z/vz/off_knee/off_hip/ph 的分相摘要与逐拍 diff 摘要。
#             用法：python -m legged_gym.scripts.sim2sim_mujoco.eval.s2b_jump_trace
# ==============================================================================
import os

import numpy as np

from legged_gym.scripts.sim2sim_mujoco.sim_loop import PKG, SimLoop, load_contract, resolve_model

MJCF = os.path.join(PKG, "assets/mjcf/wbr_v1.xml")
GOLDEN = os.path.join(PKG, "eval/golden/isaac_golden_two.npz")
H = 0.10
TRIG = 199
DUR = 6.0


def main():
    gold = np.load(GOLDEN, allow_pickle=True)
    ph_g = gold["jump10/ph"][:, 0].astype(int)
    off_g = gold["jump10/off_knee"][:, 0].astype(float)
    offh_g = gold["jump10/off_hip"][:, 0].astype(float)
    root_g = gold["jump10/root"]
    z_g = root_g[:, 2]
    vz_g = root_g[:, 10] if root_g.shape[1] >= 11 else None
    tau_g = gold["jump10/tau"]
    print(f"golden h_target[trig]={float(gold['jump10/h_target'][TRIG,0]):.3f} "
          f"cmd4[trig]={float(gold['jump10/commands'][TRIG,4]):.3f} vx={float(gold['jump10/commands'][TRIG,0]):.3f}")
    print(f"golden tau[trig] = {np.round(tau_g[TRIG], 2)}")

    contract = load_contract(os.path.join(PKG, "assets/deploy_contract_two.json"))
    ckpt = resolve_model(contract)
    loop = SimLoop(MJCF, contract, ckpt=ckpt, mode="policy")
    loop.slew = False
    loop.set_cmd("lin_vel_x", 0.0)
    loop.set_cmd("base_height", 0.28)
    loop.snap_cmds()
    loop.warmup(2)
    dt = float(contract["control_dt"])

    n = int(DUR / dt)
    trig = None
    frames = []
    for i in range(n):
        if trig is None and i >= TRIG:
            loop.jump_request(H)
            trig = i
        loop.step(record=True)
        if trig is not None:
            k = i - trig
            act = loop.last_action.astype(float)
            frames.append(dict(k=k, ph=int(loop.jump["ph"]), z=float(loop.sim.d.qpos[2]),
                               vz=float(loop.sim.d.qvel[2]), off=float(ctl_off(loop)),
                               offh=float(loop.ctl.off_hip), act=act.tolist()))
        if loop.terminated:
            print("term:", loop.terminated)
            break

    print(f"\nMJ 触发拍 = {trig}（golden {TRIG}），对齐后逐相对照（k=触发后拍数）：")
    print(f"{'k':>4} {'ph_g':>4} {'ph_mj':>5} {'z_g':>7} {'z_mj':>7} {'dz_mm':>7} "
          f"{'vz_g':>6} {'vz_mj':>6} {'off_g':>7} {'off_mj':>7} {'doff':>6}")
    z_apex_g, z_apex_m = 0, 0
    for f in frames:
        k = f["k"]
        if TRIG + k >= len(z_g):
            break
        zg, zm = float(z_g[TRIG + k]), f["z"]
        z_apex_g, z_apex_m = max(z_apex_g, zg), max(z_apex_m, zm)
        vg = float(vz_g[TRIG + k]) if vz_g is not None else float("nan")
        if k % 5 == 0 or abs(zm - zg) > 0.02 or f["ph"] != int(ph_g[TRIG + k]):
            print(f"{k:>4} {int(ph_g[TRIG+k]):>4} {f['ph']:>5} {zg:>7.4f} {zm:>7.4f} {(zm-zg)*1000:>7.1f} "
                  f"{vg:>6.2f} {f['vz']:>6.2f} {off_g[TRIG+k]:>7.3f} {f['off']:>7.3f} {f['off']-off_g[TRIG+k]:>6.3f}")
    print(f"\n顶点: golden {z_apex_g:.4f}  MJ {z_apex_m:.4f}  差 {(z_apex_m-z_apex_g)*1000:+.1f} mm")
    # 爆发段（k55~62）act/tau 对照：差异在策略侧还是被控对象侧
    act_g = gold["jump10/act"]; tau_g2 = gold["jump10/tau"]
    print("\n爆发段 act/tau 对照（左膝 dof=1；tau 单位 N·m）：")
    print(f"{'k':>4} {'act_g[Lk]':>9} {'act_m[Lk]':>9} {'tau_g[Lk]':>9} {'tau_m?':>6} {'off_g':>7} {'off_m':>7}")
    for f in frames:
        k = f["k"]
        if 54 <= k <= 64:
            am = f["act"][1] if len(f["act"]) > 1 else float('nan')
            print(f"{k:>4} {act_g[TRIG+k,1]:>9.3f} {am:>9.3f} {tau_g2[TRIG+k,1]:>9.2f} {'—':>6} "
                  f"{off_g[TRIG+k]:>7.3f} {f['off']:>7.3f}")
    # 分相首帧
    for p in (1, 2, 3, 4, 5, 0):
        kg = int(np.argmax(ph_g[TRIG:] == p)) if (ph_g[TRIG:] == p).any() else None
        km = next((f["k"] for f in frames if f["ph"] == p), None)
        if kg is not None:
            print(f"ph{p} 首帧: golden +{kg} ({kg*dt:.2f}s)  MJ +{km} ({km*dt:.2f}s)" if km is not None
                  else f"ph{p} 首帧: golden +{kg}  MJ 未达")


def ctl_off(loop):
    return loop.jump["off"]


if __name__ == "__main__":
    main()
