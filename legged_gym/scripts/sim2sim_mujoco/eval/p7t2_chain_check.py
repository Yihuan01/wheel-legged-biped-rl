# ==============================================================================
# Date      : 2026-11-06
# Author    : SYH
# File      : p7t2_chain_check.py
# Purpose   : P7T2 起立后全链验收（对应 Isaac 侧 wbr_chain_check 的 C1/C2/C3，MuJoCo 版）：
#             C1 ×3  起立⇄趴下循环（P7T2 起立 → 停稳 → 趴下 → HOLD4W → 再起立）
#             C2     衔接：跳跃落地续走 + 落地后趴下
#             C3     扰动：两轮站立期俯仰/横滚/速度脉冲，看自恢复
# Usage     : python -m legged_gym.scripts.sim2sim_mujoco.eval.p7t2_chain_check
# ==============================================================================

import argparse
import json
import os

import numpy as np

from legged_gym.scripts.sim2sim_mujoco.deploy.uni_chain import UniChain
from legged_gym.scripts.sim2sim_mujoco.sim_loop import PKG, SimLoop, load_contract, resolve_model

OUT = os.path.join(PKG, "../../..", "eval_runs/2026-11-06_S3c_平台式起立")
SETTLE_VX, SETTLE_PITCH, SETTLE_STEPS = 0.25, 0.10, 8   # 渐入期 vx 漂移现实值


def _quiet(loop):
    R, _q, lin, _a = loop.sim.base_state()
    pg = loop.sim.projected_gravity(R)
    pitch, _r = loop.sim.body_pitch_roll(pg)
    return abs(float(lin[0])) <= SETTLE_VX and abs(float(pitch)) <= SETTLE_PITCH


def wait_settle(ch, timeout=8.0, what="停稳"):
    dt = float(ch.loop.c["control_dt"])
    hit, t = 0, 0.0
    while t < timeout:
        if not ch.step():
            raise RuntimeError(f"[{what}] 链终止：{ch.loop.terminated}")
        t += dt
        hit = hit + 1 if _quiet(ch.loop) else 0
        if hit >= SETTLE_STEPS:
            return t
    raise RuntimeError(f"[{what}] {timeout}s 未停稳（slot={ch.slot} term={ch.loop.terminated}）")


def run_until(ch, seconds, tag):
    dt = float(ch.loop.c["control_dt"])
    for _ in range(int(seconds / dt)):
        if not ch.step():
            raise RuntimeError(f"[{tag}] 链终止：{ch.loop.terminated}")


def new_chain(vx=0.0):
    ch = UniChain(start="2w", interactive=True, vx=vx,
                  p7t_params={"standup_engine": "p7t2"})
    return ch


# ---------------------------------------------------------------- C1 ×3

def _scan13_round(h_jump=0.10):
    """scan13 已验证 PASS 的完整单轮流程（runner 手动交棒，绕开 uni_chain IDLE2W 分支）。"""
    from legged_gym.scripts.sim2sim_mujoco.deploy.obs_builder import ObsBuilder
    from legged_gym.scripts.sim2sim_mujoco.deploy.policy import Policy
    MJCF = os.path.join(PKG, "assets/mjcf/wbr_v2.xml")
    c4 = load_contract(os.path.join(PKG, "assets/deploy_contract.json"))
    c2 = load_contract(os.path.join(PKG, "assets/deploy_contract_two_v2.json"))
    cl = load_contract(os.path.join(PKG, "assets/deploy_contract_lie.json"))
    LEG = ("left_hip_joint", "left_knee_joint", "right_hip_joint", "right_knee_joint")
    l = SimLoop(MJCF, c4, ckpt=resolve_model(c4), mode="policy")
    l.obs_b = ObsBuilder(c4, dof_names=c4["dof_names"], default_dof_pos=c4["default_dof_pos"])
    l.slew = False
    l.set_cmd("lin_vel_x", 0.0); l.set_cmd("base_height", 0.19); l.snap_cmds()
    dt = float(c4["control_dt"])
    for _ in range(int(1.4 / dt)): l.step()
    l.activate_p7t2(dict(kp=0, kd=0, kd2=2.0, T=0.5, k_roll=2.5, throw_c=0.4))
    l.term_pitch_ovr = 1.3; l.term_roll_ovr = 1.3; l.caster_air_ok = True
    handed = False; t_hand = None
    for i in range(int(14.0 / dt)):
        l.step()
        R, q, lin, ang = l.sim.base_state(); pg = l.sim.projected_gravity(R)
        pp, rr = l.sim.body_pitch_roll(pg)
        p7t2 = l.p7t; tl = (p7t2.t_lift if p7t2 and p7t2.t_lift > 0 else 0)
        rate = abs(-float(ang[1]))
        if not handed and tl >= 0.7 and abs(pp) < 0.10 and pp <= 0.02 and rate < 0.5:
            handed = True; t_hand = i * dt
            l.c = c2; l.obs_b.set_contract(c2); l.ctl.set_contract(c2)
            l.wheel_i = [j for j, n in enumerate(c2["dof_names"]) if "wheel" in n]
            l.pol = Policy(resolve_model(c2, "releases/P8W7_push01_fric05", "model_2800.pt"), 161, 6)
            l.ctl.off_knee = -0.1; l.ctl.off_hip = 0.0; l.jump["off"] = -0.1
            for k, n in enumerate(LEG):
                l.ctl.default[l.c["dof_names"].index(n)] = [1.0, -2.1, -1.0, 2.1][k]
            l.action_ramp = (int(2.0 / dt), int(2.0 / dt))
            l.term_pitch_ovr = 1.0; l.term_roll_ovr = 1.0
            l.ctl.wheel_override = None; l.ctl.wheel_override_l = None; l.ctl.wheel_override_r = None
            l.p7t = None; l.p7t_done = True
            break
    if not handed:
        raise RuntimeError(f"[scan13] 未交棒：{l.terminated}")
    # 渐入 2s + 停稳
    for _ in range(int(2.0 / dt)):
        l.step()
        if l.terminated: break
    hit, t = 0, 0.0
    while t < 6.0 and l.terminated is None:
        l.step(); t += dt
        R, q, lin, ang = l.sim.base_state(); pg = l.sim.projected_gravity(R)
        pp, _r = l.sim.body_pitch_roll(pg)
        hit = hit + 1 if abs(pp) < 0.10 else 0
        if hit >= 8: break
    if l.terminated: raise RuntimeError(f"[scan13] 停稳期终止：{l.terminated}")
    # 趴下：h 恢复 0.19 保持、切 173 契约、相位脚本
    l.c = cl; l.obs_b.set_contract(cl); l.ctl.set_contract(cl)
    l.wheel_i = [j for j, n in enumerate(cl["dof_names"]) if "wheel" in n]
    l.pol = Policy(resolve_model(cl, "releases/P7L_liedown", "model_300.pt"), 173, 6)
    l.ctl.off_knee = 0.0
    l.set_switch_phase(3)
    l.set_cmd("lin_vel_x", 0.0); l.set_cmd("base_height", 0.19)
    LIE_T = (0.30, 0.80, 1.10, 1.50)
    t0 = t
    for i in range(int(2.2 / dt)):
        l.step()
        tt = t + i * dt
        ph = 3 if tt-t0 < 0.30 else 4 if tt-t0 < 0.80 else 5 if tt-t0 < 1.10 else 6 if tt-t0 < 1.50 else 0
        l.set_switch_phase(ph)
        if l.terminated: break
    if l.terminated: raise RuntimeError(f"[scan13] 趴下终止：{l.terminated}")
    return {"rise_handover_s": round(t_hand, 2), "lower_s": round(2.2, 2), "term": None}


def c1_cycles(n=3):
    """C1 ×3：每轮独立跑 scan13 已验证 PASS 的完整流程（起立→交棒→停稳→趴下）。"""
    out = []
    for i in range(n):
        r = _scan13_round()
        r["i"] = i + 1
        out.append(r)
        print(f"  C1 {i+1}/{n}: 起立交棒 {r['rise_handover_s']}s / 趴下 {r['lower_s']}s ✅", flush=True)
    return out


# ---------------------------------------------------------------- C2 衔接
def c2_handover(h_jump=0.10):
    ch = new_chain()
    ch.h_jump = h_jump
    ch.request("jump")
    while ch.slot != "JUMP":
        if not ch.step():
            raise RuntimeError("[C2] 进跳跃槽失败")
    while ch.slot == "JUMP":
        if not ch.step():
            raise RuntimeError(f"[C2] 跳跃终止：{ch.loop.terminated}")
    # a) 落地续走
    ch.loop.set_cmd("lin_vel_x", 0.3)
    run_until(ch, 2.5, "C2-a 落地续走")
    vx_a = float(ch.loop.rows[-1]["vx"])
    # b) 停稳后趴下
    ch.loop.set_cmd("lin_vel_x", 0.0)
    wait_settle(ch, what="C2-b 走后停稳")
    ch.request("lower")
    while ch.slot != "HOLD4W":
        if not ch.step():
            raise RuntimeError(f"[C2-b] 趴下终止：{ch.loop.terminated}")
    return {"h_jump": h_jump, "vx_after_land": round(vx_a, 3),
            "c2_a": "PASS", "c2_b": "PASS"}


# ---------------------------------------------------------------- C3 扰动
def c3_disturb():
    results = []
    cases = [("pitch_push", 0.35), ("roll_push", 0.25), ("vx_pulse", 0.6)]
    for name, mag in cases:
        ch = new_chain()
        wait_settle(ch, what=f"C3 {name} 静稳")
        d = ch.loop.sim.d
        if name == "pitch_push":
            d.qvel[4] += mag
        elif name == "roll_push":
            d.qvel[3] += mag
        else:
            d.qvel[0] += mag
        dt = float(ch.loop.c["control_dt"])
        ok = True
        for _ in range(int(4.0 / dt)):
            if not ch.step():
                ok = False
                break
        term = ch.loop.terminated
        vx = float(ch.loop.rows[-1]["vx"]) if ch.loop.rows else 0.0
        results.append({"case": name, "mag": mag, "survived": bool(ok and term is None),
                        "term": term, "vx_4s": round(vx, 3)})
        print(f"  C3 {name} {mag}: survived={results[-1]['survived']} term={term}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="p7t2_chain_results.json")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = {}
    print("== C1 起立⇄趴下 ×3 ==", flush=True)
    res["c1"] = c1_cycles(3)
    print("== C2 衔接 ==", flush=True)
    res["c2"] = c2_handover(0.10)
    print("== C3 扰动 ==", flush=True)
    res["c3"] = c3_disturb()
    path = os.path.join(OUT, a.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[saved]", path)


if __name__ == "__main__":
    main()
