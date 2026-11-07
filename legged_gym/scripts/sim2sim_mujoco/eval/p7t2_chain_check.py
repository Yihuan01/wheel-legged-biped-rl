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
    """P7T2 起立→停稳→趴下→回四轮（**全走 UniChain**，与控制台同一条路径）。

    全走 UniChain（RISE 真起立、交棒渐入分通道），与控制台同一条路径。
    ⚠ 不得绕开 IDLE2W 分支手工切契约/摆 default——那测的就不是控制台实际跑的路径。
    """
    # vx=0.0：链的默认 vx 是 0.3（行进档），交棒 2 s 后会把指令恢复成 0.3 ⇒ 机器人开始走，
    #   紧接着的「趴下」在行走中执行会前翻（实测 pitch −0.5 终止）。控制台默认 cmds.vx=0，
    #   评估按控制台口径来。
    ch = UniChain(start="4w", interactive=True, allow_rise=True, vx=0.0,
                  p7t_params={"standup_engine": "p7t2"})
    wait_settle(ch, timeout=4.0, what="4W 静稳")
    ch.request("next")                      # 4W → RISE（真起立）
    t0 = ch.t_total
    while ch.slot != "IDLE2W":
        if not ch.step():
            raise RuntimeError(f"[C1] 起立终止：{ch.loop.terminated}")
        if ch.t_total - t0 > 12.0:
            raise RuntimeError("[C1] 起立 12s 未完成")
    rise_s = ch.t_total - t0
    wait_settle(ch, timeout=8.0, what="起立后停稳")
    ch.request("lower")                     # 两轮 → 趴下
    t1 = ch.t_total
    while ch.slot != "HOLD4W":
        if not ch.step():
            raise RuntimeError(f"[C1] 趴下终止：{ch.loop.terminated}")
        if ch.t_total - t1 > 8.0:
            raise RuntimeError("[C1] 趴下 8s 未完成")
    return {"rise_s": round(rise_s, 2), "lower_s": round(ch.t_total - t1, 2), "term": None}


def c1_cycles(n=3):
    """C1 ×3：每轮独立跑 scan13 已验证 PASS 的完整流程（起立→交棒→停稳→趴下）。"""
    out = []
    for i in range(n):
        r = _scan13_round()
        r["i"] = i + 1
        out.append(r)
        print(f"  C1 {i+1}/{n}: 起立 {r['rise_s']}s / 趴下 {r['lower_s']}s ✅", flush=True)
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
