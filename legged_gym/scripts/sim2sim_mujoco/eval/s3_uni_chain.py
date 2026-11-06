# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : s3_uni_chain.py
# Purpose   : S3 五槽 uni 整链（MuJoCo）：4W 行驶(1s) → P7T 起立(~2s) → W3 行走(3s,
#             vx 0.3) → 装填跳跃(~1.6s) → 趴下(P7L, ~1.5s) → HOLD4W(1s)。三契约
#             运行时切换（164 四轮 / 161 两轮 / 173 趴下），链检 C1 同款顺序。
# ==============================================================================

import json
import os

import numpy as np
import torch

from legged_gym.scripts.sim2sim_mujoco.deploy.obs_builder import ObsBuilder
from legged_gym.scripts.sim2sim_mujoco.deploy.policy import Policy
from legged_gym.scripts.sim2sim_mujoco.sim_loop import PKG, SimLoop, load_contract, resolve_model

MJCF = os.path.join(PKG, "assets/mjcf/wbr_v2.xml")
OUT = os.path.join(PKG, "../../..", "eval_runs/2026-11-05_S3_五槽uni整链")
LEGS4 = ("left_hip_joint", "left_knee_joint", "right_hip_joint", "right_knee_joint")
POSE_2W = [1.0, -2.1, -1.0, 2.1]
Z2W = 0.2697


def main():
    c4 = load_contract(os.path.join(PKG, "assets/deploy_contract.json"))          # 164 四轮
    c2 = load_contract(os.path.join(PKG, "assets/deploy_contract_two_v2.json"))   # 161 两轮@v2（五槽 w3 槽语义）
    cl = load_contract(os.path.join(PKG, "assets/deploy_contract_lie.json"))      # 173 趴下
    ck_w4 = resolve_model(c4)
    ck_w3 = resolve_model(c2, "releases/P8W7_push01_fric05", "model_2800.pt")
    ck_lie = resolve_model(cl)

    loop = SimLoop(MJCF, c4, ckpt=ck_w4, mode="policy")     # SimLoop 内部自建 Policy
    loop.obs_b = ObsBuilder(c4, dof_names=c4["dof_names"],
                            default_dof_pos=c4["default_dof_pos"])   # 8 dof 载体（链全程连续）
    loop.slew = False
    loop.set_cmd("lin_vel_x", 0.3)
    loop.set_cmd("base_height", 0.21)
    loop.snap_cmds()

    marks = []
    rows = []
    dt = float(c4["control_dt"])

    def phase(name, dur, fn=None):
        """跑一段链，fn(i, t) 返回 True 强制结束。"""
        n = int(dur / dt)
        for i in range(n):
            if fn is not None and fn(i, i * dt):
                return True
            loop.step(record=True)
            rows.append((len(rows) * dt, name, float(loop.sim.d.qpos[2]),
                         float(np.degrees(loop.rows[-1]["pitch"])) if loop.rows else 0.0))
            if loop.terminated:
                print(f"⛔ [{name}] 终止: {loop.terminated}")
                return True
        return False

    # ── 槽1: 四轮行驶 1 s ──
    print("▶ 槽1 四轮行驶 vx0.3", flush=True)
    marks.append(("4W", len(rows), float(loop.sim.d.qpos[2])))
    phase("4W", 1.0)

    # ── 槽2: P7T 起立（契约切 lie=173？起立段 act=0 无策略，obs 无消费——保持 164 契约跑脚本） ──
    print("▶ 槽1.5 减速缓冲", flush=True)
    loop.set_cmd("lin_vel_x", 0.0)
    phase("BRAKE", 0.4)
    print("▶ 槽2 P7T 起立（软交接：伸腿末 0.1s 切 w3）", flush=True)
    loop.activate_p7t()
    loop.p7t_switched = False

    def p7t_fn(i, t):
        # 软交接：伸腿段 al≥0.8（t≥1.9）时切 161 契约＋w3 网——default 跳变小、不经历
        # "抬升段"（轮恒力矩冲高与侧倾），平衡由 w3 网无缝接管。
        if not loop.p7t_switched and loop.p7t.t >= 1.9:
            loop.c = c2
            loop.obs_b.c = c2                    # 换 obs 布局（161）；hist/default 连续
            loop.pol = Policy(ck_w3, 161, 6)
            loop.jt = None
            loop.ctl.off_knee = -0.1             # w3 训练分布（flat 语义）
            loop.jump["off"] = -0.1
            loop.ctl.off_hip = 0.0
            loop.jump["off_hip"] = 0.0
            for k, idx in enumerate([c2["dof_names"].index(n) for n in LEGS4]):
                loop.ctl.default[idx] = POSE_2W[k]
            loop.ctl.leg_override = None
            loop.ctl.wheel_override = None
            loop.ctl.leg_boost = 1.0
            loop.term_pitch_ovr = 1.0
            loop.term_roll_ovr = 0.8
            loop.p7t = None                      # 脚本退役（控制权交 w3）
            loop.p7t_switched = True
            print("  [软交接] t=%.2f z=%.4f" % (t, float(loop.sim.d.qpos[2])), flush=True)
            return True
        return False
    phase("P7T", 4.0, p7t_fn)
    marks.append(("RISED", len(rows), float(loop.sim.d.qpos[2])))

    # ── 槽3: W3 两轮行走（软交接后继续，契约/网已就位） ──
    print("▶ 槽3 W3 两轮行走 vx0.3", flush=True)
    loop.set_cmd("lin_vel_x", 0.3)
    loop.set_cmd("base_height", 0.28)
    loop.snap_cmds()

    def w3_walk(i, t):
        return i >= int(3.0 / dt)
    phase("W3", 3.0, w3_walk)
    marks.append(("W3ED", len(rows), float(loop.sim.d.qpos[2])))

    # ── 槽4: 跳跃（同契约 161？跳跃模板在 two 契约 jump_template=true——但 SimLoop.jt 已 None；
    #   两轮契约带 jump_template → 重建模板） ──
    print("▶ 槽4 跳跃", flush=True)
    loop.c = c2
    loop.obs_b.c = c2
    from legged_gym.scripts.sim2sim_mujoco.deploy.jump_template import JumpTemplate
    loop.jt = JumpTemplate(k_lim=tuple(c2["knee_limits"]), h_lim=tuple(c2["hip_limits"]), dt=dt)
    loop.jump_request(0.10)

    def wait_jump(i, t):
        return i >= int(1.8 / dt)
    phase("JUMP", 2.0, wait_jump)
    marks.append(("JUMPED", len(rows), float(loop.sim.d.qpos[2])))

    # ── 槽5: 趴下（契约切 173＋P7L；起点=当前两轮站姿） ──
    print("▶ 槽5 趴下", flush=True)
    loop.c = cl
    loop.obs_b.c = cl                    # 换 obs 布局（173）；hist/default 连续
    loop.pol = Policy(ck_lie, 173, 6)
    loop.jt = None
    loop.set_cmd("base_height", 0.19)
    loop.snap_cmds()

    def lie(i, t):
        return i >= int(1.5 / dt)      # 趴下时序 1.5s
    phase("LIE", 1.6, lie)

    # ── 槽6: HOLD4W 1 s ──
    loop.set_switch_phase(0)
    phase("H4W", 1.0)
    marks.append(("DONE", len(rows), float(loop.sim.d.qpos[2])))

    # ── 汇总 ──
    summary = {"marks": [(n, round(t, 2), round(zz, 4)) for n, t, zz in marks],
               "total_s": round(len(rows) * dt, 2),
               "frames": len(rows), "term": loop.terminated}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "chain_c1.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
