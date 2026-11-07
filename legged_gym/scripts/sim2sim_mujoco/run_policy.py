# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : run_policy.py
# Purpose   : MuJoCo 四轮批量运行 CLI 壳：固定指令跑 SimLoop，出摘要/遥测/四联图数据。
#             运行核在 sim_loop.py（与控制台 serve.py 共用一核）；本文件只做参数解析
#             与产物落盘。物理语义与 sim_loop 逐位一致（slew 关闭，golden 对账把关）。
#             用法：python -m legged_gym.scripts.sim2sim_mujoco.run_policy --vx 0.3 --h 0.21 --dur 10
# ==============================================================================

import argparse
import json
import os
import sys

from legged_gym.scripts.sim2sim_mujoco.sim_loop import PKG, ROOT, SimLoop, load_contract, resolve_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default=os.path.join(PKG, "assets/deploy_contract.json"))
    ap.add_argument("--mjcf", default=os.path.join(PKG, "assets/mjcf/wbr_v2.xml"))
    ap.add_argument("--release", default=None, help="默认取 contract.sources.release")
    ap.add_argument("--mode", choices=["policy", "zero"], default="policy")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--h", type=float, default=0.19)
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--warmup", type=int, default=1, help="起跑前的零动作控制步数（对齐 Isaac reset 协议）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", action="store_true", help="把遥测写盘（默认只打印摘要）")
    ap.add_argument("--viewer", action="store_true", help="打开 MuJoCo 窗口实时看（CPU 渲染）")
    ap.add_argument("--speed", type=float, default=1.0, help="回放倍速（0.5=慢放，2=快放）")
    a = ap.parse_args()

    contract = load_contract(a.contract)
    ckpt = resolve_model(contract, a.release)
    loop = SimLoop(a.mjcf, contract, ckpt=ckpt if a.mode == "policy" else None, mode=a.mode)
    loop.slew = False                   # 批量模式：指令一步到位（slew 关，parity 把关）
    loop.set_cmd("lin_vel_x", a.vx)
    loop.set_cmd("ang_vel_yaw", a.yaw)
    loop.set_cmd("base_height", a.h)
    if a.vy:
        loop.set_cmd("lin_vel_y", a.vy)  # 契约里 lin_vel_y 无通道时保持 0（四轮不用侧移）

    if a.viewer:
        loop.attach_viewer(a.speed)

    alive = loop.warmup(a.warmup)
    n_steps = int(round(a.dur / contract["control_dt"]))
    for _ in range(n_steps):
        if not alive:
            break
        loop.step(record=True)
        if loop.terminated:
            break
        alive = loop.sync_viewer()

    summary = loop.summary(dur_target=a.dur)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if a.save:
        out = loop.dump(out=a.out, summary=summary)
        print(f"[saved] {out}")
    if loop.viewer is not None:
        # GLFW/NVIDIA 在解释器退出阶段会段错误（模型/窗口已正常关过），直接硬退避开析构。
        loop.close_viewer()
    return 0 if loop.terminated is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
