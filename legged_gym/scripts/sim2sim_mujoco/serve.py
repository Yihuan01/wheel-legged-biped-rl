# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : serve.py
# Purpose   : 控制台 MuJoCo 引擎子进程入口（wbr_bench 按 engine=mujoco 用 mujoco env
#             拉起）：读 spec.json → 单策略 SimLoop 或五槽 uni 链 → mj_bridge + 可选 viewer。
#             用法：python -m legged_gym.scripts.sim2sim_mujoco.serve --spec <spec.json>
#             spec.wmode: four|two|jump|lie 单槽；uni 五槽链（spec.start=2w|4w）
# ==============================================================================

import argparse
import json
import os
import time

# 【CPU/响应】必须在任何 torch 模块被导入、任何并行工作开始前设置：单环境小网络推理
#   用满 OpenMP/MKL 线程池会占 700%+ CPU，且让主线程抢不到时间片（SIGINT/桥 stop 迟钝）。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from legged_gym.scripts.sim2sim_mujoco import mj_bridge
from legged_gym.scripts.sim2sim_mujoco.sim_loop import PKG, ROOT, SimLoop, load_contract, resolve_model


def _resolve_spec_path(p, default=None):
    p = p or default
    if p and not os.path.isabs(p):
        p = os.path.join(ROOT, p)
    return p


class MjTelWriter:
    """【绘图页接 MuJoCo 遥测（P4）】把 SimLoop.rows 增量落成 plot02 口径的 wbr_tel_mj_*.tsv，
    写进 play02 的遥测目录（WBR_PLAY_DATA 或 <repo>/legged_gym/scripts/wbr_play_data）——
    bench_plot 的 PATTERNS（wbr_tel_*.tsv）直接命中，绘图页数据层零改动。
    缺的 Isaac 列（关节力矩/act_h/离地等）不伪造：read_window 只画有数据的系列，缺面板留空。
    meta 只写 joints=（空）⇒ 不生成 Isaac 的关节面板；time_s/body_z/done 为 Index 必需列。"""

    COLS = ["time_s", "body_z", "done", "vz_body", "vx", "vy", "wz", "x", "y",
            "tgt_vx", "act_vx", "tgt_yaw", "act_yaw", "tgt_roll", "act_roll", "tgt_h",
            "dbg_ph", "wheel_l", "wheel_r", "tau_max", "tip"]

    def __init__(self, loop, ckpt=None):
        d = os.environ.get("WBR_PLAY_DATA") or os.path.join(ROOT, "legged_gym/scripts/wbr_play_data")
        os.makedirs(d, exist_ok=True)
        self.path = os.path.join(d, "wbr_tel_mj_%s.tsv" % time.strftime("%Y%m%d-%H%M%S"))
        self.n = 0
        self._last = 0.0
        it = os.path.splitext(os.path.basename(ckpt or ""))[0].replace("model_", "")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("#engine=mujoco\n")
            f.write("#task=%s\n" % str(loop.c.get("task") or "wbr_mj"))
            f.write("#dt=%s\n" % loop.c.get("control_dt", 0.01))
            if ckpt:
                f.write("#model_dir=%s\n" % os.path.dirname(ckpt))
                f.write("#model_iter=%s\n" % (it or "?"))
            f.write("#joints=\n")
            f.write("\t".join(self.COLS) + "\n")
        print(f"[tel] 绘图遥测：{self.path}", flush=True)

    def sync(self, loop, terminated=False, force=False):
        """追加 rows[self.n:]（0.5 s 节流；force=True 用于收尾，含 done 标记）。"""
        if not loop.rows or self.n >= len(loop.rows):
            return
        if not force and time.time() - self._last < 0.5:
            return
        self._last = time.time()
        n_end = len(loop.rows)
        with open(self.path, "a", encoding="utf-8") as f:
            for i in range(self.n, n_end):
                r = loop.rows[i]
                last = (i == n_end - 1)
                f.write("\t".join(str(v) for v in (
                    r["t"], r["z"], 1 if (terminated and last) else 0,
                    r.get("vz", 0.0), r["vx"], r["vy"], r["wz"], r["x"], r["y"],
                    r["cmd_vx"], r["vx"], r["cmd_yaw"], r["wz"], r["cmd_roll"], r["roll"],
                    r["cmd_h"], r.get("dbg_ph", 0), r["wheel_l"], r["wheel_r"], r["tau_max"], r["tip"],
                )) + "\n")
        self.n = n_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="wbr_bench 落盘的 spec.json")
    a = ap.parse_args()
    with open(a.spec, encoding="utf-8") as f:
        spec = json.load(f)

    wmode = str(spec.get("wmode") or "four")
    record = bool(spec.get("record", True))
    dur = float(spec.get("dur", 0) or 0)

    # ---------------------------------------------------------------- 五槽 uni 链
    if wmode == "uni":
        from legged_gym.scripts.sim2sim_mujoco.deploy.uni_chain import UniChain
        # 控制台＝事件驱动：槽位停住等 /api/action 指令；`dur` 是整个会话的总保险超时
        _cmds = spec.get("cmds") or {}
        _jump = float(_cmds.get("jump_height", 0.10))
        _vx = float(_cmds.get("lin_vel_x", 0.0))     # 默认 0：启动即静止，等面板给速度
        chain = UniChain(start=str(spec.get("start") or "2w"),
                         h_jump=_jump if _jump > 0 else 0.10,
                         vx=_vx if _vx else 0.0,
                         interactive=True,
                         allow_rise=bool(spec.get("allow_rise", False)),
                         terrain=bool(spec.get("terrain", False)))
        loop = chain.loop
        loop.slew = False
        tel = MjTelWriter(loop) if record else None
        # 只把「面板实时下调的指令」写进 cmd_target；未指定的通道保留链内各槽的默认值。
        # ⚠ 不能对 spec.cmds 里每个 key 都 set_cmd：配置页的 base_height 是**槽位初值**，
        # 链各槽有自己的 h（两轮 0.25 / 四轮 0.19），全量覆盖会让两轮槽一开始就站在 0.19。
        cmds = spec.get("cmds") or {}
        # 链**启动一律静止**（vx=0）：配置页的 Vx 是历史值/预设，启动就继承会让机器人一上来自己走。
        # 运行中拖滑条 / yaw / roll 仍实时生效（见 mj_bridge._cmd）。
        for key in ("ang_vel_yaw", "ang_roll"):
            if key in cmds:
                loop.set_cmd(key, float(cmds[key]))
        loop.set_cmd("lin_vel_x", 0.0)
        chain.vx = 0.0
        loop.snap_cmds()
        httpd = mj_bridge.install(loop, chain=chain)
        if httpd is None:
            print("❌ [serve] 控制桥未起来，退出", flush=True)
            return 1
        if spec.get("window", True):
            loop.attach_viewer(float(spec.get("speed", 1.0)))
        n_steps = int(round(dur / float(chain.c2["control_dt"]))) if dur > 0 else None
        print(f"▶️  [serve] engine=mujoco wmode=uni interactive=ON start={spec.get('start', '2w')} "
              f"terrain={'wbr_p8_demo' if spec.get('terrain') else '平地'} "
              f"dur={'∞（会话总保险：未设）' if n_steps is None else dur}s record={record} "
              f"— 槽位停住等面板指令", flush=True)
        try:
            while True:
                if not chain.step(record=record):
                    print("⛔ [serve] 链结束：" + str(loop.terminated or "DONE"), flush=True)
                    break
                if tel:
                    tel.sync(loop)
                if n_steps is not None and len(loop.rows) >= n_steps:
                    break
                if not loop.tick():
                    print("🚪 [serve] viewer 已关闭", flush=True)
                    break
        except KeyboardInterrupt:
            print("🛑 [serve] 收到停止信号", flush=True)
        finally:
            summary = chain.summary()
            if tel:
                tel.sync(loop, terminated=bool(loop.terminated), force=True)
            if record and loop.rows:
                out = loop.dump(summary=summary)
                print(f"[saved] {out}", flush=True)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        loop.close_viewer()
        return 0 if loop.terminated is None else 1

    # ---------------------------------------------------------------- 单槽策略
    contract = load_contract(_resolve_spec_path(spec.get("contract")))
    mjcf = _resolve_spec_path(spec.get("mjcf"), os.path.join(PKG, "assets/mjcf/wbr_v2.xml"))
    ckpt = resolve_model(contract, spec.get("release"), spec.get("model"))
    assert os.path.isfile(ckpt), f"模型不存在：{ckpt}"
    assert os.path.isfile(mjcf), f"MJCF 不存在：{mjcf}"

    # 注意：spec.mode 是控制台的 run 分类名（落 run 目录用），不是 SimLoop 的 policy/zero
    # 语义——控制台永远跑策略（曾经把 "mujoco" 传进去导致零动作沉降、机器人不走）。
    loop = SimLoop(mjcf, contract, ckpt=ckpt, mode="policy")
    loop.slew = True                       # 控制台模式：滑条指令限速渐进（vx 0.8 m/s²、h 0.15 m/s）
    ranges = loop.c["command_ranges"]
    for key in ("lin_vel_x", "ang_vel_yaw", "base_height", "ang_roll"):
        v = float((spec.get("cmds") or {}).get(key, 0.0))
        rg = ranges.get(key)
        if rg:
            v = min(max(v, float(rg[0])), float(rg[1]))
        loop.set_cmd(key, v)
    loop.snap_cmds()                       # 冷启动指令一步到位（限速只用于运行中滑条）

    httpd = mj_bridge.install(loop)
    if httpd is None:
        print("❌ [serve] 控制桥未起来，退出", flush=True)
        return 1

    if spec.get("window", True):
        loop.attach_viewer(float(spec.get("speed", 1.0)))

    n_steps = int(round(dur / float(contract["control_dt"]))) if dur > 0 else None
    tel = MjTelWriter(loop, ckpt=ckpt) if record else None
    loop.warmup(1)
    print(f"▶️  [serve] engine=mujoco wmode={wmode} model={os.path.basename(ckpt)} "
          f"mjcf={os.path.basename(mjcf)} dur={'∞' if n_steps is None else dur}s record={record}",
          flush=True)

    summary = None
    try:
        while True:
            loop.step(record=record)
            if tel:
                tel.sync(loop)
            if loop.terminated:
                print(f"⛔ [serve] 终止判定：{loop.terminated}", flush=True)
                break
            if n_steps is not None and loop.steps_done - 1 >= n_steps:
                break
            if not loop.tick():
                print("🚪 [serve] viewer 已关闭", flush=True)
                break
    except KeyboardInterrupt:
        print("🛑 [serve] 收到停止信号", flush=True)
    finally:
        summary = loop.summary(dur_target=dur)
        if tel:
            tel.sync(loop, terminated=bool(loop.terminated), force=True)
        if record and loop.rows:
            out = loop.dump(summary=summary)
            print(f"[saved] {out}", flush=True)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    loop.close_viewer()                    # 窗口模式下硬退（内部已 flush）；无窗口则正常返回
    return 0 if loop.terminated is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
