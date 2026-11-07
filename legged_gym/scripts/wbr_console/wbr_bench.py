# ==============================================================================
# Date      : 2026-10-26
# Author    : SYH
# File      : wbr_bench.py
# Purpose   : 控制台后端：静态站点 + 试验台 API（模型扫描 / 预检 / 场景 / 启停 / 绘图）+
# 控制桥代理，纯标准库实现。
# ==============================================================================

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_store as store
import bench_models as models
import bench_plot as plot

try:      # 原生窗口停靠（libX11 + ctypes）；无 X 显示时降级为不可用
    import xwin
except Exception:  # pragma: no cover
    xwin = None

STATIC = HERE
DEFAULT_PORT = 8770
DEFAULT_BRIDGE_PORT = 8771
REPO_ROOT = store.REPO_ROOT
# 引擎解释器：Isaac 仿真必须 isaacgym01（libpython3.8），MuJoCo 纯 CPU 用 mujoco env。
# 两者都允许环境变量覆盖（后端自身可能跑在系统 python，不能拿 sys.prefix）。
PY_ISAAC = os.environ.get("WBR_PYTHON") or models.env_python()
PY_MUJOCO = os.environ.get("WBR_PY_MUJOCO") or os.path.expanduser("~/miniconda3/envs/mujoco/bin/python")
RUN_SCRIPT = os.path.join(HERE, "wbr_bench_run.py")
MUJOCO_MODULE = "legged_gym.scripts.sim2sim_mujoco.serve"
SIM2SIM_DIR = os.path.join(REPO_ROOT, "legged_gym/scripts/sim2sim_mujoco")

# 【Sim2Sim P1】引擎差异收敛成一张表：启动 argv、LD_LIBRARY_PATH、训练拦截、预检都查这里，
# 不散落 if engine==。isaac 项内容＝原行为逐字保留；新引擎只加表项。
ENGINES = {
    "isaac": {"python": lambda: PY_ISAAC, "script": RUN_SCRIPT,
              "ld_isaac": True, "block_on_training": True},
    "mujoco": {"python": lambda: PY_MUJOCO, "module": MUJOCO_MODULE,
               "ld_isaac": False, "block_on_training": False},
}


def engine_of(spec):
    e = (spec or {}).get("engine", "isaac")
    return e if e in ENGINES else "isaac"      # 未知引擎按 isaac 兜底（向后兼容）

# 四槽的默认 obs/act（选不中时的兜底校验基准）
SLOT_SPEC = {
    "fourw": {"stage": "① 四轮行驶", "obs": 164, "act": 6},
    "standup": {"stage": "② 起立", "obs": 173, "act": 6},
    "w3": {"stage": "③ 两轮站立/行走", "obs": 161, "act": 6},
    "lie": {"stage": "④ 趴下", "obs": 173, "act": 6},
    # 【P8 五槽】跳跃槽（事件级）：两轮行走网常驻，跳跃事件窗（env.ph>0）内由跳跃网接管。
    # 161 维与 w3 槽同 perm（W3_ACT_PERM）；空槽=不加载（两轮无跳跃能力，预检按就绪计）。
    "jump": {"stage": "⑤ 跳跃", "obs": 161, "act": 6},
}
# 单策略：task → 期望 obs（用于校验 checkpoint 与 task 是否匹配）
# 地形任务 → 高度图（preflight 用来确认"这次真的会加载地形"）
TERRAIN_FILES = {
    "wbr_terrain": "legged_gym/envs/wbr/terrain/release/wbr_rand_curve.npz",
    "wbr_demo": "legged_gym/envs/wbr/terrain/release/wbr_p8_demo.npz",
    "wbr_combo": "legged_gym/envs/wbr/terrain/release/wbr_p8_combo_L3.npz",
    "wbr_elements": "legged_gym/envs/wbr/terrain/release/wbr_p8_L3.npz",
}
TASK_OBS = {"wbr_flat": 161, "wbr_jump": 161, "wbr_flat4": 164, "wbr_switch": 173, "wbr_terrain": 161,
            "wbr_demo": 161, "wbr_combo": 161, "wbr_elements": 161}

_proc = None          # 仿真子进程
_proc_lock = threading.Lock()
_run_meta = {}        # 当前 run 的信息


# ------------------------------------------------------------------ 进程管理
def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def training_processes():
    """找训练进程（不依赖 psutil）：扫 /proc 的 cmdline。"""
    out = []
    me = os.getpid()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "legged_gym.scripts.train" in cmd or re.search(r"\btrain\.py\b", cmd):
            out.append({"pid": pid, "cmd": cmd.strip()[:200]})
    return out


def gpu_mem():
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            timeout=3).decode().strip().split("\n")[0]
        used, total = [float(x) for x in o.split(",")]
        return {"used": round(used / 1024, 1), "total": round(total / 1024, 1)}
    except Exception:
        return {"used": None, "total": None}


def sim_status():
    global _proc
    with _proc_lock:
        alive = _proc is not None and _proc.poll() is None
        info = {
            "status": "running" if alive else "idle",
            "statusText": "仿真运行中" if alive else "仿真未启动",
            "pid": _proc.pid if alive else None,
            "run": _run_meta.get("id") if alive else None,
            "gpu": gpu_mem(),
            "training": training_processes(),
        }
    return info


def build_launch(spec):
    """按 engine 组装仿真子进程的 argv 与环境（纯函数，便于 isaac 路径快照回归）。
    isaac 分支与旧 _launch 逐字一致（脚本启动 + isaacgym 的 LD_LIBRARY_PATH）。"""
    eng = ENGINES[engine_of(spec)]
    py = eng["python"]()
    env = dict(os.environ)
    env["WBR_BRIDGE_PORT"] = str(spec.get("bridge_port", DEFAULT_BRIDGE_PORT))
    env["PYTHONUNBUFFERED"] = "1"
    if eng["ld_isaac"]:
        # 关键：libpython 在 **子进程所用解释器** 的 lib 下（后端可能跑在系统 python，
        # 不能拿 sys.prefix）；否则 import isaacgym 时找不到 libpython3.8.so.1.0。
        conda_lib = os.path.join(os.path.dirname(os.path.dirname(py)), "lib")
        env["LD_LIBRARY_PATH"] = conda_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        env["PYTHONPATH"] = REPO_ROOT + os.pathsep + os.path.join(REPO_ROOT, "rsl_rl-1.0.2") + os.pathsep + env.get("PYTHONPATH", "")
    else:
        env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    if eng.get("script"):
        argv = [py, eng["script"], "--spec"]
    else:
        argv = [py, "-m", eng["module"], "--spec"]
    return argv, env


def _launch(spec):
    """起仿真子进程，落 argv 快照 + console.log。"""
    global _proc, _run_meta
    run_dir, run_id = store.new_run_dir(spec.get("mode", "run"))
    spec_path = os.path.join(run_dir, "spec.json")
    with open(spec_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "argv.json"), "w", encoding="utf-8") as fh:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": spec.get("mode"),
                   "spec": spec}, fh, ensure_ascii=False, indent=2)

    argv, env = build_launch(spec)
    # 【Sim2Sim P3】停靠目标窗口名随引擎切换（isaac gym 默认不变；MuJoCo viewer 窗口按标题找）
    if xwin is not None:
        try:
            xwin.set_dock_match("mujoco" if engine_of(spec) == "mujoco" else "isaac gym")
        except Exception:
            pass
    console_path = os.path.join(run_dir, "console.log")
    con = open(console_path, "wb")
    proc = subprocess.Popen(argv + [spec_path],
                            cwd=os.path.dirname(HERE), env=env,
                            stdout=con, stderr=subprocess.STDOUT, start_new_session=True)
    with _proc_lock:
        _proc = proc
        _run_meta = {"id": run_id, "dir": run_dir, "spec": spec, "console": console_path,
                     "engine": engine_of(spec)}
    print(f"▶️  [bench] 起仿真 pid={proc.pid} run={run_id} engine={engine_of(spec)}")
    return {"ok": True, "pid": proc.pid, "run": run_id, "engine": engine_of(spec)}


def _stop():
    """停仿真：先走桥的确定性停止（loop.stop_requested，主循环 ≤10 ms 退出并 flush），
    再发 SIGINT 作兜底；等待链从 21 s 缩到 ~1.5 s。Isaac 侧仍走原信号路径（play02 收尾）。"""
    global _proc
    with _proc_lock:
        proc = _proc
        meta = dict(_run_meta)
    if proc is None or proc.poll() is not None:
        return {"ok": True, "note": "没有在跑的仿真"}
    eng = meta.get("engine")
    if eng == "mujoco":
        try:
            bridge_post("/api/stop", {})          # 置 stop_requested；桥自己延迟发 SIGINT 兜底
        except Exception:
            pass
        # 正常收尾（finally flush + dump）通常 <0.3 s；给 2 s 足矣
        for _ in range(20):
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except Exception:
                pass
            for _ in range(10):
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    else:
        # Isaac：原路径（play02 finally 收尾较慢，保留宽裕等待）
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except Exception:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
        for _ in range(80):
            if proc.poll() is not None:
                break
            time.sleep(0.15)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except Exception:
                pass
            for _ in range(60):
                if proc.poll() is not None:
                    break
                time.sleep(0.15)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    with _proc_lock:
        _proc = None
    return {"ok": True}


# ------------------------------------------------------------------ 桥代理
def bridge_get(path, timeout=2.0):
    url = f"http://127.0.0.1:{_bridge_port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        return {"bridgeDown": True, "error": str(e)}


def bridge_post(path, obj, timeout=2.0):
    url = f"http://127.0.0.1:{_bridge_port}{path}"
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        return {"bridgeDown": True, "error": str(e)}


# ------------------------------------------------------------------ 预检
def _query_param(query, key):
    """从 '?a=1&name=xx' 里取一个参数（URL 解码）。"""
    from urllib.parse import parse_qs
    qs = parse_qs(query.split("?", 1)[1] if "?" in query else "")
    vals = qs.get(key)
    return vals[0] if vals else None


def _save_pngs(body):
    """把前端导出的 PNG（base64）存到 ~/Downloads/<dir>/，与 plot02 的保存行为一致。"""
    import base64
    out_dir = os.path.join(os.path.expanduser("~/Downloads"), store.safe_name(body.get("dir") or "WBR遥测"))
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for item in body.get("files") or []:
        name = store.safe_name(item.get("name") or "panel")
        if not name.lower().endswith(".png"):
            name += ".png"
        try:
            with open(os.path.join(out_dir, name), "wb") as fh:
                fh.write(base64.b64decode(item.get("b64") or ""))
            n += 1
        except Exception:
            pass
    return {"ok": True, "dir": out_dir, "n": n}


def _tel_logs():
    """列 wbr_play_data 下的遥测文件（新→旧）。
    新前缀 Isaac_1_/Isaac_2_/MJ_ 不含 "_tel_" —— 过滤改为
    前缀白名单（旧 "_tel_" in n 会漏掉全部新日志 ⇒ 运行页抽屉列表不更新）。"""
    d = os.path.normpath(os.path.join(HERE, "..", "wbr_play_data"))
    out = []
    if os.path.isdir(d):
        _PREFIX = ("Isaac_1_", "Isaac_2_", "MJ_", "wbr_tel_", "wbr_switch_tel_")
        names = [n for n in os.listdir(d)
                 if n.endswith(".tsv") and n.startswith(_PREFIX)]
        names.sort(key=lambda n: os.path.getmtime(os.path.join(d, n)), reverse=True)
        for i, n in enumerate(names):
            full = os.path.join(d, n)
            out.append({"name": n, "sizeMB": round(os.path.getsize(full) / 1e6, 2), "current": i == 0})
    return out




# ---------------------------------------------------------------- 预检辅助（2026-11-02）
def _terrain_dims(npz_path, num_rows, num_cols, length, width, h_scale, border):
    """按 Terrain.__init__ 的口径算"配置推出的高度图尺寸"。"""
    import math
    try:
        npx = int(round(length / h_scale)) * int(num_rows)
        npy = int(round(width / h_scale)) * int(num_cols)
        b = int(round(border / h_scale))
        return (2 * b + npx, 2 * b + npy)
    except Exception:
        return None


def _check_terrain_geometry(single, items):
    """检查"会读哪张高度图 + 尺寸是否匹配任务 cfg"。

    ⚠ 不能从高度图 meta 推 cfg：_load_height_field 是拿**cfg** 去算出 tot_rows/tot_cols
      再和文件比，所以 cfg 与文件不一致时它直接抛错（011614 那次就是 840×580 vs 750×580）。
      这里用任务 cfg 的几何值与 height_field_file 的尺寸对账，提前拦下。
    """
    import json as _json
    import numpy as _np
    task = single.get("task")
    _tf = TERRAIN_FILES.get(task)
    if not _tf:
        return
    full = _tf if os.path.isabs(_tf) else os.path.join(REPO_ROOT, _tf)
    if not os.path.isfile(full):
        items.append({"level": "bad", "text": f"地形：{os.path.basename(_tf)}（文件不存在）"})
        return
    # 任务 cfg 的几何值必须 import isaacgym，预检在服务端进程里不做 —— 用文件 meta 反查
    # "有没有 cfg 会推导出与文件不符的尺寸"，只在已知任务上做静态表校验。
    try:
        d = _np.load(full, allow_pickle=False)
        hf = d["height_field_raw"]
        meta = _json.loads(str(d["meta"])) if "meta" in d else {}
        expect = _terrain_dims(full, meta.get("num_rows"), meta.get("num_cols"),
                               meta.get("env_len"), meta.get("env_wid"),
                               meta.get("h_scale"), meta.get("border"))
        if expect and tuple(hf.shape) != tuple(expect):
            items.append({"level": "bad",
                          "text": f"地形：{os.path.basename(_tf)} 文件 {hf.shape} 与 meta 推出 {expect} 不符"})
        else:
            items.append({"level": "ok",
                          "text": f"地形：{os.path.basename(_tf)} {int(meta.get('num_rows', 0))}×"
                                  f"{int(meta.get('num_cols', 0))} 格 / {meta.get('env_len')}×{meta.get('env_wid')} m"})
    except Exception as e:
        items.append({"level": "warn", "text": f"地形：读取 {os.path.basename(_tf)} 失败（{type(e).__name__}）"})


# 【P8 五槽·点击校准】页面 CSS 坐标 ↔ 屏幕设备坐标 配对（/api/point 采集）。
# 持久化到 wbr_bench_store/calibration.json：8770 重启不丢（校准一次长期有效；
# 浏览器挪屏/改缩放后再点几下即自动更新）。
_PAIRS = []
_CAL_FILE = None


def _cal_load():
    global _CAL_FILE
    try:
        _CAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "wbr_bench_store", "calibration.json")
        if os.path.isfile(_CAL_FILE):
            with open(_CAL_FILE, encoding="utf-8") as _fh:
                _PAIRS[:] = [tuple(p) for p in json.load(_fh).get("pairs", [])]
    except Exception as _e:
        print(f"[bench] 校准数据加载失败：{_e}")


def _cal_save():
    try:
        _d = os.path.dirname(_CAL_FILE)
        os.makedirs(_d, exist_ok=True)
        with open(_CAL_FILE, "w", encoding="utf-8") as _fh:
            json.dump({"pairs": _PAIRS[-40:]}, _fh)
    except Exception as _e:
        print(f"[bench] 校准数据保存失败：{_e}")


def _solve_xform():
    """RANSAC-lite 解「页面 CSS → 屏幕设备」变换（dev ≈ css·s + origin）。

    为什么不能裸最小二乘：键盘触发/特殊点击的配对里指针不在点击处（一对坏点），
    全体最小二乘直接被拖飞（11-04 实测 x=15928 ⇒ 飞出屏幕）。RANSAC 每对做种子、
    数内点（残差 <40 设备px），取内点最多的解再对内点精化。"""
    if len(_PAIRS) < 2:
        return None
    pts = list(_PAIRS)[-40:]
    best, best_in = None, 0
    for i in range(len(pts)):
        c1x, c1y, d1x, d1y = pts[i]
        for j2 in range(len(pts)):
            if j2 == i:
                continue
            c2x, c2y, d2x, d2y = pts[j2]
            dcx, dcy = c2x - c1x, c2y - c1y
            if abs(dcx) < 30 and abs(dcy) < 30:
                continue                              # 种子太近，解不稳
            if abs(dcx) >= 30:
                sx = (d2x - d1x) / dcx
                if not (0.2 < sx < 4.0):
                    continue
            if abs(dcy) >= 30:
                sy = (d2y - d1y) / dcy
                if not (0.2 < sy < 4.0):
                    continue
            # 统一 scale（DPR 均匀假设）：按最小二乘权重的点积
            den = dcx * dcx + dcy * dcy
            if den < 1e-6:
                continue
            sc = ((d2x - d1x) * dcx + (d2y - d1y) * dcy) / den
            if not (0.2 < sc < 4.0):
                continue
            ox = d1x - c1x * sc
            oy = d1y - c1y * sc
            inl = sum(1 for cx, cy, dx, dy in pts
                      if abs(cx * sc + ox - dx) < 40 and abs(cy * sc + oy - dy) < 40)
            if inl > best_in:
                best_in, best = inl, (sc, ox, oy)
    if not best or best_in < 2:
        return None
    sc, ox, oy = best
    # 内点精化（最小二乘，仅内点）
    inl = [(cx, cy, dx, dy) for cx, cy, dx, dy in pts
           if abs(cx * sc + ox - dx) < 40 and abs(cy * sc + oy - dy) < 40]
    _n = len(inl)
    mcx = sum(p[0] for p in inl) / _n
    mcy = sum(p[1] for p in inl) / _n
    mdx = sum(p[2] for p in inl) / _n
    mdy = sum(p[3] for p in inl) / _n
    num = den = 0.0
    for cx, cy, dx, dy in inl:
        num += (cx - mcx) * (dx - mdx) + (cy - mcy) * (dy - mdy)
        den += (cx - mcx) ** 2 + (cy - mcy) ** 2
    if den > 1e-6:
        sc2 = num / den
        if 0.2 < sc2 < 4.0:
            sc = sc2
    ox = mdx - mcx * sc
    oy = mdy - mcy * sc
    return {"s": sc, "ox": ox, "oy": oy, "n": best_in}


_cal_load()


def preflight(spec):
    """按引擎分发预检（isaac 分支＝原逻辑逐字保留）。"""
    if engine_of(spec) == "mujoco":
        return _preflight_mujoco(spec)
    return _preflight_isaac(spec)


def _mujoco_contract(path=None):
    """读部署契约 JSON（纯 stdlib；后端进程不 import sim2sim_mujoco——那会拖进 torch/mujoco）。"""
    p = path or os.path.join(SIM2SIM_DIR, "assets/deploy_contract.json")
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


# MuJoCo 引擎可用的部署契约（相对 REPO_ROOT；键＝前端模式选择 wmode）
MUJOCO_CONTRACTS = {
    "four": "legged_gym/scripts/sim2sim_mujoco/assets/deploy_contract.json",
    "two": "legged_gym/scripts/sim2sim_mujoco/assets/deploy_contract_two.json",
    "jump": "legged_gym/scripts/sim2sim_mujoco/assets/deploy_contract_two.json",
    "lie": "legged_gym/scripts/sim2sim_mujoco/assets/deploy_contract_lie.json",
    "uni": "legged_gym/scripts/sim2sim_mujoco/assets/deploy_contract.json",
    "uni4": "legged_gym/scripts/sim2sim_mujoco/assets/deploy_contract.json",
}

# 五槽链固定用到的四份交付（契约文件, release, model, 标签）
MUJOCO_CHAIN_SLOTS = (
    ("legged_gym/scripts/sim2sim_mujoco/assets/deploy_contract.json",
     "releases/P7F3e_yaw1_dr15", "model_11800.pt", "四轮行驶 164"),
    ("legged_gym/scripts/sim2sim_mujoco/assets/deploy_contract_two_v2.json",
     "releases/P8W7_push01_fric05", "model_2800.pt", "两轮行走 161"),
    ("legged_gym/scripts/sim2sim_mujoco/assets/deploy_contract_two_v2.json",
     "releases/P8J3d_wheelfollow_lite", "model_5650.pt", "两轮跳跃 161"),
    ("legged_gym/scripts/sim2sim_mujoco/assets/deploy_contract_lie.json",
     "releases/P7L_liedown", "model_300.pt", "趴下 173"),
)


def _mujoco_contract_of(spec=None):
    """spec.contract（相对 REPO_ROOT 或绝对）→ 契约文件路径；缺省四轮（向后兼容旧 spec）。"""
    p = (spec or {}).get("contract") or MUJOCO_CONTRACTS["four"]
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _preflight_mujoco(spec):
    """MuJoCo 预检：解释器 / MJCF / 权重与契约对账 / 桥端口。纯 CPU，不查 CUDA 与训练进程。"""
    import socket
    items = []
    if str(spec.get("wmode")) in ("uni", "uni4"):
        terrain = bool(spec.get("terrain"))
        items = _preflight_mujoco_chain(spec)
        if str(spec.get("start") or "2w") == "4w":
            items.append({"level": "ok", "text": "四轮起步：起立走 RISE（P7T2 脚本真起立，2.9 s 完成，无瞬移）"})
        else:
            items.append({"level": "ok", "text": "两轮起步：槽位停住等面板指令（起立/趴下/跳跃/回四轮/复位），不自动放完" +
                          ("；地形模式自动播放链跳过跳跃槽" if terrain else "")})
        items.append({"level": "info", "text": "事件驱动链：时长＝会话总保险超时（0＝不限时）；每槽停住等指令"})
        return items
    if os.path.isfile(PY_MUJOCO) and os.access(PY_MUJOCO, os.X_OK):
        items.append({"level": "ok", "text": f"MuJoCo 解释器：{PY_MUJOCO}"})
    else:
        items.append({"level": "bad", "text": f"MuJoCo 解释器不可用：{PY_MUJOCO}（可用 WBR_PY_MUJOCO 覆盖）"})

    contract = _mujoco_contract(_mujoco_contract_of(spec))
    src = contract.get("sources") or {}
    mode_cn = "两轮" if contract.get("mode") == "two_wheel" else "四轮"
    mjcf = spec.get("mjcf") or contract.get("mjcf") or os.path.join(SIM2SIM_DIR, "assets/mjcf/wbr_v2.xml")
    if not os.path.isabs(mjcf):
        mjcf = os.path.join(REPO_ROOT, mjcf)
    if os.path.isfile(mjcf):
        items.append({"level": "ok", "text": f"MJCF：{os.path.basename(mjcf)} · sha {_sha256_file(mjcf)[:8]}"})
    else:
        items.append({"level": "bad", "text": f"MJCF 不存在：{mjcf}"})

    rel = spec.get("release") or src.get("release")
    name = spec.get("model") or src.get("release_model")
    if rel and not os.path.isabs(rel):
        rel = os.path.join(REPO_ROOT, rel)
    ckpt = os.path.join(rel, name) if (rel and name and not os.path.isabs(name)) else (name or "")
    if ckpt and os.path.isfile(ckpt):
        items.append({"level": "ok", "text": f"权重：{name} · {rel}"})
        dims = models.probe_one(ckpt)
        if dims:
            obs, act = dims
            want = contract.get("obs_dim")
            if want and obs != int(want):
                items.append({"level": "bad", "text": f"obs {obs} 与契约 {want} 不符——不是{mode_cn} {want} 维策略"})
            else:
                items.append({"level": "ok", "text": f"obs {obs} / act {act} 与契约一致"})
        else:
            items.append({"level": "warn", "text": "读不出权重维度（torch 探测失败，不强拦）"})
        cfg = os.path.join(rel or "", "config.json")
        want_sha = src.get("release_config_sha256")
        if os.path.isfile(cfg) and want_sha:
            if _sha256_file(cfg) == want_sha:
                items.append({"level": "ok", "text": "release config sha 与契约一致"})
            else:
                items.append({"level": "warn", "text": "release config sha 与契约不一致——S0 对账结论可能不适用于该权重"})
    else:
        items.append({"level": "bad", "text": f"权重不存在：{ckpt or '（未指定）'}"})

    port = int(spec.get("bridge_port", DEFAULT_BRIDGE_PORT))
    s = socket.socket()
    # 真桥是 ThreadingHTTPServer（allow_reuse_address=True，TIME_WAIT 不拦重启）——
    # 探测 socket 必须同语义，否则上一场仿真刚收尾会误报「端口被占用」。
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
        free = True
    except OSError:
        free = False
    finally:
        s.close()
    items.append({"level": "ok" if free else "bad", "text": f"桥端口 {port}：" + ("空闲" if free else "被占用")})
    items.append({"level": "info", "text": "MuJoCo 纯 CPU 运行，可与训练并行（不查 CUDA/训练进程）"})
    return items


def _preflight_mujoco_chain(spec):
    """五槽链预检：四份交付权重＋契约、MJCF、桥端口（链内权重固定，不吃 spec.release）。"""
    import socket
    items = []
    if os.path.isfile(PY_MUJOCO) and os.access(PY_MUJOCO, os.X_OK):
        items.append({"level": "ok", "text": f"MuJoCo 解释器：{PY_MUJOCO}"})
    else:
        items.append({"level": "bad", "text": f"MuJoCo 解释器不可用：{PY_MUJOCO}"})
    for cfile, rel, name, label in MUJOCO_CHAIN_SLOTS:
        cpath = os.path.join(REPO_ROOT, cfile)
        contract = _mujoco_contract(cpath)
        ckpt = os.path.join(REPO_ROOT, rel, name)
        if not os.path.isfile(ckpt):
            items.append({"level": "bad", "text": f"{label} 权重缺失：{rel}/{name}"})
            continue
        dims = models.probe_one(ckpt)
        want = contract.get("obs_dim")
        okdim = (not dims) or int(dims[0]) == int(want or -1)
        items.append({"level": "ok" if okdim else "bad",
                      "text": f"{label}：{name} · obs {dims[0] if dims else '?'}/{want} · 契约 "
                              + ("一致" if okdim else "不符")})
    terrain = bool(spec.get("terrain"))
    mjcf_name = "wbr_v2_demo.xml" if terrain else "wbr_v2.xml"
    mjcf = os.path.join(SIM2SIM_DIR, "assets/mjcf", mjcf_name)
    items.append({"level": "ok" if os.path.isfile(mjcf) else "bad",
                  "text": ("MJCF：" + mjcf_name + " · sha " + _sha256_file(mjcf)[:8]) if os.path.isfile(mjcf)
                          else "MJCF 缺失：" + mjcf_name})
    if terrain:
        meta = os.path.join(SIM2SIM_DIR, "assets/terrain/wbr_v2_demo.json")
        png = os.path.join(SIM2SIM_DIR, "assets/terrain/wbr_v2_demo.png")
        ok = os.path.isfile(meta) and os.path.isfile(png)
        items.append({"level": "ok" if ok else "bad",
                      "text": "复杂地形 wbr_p8_demo：高度图 " +
                              ("已就位（8-bit PNG，量化 ≈3.8 mm）" if ok else "资产缺失（先跑 make_terrain_asset.py）")})
        if ok:
            items.append({"level": "warn", "text": "跳跃策略（P8J3d）在复杂地形未验证：装填跳建议到中央十字走廊平地，或切回平地配置"})
    port = int(spec.get("bridge_port", DEFAULT_BRIDGE_PORT))
    sk = socket.socket()
    sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sk.bind(("127.0.0.1", port))
        free = True
    except OSError:
        free = False
    finally:
        sk.close()
    items.append({"level": "ok" if free else "bad", "text": f"桥端口 {port}：" + ("空闲" if free else "被占用")})
    if str(spec.get("start") or "2w") == "4w":
        items.append({"level": "ok", "text": "四轮起步的链含 P7T2 起立：4W→RISE（真起立）→IDLE2W，验收 C1×3 全过"})
    else:
        items.append({"level": "ok", "text": "两轮起步：行走→跳跃→趴下→四轮，四槽已验证零终止"})
    items.append({"level": "info", "text": "MuJoCo 纯 CPU 运行，可与训练并行（不查 CUDA/训练进程）"})
    return items


def _preflight_isaac(spec):
    items = []
    mode = spec.get("mode", "uni")
    if mode in ("switch", "uni"):
        ok = 0
        n_slots = 0
        dims = []
        for key, meta in SLOT_SPEC.items():
            n_slots += 1
            path = (spec.get("slots") or {}).get(key)
            # 起立槽支持相位脚本：值可写成 {"type":"script"}（无权重），
            # 此时该槽不查权重，按"已就绪"计（四格保留，仍显示 4/4）。字符串/字典两种都认。
            if key == "standup" and isinstance(path, dict) and str(path.get("type", "")).lower() == "script":
                ok += 1
                items.append({"level": "ok", "text": f"{meta['stage']}：相位脚本（P7T2，无需权重）"})
                dims.append("script")
                continue
            if not path:
                if key == "jump":
                    # 跳跃槽允许空（=两轮无跳跃能力的旧四槽形态），按就绪计
                    ok += 1
                    items.append({"level": "ok", "text": f"{meta['stage']}：未配置（无事件级跳跃网）"})
                    dims.append("-")
                    continue
                items.append({"level": "bad", "text": f"{meta['stage']}：未选择 checkpoint"})
                continue
            full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
            if not os.path.isfile(full):
                items.append({"level": "bad", "text": f"{meta['stage']}：文件不存在 {path}"})
                continue
            dims_ = models.probe_one(path)
            if dims_ is None:
                items.append({"level": "warn", "text": f"{meta['stage']}：读不出维度（{path}）"})
                continue
            obs, act = dims_
            dims.append(str(obs))
            if obs == meta["obs"] and act == meta["act"]:
                ok += 1
            else:
                items.append({"level": "bad",
                              "text": f"{meta['stage']}：维度不符（obs {obs}/act {act}，需 {meta['obs']}/{meta['act']}）"})
        items.insert(0, {"level": "ok" if ok == n_slots else "bad", "text": f"{ok}/{n_slots} 权重存在且维度匹配"})
        if dims:
            items.append({"level": "ok", "text": "obs 校验：" + " / ".join(dims) + " · act 6"})
    else:
        single = spec.get("single") or {}
        path = single.get("ckpt_path")
        if path and os.path.isfile(path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)):
            dims_ = models.probe_one(path)
            if dims_:
                obs, act = dims_
                task = single.get("task")
                want = TASK_OBS.get(task)
                if want is not None and obs != want:
                    items.append({"level": "bad",
                                  "text": f"单策略：obs {obs} 与 task={task}（需 {want}）不符，请改 task 或换 checkpoint"})
                else:
                    items.append({"level": "ok", "text": f"单策略 obs {obs} / act {act} · {os.path.basename(path)}"})
            else:
                items.append({"level": "warn", "text": "单策略：读不出维度"})
            # 地形任务：确认会读哪张高度图 + 尺寸对账（011614 那次就是这里没拦住）
            _check_terrain_geometry(single, items)
            if single.get("from_run_config"):
                items.append({"level": "info",
                              "text": "from_run_config 只回填奖励/指令参数（地形几何已锁定为任务自己的图）"})
        else:
            _p = path or "（空）"
            hint = ""
            if path:
                _run = os.path.dirname(path if os.path.isabs(path) else os.path.join(REPO_ROOT, path))
                if not os.path.isdir(_run):
                    hint = f"；run 目录不存在：{_run}"
                else:
                    _ms = sorted([f for f in os.listdir(_run) if f.startswith("model_") and f.endswith(".pt")])
                    if _ms:
                        hint = f"；该目录现有：{', '.join(_ms[-5:])}"
            items.append({"level": "bad", "text": f"单策略：checkpoint 不存在 {_p}{hint}"})

    t = training_processes()
    if t:
        items.append({"level": "warn", "text": f"检测到训练进程 PID {t[0]['pid']} → 启动将被拦截（可强制）"})
    else:
        items.append({"level": "ok", "text": "没有检测到训练进程，显卡可独占"})
    items.extend(_check_last_failure())
    items.extend(_check_cuda())
    g = gpu_mem()
    if g["used"] is not None:
        items.append({"level": "info", "text": f"显卡 {g['used']} / {g['total']} GB"})
    return items


def _check_last_failure():
    """最近一次单策略 run 若以异常退出，把最后一行错误提到预检里（不用翻控制台抽屉）。"""
    try:
        runs = store.list_runs()
    except Exception:
        return []
    for r in runs[:3]:
        try:
            log = store.tail_console(r["id"], max_bytes=8000)
        except Exception:
            continue
        if not log:
            continue
        if "Traceback" not in log:
            return []                      # 最近一次没报错 ⇒ 不提示
        lines = [ln.strip() for ln in log.strip().splitlines() if ln.strip()]
        err = next((ln for ln in reversed(lines)
                    if ln.startswith(("ValueError", "RuntimeError", "FileNotFoundError",
                                      "AttributeError", "KeyError", "TypeError",
                                      "IndexError", "ImportError"))), lines[-1][:160])
        return [{"level": "warn", "text": f"上次运行（{r['id']}）报错：{err}"}]
    return []


def _check_cuda():
    """CUDA 上下文能不能建起来（2026-11-02：笔记本休眠唤醒后出 Xid 154，nvidia-smi 正常但
    所有 CUDA 程序报 `CUDA unknown error` —— 只看显存会发现不了）。用探测缓存，代价 <200 ms。"""
    import subprocess
    import sys
    py = models.env_python()
    code = ("import torch;"
            "print('OK' if torch.cuda.is_available() else 'NO');"
            "\ntry:\n torch.zeros(1, device='cuda:0'); print('ALLOC_OK')\n"
            "except Exception as e:\n print('ALLOC_FAIL', str(e)[:80])")
    try:
        r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=20)
        out = (r.stdout or "") + (r.stderr or "")
        if "ALLOC_OK" in out:
            return [{"level": "ok", "text": "CUDA 可用（上下文创建成功）"}]
        if "ALLOC_FAIL" in out or "\nNO" in ("\n" + out) or out.startswith("NO"):
            return [{"level": "bad",
                     "text": "CUDA 不可用 —— 若 nvidia-smi 正常但这里失败，多半是休眠唤醒后 GPU 进入"
                             "错误态（Xid 154），需重启机器；驱动层报错见 journalctl -k | grep Xid"}]
        return [{"level": "warn", "text": f"CUDA 自检无结论（{out.strip()[:60]}）"}]
    except Exception as e:
        return [{"level": "warn", "text": f"CUDA 自检失败（{type(e).__name__}）"}]


def _mujoco_export():
    """导出图：用 mujoco env 跑 eval/plot_run.py --latest，把最新 run 的遥测画成 PNG。
    后端进程没有 matplotlib/mujoco —— 必须子进程（与仿真相同解释器）。"""
    if not os.path.isfile(PY_MUJOCO):
        return 400, {"error": f"MuJoCo 解释器不可用：{PY_MUJOCO}"}
    try:
        r = subprocess.run([PY_MUJOCO, "-m", "legged_gym.scripts.sim2sim_mujoco.eval.plot_run", "--latest"],
                           cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, PYTHONPATH=REPO_ROOT, PYTHONUNBUFFERED="1"))
    except subprocess.TimeoutExpired:
        return 500, {"error": "导出超时（120s）"}
    out = ""
    for ln in (r.stdout or "").splitlines():
        if ln.startswith("[saved] "):
            out = ln[len("[saved] "):].strip()
    if not out:
        tail = ((r.stderr or "") + (r.stdout or "")).strip().splitlines()
        return 500, {"error": "导出失败：" + (tail[-1] if tail else f"exit {r.returncode}")}
    png = os.path.join(REPO_ROOT, out) if not os.path.isabs(out) else out
    try:
        import base64
        with open(png, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return {"ok": True, "path": out, "dataUrl": "data:image/png;base64," + b64}
    except OSError as e:
        return 500, {"error": f"PNG 读取失败：{e}"}


def mujoco_manifest():
    """MuJoCo 配置页静态清单：MJCF 列表+sha、S0 parity 对账读数、契约要点。
    纯 stdlib 读文件——后端进程不 import sim2sim_mujoco（会拖进 torch/mujoco）。
    双模式：contracts/parities 按前端 wmode（four/two）各给一份；顶层 contract/parity
    保留＝四轮（旧字段，向后兼容）。"""
    mjcfs = []
    mj_dir = os.path.join(SIM2SIM_DIR, "assets/mjcf")
    if os.path.isdir(mj_dir):
        for n in sorted(os.listdir(mj_dir)):
            if n.endswith(".xml"):
                full = os.path.join(mj_dir, n)
                mjcfs.append({"name": n, "path": os.path.relpath(full, REPO_ROOT),
                              "mesh": "nomesh" not in n.lower(),
                              "sha256": _sha256_file(full)[:8]})

    def _read_json(p):
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
        return {}

    def _cinfo(c):
        src = c.get("sources") or {}
        return {"obs_dim": c.get("obs_dim"), "act_dim": c.get("act_dim"),
                "mode": c.get("mode"), "mjcf": c.get("mjcf"),
                "release": src.get("release"), "release_model": src.get("release_model"),
                "command_ranges": c.get("command_ranges")}

    def _pinfo(p):
        return {"passed": p.get("passed"), "gates": p.get("gates"),
                "scenarios": p.get("scenarios") or {}, "policy": p.get("policy")}

    gdir = os.path.join(SIM2SIM_DIR, "eval/golden")
    c4 = _mujoco_contract(os.path.join(SIM2SIM_DIR, "assets/deploy_contract.json"))
    c2 = _mujoco_contract(os.path.join(SIM2SIM_DIR, "assets/deploy_contract_two.json"))
    cl = _mujoco_contract(os.path.join(SIM2SIM_DIR, "assets/deploy_contract_lie.json"))
    src = c4.get("sources") or {}
    return {
        "mjcfs": mjcfs,
        "contracts": {"four": _cinfo(c4), "two": _cinfo(c2), "jump": _cinfo(c2), "lie": _cinfo(cl)},
        "parities": {"four": _pinfo(_read_json(os.path.join(gdir, "parity.json"))),
                     "two": _pinfo(_read_json(os.path.join(gdir, "parity_two.json")))},
        # —— 旧字段（四轮口径，保留兼容）——
        "parity": _pinfo(_read_json(os.path.join(gdir, "parity.json"))),
        "contract": {"obs_dim": c4.get("obs_dim"), "act_dim": c4.get("act_dim"),
                     "release": src.get("release"), "release_model": src.get("release_model"),
                     "command_ranges": c4.get("command_ranges")},
    }


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def log_message(self, *a):
        pass

    # ---------- 静态 ----------
    def _serve_static(self, rel):
        if rel in ("", "/", "/index.html"):
            rel = "app.html"
        rel = rel.lstrip("/")
        # 防目录穿越
        full = os.path.normpath(os.path.join(STATIC, rel))
        if not full.startswith(STATIC) or not os.path.isfile(full):
            self._send(404, "not found", "text/plain; charset=utf-8")
            return
        ext = os.path.splitext(full)[1].lower()
        ctype = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "application/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
                 ".jpg": "image/jpeg", ".png": "image/png", ".svg": "image/svg+xml"}.get(
            ext, "application/octet-stream")
        with open(full, "rb") as fh:
            self._send(200, fh.read(), ctype)

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path == "/api/status":
                self._json(200, sim_status())
            elif path == "/api/window":
                if xwin is None:
                    self._json(200, {"available": False, "status": {"found": False}})
                else:
                    try:
                        self._json(200, {"available": True, "status": xwin.status()})
                    except Exception as exc:
                        self._json(500, {"available": True, "error": f"{type(exc).__name__}: {exc}"})
            elif path == "/api/window/list":
                if xwin is None:
                    self._json(200, {"available": False, "windows": []})
                else:
                    try:
                        self._json(200, {"available": True, "windows": xwin.list_visible()})
                    except Exception as exc:
                        self._json(500, {"available": True, "error": f"{type(exc).__name__}: {exc}"})
            elif path == "/api/window/pointer":
                # 2 点校准用：返回鼠标此刻的全局设备像素坐标（跨多显示器/分数缩放下 X11 真实坐标）
                if xwin is None:
                    self._json(200, {"ok": False})
                else:
                    try:
                        self._json(200, xwin.pointer())
                    except Exception as exc:
                        self._json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            elif path == "/api/state":
                st = bridge_get("/api/state")
                st["sim"] = sim_status()
                self._json(200, st)
            elif path == "/api/models":
                self._json(200, {"checkpoints": models.scan(probe=True)})
            elif path == "/api/scenarios":
                self._json(200, {"scenarios": store.list_scenarios()})
            elif path == "/api/saved":
                self._json(200, {"saved": store.list_saved()})
            elif path == "/api/runs":
                self._json(200, {"runs": store.list_runs()})
            elif path == "/api/plot/list":
                self._json(200, {"logs": plot.list_logs()})
            elif path == "/api/plot/meta":
                nm = _query_param(self.path, "name")
                self._json(200, plot.meta(nm) or {})
            elif path == "/api/plot/overview":
                nm = _query_param(self.path, "name")
                self._json(200, plot.overview(nm) or {})
            elif path == "/api/plot/window":
                nm = _query_param(self.path, "name")
                try:
                    t0 = float(_query_param(self.path, "t0"))
                    t1 = float(_query_param(self.path, "t1"))
                except (TypeError, ValueError):
                    self._json(400, {"error": "t0/t1 需要是数字"})
                    return
                thin = int(_query_param(self.path, "thin") or 700)
                seg = plot.window(nm, t0, t1, thin)
                self._json(200 if seg else 404, seg or {})
            elif path == "/api/plot/kinds":
                self._json(200, {"kinds": plot.kind_json()})
            elif path == "/api/tellogs":
                self._json(200, {"logs": _tel_logs()})
            elif path == "/api/runlog":
                q = self.path.split("?", 1)[1] if "?" in self.path else ""
                rid = ""
                for kv in q.split("&"):
                    if kv.startswith("id="):
                        rid = kv[3:]
                self._json(200, {"id": rid, "text": store.tail_console(rid)})
            elif path == "/api/settings":
                self._json(200, store.get_settings())
            elif path == "/api/mujoco/manifest":
                self._json(200, mujoco_manifest())
            else:
                self._serve_static(path)
        except Exception as exc:
            traceback.print_exc()
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._body()
        try:
            if path == "/api/start":
                spec = body.get("spec") or {}
                if ENGINES[engine_of(spec)]["block_on_training"]:
                    t = training_processes()
                    if t and not body.get("force"):
                        self._json(409, {"error": "training", "training": t,
                                         "text": f"检测到训练进程 PID {t[0]['pid']}，按纪律拦截"})
                        return
                if sim_status()["status"] == "running":
                    self._json(409, {"error": "already running"})
                    return
                spec.setdefault("bridge_port", _bridge_port)
                pre = preflight(spec)
                hard = [i for i in pre if i["level"] == "bad"]
                if hard and not body.get("force"):
                    self._json(400, {"error": "preflight", "items": pre})
                    return
                info = _launch(spec)
                # 【P8 五槽】窗口对齐＝前端自动停靠（dockWhenReady：仿真运行即把 Isaac 窗口
                # 贴到 仿真视口 面板矩形下，dock 自校准 3 次收敛）。不另起 win_align 进程——
                # 两个写入者抢窗口位置会互相破坏停靠；win_align.py 保留作诊断/独立摆放工具。
                # ⚠ 必须回包：漏了这行客户端拿到的是"连接被关闭无响应"，/api/start 的返回体就是 _launch 的 info。
                self._json(200, info)
            elif path == "/api/stop":
                self._json(200, _stop())
            elif path == "/api/preflight":
                self._json(200, {"items": preflight(body.get("spec") or {})})
            elif path == "/api/cmd":
                self._json(200, bridge_post("/api/cmd", body))
            elif path == "/api/action":
                self._json(200, bridge_post("/api/action", body))
            elif path == "/api/point":
                # 【P8 五槽·点击校准】前端把每次点击的页面坐标发上来，后端同步读鼠标
                # 指针的真实设备坐标（xwin.pointer，全局设备像素）配成一对。两组不同
                # 位置的点击即可解出「页面 CSS → 屏幕设备」的变换（统一 scale + 每轴
                # 原点，最小二乘）——比 window.screenX 跨屏估算可靠（指针是 X11 真值）。
                try:
                    cx = float(body.get("cx")); cy = float(body.get("cy"))
                except (TypeError, ValueError):
                    self._json(400, {"error": "cx/cy 需要数字"})
                    return
                pt = xwin.pointer()
                if not pt.get("ok"):
                    self._json(200, {"ok": False, "error": "指针读数不可用"})
                    return
                _PAIRS.append((cx, cy, float(pt["x"]), float(pt["y"])))
                del _PAIRS[:-40]
                _cal_save()
                _xf = _solve_xform()
                self._json(200, {"ok": True, "pairs": len(_PAIRS), "xform": _xf})
            elif path == "/api/window/dock":
                if xwin is None:
                    self._json(503, {"error": "无 X 显示，窗口停靠不可用"})
                    return
                # 【P8 五槽·页面基准停靠】目标矩形优先用「点击校准」解出的变换
                # （页面 CSS → 屏幕设备，指针真值配对，多屏/分数缩放下精确）；
                # 未解出时回退浏览器 X 原点合成（次准）→ 再退前端绝对估算（末选）。
                _pg = body.get("page")
                _xf = _solve_xform()
                if _pg and not _xf:
                    self._json(409, {"needcal": True,
                                     "error": "点击校准数据不足或被污染——请用鼠标在页面上点几处不同位置的按钮后再试"})
                    return
                if _pg and _xf:
                    _s, _ox, _oy = _xf["s"], _xf["ox"], _xf["oy"]
                    x = int(round(_ox + float(_pg.get("l", 0)) * _s))
                    y = int(round(_oy + float(_pg.get("t", 0)) * _s))
                    w = int(round(float(_pg.get("w", 800)) * _s))
                    h = int(round(float(_pg.get("h", 600)) * _s))
                elif _pg:
                    # 浏览器按标题匹配不可靠（实测匹配到隐藏窗口 (0,0)）⇒ 不回退，
                    # 明确要求先完成点击校准（配对持久化，重启不丢）
                    self._json(409, {"needcal": True,
                                     "error": "视口未校准——请用鼠标在页面上点 2~3 处不同位置的按钮（一次校准长期有效）"})
                    return
                else:
                    try:
                        x, y = int(body.get("x")), int(body.get("y"))
                        w, h = int(body.get("w")), int(body.get("h"))
                    except (TypeError, ValueError):
                        self._json(400, {"error": "x/y/w/h 需要是整数（设备像素）"})
                        return
                if w < 160 or h < 120:
                    self._json(400, {"error": "区域太小"})
                    return
                if not _pg:
                    host = xwin.rect_inside_window(x, y, w, h)
                    if host is None:
                        self._json(409, {"error": "视口矩形下面没有任何可见窗口（嵌入式面板拿不到屏幕坐标）"})
                        return
                st = xwin.dock(x, y, w, h)
                self._json(200, {"ok": True, "host": (_pg and "page") or host, "status": st,
                                 "x": x, "y": y, "w": w, "h": h, "page": bool(_pg)})
            elif path == "/api/window/undock":
                if xwin is None:
                    self._json(503, {"error": "无 X 显示"})
                    return
                self._json(200, {"ok": True, "status": xwin.undock()})
            elif path == "/api/window/reset":
                if xwin is None:
                    self._json(503, {"error": "无 X 显示"})
                    return
                wid = xwin.find_window()
                if wid:
                    xwin.set_decorated(wid, True)
                    xwin.set_above(wid, False)
                self._json(200, {"ok": True, "status": xwin.status()})
            elif path == "/api/scenarios":
                self._json(200, {"ok": True, "scenario": store.save_scenario(body.get("name"), body.get("payload"))})
            elif path == "/api/saved":
                try:
                    meta = store.solidify(body.get("path"),
                                          extra={"obs": body.get("obs"), "act": body.get("act")},
                                          folder=body.get("name"))
                    self._json(200, {"ok": True, "meta": meta})
                except FileNotFoundError:
                    self._json(404, {"error": "file not found"})
            elif path == "/api/settings":
                self._json(200, store.save_settings(body))
            elif path == "/api/mujoco/export":
                code_obj = _mujoco_export()
                self._json(code_obj[0] if isinstance(code_obj, tuple) else 200,
                           code_obj[1] if isinstance(code_obj, tuple) else code_obj)
            elif path == "/api/plot/save":
                self._json(200, _save_pngs(body))
            elif path == "/api/app/open":
                # 用真实浏览器窗口打开控制台：嵌入式面板里拿不到屏幕坐标，停靠只能在这种窗口里做
                port = self.server.server_address[1]
                url = body.get("url") or f"http://127.0.0.1:{port}/?tab=run&dock=1"
                exe = next((shutil.which(b) for b in ("microsoft-edge", "google-chrome", "chromium", "chromium-browser") if shutil.which(b)), None)
                if not exe:
                    self._json(503, {"error": "没找到可用的 Chromium 内核浏览器"})
                    return
                prof = os.path.join("/tmp", "wbr_console_app_profile")
                subprocess.Popen([exe, "--app=" + url, "--window-size=1280,900",
                                  "--window-position=120,80", "--user-data-dir=" + prof],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True)
                self._json(200, {"ok": True, "exe": exe, "url": url})
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:
            traceback.print_exc()
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_DELETE(self):
        path = self.path.split("?")[0]
        body = self._body()
        try:
            if path == "/api/scenarios":
                self._json(200, {"ok": store.delete_scenario(body.get("name"))})
            elif path == "/api/saved":
                self._json(200, {"ok": store.delete_saved(body.get("name"))})
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:
            self._json(500, {"error": str(exc)})


_bridge_port = DEFAULT_BRIDGE_PORT


def main():
    global _bridge_port
    ap = argparse.ArgumentParser(description="WBR Console 后端（静态站点 + 试验台 API）")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--bridge-port", type=int, default=DEFAULT_BRIDGE_PORT)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    _bridge_port = a.bridge_port

    httpd = None
    for p in range(a.port, a.port + 20):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError as e:
            print(f"端口 {p} 不可用（{e.strerror}），顺延")
    if httpd is None:
        raise SystemExit(f"❌ {a.port}~{a.port+19} 都不可用")

    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"🧭 WBR Console  {url}")
    print(f"   存储：{store.STORE_DIR}")
    print(f"   控制桥端口：{_bridge_port}（仿真进程内）")

    def _term(_signum, _frame):          # kill / nohup 被 kill 时别把仿真子进程留成孤儿
        print("\n⏹️ 收到 SIGTERM，先停仿真再退出")
        _stop()
        os._exit(0)

    signal.signal(signal.SIGTERM, _term)
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️ 关闭 WBR Console")
    finally:
        _stop()


if __name__ == "__main__":
    main()
