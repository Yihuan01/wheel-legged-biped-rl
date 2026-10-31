# ==============================================================================
# Date      : 2026-10-26
# Author    : SYH
# File      : wbr_bench.py
# Purpose   : 控制台后端：静态站点 + 试验台 API（模型扫描 / 预检 / 场景 / 启停 / 绘图）+
#             控制桥代理，纯标准库实现。
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
PY = os.environ.get("WBR_PYTHON") or models.env_python()
RUN_SCRIPT = os.path.join(HERE, "wbr_bench_run.py")

# 四槽的默认 obs/act（选不中时的兜底校验基准）
SLOT_SPEC = {
    "fourw": {"stage": "① 四轮行驶", "obs": 164, "act": 6},
    "standup": {"stage": "② 起立", "obs": 173, "act": 6},
    "w3": {"stage": "③ 两轮站立/行走", "obs": 161, "act": 6},
    "lie": {"stage": "④ 趴下", "obs": 173, "act": 6},
}
# 单策略：task → 期望 obs（用于校验 checkpoint 与 task 是否匹配）
# 地形任务 → 高度图（preflight 用来确认"这次真的会加载地形"）
TERRAIN_FILES = {
    "wbr_terrain": "legged_gym/envs/wbr/terrain/release/wbr_rand_curve.npz",
    "wbr_demo": "legged_gym/envs/wbr/terrain/release/wbr_p8_demo.npz",
    "wbr_combo": "legged_gym/envs/wbr/terrain/release/wbr_p8_combo_L3.npz",
    "wbr_elements": "legged_gym/envs/wbr/terrain/release/wbr_p8_L3.npz",
}
TASK_OBS = {"wbr_flat": 161, "wbr_flat4": 164, "wbr_switch": 173, "wbr_terrain": 161,
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

    env = dict(os.environ)
    env["WBR_BRIDGE_PORT"] = str(spec.get("bridge_port", DEFAULT_BRIDGE_PORT))
    env["PYTHONUNBUFFERED"] = "1"
    # 关键：libpython 在 **子进程所用解释器** 的 lib 下（后端可能跑在系统 python，
    # 不能拿 sys.prefix）；否则 import isaacgym 时找不到 libpython3.8.so.1.0。
    conda_lib = os.path.join(os.path.dirname(os.path.dirname(PY)), "lib")
    env["LD_LIBRARY_PATH"] = conda_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + os.path.join(REPO_ROOT, "rsl_rl-1.0.2") + os.pathsep + env.get("PYTHONPATH", "")

    console_path = os.path.join(run_dir, "console.log")
    con = open(console_path, "wb")
    proc = subprocess.Popen([PY, RUN_SCRIPT, "--spec", spec_path],
                            cwd=os.path.dirname(HERE), env=env,
                            stdout=con, stderr=subprocess.STDOUT, start_new_session=True)
    with _proc_lock:
        _proc = proc
        _run_meta = {"id": run_id, "dir": run_dir, "spec": spec, "console": console_path}
    print(f"▶️  [bench] 起仿真 pid={proc.pid} run={run_id}")
    return {"ok": True, "pid": proc.pid, "run": run_id}


def _stop():
    global _proc
    with _proc_lock:
        proc = _proc
    if proc is None or proc.poll() is not None:
        return {"ok": True, "note": "没有在跑的仿真"}
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except Exception:
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
    # 给足时间走 play02 的 finally 收尾 flush（把未满一段的零头写盘）；Isaac Gym 收尾较慢，放宽到 ~10s
    for _ in range(80):
        if proc.poll() is not None:
            break
        time.sleep(0.15)
    if proc.poll() is None:
        # SIGINT 迟迟不退：再补一次 SIGINT，多等一会，尽量别用 SIGKILL（会跳过收尾 flush 丢数据）
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
    """列 wbr_play_data 下的遥测文件（新→旧）。"""
    d = os.path.normpath(os.path.join(HERE, "..", "wbr_play_data"))
    out = []
    if os.path.isdir(d):
        names = [n for n in os.listdir(d) if n.endswith(".tsv") and "_tel_" in n]
        names.sort(key=lambda n: os.path.getmtime(os.path.join(d, n)), reverse=True)
        for i, n in enumerate(names):
            full = os.path.join(d, n)
            out.append({"name": n, "sizeMB": round(os.path.getsize(full) / 1e6, 2), "current": i == 0})
    return out


def preflight(spec):
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
            #   此时该槽不查权重，按"已就绪"计（四格保留，仍显示 4/4）。字符串/字典两种都认。
            if key == "standup" and isinstance(path, dict) and str(path.get("type", "")).lower() == "script":
                ok += 1
                items.append({"level": "ok", "text": f"{meta['stage']}：相位脚本（P7T，无需权重）"})
                dims.append("script")
                continue
            if not path:
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
            # 地形任务显式提示：确认会读哪张高度图（防止"以为加载了其实没加载"）
            _tf = TERRAIN_FILES.get(single.get("task"))
            if _tf:
                _full = _tf if os.path.isabs(_tf) else os.path.join(REPO_ROOT, _tf)
                items.append({"level": "ok" if os.path.isfile(_full) else "bad",
                              "text": f"地形：{os.path.basename(_tf)}"
                                      + ("" if os.path.isfile(_full) else "（文件不存在）")})
            if single.get("from_run_config"):
                items.append({"level": "info",
                              "text": "from_run_config 只回填奖励/指令参数（地形几何已锁定为任务自己的图）"})
        else:
            items.append({"level": "warn", "text": "单策略：未指定可用的 checkpoint 路径"})

    t = training_processes()
    if t:
        items.append({"level": "warn", "text": f"检测到训练进程 PID {t[0]['pid']} → 启动将被拦截（可强制）"})
    else:
        items.append({"level": "ok", "text": "没有检测到训练进程，显卡可独占"})
    g = gpu_mem()
    if g["used"] is not None:
        items.append({"level": "info", "text": f"显卡 {g['used']} / {g['total']} GB"})
    return items


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
                t = training_processes()
                if t and not body.get("force"):
                    self._json(409, {"error": "training", "training": t,
                                     "text": f"检测到训练进程 PID {t[0]['pid']}，按纪律拦截"})
                    return
                if sim_status()["status"] == "running":
                    self._json(409, {"error": "already running"})
                    return
                spec = body.get("spec") or {}
                spec.setdefault("bridge_port", _bridge_port)
                pre = preflight(spec)
                hard = [i for i in pre if i["level"] == "bad"]
                if hard and not body.get("force"):
                    self._json(400, {"error": "preflight", "items": pre})
                    return
                self._json(200, _launch(spec))
            elif path == "/api/stop":
                self._json(200, _stop())
            elif path == "/api/preflight":
                self._json(200, {"items": preflight(body.get("spec") or {})})
            elif path == "/api/cmd":
                self._json(200, bridge_post("/api/cmd", body))
            elif path == "/api/action":
                self._json(200, bridge_post("/api/action", body))
            elif path == "/api/window/dock":
                if xwin is None:
                    self._json(503, {"error": "无 X 显示，窗口停靠不可用"})
                    return
                try:
                    x, y = int(body.get("x")), int(body.get("y"))
                    w, h = int(body.get("w")), int(body.get("h"))
                except (TypeError, ValueError):
                    self._json(400, {"error": "x/y/w/h 需要是整数（设备像素）"})
                    return
                if w < 160 or h < 120:
                    self._json(400, {"error": "区域太小"})
                    return
                host = xwin.rect_inside_window(x, y, w, h)
                if host is None:
                    self._json(409, {"error": "视口矩形下面没有任何可见窗口（嵌入式面板拿不到屏幕坐标）"})
                    return
                self._json(200, {"ok": True, "host": host, "status": xwin.dock(x, y, w, h)})
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
