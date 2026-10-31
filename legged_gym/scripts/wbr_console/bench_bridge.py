# ==============================================================================
# Date      : 2026-10-26
# Author    : SYH
# File      : bench_bridge.py
# Purpose   : 仿真进程内的 HTTP 控制口：读写 gui_cmds / mode_shared / jump_flags / readout，
#             把控制台的指令送进正在跑的仿真。
# ==============================================================================

import json
import os
import signal
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_PORT = int(os.environ.get("WBR_BRIDGE_PORT", "8771"))
_STATE = {
    "switch": False,
    "ranges": None,
    "env": None,
    "camera": False,   # 相机跟随（默认关闭，前端可手动开关）
    "started": time.time(),
    "telemetry": {"file": "—", "sizeMB": 0.0, "rows": 0},
}
_tel_cache = {"path": None, "size": -1, "rows": 0, "at": 0.0}
_APPLIED_PARAMS = {}   # 启动时真正写进 env.sw 的相位参数（供前端回显）


def set_applied_params(d):
    _APPLIED_PARAMS.clear()
    _APPLIED_PARAMS.update(d or {})

# 合法通道（与 play02.CHANNELS 的 key 一致）
_CMD_KEYS = ("lin_vel_x", "ang_vel_yaw", "base_height", "ang_roll", "jump_height")


def _latest_telemetry():
    """找 wbr_play_data 下最新的 wbr_*_tel_*.tsv，带缓存地数行数（≤2s 一次）。"""
    from deps import play02_core as play02
    d = play02.DATA_DIR
    try:
        names = [n for n in os.listdir(d) if n.endswith(".tsv") and "_tel_" in n]
    except OSError:
        return _STATE["telemetry"]
    if not names:
        return _STATE["telemetry"]
    names.sort(key=lambda n: os.path.getmtime(os.path.join(d, n)))
    name = names[-1]
    full = os.path.join(d, name)
    try:
        size = os.path.getsize(full)
    except OSError:
        return _STATE["telemetry"]
    now = time.time()
    if _tel_cache["path"] != full or now - _tel_cache["at"] > 2.0:
        rows = _tel_cache["rows"] if _tel_cache["path"] == full else 0
        if size != _tel_cache["size"]:
            rows = 0
            try:
                with open(full, "rb") as fh:
                    for _ in fh:
                        rows += 1
                rows = max(0, rows - 12)          # 减掉表头注释行
            except OSError:
                pass
        _tel_cache.update(path=full, size=size, rows=rows, at=now)
    _STATE["telemetry"] = {"file": name, "sizeMB": round(size / 1e6, 2), "rows": _tel_cache["rows"]}
    return _STATE["telemetry"]


def _cmd_ranges():
    """当前实际生效的指令量程（随模式/加载模型动态）。
    vx/yaw/jump 用面板播放量程（_STATE['ranges']，启动时给）；
    base_height / ang_roll 用 env 当前的 command_ranges（随模式切换动态：四轮 h[0.175,0.245]、roll 固定；
    两轮 h[0.2,0.35]、roll ±0.2；起立/趴下 h 固定）。量程退化(min==max)⇒前端把该滑块置灰。"""
    env = _STATE.get("env")
    out = {k: list(v) for k, v in (_STATE.get("ranges") or {}).items()}
    if env is not None:
        try:
            cr = getattr(env, "command_ranges", None) or {}
            for k in ("base_height", "ang_roll"):
                v = cr.get(k)
                if v:
                    out[k] = [float(v[0]), float(v[1])]
        except Exception:
            pass
    return out


def _build_state():
    from deps import play02_core as play02
    r = play02.readout or {}
    mode_key = play02.mode_shared.get("mode") if _STATE["switch"] else None
    dbg = r.get("dbg") or {}
    mode_line = r.get("mode_line")
    state = {
        "running": True,
        "switch": _STATE["switch"],
        "pid": os.getpid(),
        "uptime": round(time.time() - _STATE["started"], 1),
        "modeKey": mode_key,
        "modeCN": play02.MODE_CN.get(mode_key, mode_key) if mode_key else None,
        "toggleLabel": play02.TOGGLE_LABEL.get(mode_key, "") if mode_key else "",
        "ph": int(round(dbg.get("ph", 0))) if dbg else 0,
        "phUni": (r.get("uni_ph") if r.get("uni_ph") is not None else int(round(dbg.get("ph", 0))) if dbg else 0),
        "modeLine": mode_line,
        "evUp": ("✓" in mode_line) if mode_line else None,
        "cmd": r.get("cmd") or {},
        "cmdJump": r.get("cmd_jump"),
        "armed": r.get("armed"),
        "auto": r.get("auto"),
        "gui": {k: float(play02.gui_cmds.get(k, 0.0)) for k in _CMD_KEYS},
        "cmdRanges": _cmd_ranges(),
        "camera": bool(_STATE.get("camera")),
        "telemetry": _latest_telemetry(),
        "hasReadout": bool(r),
        "params": dict(_APPLIED_PARAMS),
    }
    return state


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

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

    def do_GET(self):
        try:
            if self.path.startswith("/api/ping"):
                self._send(200, {"ok": True, "bridge": True})
            elif self.path.startswith("/api/state"):
                self._send(200, _build_state())
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        try:
            from deps import play02_core as play02
            body = self._body()
            if self.path.startswith("/api/cmd"):
                key = body.get("key")
                if key not in _CMD_KEYS:
                    self._send(400, {"error": f"bad key {key}"})
                    return
                play02.gui_cmds[key] = float(body.get("value", 0.0))
                self._send(200, {"ok": True, "gui": {k: float(play02.gui_cmds.get(k, 0.0)) for k in _CMD_KEYS}})
            elif self.path.startswith("/api/action"):
                act = body.get("action")
                msg = act
                if act == "toggle":
                    play02.mode_shared["request"] = "toggle"
                elif act == "rise":
                    if play02.mode_shared.get("mode") == play02.MODE_FOURW:
                        play02.mode_shared["request"] = "toggle"
                    else:
                        msg = "当前不是四轮模式，无法起立"
                elif act == "lower":
                    if play02.mode_shared.get("mode") == play02.MODE_TWOW:
                        play02.mode_shared["request"] = "toggle"
                    else:
                        msg = "当前不是两轮模式，无法趴下"
                elif act == "reset":
                    play02.mode_shared["reset"] = True
                elif act == "armjump":
                    # uni 只有两轮能跳（后端两轮才打开跳跃机）；非两轮给明确提示，不静默失败。
                    if _STATE.get("switch") and play02.mode_shared.get("mode") != play02.MODE_TWOW:
                        msg = "仅两轮模式可跳跃（先切到两轮）"
                    else:
                        play02.jump_flags["arm_once"] = True
                elif act == "auto":
                    play02.jump_flags["auto"] = bool(body.get("value"))
                elif act == "camera":
                    _STATE["camera"] = bool(body.get("value"))
                    msg = "相机跟随" + ("开启" if _STATE["camera"] else "关闭")
                else:
                    self._send(400, {"error": f"bad action {act}"})
                    return
                self._send(200, {"ok": True, "action": act, "msg": msg})
            elif self.path.startswith("/api/stop"):
                self._send(200, {"ok": True, "stopping": True})
                threading.Thread(target=lambda: (time.sleep(0.15), os.kill(os.getpid(), signal.SIGINT)),
                                 daemon=True).start()
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def _serve():
    httpd = None
    for p in range(_PORT, _PORT + 10):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), _Handler)
            break
        except OSError:
            continue
    if httpd is None:
        print(f"❌ [bridge] {_PORT}~{_PORT+9} 端口都不可用，控制台将无法连接")
        return
    _STATE["port"] = httpd.server_address[1]
    print(f"🔌 [bridge] 控制口 http://127.0.0.1:{httpd.server_address[1]}/  （面板已由网页接管）")
    try:
        httpd.serve_forever()
    except Exception:
        pass


def install(env, ranges=None, switch=False):
    """替换 play02.start_gui_thread：签名与原来一致，只起 HTTP 控制口、不开 tkinter。"""
    _STATE["switch"] = bool(switch)
    _STATE["ranges"] = ranges
    _STATE["env"] = env
    _STATE["started"] = time.time()
    _install_camera_follow(env)
    threading.Thread(target=_serve, daemon=True).start()
    # 打印一次等价“面板已就绪”，方便 console.log 里看到
    print("🎛️  [bench] 网页控制台已接管（原生 tkinter 面板未启动）")
    return True


def _install_camera_follow(env):
    """相机跟随（控制台专用，play02 只在第 1 步设一次相机）。

    运行时包装 env.step / env.playback_reset —— 不改 play02.py，也不新开线程：
    相机更新发生在主线程的 step 内，和朋友原本就有的 env.set_camera 同一路径。
    """
    if env is None or getattr(env, "viewer", None) is None:
        return
    try:
        import numpy as np
    except Exception:
        return

    def _look():
        try:
            pos = env.root_states[0, 0:3].cpu().numpy()
            env.set_camera(pos + np.array([-1.5, 1.0, 0.8]), pos + np.array([0.5, 0.0, 0.0]))
        except Exception:
            pass

    try:
        _orig_step = env.step
        _last = [0.0]                   # 上次相机更新时刻（monotonic 秒）
        # 状态刷新 120 Hz（每步都更，控制 1/120 s 节流，
        #   连续指令下发时的相机滞后感明显减小；开销可忽略——只是 set_camera 两个三元组）
        _interval = 1.0 / 120.0

        def _step(actions):
            out = _orig_step(actions)
            # 仅在开关打开时才更新相机（默认关闭，前端手动开关）
            if _STATE.get("camera"):
                now = time.monotonic()
                if now - _last[0] >= _interval:
                    _last[0] = now
                    _look()
            return out

        env.step = _step
    except Exception:
        pass

    orig_reset = getattr(env, "playback_reset", None)
    if callable(orig_reset):
        try:
            def _reset(*a, **k):
                r = orig_reset(*a, **k)
                if _STATE.get("camera"):
                    _look()
                return r
            env.playback_reset = _reset
        except Exception:
            pass
    print("🎥 [bench] 相机跟随可用（默认关闭，可手动开关）")
