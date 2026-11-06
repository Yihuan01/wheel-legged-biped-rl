# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : bridge_common.py
# Purpose   : 控制桥 HTTP 骨架（单一实现）：路由分发、端口扫描、daemon 线程安装、
#             延迟 SIGINT 自杀。bench_bridge.py（Isaac）与 sim2sim_mujoco/mj_bridge.py
#             （MuJoCo）共用，两端只注册自己的路由表，协议形状不会漂移。
# ==============================================================================

import json
import os
import signal
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(routes):
    """routes: {(method, path): fn(body) -> dict | (code, dict)}。fn 返回 dict 即回 200，
    返回 (code, dict) 可指定非 200（如 400 参数错）；抛异常回 500；无此路由回 404。
    fn 内可先起后台线程再 return（响应先达、动作后至，如 stop）。"""

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

        def _route(self, method):
            path = self.path.split("?")[0]
            fn = routes.get((method, path))
            if fn is None:
                self._send(404, {"error": "not found"})
                return
            try:
                out = fn(self._body())
                code, obj = out if isinstance(out, tuple) else (200, out)
                self._send(code, obj)
            except Exception as exc:
                traceback.print_exc()
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

    return _Handler


def delayed_sigint(delay=0.15):
    """先让响应出网，再给自己 SIGINT（仿真进程的统一自杀路径，走正常收尾 flush）。"""
    threading.Thread(target=lambda: (time.sleep(delay), os.kill(os.getpid(), signal.SIGINT)),
                     daemon=True).start()


def start_bridge(routes, port_env="WBR_BRIDGE_PORT"):
    """按 WBR_BRIDGE_PORT 起 HTTP 桥（端口占用向后顺延 10 个），daemon 线程跑起来。
    返回 httpd；全部端口不可用返回 None（控制台将无法连接，进程内已打印提示）。"""
    port = int(os.environ.get(port_env, "8771"))
    httpd = None
    for p in range(port, port + 10):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), make_handler(routes))
            break
        except OSError:
            continue
    if httpd is None:
        print(f"❌ [bridge] {port}~{port+9} 端口都不可用，控制台将无法连接")
        return None
    print(f"🔌 [bridge] 控制口 http://127.0.0.1:{httpd.server_address[1]}/  （面板已由网页接管）")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
