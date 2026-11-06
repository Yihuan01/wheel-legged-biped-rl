# ==============================================================================
# Date      : 2026-11-05（2026-11-06 事件驱动改造）
# Author    : SYH
# File      : mj_bridge.py
# Purpose   : MuJoCo 仿真进程内的 HTTP 控制桥：协议与 wbr_console/bench_bridge.py 逐字
#             对齐（/api/ping|state|cmd|action|stop），仅 engine 报 "mujoco"。
#             HTTP 管道用共享骨架 bridge_common，不另写一份。
#             链模式（chain 非 None）：/api/action 把面板按钮转成 chain.request()；
#             单槽模式保留原有语义（四轮/两轮/跳跃/趴下各自可用通道）。
# ==============================================================================

import os
import time

from legged_gym.scripts.wbr_console import bridge_common

# 通道 → 指令槽位（与 SimLoop.CMD_IDX 一致）
_KEYS = ("lin_vel_x", "ang_vel_yaw", "base_height", "ang_roll", "jump_height")
_UNSAFE_ACTIONS = ("toggle", "rise", "lower", "armjump", "auto")   # 单策略默认拒

_STATE = {
    "loop": None,
    "chain": None,         # 【S3】五槽 uni 链（非 None 时 state 带 slot/phase/marks）
    "camera": True,        # MuJoCo 窗口默认相机跟随机身
    "started": time.time(),
    "gui": {k: 0.0 for k in _KEYS},   # 前端最近一次下发的原始值（回显用）
}


def _ranges():
    """契约里的指令量程，只给前端通道要的五个 key。roll [0,0] 退化 ⇒ 前端自动置灰。"""
    cr = _STATE["loop"].c["command_ranges"]
    return {k: [float(cr[k][0]), float(cr[k][1])] for k in _KEYS if k in cr}


def _clamp(key, value):
    rg = _ranges().get(key)
    if rg:
        return min(max(float(value), rg[0]), rg[1])
    return float(value)


def _telemetry():
    loop = _STATE["loop"]
    if loop.out_tsv:
        try:
            return {"file": os.path.basename(loop.out_tsv),
                    "sizeMB": round(os.path.getsize(loop.out_tsv) / 1e6, 2),
                    "rows": len(loop.rows)}
        except OSError:
            pass
    return {"file": "—", "sizeMB": 0.0, "rows": len(loop.rows)}


def _can_jump():
    loop = _STATE["loop"]
    chain = _STATE["chain"]
    if chain is not None:
        return "jump" in chain.status().get("allowed", ())   # 链模式按守卫表（当前槽是否接受装填跳）
    return bool(loop.c.get("jump_hi", 0.0) > 0.06 or loop.jt is not None)


def _build_state():
    loop = _STATE["loop"]
    chain = _STATE["chain"]
    g = _STATE["gui"]
    # cmd 三元组 [in, tgt, act]：in=前端原始下发，tgt=限速目标，act=限速后实际值
    cmd = {
        "vx": [g["lin_vel_x"], float(loop.cmd_target[0]), float(loop.cmds[0])],
        "yaw": [g["ang_vel_yaw"], float(loop.cmd_target[2]), float(loop.cmds[2])],
        "h": [g["base_height"], float(loop.cmd_target[5]), float(loop.cmds[5])],
        "roll": [g["ang_roll"], float(loop.cmd_target[6]), float(loop.cmds[6])],
    }
    st = {
        "running": True,
        "engine": "mujoco",
        "switch": bool(chain),               # 五槽链：前端按槽位布局渲染
        "pid": os.getpid(),
        "uptime": round(time.time() - _STATE["started"], 1),
        "ph": 0,
        "hasReadout": True,
        "cmd": cmd,
        "gui": dict(g),
        "cmdRanges": _ranges(),
        "camera": bool(_STATE["camera"]),
        "telemetry": _telemetry(),
        "terminated": loop.terminated,
        "model": loop.model_desc,
        "mjcf": loop.mjcf_name,
        "slot": None,
        "phase": 0,
        "marks": [],
        "chain": bool(chain),
        "canJump": _can_jump(),
    }
    if chain is not None:
        # 跳跃高度三联 [in, tgt, act]：面板预设值 / 链记下的下一跳高度
        st["cmdJump"] = [float(g.get("jump_height", 0.0)),
                         float(chain.h_jump), float(chain.h_jump)]
    if chain:
        cs = chain.status()
        for k in ("slot", "phase", "events", "interactive", "allowed",
                  "lastRequest", "nRequests", "nRejected", "allowRise", "z", "vx", "pitch_deg"):
            st[k] = cs.get(k)
        # marks 给前端时序条用事件流（含指令回执）；槽位切换的旧元组格式不再单独下发
        st["marks"] = list(cs.get("events", []))
    return st


# ------------------------------------------------------------------ 路由
def _ping(body):
    return {"ok": True, "bridge": True}


def _state(body):
    return _build_state()


def _cmd(body):
    loop = _STATE["loop"]
    key = body.get("key")
    if key == "jump_height":
        v = float(body.get("value", 0.0))
        _STATE["gui"][key] = v
        chain = _STATE["chain"]
        if chain is not None:
            # 【A+B】链模式：Jump 滑条＝「下一跳高度」，写进链的 h_jump（装填仍走「装填跳」按钮）
            if v > 0.0:
                chain.h_jump = v
            return {"ok": True, "gui": dict(_STATE["gui"]), "note": f"下一跳高度 {chain.h_jump:.2f} m（点「装填跳」执行）"}
        if not _can_jump():
            return 400, {"error": "jump_height: 当前槽位无跳跃通道"}
        if loop.jt is not None:
            loop.jump_request(v)
        return {"ok": True, "gui": dict(_STATE["gui"])}
    if key not in ("lin_vel_x", "ang_vel_yaw", "base_height", "ang_roll"):
        return 400, {"error": f"bad key {key}"}
    # 【A+B】链模式 base_height 实时放行，但：
    #   · 0 视为无效（前端启动会把滑条 0 回写，若照收会让策略吃分布外高度 —— 实测跳跃槽蹲不下去）；
    #   · 只在两轮槽记录 h_2w（切槽沿用）；四轮/趴下槽仍由契约管自己的 0.19。
    if _STATE["chain"] is not None and key == "base_height":
        chain = _STATE["chain"]
        v = float(body.get("value", 0.0))
        _STATE["gui"][key] = v
        if v <= 0.0:
            return {"ok": True, "gui": dict(_STATE["gui"]), "note": "高度 0 已忽略（保持槽位默认）"}
        v = _clamp(key, v)                      # 夹到当前槽契约量程（两轮 0.20~0.35）
        loop.set_cmd(key, v)
        if chain.slot == "IDLE2W":
            chain.h_2w = v
        return {"ok": True, "gui": dict(_STATE["gui"]), "note": f"高度 {v:.3f} m"}
    value = _clamp(key, body.get("value", 0.0))
    _STATE["gui"][key] = float(value)
    loop.set_cmd(key, value)
    if _STATE["chain"] is not None and key == "lin_vel_x":
        _STATE["chain"].vx = float(value)   # 切槽后仍保持面板设定的行走速度
    return {"ok": True, "gui": dict(_STATE["gui"])}


def _action(body):
    loop = _STATE["loop"]
    chain = _STATE["chain"]
    act = body.get("action")
    msg = act

    # —— 链模式：所有面板按钮统一转成链请求，由守卫决定接受/拒绝 ——
    if chain is not None:
        from legged_gym.scripts.sim2sim_mujoco.deploy.uni_chain import ACTION_TO_REQ, REQ_LABEL
        if act == "camera":
            _STATE["camera"] = bool(body.get("value"))
            loop.follow_cam = _STATE["camera"]
            msg = "相机跟随" + ("开启" if _STATE["camera"] else "关闭")
        elif act == "stop":
            loop.stop_requested = True
        elif act == "auto":
            msg = "链模式不支持连跳（请单槽使用）"
        elif act in ACTION_TO_REQ or act in ("two", "four", "next", "jump", "reset"):
            req = ACTION_TO_REQ.get(str(act), str(act))
            chain.request(req, body.get("value"))
            msg = "已下发：" + REQ_LABEL.get(req, req) + "（结果看状态条）"
        else:
            return 400, {"error": f"bad action {act}"}
        return {"ok": True, "action": act, "msg": msg}

    # —— 单槽模式（原有语义） ——
    if act == "reset":
        loop.request_reset()                  # 主循环在下一 step 安全点执行
        msg = "复位已请求（下一步生效，零历史冷启动）"
    elif act == "camera":
        _STATE["camera"] = bool(body.get("value"))
        loop.follow_cam = _STATE["camera"]
        msg = "相机跟随" + ("开启" if _STATE["camera"] else "关闭")
    elif act == "stop":
        pass                                  # 走 /api/stop；这里只为不落 bad action
    elif act == "armjump" and _can_jump():
        loop.jump_request(float(_STATE["gui"].get("jump_height", 0.10) or 0.10))
        msg = "已装填一跳"
    elif act in _UNSAFE_ACTIONS:
        msg = "单策略不支持"
    else:
        return 400, {"error": f"bad action {act}"}
    return {"ok": True, "action": act, "msg": msg}


def _stop(body):
    """【确定性停止】先置 loop.stop_requested，主循环下一拍（≤10 ms）退出并正常收尾；
    SIGINT 只作兜底（实测 MuJoCo 进程对 SIGINT 不敏感，单靠信号要等 21 s 才被 SIGKILL）。"""
    loop = _STATE["loop"]
    if loop is not None:
        loop.stop_requested = True
    bridge_common.delayed_sigint(1.5)
    return {"ok": True, "stopping": True}


_ROUTES = {
    ("GET", "/api/ping"): _ping,
    ("GET", "/api/state"): _state,
    ("POST", "/api/cmd"): _cmd,
    ("POST", "/api/action"): _action,
    ("POST", "/api/stop"): _stop,
}


def install(loop, chain=None):
    """把桥接到 SimLoop 上（serve.py 在建好 loop 后调用）。chain 非 None 即五槽链。"""
    _STATE["loop"] = loop
    _STATE["chain"] = chain
    _STATE["started"] = time.time()
    loop.follow_cam = _STATE["camera"]
    print("🎛️  [bench] MuJoCo 控制桥已就绪（网页控制台接管，无原生面板）")
    return bridge_common.start_bridge(_ROUTES)
