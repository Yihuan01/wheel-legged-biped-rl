#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仿真窗口对齐：IsaacGym API 不暴露窗口在屏幕上的坐标/位置 ⇒ ctypes 直调 libX11 后处理。

零外部依赖（libX11 桌面必有；不装 xdotool/wmctrl/python-xlib）。
用法：find_and_align("isaac", x, y, w, h, timeout=120, log=print) —— 轮询等窗口出现，
按标题子串匹配后 XMoveResizeWindow 到指定位置/尺寸（连摆两次防窗口管理器拉回）。
属主：[Ubuntu/zcode] 2026-11-04，P8 状态机五槽配套（作者需求：仿真窗口默认对齐）。
"""
import ctypes
import ctypes.util
import threading
import time

_X11 = None
_X11_ONCE = threading.Lock()


def _load():
    global _X11
    with _X11_ONCE:
        if _X11 is not None:
            return _X11
        path = ctypes.util.find_library("X11") or "libX11.so.6"
        lib = ctypes.CDLL(path)
        # ---- 签名 ----
        lib.XOpenDisplay.restype = ctypes.c_void_p
        lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
        lib.XDefaultRootWindow.restype = ctypes.c_ulong
        lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.XInternAtom.restype = ctypes.c_ulong
        lib.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
        lib.XGetWindowProperty.restype = ctypes.c_int
        lib.XGetWindowProperty.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
            ctypes.c_long, ctypes.c_long, ctypes.c_bool, ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_void_p)]
        lib.XFree.argtypes = [ctypes.c_void_p]
        lib.XFetchName.restype = ctypes.c_int
        lib.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_char_p)]
        lib.XMoveResizeWindow.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint, ctypes.c_uint]
        lib.XMoveWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int]
        lib.XFlush.argtypes = [ctypes.c_void_p]
        lib.XGetGeometry.restype = ctypes.c_int
        _X11 = lib
        return _X11


def _get_prop(disp, win, atom_name):
    """读窗口属性，返回 (类型 atom, bytes)；无/失败返回 None。"""
    lib = _X11
    a = lib.XInternAtom(disp, atom_name.encode(), False)
    if not a:
        return None
    t = ctypes.c_ulong()
    fmt = ctypes.c_int()
    n = ctypes.c_ulong()
    br = ctypes.c_ulong()
    v = ctypes.c_void_p()
    if lib.XGetWindowProperty(disp, win, a, 0, 65536, False, ctypes.c_ulong(0),
                              ctypes.byref(t), ctypes.byref(fmt),
                              ctypes.byref(n), ctypes.byref(br), ctypes.byref(v)) != 0:
        return None
    if n.value == 0 or not v.value:
        return None
    step = 8 if fmt.value == 32 else max(fmt.value // 8, 1)   # format 32 在 64 位上是 long
    raw = ctypes.string_at(v.value, n.value * step)
    lib.XFree(v)
    return t.value, raw


_X_EH_TYPE = None


def _with_err_guard(fn):
    """Xlib 全局错误处理器是进程级的 ⇒ 只在本次调用期间临时换成忽略错误的版本，
    调完立即恢复（isaacgym 渲染循环依赖自己的 handler，常驻替换会让它 ~3s 崩）。"""
    global _X_EH_TYPE
    lib = _X11
    if _X_EH_TYPE is None:
        _X_EH_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    _h = _X_EH_TYPE(lambda d, e: 0)
    globals()["_X_ERR_HANDLER"] = _h                       # 持引用防 GC
    prev = lib.XSetErrorHandler(_h)
    try:
        return fn()
    finally:
        lib.XSetErrorHandler(prev)


def _client_windows(disp, root):
    """_NET_CLIENT_LIST → 窗口 id 列表（X 窗口管理器/EWMH 必有）。"""
    return _with_err_guard(lambda: _client_windows_inner(disp, root))


def _client_windows_inner(disp, root):
    r = _get_prop(disp, root, "_NET_CLIENT_LIST")
    if not r or r[0] != 33:                                 # XA_WINDOW
        return []
    b = r[1]
    return [ctypes.c_ulong.from_buffer_copy(b, i * 8).value for i in range(len(b) // 8)]


def _win_name(disp, win):
    return _with_err_guard(lambda: _win_name_inner(disp, win))


def _win_name_inner(disp, win):
    r = _get_prop(disp, win, "_NET_WM_NAME")                # UTF8 优先
    if r and r[0] != 0:
        try:
            return r[1].decode("utf-8", "ignore")
        except Exception:
            return ""
    buf = ctypes.c_char_p()
    if _X11.XFetchName(disp, win, ctypes.byref(buf)) and buf.value:
        nm = buf.value.decode("utf-8", "ignore")
        _X11.XFree(buf.value)
        return nm
    return ""


def list_windows():
    """枚举 (窗口id, 标题)——诊断用。"""
    lib = _load()
    disp = lib.XOpenDisplay(None)
    if not disp:
        return []
    try:
        root = lib.XDefaultRootWindow(disp, 0)
        return [(w, _win_name(disp, w)) for w in _client_windows(disp, root)]
    finally:
        lib.XCloseDisplay(disp)


def _query_monitors(disp, root):
    """显示器布局：Xinerama → XRandR → 空列表（空＝不做跨屏移动，安全跳过）。"""
    lib = _X11

    class _Mon(ctypes.Structure):
        _fields_ = [("sx", ctypes.c_int), ("sy", ctypes.c_int),
                    ("sw", ctypes.c_int), ("sh", ctypes.c_int)]

    # 1) Xinerama
    _xine = globals().get("_XINERAMA")
    if _xine:
        _n = ctypes.c_int()
        _arr = ctypes.c_void_p()
        if _xine.XineramaQueryScreens(disp, ctypes.byref(_n)) and _n.value > 0 and _arr.value:
            _mons = [_Mon.from_buffer_copy(_arr.value, i * 8) for i in range(_n.value)]
            lib.XFree(_arr)
            return _mons
    # 2) 桌面壳窗口位置推断（@!x,y;BDHF 每台显示器一个，桌面环境必有）——从 _NET_CLIENT_LIST 取原点
    _mons = []
    for win in _client_windows(disp, root):
        nm = _win_name(disp, win)
        if nm.startswith("@!") and ";" in nm:
            try:
                _xy = nm.split("@!")[1].split(";")[0]
                _mx, _my = (int(v) for v in _xy.split(","))
                _mons.append(_Mon(_mx, _my, 0, 0))       # 只用原点，尺寸未知
            except (ValueError, IndexError):
                pass
    return _mons


def find_and_align(substr, x=None, y=None, w=None, h=None, timeout=120, log=print):
    """轮询等标题含 `substr` 的窗口出现，摆到 (x,y) 并缩放到 w×h；连摆两次防 WM 拉回。
    x/y/w/h 缺省＝屏幕水平居中、顶部 40px、1280×720。返回是否摆过。"""
    lib = _load()
    disp = lib.XOpenDisplay(None)
    if not disp:
        log("win_align: XOpenDisplay 失败（无 X 会话）⇒ 跳过窗口对齐")
        return False
    try:
        sw = lib.XWidthOfScreen
        lib.XWidthOfScreen.restype = ctypes.c_int
        lib.XWidthOfScreen.argtypes = [ctypes.c_void_p]
        root = lib.XDefaultRootWindow(disp, 0)
        # 屏幕尺寸（缺省定位用）
        scr = None
        _sd = lib.XInternAtom(disp, b"RESOURCE_MANAGER", False)
        try:
            lib.XGetGeometry.restype = ctypes.c_int
            # 用根窗口的 XWidthOfScreen：需要 Screen*，改走 XDisplayWidth
            lib.XDisplayWidth.restype = ctypes.c_int
            lib.XDisplayWidth.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.XDisplayHeight.restype = ctypes.c_int
            lib.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
            scr = (lib.XDisplayWidth(disp, 0), lib.XDisplayHeight(disp, 0))
        except Exception:
            scr = None
        log(f"win_align: 等待标题含「{substr}」的窗口（屏幕 {scr}）")
        # 显示器布局（Xinerama）：跨显示器移动会触发 Vulkan surface 重建 ⇒ isaacgym 崩溃
        #   （11-04 实测两连）⇒ 只在窗口**当前所在显示器**内对齐到左上角＋小偏移。
        _mons = _query_monitors(disp, root)
        log(f"win_align: 显示器布局 {[(m.sx, m.sy, m.sw, m.sh) for m in _mons]}")
        t0 = time.time()
        moved = 0
        names_logged = False
        while time.time() - t0 < timeout:
            wins = _client_windows(disp, root)
            hit = 0
            names = {}
            for win in wins:
                nm = _win_name(disp, win)
                names[win] = nm
                if substr.lower() in nm.lower():
                    hit = win
            if not hit and not names_logged:
                log("win_align: 未匹配到窗口，当前窗口列表 " +
                    " | ".join(f"「{nm}」" for nm in names.values() if nm))
                names_logged = True
            if hit:
                # 窗口当前位置（根坐标）→ 所在显示器 → 该显示器内左上角＋小偏移对齐
                _rx, _ry = ctypes.c_int(), ctypes.c_int()
                _child = ctypes.c_ulong()
                lib.XTranslateCoordinates(disp, hit, root, 0, 0,
                                          ctypes.byref(_rx), ctypes.byref(_ry), ctypes.byref(_child))
                # 监视器选择：有尺寸的按区间，只有原点的（壳窗口推断）取 ≤rx 的最大原点
                _mon = None
                for m in _mons:
                    if m.sx <= _rx.value and (m.sw == 0 or _rx.value < m.sx + m.sw):
                        if _mon is None or m.sx > _mon.sx:
                            _mon = m
                if _mon is not None:
                    _ox, _oy = _mon.sx + 40, _mon.sy + 40
                else:
                    log("win_align: 无显示器布局信息 ⇒ 不移动（防跨屏崩溃）")
                    lib.XCloseDisplay(disp)
                    return True
                # ⚠ 只移位置（XMoveWindow）：XMoveResizeWindow 会触发 Vulkan 交换链重建，
                #   isaacgym 渲染循环不兼容 ⇒ 进程 ~1-2s 内静默崩溃（11-04 实测两连）。
                #   已在目标位则跳过（零操作零风险）；跨显示器移动禁止（同上崩溃机理）。
                if abs(_rx.value - _ox) > 4 or abs(_ry.value - _oy) > 4:
                    _with_err_guard(lambda: (lib.XMoveWindow(disp, hit, _ox, _oy), lib.XFlush(disp)))
                    moved += 1
                    log(f"win_align: 窗口「{names[hit]}」→ ({_ox},{_oy})（显示器 {_mon and (_mon.sx, _mon.sw)} 内对齐）"
                        + ("（复摆确认）" if moved >= 2 else ""))
                    if moved >= 2:                # 连摆两次：窗口管理器若拉回则再压一次
                        time.sleep(3.0)
                        _with_err_guard(lambda: (lib.XMoveWindow(disp, hit, _ox, _oy), lib.XFlush(disp)))
                        lib.XCloseDisplay(disp)
                        return True
                else:
                    log(f"win_align: 窗口「{names[hit]}」已在目标位（{_rx.value},{_ry.value}）")
                    lib.XCloseDisplay(disp)
                    return True
            time.sleep(3.0)
        log(f"win_align: {timeout}s 内没等到窗口「{substr}」")
        lib.XCloseDisplay(disp)
        return False
    finally:
        lib.XCloseDisplay(disp)


def align_async(substr="isaac", x=None, y=None, w=None, h=None, timeout=120, log=print):
    """后台线程版（控制台仿真子进程在 play02 主循环阻塞前/后调用皆可）。"""
    th = threading.Thread(target=find_and_align,
                          args=(substr, x, y, w, h, timeout, log), daemon=True)
    th.start()
    return th


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="仿真窗口对齐（独立进程：与 isaacgym 的 X 错误处理器完全隔离）")
    _ap.add_argument("--align", default=None, metavar="TITLE子串",
                     help="轮询等标题含此子串的窗口出现，摆到 --x/--y（缺省＝所在显示器内左上角）")
    _ap.add_argument("--x", type=int, default=None)
    _ap.add_argument("--y", type=int, default=None)
    _ap.add_argument("--w", type=int, default=None, help="预留（resize 会崩 Vulkan，不启用）")
    _ap.add_argument("--h", type=int, default=None)
    _ap.add_argument("--timeout", type=int, default=120)
    _args = _ap.parse_args()
    if _args.align:
        _ok = find_and_align(_args.align, _args.x, _args.y, _args.w, _args.h,
                             timeout=_args.timeout, log=lambda m: print("[win] " + m, flush=True))
        raise SystemExit(0 if _ok else 3)
    print("当前窗口列表：")
    for wid, nm in list_windows():
        print(f"  0x{wid:x}  {nm}")
