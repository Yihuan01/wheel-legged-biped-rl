# ==============================================================================
# Date      : 2026-10-26
# Author    : SYH
# File      : xwin.py
# Purpose   : X11 窗口操作：按名字查找 Isaac Gym 窗口并移动 / 缩放（ctypes 调 libX11）。
# ==============================================================================

import ctypes
import ctypes.util
import time

_lib = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")

c_void_p, c_ulong, c_int, c_uint, c_long, c_char_p = (
    ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_uint, ctypes.c_long, ctypes.c_char_p)


class XWindowAttributes(ctypes.Structure):
    _fields_ = [
        ("x", c_int), ("y", c_int), ("width", c_int), ("height", c_int),
        ("border_width", c_int), ("depth", c_int), ("visual", c_void_p),
        ("root", c_ulong), ("class_", c_int), ("bit_gravity", c_int),
        ("win_gravity", c_int), ("backing_store", c_int), ("backing_planes", c_ulong),
        ("backing_pixel", c_ulong), ("save_under", c_int), ("colormap", c_ulong),
        ("map_installed", c_int), ("map_state", c_int), ("all_event_masks", c_long),
        ("your_event_mask", c_long), ("do_not_propagate_mask", c_long),
        ("override_redirect", c_int), ("screen", c_void_p),
    ]


class XWindowChanges(ctypes.Structure):
    _fields_ = [
        ("x", c_int), ("y", c_int), ("width", c_int), ("height", c_int),
        ("border_width", c_int), ("sibling", c_ulong), ("stack_mode", c_int),
    ]


class XClientMessageEvent(ctypes.Structure):
    _fields_ = [
        ("type", c_int), ("serial", c_ulong), ("send_event", c_int),
        ("display", c_void_p), ("window", c_ulong), ("message_type", c_ulong),
        ("format", c_int), ("data", c_long * 5),
    ]


class XEvent(ctypes.Union):
    _fields_ = [("type", c_int), ("xclient", XClientMessageEvent), ("pad", c_long * 24)]


_lib.XOpenDisplay.restype = c_void_p
_lib.XOpenDisplay.argtypes = [c_char_p]
_lib.XDefaultRootWindow.restype = c_ulong
_lib.XDefaultRootWindow.argtypes = [c_void_p]
_lib.XQueryTree.argtypes = [c_void_p, c_ulong, ctypes.POINTER(c_ulong), ctypes.POINTER(c_ulong),
                            ctypes.POINTER(ctypes.POINTER(c_ulong)), ctypes.POINTER(c_uint)]
_lib.XFetchName.argtypes = [c_void_p, c_ulong, ctypes.POINTER(c_char_p)]
_lib.XGetWindowAttributes.argtypes = [c_void_p, c_ulong, ctypes.POINTER(XWindowAttributes)]
_lib.XInternAtom.restype = c_ulong
_lib.XInternAtom.argtypes = [c_void_p, c_char_p, c_int]
_lib.XChangeProperty.argtypes = [c_void_p, c_ulong, c_ulong, c_ulong, c_int, c_int, c_void_p, c_int]
_lib.XConfigureWindow.argtypes = [c_void_p, c_ulong, c_uint, c_void_p]
_lib.XMapWindow.argtypes = [c_void_p, c_ulong]
_lib.XUnmapWindow.argtypes = [c_void_p, c_ulong]
_lib.XRaiseWindow.argtypes = [c_void_p, c_ulong]
_lib.XFlush.argtypes = [c_void_p]
_lib.XSendEvent.argtypes = [c_void_p, c_ulong, c_int, c_long, c_void_p]
_lib.XSync.argtypes = [c_void_p, c_int]
_lib.XTranslateCoordinates.argtypes = [c_void_p, c_ulong, c_ulong, c_int, c_int,
                                       ctypes.POINTER(c_int), ctypes.POINTER(c_int), ctypes.POINTER(c_ulong)]
_lib.XQueryPointer.argtypes = [c_void_p, c_ulong, ctypes.POINTER(c_ulong), ctypes.POINTER(c_ulong),
                               ctypes.POINTER(c_int), ctypes.POINTER(c_int),
                               ctypes.POINTER(c_int), ctypes.POINTER(c_int), ctypes.POINTER(c_uint)]

CWX, CWY, CWW, CWH, CWBORDER, CWSIBLING, CWSTACK = 1, 2, 4, 8, 16, 32, 64
ABOVE, BELOW, TOPIF, BOTTOMIF = 0, 1, 2, 3
MAP_STATE_VIEWABLE = 2

_dpy = None
_lk = None


def _lock():
    global _lk
    if _lk is None:
        import threading
        _lk = threading.RLock()
    return _lk

# X11 错误默认会终止进程（BadWindow 在窗口销毁竞态里很常见）；装一个静默处理器。
_ERROR_HANDLER = ctypes.CFUNCTYPE(c_int, c_void_p, c_void_p)


@_ERROR_HANDLER
def _ignore_x_error(_d, _e):
    return 0


_lib.XSetErrorHandler.argtypes = [_ERROR_HANDLER]
_lib.XSetErrorHandler(_ignore_x_error)


def display():
    global _dpy
    if _dpy is None:
        _dpy = _lib.XOpenDisplay(None)
        if not _dpy:
            raise RuntimeError("XOpenDisplay 失败（DISPLAY/XAUTHORITY？）")
    return _dpy


def _name(win):
    nm = c_char_p()
    if _lib.XFetchName(display(), win, ctypes.byref(nm)) and nm.value:
        return nm.value.decode("utf-8", "replace")
    return ""


def _attrs(win):
    a = XWindowAttributes()
    if not _lib.XGetWindowAttributes(display(), win, ctypes.byref(a)):
        return None
    return a


def _root_children():
    d = display()
    root = _lib.XDefaultRootWindow(d)
    r = c_ulong(); p = c_ulong()
    kids = ctypes.POINTER(c_ulong)(); n = c_uint()
    if not _lib.XQueryTree(d, root, ctypes.byref(r), ctypes.byref(p), ctypes.byref(kids), ctypes.byref(n)):
        return []
    try:
        return [kids[i] for i in range(n.value)]
    finally:
        _lib.XFree(kids) if hasattr(_lib, "XFree") else None


def _walk(depth=4):
    """广度遍历整棵窗口树（WM 会给顶层套一个 frame，直接子节点里找不到真正的客户窗口）。"""
    d = display()
    root = _lib.XDefaultRootWindow(d)
    seen, out, level = set(), [], [(root, 0)]
    while level:
        nxt = []
        for win, dep in level:
            if win in seen or dep > depth:
                continue
            seen.add(win)
            if dep:
                out.append(win)
            if dep == depth:
                continue
            r = c_ulong(); p = c_ulong()
            kids = ctypes.POINTER(c_ulong)(); n = c_uint()
            if _lib.XQueryTree(d, win, ctypes.byref(r), ctypes.byref(p), ctypes.byref(kids), ctypes.byref(n)):
                nxt.extend((kids[i], dep + 1) for i in range(n.value))
        level = nxt
    return out


# 【Sim2Sim P3】停靠目标窗口名按引擎切换（isaac 默认不变；wbr_bench 起 MuJoCo 时调 set_dock_match）
_DEFAULT_MATCH = ["isaac gym"]


def set_dock_match(m):
    _DEFAULT_MATCH[0] = m


def find_window(match=None):
    """找标题含 match 的窗口里面积最大的那个（可视图层深度 4 内）；找不到返回 None。
    match 缺省用 _DEFAULT_MATCH（引擎相关，默认 isaac gym）。"""
    match = match or _DEFAULT_MATCH[0]
    best, best_area = None, -1
    for w in _walk():
        nm = _name(w)
        if match.lower() not in nm.lower():
            continue
        a = _attrs(w)
        if not a:
            continue
        area = a.width * a.height
        if area > best_area:
            best, best_area = w, area
    return best


def _abs_pos(win):
    """窗口左上角在 root 坐标系的位置（含 WM 边框偏移）。"""
    d = display()
    root = _lib.XDefaultRootWindow(d)
    x, y, child = c_int(), c_int(), c_ulong()
    _lib.XTranslateCoordinates(d, win, root, 0, 0, ctypes.byref(x), ctypes.byref(y), ctypes.byref(child))
    return x.value, y.value


def pointer():
    """鼠标指针在 root 坐标系的位置（**全局设备像素**，跨多显示器 + 分数缩放下 X11 的真实坐标）。
    供前端做「2 点校准」：用户点击时取此刻真实设备坐标，与页面 CSS 坐标配对，
    反解出本窗口/本显示器独有的 CSS→设备像素 变换（DPR + 显示器原点），比 window.screenX 可靠得多
    （screenX/screenY 是「相对所在显示器」的 CSS 像素，多屏/分数缩放下与 X11 全局设备像素对不上）。"""
    with _lock():
        d = display()
        root = _lib.XDefaultRootWindow(d)
        rr = c_ulong(); cr = c_ulong()
        rx = c_int(); ry = c_int(); wx = c_int(); wy = c_int(); mk = c_uint()
        ok = _lib.XQueryPointer(d, root, ctypes.byref(rr), ctypes.byref(cr),
                                ctypes.byref(rx), ctypes.byref(ry),
                                ctypes.byref(wx), ctypes.byref(wy), ctypes.byref(mk))
        if not ok:
            return {"ok": False}
        return {"ok": True, "x": rx.value, "y": ry.value}


def _wm_class(win):
    """WM_CLASS 的 class 段（如 microsoft-edge / google-chrome / Gimp）。"""
    a = _lib.XInternAtom(display(), b"WM_CLASS", False)
    t = c_ulong(); f = c_int(); n = c_ulong(); br = c_ulong(); v = c_void_p()
    if _lib.XGetWindowProperty(display(), win, a, 0, 65536, False, c_ulong(0),
                               ctypes.byref(t), ctypes.byref(f), ctypes.byref(n),
                               ctypes.byref(br), ctypes.byref(v)) != 0:
        return ""
    if n.value == 0 or not v.value:
        return ""
    raw = ctypes.string_at(v.value, n.value)          # format 8，instance\0class
    _lib.XFree(v)
    parts = raw.decode("utf-8", "replace").split("\x00")
    return parts[1] if len(parts) > 1 else (parts[0] if parts else "")


def find_by_class(match, exclude_title=None):
    """按 WM_CLASS 子串找窗口（取面积最大）；排除标题含 exclude_title 的（如 isaac 自己）。"""
    best, best_area = None, -1
    for w in _walk():
        cls = _wm_class(w)
        if match.lower() not in cls.lower():
            continue
        if exclude_title and exclude_title.lower() in _name(w).lower():
            continue
        a = _attrs(w)
        if not a:
            continue
        if a.width * a.height > best_area:
            best, best_area = w, a.width * a.height
    return best


def browser_origin(match):
    """浏览器窗口（按标题/类名子串匹配，取面积最大）左上角的 root 坐标（设备像素）。
    供「页面基准停靠」：前端报面板在页面内的相对矩形＋DPR＋浏览器 chrome 高度，
    后端用浏览器窗口的 X 原点合成屏幕绝对矩形——绕开多屏/分数缩放下
    window.screenX 估算不准的问题（P8 五槽，11-04）。"""
    w = find_window(match)
    if not w:
        # 标题匹配失败（浏览器窗口常无标题）⇒ 按 WM_CLASS 找
        w = find_by_class(match.replace("edge", "microsoft-edge")
                          .replace("chrome", "google-chrome")
                          .replace("firefox", "Navigator"))
    if not w:
        return None
    return _abs_pos(w)


def status(match=None):
    """浏览器窗口（标题含 match，取面积最大）左上角的 root 坐标（设备像素）。
    供「页面基准停靠」：前端报面板在页面内的相对矩形＋DPR＋浏览器 chrome 高度，
    后端用浏览器窗口的 X 原点合成屏幕绝对矩形——绕开多屏/分数缩放下
    window.screenX 估算不准的问题（P8 五槽，11-04）。"""
    w = find_window(match)
    if not w:
        return None
    return _abs_pos(w)


def status(match=None):
    match = match or _DEFAULT_MATCH[0]
    with _lock():
        for _ in range(3):                     # 同款瞬态失败重试
            r = _status(match)
            if r.get("found"):
                return r
            time.sleep(0.2)
        return _status(match)


def _status(match=None):
    w = find_window(match or _DEFAULT_MATCH[0])
    if not w:
        return {"found": False}
    a = _attrs(w)
    ax, ay = _abs_pos(w)
    return {"found": True, "id": hex(w), "title": _name(w),
            "x": ax, "y": ay, "w": a.width, "h": a.height,
            "mapped": a.map_state == MAP_STATE_VIEWABLE,
            "override_redirect": bool(a.override_redirect)}


def list_visible(min_size=200):
    """列出所有可见顶层窗口（标题+绝对几何+是否 Isaac），供停靠诊断：
    真浏览器里点「前置窗口」失败时，可看本机 X11 实际能看到哪些窗口、坐标是否与预期一致。"""
    out = []
    with _lock():
        d = display()
        root = _lib.XDefaultRootWindow(d)
        r = c_ulong(); p = c_ulong()
        kids = ctypes.POINTER(c_ulong)(); n = c_uint()
        if not _lib.XQueryTree(d, root, ctypes.byref(r), ctypes.byref(p), ctypes.byref(kids), ctypes.byref(n)):
            return out
        for i in range(n.value):
            win = kids[i]
            a = _attrs(win)
            if not a or a.map_state != MAP_STATE_VIEWABLE:
                continue
            if a.width < min_size or a.height < min_size:
                continue
            ax, ay = _abs_pos(win)
            nm = _name(win) or ""
            out.append({"title": nm or "(无标题)", "x": ax, "y": ay, "w": a.width, "h": a.height,
                        "isaac": "isaac gym" in nm.lower()})
    out.sort(key=lambda w: w["w"] * w["h"], reverse=True)
    return out


def set_decorated(win, decorated):
    """去/加窗口边框（EWMH 的 _MOTIF_WM_HINTS，GNOME/mutter 认这个）。"""
    d = display()
    atom = _lib.XInternAtom(d, b"_MOTIF_WM_HINTS", 0)
    hints = (c_long * 5)(2 if not decorated else 2, 1 if decorated else 0, 0, 0, 0)
    _lib.XChangeProperty(d, win, atom, atom, 32, 1, ctypes.byref(hints), 5)
    _lib.XFlush(d)


def set_above(win, above=True):
    """置顶/取消置顶（_NET_WM_STATE_ABOVE 客户端消息 → root）。"""
    d = display()
    root = _lib.XDefaultRootWindow(d)
    ev = XEvent()
    ev.xclient.type = 33                      # ClientMessage
    ev.xclient.serial = 0
    ev.xclient.send_event = 1
    ev.xclient.display = d
    ev.xclient.window = win
    ev.xclient.message_type = _lib.XInternAtom(d, b"_NET_WM_STATE", 0)
    ev.xclient.format = 32
    ev.xclient.data[0] = 1 if above else 0    # 1=add 0=remove
    ev.xclient.data[1] = _lib.XInternAtom(d, b"_NET_WM_STATE_ABOVE", 0)
    ev.xclient.data[2] = 0
    ev.xclient.data[3] = 1                    # source indication: normal app
    ev.xclient.data[4] = 0
    _lib.XSendEvent(d, root, 0, 0x00040000 | 0x00080000, ctypes.byref(ev))  # SubstructureRedirect|Notify
    _lib.XFlush(d)


def move_resize(win, x, y, w, h):
    d = display()
    ch = XWindowChanges(x=int(x), y=int(y), width=int(w), height=int(h),
                        border_width=0, sibling=0, stack_mode=ABOVE)
    _lib.XConfigureWindow(d, win, CWX | CWY | CWW | CWH | CWBORDER | CWSTACK, ctypes.byref(ch))
    _lib.XFlush(d)


def show(win):
    _lib.XMapWindow(display(), win)
    _lib.XFlush(display())


def hide(win):
    _lib.XUnmapWindow(display(), win)
    _lib.XFlush(display())


def dock(x, y, w, h, match=None, borderless=True):
    match = match or _DEFAULT_MATCH[0]
    with _lock():
        w_id = None
        for _ in range(4):                     # find_window 偶发瞬态失败（X 查询竞态）⇒ 重试
            w_id = find_window(match)
            if w_id:
                break
            time.sleep(0.3)
        if not w_id:
            return {"found": False}
        if borderless:
            set_decorated(w_id, False)
        show(w_id)
        set_above(w_id, True)
        move_resize(w_id, x, y, w, h)
        _lib.XRaiseWindow(display(), w_id)
        _lib.XSync(display(), 0)
        # 【Sim2Sim】返回**实测**几何：dockWindow 的自校正拿 st.x/st.y 与目标比对，贴歪了自动重贴。
        #   11-04 的「返回请求值」是为避开再 find_window 的瞬态误判——这里窗口 id 在手直接 _attrs，
        #   不重复找窗，无该风险；attrs 偶尔拿不到时退回请求值（自校正退化为不动作，无害）。
        a = _attrs(w_id)
        if a is not None:
            ax, ay = _abs_pos(w_id)
            return {"found": True, "x": ax, "y": ay, "w": a.width, "h": a.height}
        return {"found": True, "x": x, "y": y, "w": w, "h": h}


def undock(match=None):
    match = match or _DEFAULT_MATCH[0]
    # ⚠【P8 五槽】不能 hide()（XUnmapWindow）：对 GL/Vulkan 窗口 unmap 会摧毁渲染上下文
    #   ⇒ 仿真变 headless、窗口从此消失。
    #   降下＝只降层级（set_above False，窗口退到后面保持映射），GL 存活；重停靠时再置顶。
    with _lock():
        w_id = find_window(match)
        if not w_id:
            return {"found": False}
        set_above(w_id, False)
        return _status(match)


def rect_inside_window(x, y, w, h, min_size=120, skip_title=None):
    """请求的矩形（设备像素）是否真的落在某个可见顶层窗口里。

    嵌入式面板（webview）里 window.screenX/outerWidth 读不到真值，前端会算出荒唐的坐标；
    如果那个矩形下面根本没有窗口，就说明坐标不可信。判据用矩形中心点 + 面积足够大。
    """
    skip_title = skip_title or _DEFAULT_MATCH[0]
    cx, cy = x + w // 2, y + h // 2
    with _lock():
        d = display()
        root = _lib.XDefaultRootWindow(d)
        r = c_ulong(); p = c_ulong()
        kids = ctypes.POINTER(c_ulong)(); n = c_uint()
        if not _lib.XQueryTree(d, root, ctypes.byref(r), ctypes.byref(p), ctypes.byref(kids), ctypes.byref(n)):
            return None
        ids = [kids[i] for i in range(n.value)]
        for win in ids:
            a = _attrs(win)
            if not a or a.map_state != MAP_STATE_VIEWABLE:
                continue
            if a.width < min_size or a.height < min_size:
                continue
            nm = _name(win)
            if skip_title and skip_title in nm.lower():
                continue
            ax, ay = _abs_pos(win)
            if ax <= cx < ax + a.width and ay <= cy < ay + a.height:
                return {"title": nm or "(无标题)", "x": ax, "y": ay, "w": a.width, "h": a.height}
        return None


def _main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    w = find_window()
    if cmd == "status":
        print(status())
    elif cmd == "dock":
        x, y, ww, hh = (int(v) for v in sys.argv[2:6])
        print(dock(x, y, ww, hh))
    elif cmd == "hide":
        print(undock())
    elif cmd == "show":
        print(dock(*[int(v) for v in sys.argv[2:6]]) if len(sys.argv) > 5 else (show(w) or status()))
    elif cmd == "decorated":
        set_decorated(w, True); print(status())
    else:
        print("用法: xwin.py status|dock X Y W H|hide|show X Y W H|decorated")


if __name__ == "__main__":
    _main()
