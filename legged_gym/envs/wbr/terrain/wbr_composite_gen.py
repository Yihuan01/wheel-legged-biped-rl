# ==============================================================================
# Date      : 2026-10-29
# Author    : SYH
# File      : wbr_composite_gen.py
# Purpose   : 组合地形生成器：元素首尾相接的演示走廊，以及 3 档 × 3 种排列的组合图。
#             几何常量复用 wbr_terrain_gen.py，避免两套口径漂移。
# ==============================================================================

import argparse
import json
import math
import os

import numpy as np

# 几何常量与单元素图共用一份，避免两套口径漂移
from wbr_terrain_gen import (H_SCALE, V_SCALE, BORDER, SLOPE_SOFT,
                             SLOPE_SOFT_LONG, H_CAP, T7_TOP, T7_PIT, LEVELS,
                             T6_LEDGE_FRAC, T6_MIN_LEDGE)

# ---------------------------------------------------------------- 走廊几何
DEMO_LEN = 47.0          # 出生平台 3 + 走廊 44
DEMO_WID = 8.0
DEMO_SPAWN = 3.0
COMBO_LEN = 20.0         # 组合地形每格 20 m：够走 8~10 段，又让课程按整格升档
COMBO_WID = 8.0
COMBO_SPAWN = 2.0
END_PAD = 1.0            # 每格末端缓冲平地


# ============================================================ 元素剖面（沿 +x）
def _px(m):
    return max(int(round(m / H_SCALE)), 1)


def _rounded_step(n, h, lu, ld, hold=0.0):
    """一段完整元素：原地 → 缓升 lu → 台面 hold → 缓降 ld → 回 0。

    `lu/ld/hold` 放不下时按比例压缩缓坡（台面优先保住），保证整段严格落在 [0, n) 内、
    末格正好是 0（段与段之间高度连续，机器人可来回走）。
    """
    seg = np.zeros(n)
    lu, ld = max(int(lu), 1), max(int(ld), 1)
    hp = max(int(round(hold / H_SCALE)), 0)
    if lu + hp + ld > n:
        avail = max(n - hp, 2)
        k = avail / float(lu + ld)
        lu = max(int(lu * k), 1)
        ld = max(int(ld * k), 1)
        while lu + hp + ld > n:      # 收尾对齐，多出来的一格给下坡
            if lu > ld:
                lu -= 1
            else:
                ld -= 1
    i = 0
    seg[i:i + lu] = h * np.arange(1, lu + 1) / lu
    i += lu
    if hp > 0:
        seg[i:i + hp] = h
        i += hp
    seg[i:i + ld] = h * (1.0 - np.arange(1, ld + 1) / ld)
    return seg


def seg_flat(n, L, p):
    return np.zeros(n)


def seg_long_slope(n, L, p):
    """纵坡：缓升 → 台面 → 缓降（一段土丘），回来落在 0。

    ⚠ `_rounded_step` 按**采样格**线性爬升，实际坡度 = 峰高/爬升格数/0.1 m，会比档位角陡
    ⇒ 这里按精确格数换算，峰高与坡角同时成立（n 不够时自动降峰）。
    """
    a = min(float(p.get("deg", SLOPE_SOFT_LONG)), SLOPE_SOFT_LONG)
    # 坡度按 15% 的纵向余量算（给横向包络与 clamp 留空间，避免 15° 档被压出假台阶）
    ramp = max(int(n * 0.30), 1)
    h = math.tan(math.radians(a * 0.85)) * ramp * H_SCALE
    return _rounded_step(n, h, ramp, ramp, hold=0.0)


def seg_side_slope(n, L, p):
    """单边横坡：纵向包络 bell（两端缓入缓出），横向 +y 侧线性抬升。"""
    return None          # 见 build_cell：横向元素单独处理


def seg_uneven(n, L, p):
    """左右高低不平：+y 侧整块抬 h，横向一条 ≤10° 缓坡接上去。"""
    return None


def seg_wave(n, L, p):
    amp = float(p.get("amp", 0.03))
    lam = max(float(p.get("lam", 1.2)),
              2 * math.pi * amp / math.tan(math.radians(SLOPE_SOFT)))
    xs = (np.arange(n) + 0.5) * H_SCALE
    w = amp * (1 - np.cos(2 * math.pi * xs / lam))
    r = max(_px(0.8), 1)                       # 两端各 0.8 m 收尾，留出段尾归零的余量
    env = np.clip(np.arange(n) / r, 0, 1) * np.clip((n - 1 - np.arange(n)) / r, 0, 1)
    return w * env


def seg_step_up(n, L, p):
    """台阶跳上：缓升到 ramp_h → **一格真实台沿**（竖向 ledge）→ 台面 → 缓降回 0。

    台沿是"相邻两个采样点之间的竖向落差"，不占水平长度 ⇒ 缓升段末格 = ramp_h、
    下一个采样点直接 = h。
    """
    h = float(p.get("h", 0.10))
    ledge = h * float(p.get("ledge_frac", 0.40))
    if ledge < T6_MIN_LEDGE and h > T6_MIN_LEDGE:
        ledge = T6_MIN_LEDGE
    ramp_h = max(h - ledge, 0.0)
    lu = max(int(round(ramp_h / math.tan(math.radians(SLOPE_SOFT)) / H_SCALE)), 1) if ramp_h > 0 else 0
    ld = max(int(round(h / math.tan(math.radians(SLOPE_SOFT)) / H_SCALE)), 1)
    hold = max(_px(0.8), 1)
    while lu + 1 + hold + ld > n:           # +1 = 台沿那一格
        if hold > 1:
            hold -= 1
        elif ld > 1:
            ld -= 1
        elif lu > 1:
            lu -= 1
        else:
            break
    seg = np.zeros(n)
    i = min(lu, n - 1)
    if i > 0:
        seg[:i] = ramp_h * np.arange(1, i + 1) / max(lu, 1)
    top0 = i                                   # 台沿：i-1 → i 是竖向跳变
    top1 = min(top0 + hold, n)
    seg[top0:top1] = h
    k = min(ld, n - top1)
    if k > 0:
        seg[top1:top1 + k] = h * (1.0 - np.arange(1, k + 1) / ld)
    return seg


def seg_step_down(n, L, p):
    """台阶跳下：高台 → 一格真实台沿 → 缓坡落地 → 回 0。"""
    h = float(p.get("h", 0.10))
    lu = max(int(round(h / math.tan(math.radians(SLOPE_SOFT)) / H_SCALE)), 1)
    ld = lu
    hold = max(_px(0.8), 1)
    while lu + hold + ld > n:
        if hold > 1:
            hold -= 1
        elif ld > 1:
            ld -= 1
        elif lu > 1:
            lu -= 1
        else:
            break
    seg = np.zeros(n)
    seg[:lu] = h * np.arange(1, lu + 1) / max(lu, 1)
    top0 = lu
    top1 = min(lu + hold, n)
    seg[top0:top1] = h
    k = min(ld, n - top1)
    if k > 0:
        seg[top1:top1 + k] = h * (1.0 - np.arange(1, k + 1) / ld)
    if ld > 0:
        seg[hold:hold + ld] = h * (1.0 - np.arange(1, ld + 1) / ld)
    return seg


def seg_obstacles(n, L, p):
    """离散障碍：交替高矮的横条，条间留平地（每条都回 0）。"""
    h = float(p.get("h", 0.09))
    wl, gp = _px(p.get("obs_len", 0.7)), _px(p.get("gap", 0.7))
    r = max(int(round(h / math.tan(math.radians(SLOPE_SOFT)) / H_SCALE)), 1)
    seg = np.zeros(n)
    x, k = r + _px(0.2), 0
    last = None
    while x + wl <= n - r - 2:
        hv = h * (0.7 if k % 2 else 1.0)
        seg[x, :] if False else None
        seg[x:x + r] = hv * np.arange(1, r + 1) / r        # 每条障碍前有 ≤10° 上台坡
        seg[x:x + wl] = hv
        last = (x, wl, h * (0.7 if k % 2 else 1.0))
        x += wl + gp
        k += 1
    if last is not None:                     # 末条障碍两侧各留 r 格缓坡
        x0, w0, hv = last
        seg[x0 + w0:x0 + w0 + r] = hv * (1.0 - np.arange(1, r + 1) / r)
    return seg


def seg_stones(n, L, p):
    """跳石：石墩 + 浅坑（−6 cm），两端各留一段地面便于上下。

    结构：平地 | 斜坡入坑 | 坑底 | 石墩（带 ≤10° 上台坡）| 坑 | … | 斜坡出坑 | 平地。
    ⚠ 坑底必须比段末地面低 → 坑的收尾要留出"抬回 0"的斜坡，否则段尾归零会造陡坡。
    """
    h = float(p.get("h", 0.08))
    s_, gp = _px(p.get("stone", 0.9)), _px(p.get("gap", 0.3))
    edge = _px(0.5)
    # 入坑坡与出坑坡：抬升量 = h - T7_PIT（从坑底抬到石顶）
    r = max(int(round((h - T7_PIT) / math.tan(math.radians(SLOPE_SOFT)) / H_SCALE)), 1)
    seg = np.zeros(n)
    x = edge
    # 入坑：0 → T7_PIT
    r_in = max(int(round((-T7_PIT) / math.tan(math.radians(SLOPE_SOFT)) / H_SCALE)), 1)
    if x + r_in < n:
        seg[x:x + r_in] = T7_PIT * np.arange(1, r_in + 1) / r_in
        x += r_in
    while True:
        # 坑（至少 0.2 m）
        pit = max(_px(0.2), gp)
        if x + pit + r + s_ + r_in > n - edge:
            break
        seg[x:x + pit] = float(T7_PIT)
        x += pit
        seg[x:x + r] = T7_PIT + (h - T7_PIT) * np.arange(1, r + 1) / r     # 上石墩
        x += r
        seg[x:x + s_] = h
        x += s_
        seg[x:x + r_in] = h - (h - T7_PIT) * np.arange(1, r_in + 1) / r_in  # 下坑
        x += r_in
    # 收尾：从坑底缓抬回 0
    rem = n - x                       # 收尾一直走到段末
    if rem > 0:
        seg[x:x + rem] = T7_PIT * (1.0 - np.arange(1, rem + 1) / rem)
    seg[-1] = 0.0                     # 段末必须归零
    return seg


SEG_BUILDERS = {
    "flat": seg_flat,
    "t4": seg_long_slope,
    "t2": seg_wave,
    "t5": seg_step_up,
    "t6": seg_step_down,
    "obs": seg_obstacles,
    "stones": seg_stones,
}

SEG_CN = {
    "flat": "平地", "t1": "左右高低不平", "t2": "波浪", "t3": "单边横坡",
    "t4": "纵坡", "t5": "台阶跳上", "t6": "台阶跳下", "obs": "离散障碍",
    "stones": "跳石(浅坑)",
}


# ============================================================ 走廊（一维剖面 + 横向元素）
def build_corridor(segs, length_m, width_m, spawn_m, flat_m=1.0):
    """segs = [(kind, 长度 m, 参数dict), ...]；返回 (hf_cell, spans)。

    hf_cell 单位 = V_SCALE 的整数，形状 (len_px, wid_px)，起点在格子的 (0,0)。
    横向元素（t1/t3）在 corpus 里单独叠加，且**在整条走廊上共用同一个包络**。
    """
    n_px, n_py = _px(length_m), _px(width_m)
    n_spawn = _px(spawn_m)
    z = np.zeros(n_px)
    spans = []
    lateral = []          # [(kind, x0, x1, params)]
    x = n_spawn
    step = math.tan(math.radians(SLOPE_SOFT)) * H_SCALE
    for kind, L, p in segs:
        n = max(_px(L), 2)
        if kind in ("t1", "t3"):
            lateral.append((kind, x, x + n, p))
            spans.append((kind, x, x + n))
        else:
            seg = SEG_BUILDERS[kind](n, L, p)
            # 段尾归零：末值 e 用整整一段线性扣掉，每格扣 e/n（≤1 格步长方为合格，
            #   不合格说明该元素排得太挤，clamp 会再兜底）
            tail = float(seg[-1])
            if abs(tail) > 1e-9:
                seg = seg - tail * (np.arange(1, n + 1) / n)
            z[x:x + n] = seg
            spans.append((kind, x, x + n))
        x += n
    if x > n_px:                      # 超出就截断（生成器内部已经按总长配好，防御性）
        x = n_px
    Z = np.repeat(z[:, None], n_py, axis=1)

    for kind, x0, x1, p in lateral:
        n = x1 - x0
        xs = np.arange(n)
        # 纵向包络：两端各 0.5 m 缓入缓出，并要求**相邻两行高差 ≤ 1 格**（≤10°）。
        #   ⚠ 不能用 clip 包络：n=25 格、r=5 格时首四行 0.2/0.6/1.0/1.4 ⇒ 单格 6~8 cm 假墙。
        k = _px(0.5)
        if kind == "t3":
            deg = float(p.get("deg", 5.0))
            ys = (np.arange(n_py) + 0.5) * H_SCALE - width_m / 2.0
            lat = np.minimum(np.maximum(ys, 0) * math.tan(math.radians(deg)), H_CAP)
            lat = np.round(lat / H_SCALE) * H_SCALE
            full = float(lat.max())
        else:                          # t1：+y 半侧整块抬起，横向 ≤10° 缓坡接上
            h_want = min(float(p.get("h", 0.06)), H_CAP)
            ys_m = (np.arange(n_py) + 0.5) * H_SCALE
            y_mid = width_m / 2.0
            avail = max(int(round((width_m / 2.0) / H_SCALE)) - 2, 2)
            h = min(h_want, math.tan(math.radians(SLOPE_SOFT)) * avail * H_SCALE)
            lat = np.round(h * np.clip((ys_m - y_mid) / (avail * H_SCALE), 0.0, 1.0) / H_SCALE) * H_SCALE
            full = float(lat.max())
        # 纵向剖面的界（×full）：段首必须在 0、段末必须在 0，坡度 ≤1 格/行，
        #   ⚠ 不把剖面四舍五入到 0.1 m（那会让小坡变成"1 格墙"）
        prof = np.clip(np.minimum(np.minimum(xs, n - 1 - xs), k) * full / k, 0.0, full)
        prof[-1] = 0.0
        Z[x0:x1, :] += prof[:, None] * np.clip(lat, 0, H_CAP)[None, :] / max(full, 1e-6)

    # 末端缓冲 + 出生平台强制压平（所有元素都不许侵入）
    Z[n_px - _px(END_PAD):, :] = 0.0
    Z[:n_spawn, :] = 0.0
    # 竖向差限幅：每采样格 ≤ 1 格台沿（17.6 mm）。physx 在 1 格 80~140 mm 的竖面上
    #   会开出假坡度 ⇒ 组合地形统一按"≤10°/格"处理，不做单格竖墙。
    _clamp_cross_row(Z, flat_m=flat_m)

    # 段末 0.3 m 强制归零（限幅可能把末格抬到 ≤1 格：这一小段平地足够接缝）
    for (k, a, b) in spans:
        Z[max(b - _px(0.3), a):min(b, n_px), :] = 0.0
    span_m = [(k, a * H_SCALE, b * H_SCALE) for k, a, b in spans]
    return (np.round(Z / V_SCALE).astype(np.int16), span_m)


def _clamp_cross_row(Z, flat_m=1.0, limit_mm=None):
    """把沿 x 的相邻高差限制在 ≤ H_SCALE×tan(SLOPE_SOFT)（或 `limit_mm`）；
    超过的部分在该格**横向局部压平**（离台阶中心 ±0.5 m 内取两端之一）。

    2026-10-29：physx 在 1 格内 80~140 mm 的竖面上会开出假坡度（机器人直接翻），
    所以**连台阶元素也**必须压到 ≤17.6 mm/格 的工作区间；需要"真台沿"的验收改用
    x 方向 2~3 格、每格 ≤17.6 mm 的陡坡（合计可达 3.5~5 cm），不用单格竖面。
    """
    # 单个采样格允许的最大竖向差：10°（17.6 mm）。台阶元素自带的"一格台沿"也会被压到
    #   这个值 —— 不压的话 PhysX 在 1 格内 80~140 mm 的竖面上会开出假坡度，机器人直接翻。
    step = (limit_mm / 1000.0) if limit_mm else (math.tan(math.radians(SLOPE_SOFT)) * H_SCALE)
    n_px, n_py = Z.shape
    yc = n_py // 2
    def _d(i, lo, hi):
        """第 i 格相对前一格的横断高差（取包络内最大**上跳**与最大**下跳**）。"""
        if i <= 0:
            return np.inf, -np.inf
        d = Z[i, lo:hi] - Z[i - 1, lo:hi]
        return float(d.max()), float(d.min())

    def _set_row(i, lo, hi, v):
        """把第 i 格横断 [lo,hi) 压平到 v（v 是标量或 [hi-lo] 数组）。"""
        Z[i, lo:hi] = v

    # 一维剖面式限幅：只动超限那一格，上跳压回低值、下跳压回高值，逐格推进不回溯。
    lo, hi = 0, n_py
    # 正向扫一遍：上跳超限 ⇒ **只压超限的那几列**（不整格覆盖，避免整段被抹平）
    for i in range(1, n_px):
        a = Z[i - 1, lo:hi]
        b = Z[i, lo:hi]
        over = (b - a) > step
        if over.any():
            b[over] = a[over]
            Z[i, lo:hi] = b


def _wrap(hf_cell, num_rows, num_cols, row, col):
    """把一格贴进整图，返回整图与行列盒。"""
    bord = _px(BORDER)
    nr = max(num_rows, row + 1)
    nc = max(num_cols, col + 1)
    npx, npy = hf_cell.shape
    H_ = 2 * bord + nr * npx
    W_ = 2 * bord + nc * npy
    return H_, W_, bord, npx, npy


# ============================================================ ① 演示走廊
DEMO_SEGS = [
    ("flat", 4.0, {}),
    ("t4", 6.0, {"deg": 10.0}),
    ("t3", 5.0, {"deg": 8.0}),
    ("t1", 5.0, {"h": 0.08}),
    ("t2", 5.0, {"amp": 0.03, "lam": 1.2}),
    ("t5", 5.0, {"h": 0.086}),
    ("t6", 4.0, {"h": 0.086}),
    ("obs", 5.0, {"h": 0.09}),
    ("stones", 5.0, {"h": 0.08}),
]


def build_demo():
    cell, spans = build_corridor(DEMO_SEGS, DEMO_LEN, DEMO_WID, DEMO_SPAWN)
    bord = _px(BORDER)
    npx, npy = cell.shape
    hf = np.zeros((2 * bord + npx, 2 * bord + npy), dtype=np.int16)
    hf[bord:bord + npx, bord:bord + npy] = cell
    meta = {
        "name": "wbr_p8_demo", "kind": "demo_corridor",
        "h_scale": H_SCALE, "v_scale": V_SCALE, "border": BORDER,
        "env_len": DEMO_LEN, "env_wid": DEMO_WID,
        "num_rows": 1, "num_cols": 1,
        "tot_rows": int(hf.shape[0]), "tot_cols": int(hf.shape[1]),
        # 出生平台在走廊入口端，中心 = 平台一半处
        "spawn_center_m": [DEMO_SPAWN / 2.0, DEMO_WID / 2.0],
        "spawn_platform_m": DEMO_SPAWN,
        "span": DEMO_LEN,          # wbr_terrain_probe 等按此读格长
        "spans": [{"kind": k, "cn": SEG_CN[k], "x0_m": a, "x1_m": b, **{kk: vv for kk, vv in pp.items() if kk != "_seg_len"}}
                  for (k, a, b), pp in zip(spans, [s[2] for s in DEMO_SEGS])],
    }
    return hf, meta


# ============================================================ ② 组合地形（3 档 × 3 种排列）
# 三种排列只换元素顺序（L1→L3 都按同一顺序，难度由 LEVELS 控制）
COMBO_ORDERS = [
    [("flat", 3.0, {}),
     ("t4", 2.5, {}),
     ("t2", 2.5, {}),
     ("t5", 2.5, {}),
     ("obs", 2.5, {}),
     ("stones", 2.5, {}),
     ("t6", 2.5, {})],
    [("t3", 2.5, {}),
     ("t1", 2.5, {}),
     ("t6", 2.5, {}),
     ("t2", 2.5, {}),
     ("t5", 2.5, {}),
     ("obs", 2.5, {}),
     ("stones", 2.5, {})],
    [("t1", 2.5, {}),
     ("t2", 2.5, {}),
     ("t3", 2.5, {}),
     ("t4", 2.5, {}),
     ("obs", 2.5, {}),
     ("t5", 2.5, {}),
     ("t6", 2.5, {})],
]


def _order_params(order, level):
    """把档位参数摊进每段：返回 [(kind, L, params)]。"""
    lv = LEVELS[level]
    out = []
    for kind, L, p in order:
        q = dict(p)
        if kind in ("t4",):
            q.setdefault("deg", lv["t4_deg"])
        elif kind == "t2":
            q.setdefault("amp", lv["t2_amp"])
            q.setdefault("lam", lv["t2_lam"])
        elif kind in ("t5", "t6"):
            q.setdefault("h", lv["t5_h"])
        elif kind == "obs":
            q.setdefault("h", min(lv["t5_h"], 0.12))
        elif kind == "stones":
            q.setdefault("h", min(lv["t5_h"], 0.10))
        elif kind == "t1":
            q.setdefault("h", lv["t1_h"])
        elif kind == "t3":
            q.setdefault("deg", lv["t3_deg"])
        q["_seg_len"] = L
        out.append((kind, L, q))
    return out


def build_combo(level):
    bord = _px(BORDER)
    npx, npy = _px(COMBO_LEN), _px(COMBO_WID)
    num_rows, num_cols = len(LEVELS), len(COMBO_ORDERS)
    hf = np.zeros((2 * bord + num_rows * npx, 2 * bord + num_cols * npy), dtype=np.int16)
    cells = []
    for r, lv in enumerate(["L1", "L2", "L3"]):
        for c, order in enumerate(COMBO_ORDERS):
            segs = _order_params(order, lv)
            cell, spans = build_corridor(segs, COMBO_LEN, COMBO_WID, COMBO_SPAWN)
            hf[bord + r * npx: bord + (r + 1) * npx,
               bord + c * npy: bord + (c + 1) * npy] = cell
            cells.append({"row": r, "col": c, "level": lv,
                          "order": [k for k, _L, _p in segs],
                          "seg_params": [{k: _p[k] for k in _p if k != "_seg_len"}
                                         for _k, _L, _p in segs],
                          "max_height_m": float(cell.max() * V_SCALE),
                          "spans": [{"kind": k, "cn": SEG_CN[k], "x0_m": round(a, 2),
                                     "x1_m": round(b, 2)} for k, a, b in spans]})
    meta = {
        "name": "wbr_p8_combo", "kind": "combo", "level": level,
        "file_level": level,
        "h_scale": H_SCALE, "v_scale": V_SCALE, "border": BORDER,
        "env_len": COMBO_LEN, "env_wid": COMBO_WID,
        "num_rows": num_rows, "num_cols": num_cols,
        "tot_rows": int(hf.shape[0]), "tot_cols": int(hf.shape[1]),
        "spawn_center_m": [COMBO_SPAWN / 2.0, COMBO_WID / 2.0],
        "spawn_platform_m": COMBO_SPAWN,
        "levels": ["L1", "L2", "L3"], "params": LEVELS,
        "cells": cells,
    }
    return hf, meta


# ============================================================ 落盘
def default_out_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "release")


def save(obj, out_dir=None, name=None):
    """obj = (hf, meta) 或 (path, hf, meta)；返回 (path, hf, meta)。"""
    if isinstance(obj, tuple) and len(obj) == 3:
        obj = (obj[1], obj[2])
    hf, meta = obj
    out_dir = out_dir or default_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    fname = name or (f"{meta['name']}_{meta['file_level']}"
                     if meta.get("file_level") else meta["name"])
    path = os.path.join(out_dir, f"{fname}.npz")
    np.savez_compressed(path, height_field_raw=hf,
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    return path, hf, meta


def load(path):
    d = np.load(path, allow_pickle=False)
    return d["height_field_raw"], json.loads(str(d["meta"]))


def _report(tag, hf, meta):
    print(f"[gen] {tag}: shape={hf.shape}  "
          f"z={hf.min() * V_SCALE:+.3f}~{hf.max() * V_SCALE:+.3f} m")
    if meta.get("kind") == "demo_corridor":
        for s in meta["spans"]:
            print(f"       {s['x0_m']:5.1f}~{s['x1_m']:5.1f} m  {s['cn']}")


def main():
    ap = argparse.ArgumentParser(description="WBR P8 组合地形生成器")
    ap.add_argument("--what", default="all", choices=["demo", "combo", "all"])
    ap.add_argument("--level", default="all", choices=["L1", "L2", "L3", "all"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.what in ("demo", "all"):
        path, hf, meta = save(build_demo(), a.out)
        _report(path, hf, meta)
    if a.what in ("combo", "all"):
        for lv in (["L1", "L2", "L3"] if a.level == "all" else [a.level]):
            path, hf, meta = save(build_combo(lv), a.out)
            print(f"[gen] {path}  shape={hf.shape}  "
                  f"z={hf.min() * V_SCALE:+.3f}~{hf.max() * V_SCALE:+.3f} m")
            for c in meta["cells"]:
                print(f"       row{c['row']} col{c['col']} {c['level']}  "
                      f"h={c['max_height_m']:.3f} m  {'-'.join(c['order'])}")


if __name__ == "__main__":
    main()
