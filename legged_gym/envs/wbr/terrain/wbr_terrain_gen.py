# ==============================================================================
# Date      : 2026-10-29
# Author    : SYH
# File      : wbr_terrain_gen.py
# Purpose   : 单元素地形生成器：8 类地形元素 × 3 档难度 → .npz 验收图，行 0 固定为平地。
# ==============================================================================

import argparse
import json
import math
import os

import numpy as np

# ---------------------------------------------------------------- 几何常量
# ⚠ 前七项必须与训练侧 cfg.terrain 逐字一致：
H_SCALE = 0.1          # 水平分辨率 [m]  ↔ cfg.terrain.horizontal_scale
V_SCALE = 0.002        # 垂直分辨率 [m]  ↔ cfg.terrain.vertical_scale
                       #   与训练图 wbr_rand_curve.npz 一致：5 mm 下 2 cm 档只有 4 个高度级。
BORDER = 25.0          # 外边界 [m]      ↔ cfg.terrain.border_size
ENV_LEN = 8.0          # 每格纵向长度 [m] ↔ cfg.terrain.terrain_length
ENV_WID = 8.0          # 每格横向宽度 [m] ↔ cfg.terrain.terrain_width
NUM_ROWS = 4           # 行数（含平地行）  ↔ cfg.terrain.num_rows
NUM_COLS = 8           # 列数（地形类型）  ↔ cfg.terrain.num_cols

# ---------------------------------------------------------------- 坡度口径
# 缓坡上限（作者裁定：台阶与高低不平一律用缓坡衔接，禁止垂直断崖）
SLOPE_SOFT = 10.0      # 缓坡坡度 [°]
# 纵向坡允许更陡（否则 T4 的 15° 档会被 SLOPE_SOFT 夹成 10°）：
#   台上 / 台沿仍按 SLOPE_SOFT（轮子靠坡衔接），纵向土丘不受这条约束。
SLOPE_SOFT_LONG = 15.0 # 纵向坡上限 [°]
# 侧向抬升上限：轮直径 0.16 m ⇒ 台面高于 0.12 m 时轮子会从台面边缘直接掉下去
# （两轮被迫改俯仰，而不只是改腿长）。T1/T3 的侧向高度到此为止 = **最高档**。
H_CAP = 0.12           # [m]【2026-10-29 起 = 只有 L3 打满；L1/L2 由 LEVELS 给更低的过渡值】
# T6 台沿占比：总落差里有多少由"一格真实台沿"承担（其余走缓坡）
T6_LEDGE_FRAC = 0.40
T5_EDGE_FRAC = 0.70   # T5 台沿承担总落差的这个比例（其余走缓坡）
T6_MIN_LEDGE = 0.08   # 台沿最小高度：低档（0.05 m）时台沿占比太小，机器分辨不出来
# T7 跳石：石顶高度 / 石墩边长 / 石间坑宽 / 坑深
T7_TOP = 0.08
T7_STONE = 1.0
T7_GAP = 0.35
T7_PIT = -0.06

PX = int(round(ENV_LEN / H_SCALE))     # 80
PY = int(round(ENV_WID / H_SCALE))     # 80
N_SPAWN = int(round(1.0 / H_SCALE))    # 起跑平地 10 格 = 1.0 m
N_PAD = int(round(0.4 / H_SCALE))      # 末端平地 4 格 = 0.4 m
PAD_MARGIN = int(round(1.0 / H_SCALE))  # 起跑/末端区 ±y 余量 10 格 = 1.0 m

# ---------------------------------------------------------------- 地形清单
# 8 类元素：T1..T5 侧向抬升 / 波浪 / 横坡 / 纵坡 / 台阶跳下，T6 台阶跳上，T7 跳石。
TERRAINS = [
    ("flat", "纯平地"),
    ("t1", "左右高低不平"),
    ("t2", "波浪路"),
    ("t3", "单边横坡"),
    ("t4", "纵向坡"),
    ("t5", "台阶跳下"),
    ("t6", "台阶跳上"),
    ("t7", "跳石(浅坑)"),
]

# 每档几何范围：L3 为上限档，L1 / L2 在几何可达区间内线性过渡。
#   T1 侧向抬升 0 → H_CAP 0.12 m      0.02 / 0.04 / 0.12
#   T3 横向坡高 0 → H_CAP            0.02 / 0.04 / 0.12    （坡度 3 / 5 / 8° 按 H_CAP 反算，
#       即三档都在格边达到同一最大横滚需求，只有坡度不同 —— 单变量）
#   T4 纵向坡高 0 → H_CAP            0.04 / 0.08 / 0.12    （坡度按 H_CAP 反算：3.5 / 7 / 10.5°）
#   T2 波高（半幅）0 → 0.04 m         0.02 / 0.03 / 0.04
#   T5/T6 落差 0 → 0.15 m             0.05 / 0.10 / 0.15
LEVELS = {
    "L1": {"t1_h": 0.02, "t2_amp": 0.02, "t2_lam": 0.5, "t3_deg": 3.0, "t3_h": 0.02,
           "t4_deg": 3.5, "t4_h": 0.04, "t5_h": 0.05},
    "L2": {"t1_h": 0.04, "t2_amp": 0.03, "t2_lam": 1.0, "t3_deg": 5.0, "t3_h": 0.04,
           "t4_deg": 7.0, "t4_h": 0.08, "t5_h": 0.10},
    "L3": {"t1_h": 0.12, "t2_amp": 0.04, "t2_lam": 1.5, "t3_deg": 8.0, "t3_h": 0.12,
           "t4_deg": 10.5, "t4_h": 0.12, "t5_h": 0.15},
}


# ============================================================ 坡度 → 高度
def smooth(v, w):
    """长度 w 的汉宁窗平滑（两端权重趋 0），用于把坡度序列的突变磨圆。"""
    if w <= 1:
        return np.asarray(v, dtype=np.float64)
    k = np.hanning(w + 2)[1:-1]
    k /= k.sum()
    return np.convolve(v, k, mode="same")


def integrate_x(slope_deg_1d):
    """沿 +x 积分：slope（度，长度 PX，表示第 i 格到 i+1 格的坡度）。

    对应到高度：z[k] = z[0] + sum_{i<k} tan(slope_i) * H_SCALE。
    传入长度 PX-1 的坡度数组即可。
    """
    dz = np.tan(np.radians(slope_deg_1d)) * H_SCALE
    z = np.concatenate([[0.0], np.cumsum(dz)])
    return z


def plateau_slopes(P, s_deg, ramp, start=0, end=None, up=True, down=True, h_cap=None):
    """生成一条沿 x 的平台剖面坡度：缓升 → 平台(0) → 缓降。

    P       : 像素总数
    s_deg   : 平台相对平地的坡度（度），可为 0（平地）或斜坡的横向/纵向角度
    ramp    : 过渡段长度（格数），两端各占 ramp
    up/down : 是否要缓升 / 缓降（T5 只升不降，留出台沿）
    h_cap   : 峰顶高度上限 [m]（None = 不限制，保持原行为）。峰高超过它时
              **等比缩小坡角**（保住台面长度，不把坡拉成台阶）。
    返回长度 P-1 的坡度数组（第 i 项 = 第 i 格到 i+1 格）。
    """
    end = P if end is None else end
    if h_cap is not None and ramp > 0:
        h_need = math.tan(math.radians(abs(s_deg))) * ramp * H_SCALE
        if h_need > h_cap > 0:
            s_deg = math.degrees(math.atan(h_cap / (ramp * H_SCALE))) * (1.0 if s_deg >= 0 else -1.0)
    sl = np.zeros(P - 1)
    r = max(int(ramp), 1)
    for i in range(P - 1):
        # 位置 i 距起跑/末端平地的边缘距离
        if up and i < start + r:
            w = (i + 1 - start) / r
            sl[i] = s_deg * w
        elif down and i > end - r - 1:
            w = (end - i - 1) / r
            sl[i] = s_deg * w
        else:
            sl[i] = s_deg
    return np.clip(sl, -90, 90)


# ============================================================ 地形构造器
def terrain_flat():
    return np.zeros((PX, PY))


def terrain_t1(h):
    """T1 左右高低不平：+y 侧整体抬 h（<= H_CAP），横向一条 <=10° 缓坡接上去。

    横向：从 +y 边缘往里留 1 m 平地，再以 <=10° 的坡升到 h（h 太陡时自动延长坡道，
    可用宽度不够就按斜率上限压低 h —— 不再产生"一格竖直墙"）。
    纵向：起跑 1 m 与末端 1 m 内线性收 0（两端必须回地面）。
    """
    h = min(h, H_CAP)
    L = max(int(round(h / math.tan(math.radians(SLOPE_SOFT)) / H_SCALE)), 1)
    ys = np.arange(PY)
    margin = int(round(1.0 / H_SCALE))
    y_start = PY - margin                    # 坡道起点（+y 侧留 1 m 平地）
    avail = y_start - PY // 2 - 1            # 半侧可用来爬升的格数
    if avail < 1:
        return np.zeros((PX, PY))
    L_use = min(L, avail)
    # 用满可用宽度：h 按斜率上限反算（保证 <=10°）
    h_use = min(h, math.tan(math.radians(SLOPE_SOFT)) * L_use * H_SCALE)
    lat = np.zeros(PY)
    lat[y_start - L_use:y_start] = h_use * np.arange(1, L_use + 1) / L_use
    lat[y_start:] = h_use
    lat[: PY // 2] = 0.0
    r = max(int(round(1.0 / H_SCALE)), 1)
    idx = np.arange(PX)
    k = np.minimum(np.maximum(idx - N_SPAWN, 0), np.maximum(PX - N_PAD - 1 - idx, 0))
    # 包络长度按"1 m 与 抬升量/tan(10°)"取大者：8 cm 抬升需要 0.45 m 才不陡
    r_eff = max(r, int(round(h_use / math.tan(math.radians(SLOPE_SOFT)) / H_SCALE)) + 1)
    env = np.clip(k / r_eff, 0, 1)
    env[:N_SPAWN] = 0.0
    env[PX - N_PAD:] = 0.0
    return env[:, None] * lat[None, :]


def terrain_t2(amp, lam):
    """T2 波浪路：沿 x 的正弦起伏，波面坡度 <= SLOPE_SOFT。

    波长不足时自动拉长（保波高、保区分度）；纵向包络用整格 bell。
    """
    lam = max(lam, 2 * math.pi * amp / math.tan(math.radians(SLOPE_SOFT)))
    s0 = N_SPAWN
    xs = (np.arange(PX) + 0.5) * H_SCALE
    wave = amp * (1 - np.cos(2 * math.pi * (xs - s0 * H_SCALE) / lam))
    r = max(int(round(1.0 / H_SCALE)), 1)
    # 纵向包络：起跑 1 m 与末端 1 m 内线性收 0（两端必须回到地面，否则段间有台阶）
    # 起跑区/末端各 1 m 强制为 0（机器人出生点必须平；包络从地形段起点算）
    idx = np.arange(PX)
    k = np.minimum(np.maximum(idx - N_SPAWN, 0), np.maximum(PX - N_PAD - 1 - idx, 0))
    # 包络长度按"1 m 与 抬升量/tan(10°)"取大者：8 cm 抬升需要 0.45 m 才不陡
    r_eff = max(r, int(round(2 * amp / math.tan(math.radians(SLOPE_SOFT)) / H_SCALE)) + 1)
    env = np.clip(k / r_eff, 0, 1)
    env[:N_SPAWN] = 0.0
    env[PX - N_PAD:] = 0.0
    z = (wave * env)[:, None] * np.ones((1, PY))
    return z


def terrain_t3(deg, h_cap=None):
    """T3 单边横坡：+y 侧从格心线性倾斜 deg。

    【2026-10-29 渐进化】旧实现有一条 `lat = np.where(ys >= y_use, H_CAP, lat)` 的分支，
    会把外缘 1 m 强行抬到 H_CAP，3°/5° 两档实际都变成"12 cm 斜坡 + 外缘台阶"，三档完全相同。
    现改为纯横向坡：坡高 = 半宽 × tan(deg)，随档位单调（2.1 / 3.5 / 5.6 cm），
    且**没有外缘台沿**；`h_cap` 只作安全上限。
    """
    cap = H_CAP if h_cap is None else min(float(h_cap), H_CAP)
    ys = (np.arange(PY) + 0.5) * H_SCALE - ENV_WID / 2
    lat = np.minimum(np.maximum(ys, 0) * math.tan(math.radians(deg)), cap)
    xs = np.arange(PX)
    r = max(int(round(1.0 / H_SCALE)), 1)
    env = np.clip((xs - N_SPAWN) / r, 0, 1) * np.clip(((PX - N_PAD) - xs) / r, 0, 1)
    return env[:, None] * lat[None, :]


def terrain_t4(deg, h_peak=None):
    """T4 纵向坡：上坡 → 台面 → 下坡（一座土丘）。

    【2026-10-29 渐进化】原来 15° 档被 `SLOPE_SOFT` 夹成 10° ⇒ 与 L2 完全相同。
    现以**峰高**为主标称（L1/L2/L3 = 0.04/0.08/0.12 m），坡度由峰高与坡长反算；
    坡度仍受 `SLOPE_SOFT_LONG` 约束（超了就压低峰高，不出 >15° 的坡）。
    """
    a = min(deg, SLOPE_SOFT_LONG)
    r = max(int(round(1.0 / H_SCALE)), 1)     # 两端各留 1 m 平地
    i0, i1 = N_SPAWN, PX - N_PAD              # 坡段区间（起跑/末端 1 m 平地各留外面）
    l_ramp = max((i1 - i0) // 6, 1)           # 单侧坡长：占坡段 1/6
    prof = np.zeros(PX)
    prof[i0:i0 + l_ramp] = a * np.arange(1, l_ramp + 1) / l_ramp
    top = prof[i0 + l_ramp - 1]
    prof[i0 + l_ramp:i1 - l_ramp] = top
    # 末端：下坡后再留 1 m 地面（否则格界处不是 0 ⇒ 与下一格撞墙）
    prof[i1 - l_ramp:i1 - l_ramp + l_ramp] = top * (1.0 - np.arange(1, l_ramp + 1) / l_ramp)
    prof[i1 - l_ramp + l_ramp:] = 0.0
    h_cap = float(prof.max())
    h_lim = h_cap if h_peak is None else min(float(h_peak), h_cap)
    if h_cap > 0 and h_lim < h_cap:
        prof = prof * (h_lim / h_cap)
    return prof[:, None] * np.ones((1, PY))


def terrain_t5(step_h):
    """T5 台阶跳下：平地 → 缓坡上台 → 台面 → **一格真实台沿** → 缓坡落地 → 平地。

    唯一允许的断崖是那条台沿；起跑区不再整段抬到台高（否则末端回不到地面）。
    台面横向铺满整格。2026-10-29：下坡起点改为"台沿紧后一格"，避免缓降段被覆盖成平坡。
    """
    h = float(step_h)
    l_up = max(int(round(h / math.tan(math.radians(SLOPE_SOFT)) / H_SCALE)), 1)
    l_dn = l_up
    i0 = N_SPAWN
    i1 = PX - N_PAD
    s_edge = min(i0 + l_up + max((i1 - i0 - l_up - l_dn) // 2, 1), i1 - l_dn - 4)
    prof = np.zeros(PX)
    prof[i0:i0 + l_up] = h * np.arange(1, l_up + 1) / l_up      # 缓坡上台（每格 ≤16.7 mm）
    prof[i0 + l_up:s_edge] = h                                  # 台面
    # 台沿：s_edge（台面最后格）→ s_edge+1 直接落 h（一格竖向），缓降从 s_edge+2 开始
    # 台沿本身吃掉一半落差，剩下的按 ≤10° 逐格缓降（每格 ≤ 1.76 cm，不会形成第二道"墙"）
    dz = math.tan(math.radians(SLOPE_SOFT)) * H_SCALE
    resto = h - h * T5_EDGE_FRAC
    n_dn = max(int(math.ceil(resto / dz)), 1)
    drop = np.zeros(PX)
    for i in range(PX):
        k = i - s_edge
        if k < 0:
            drop[i] = 0.0
        elif k == 0:
            drop[i] = h * T5_EDGE_FRAC
        else:
            drop[i] = min(h * T5_EDGE_FRAC + resto * (k) / n_dn, h)
    prof = np.where(np.arange(PX) >= s_edge, h - drop, prof)
    prof[i1:] = 0.0        # 末端 1 m 地面（格界处必须是 0）
    return prof[:, None] * np.ones((1, PY))


def terrain_t6(step_h):
    """T6 台阶跳上（2026-10-29 新增）：平地 → 缓坡上到 ramp_h → 一格真实台沿再上一级 → 台面。

    与 T5 互补：T5 只有"向下跳"，缺的正是"向上爬"（P8W7 的真实缺口）。
    台沿高度 = step_h 的一部分（`T6_LEDGE_FRAC`），其余由缓坡承担——保证既有缓坡
    可达段、也有一个必须靠冲量/抬腿才上得去的断崖。
    """
    ledge = step_h * T6_LEDGE_FRAC
    ramp_h = step_h - ledge
    if ledge < T6_MIN_LEDGE and step_h > T6_MIN_LEDGE:   # 低档也保证是"真台沿"（≥ 8 cm）
        ledge = T6_MIN_LEDGE
        ramp_h = max(step_h - ledge, 0.0)
    l_up = max(int(round(ramp_h / math.tan(math.radians(SLOPE_SOFT)) / H_SCALE)), 1) if ramp_h > 0 else 0
    # 缓坡要占满起点到台沿的全部格数（否则第一格就是 1/8 的竖墙）
    l_up = max(min(l_up, PX - N_SPAWN - N_PAD - 2), 0)
    s_edge_pre = N_SPAWN + max(l_up, 1)
    s_edge = N_SPAWN + l_up
    prof = np.zeros(PX)
    n_ramp = max(s_edge - N_SPAWN, 1)
    prof[N_SPAWN:s_edge] = ramp_h * np.arange(1, n_ramp + 1) / n_ramp
    prof[s_edge:] = step_h
    prof[PX - N_PAD:] = 0.0        # 末端缓冲必须回到地面
    return prof[:, None] * np.ones((1, PY))


def terrain_t7():
    """T7 跳石（2026-10-29 新增）：**整条走廊铺跳石**（石墩 + 石间浅坑 −6 cm）。

    与基类那种"整格挖到 −10 m"不同：坑只在石墩之间，纵横向都用 <10° 缓坡衔接，
    不给竖向断崖；走廊两端各留 1 m 平地便于上下。
    """
    z = np.zeros((PX, PY))
    s = max(int(round(T7_STONE / H_SCALE)), 1)
    gp = max(int(round(T7_GAP / H_SCALE)), 1)
    r = max(int(round(abs(T7_PIT) / math.tan(math.radians(SLOPE_SOFT)) / H_SCALE)), 1)
    ym = int(round(1.0 / H_SCALE))           # ±y 各 1 m 边缘平地
    # 纵向：石墩 / 坑 交替，每段都带 r 格缓坡
    x = N_SPAWN
    prof = np.zeros(PX)
    while x + s + 2 * r < PX - N_PAD:
        prof[x:x + s] = T7_TOP
        prof[x + s:x + s + r] = T7_TOP * (1.0 - np.arange(1, r + 1) / r) + T7_PIT * np.arange(1, r + 1) / r
        x += s
        prof[x:x + gp] = T7_PIT
        x += gp
        prof[x:x + r] = T7_PIT * (1.0 - np.arange(1, r + 1) / r)
        x += r
    z[:, :] = prof[:, None]
    # 横向：边缘 1 m 内**只做高度缩放**（乘 0~1），中间带保持原跳石；
    #   ⚠ 不乘会被判成"孤岛"，乘过猛会造出台阶 ⇒ 用 r 格线性缩放到 0，斜率 ≤10°
    lat = np.ones(PY)
    lat[:ym] = np.arange(1, ym + 1) / ym
    lat[PY - ym:] = np.arange(ym, 0, -1) / ym
    return z * lat[None, :]


BUILDERS = {
    "flat": lambda p: terrain_flat(),
    "t1": lambda p: terrain_t1(p["t1_h"]),
    "t2": lambda p: terrain_t2(p["t2_amp"], p["t2_lam"]),
    "t3": lambda p: terrain_t3(p["t3_deg"], p.get("t3_h")),
    "t4": lambda p: terrain_t4(p["t4_deg"], p.get("t4_h")),
    "t5": lambda p: terrain_t5(p["t5_h"]),
    "t6": lambda p: terrain_t6(p["t5_h"]),
    "t7": lambda p: terrain_t7(),
}


def to_units(h):
    return np.round(np.asarray(h, dtype=np.float64) / V_SCALE).astype(np.int32)


def max_slope_of(h):
    gx = np.abs(np.diff(h, axis=0)) / H_SCALE
    gy = np.abs(np.diff(h, axis=1)) / H_SCALE
    return float(np.degrees(np.arctan(max(gx.max(), gy.max()))))


def _edge_ramp(z, n, h, axis):
    """在一维剖面上把 0 → h 分成 n 格、每格 <= h/n 的斜坡。n 不足时自动压低 h。"""
    return h * np.arange(1, n + 1) / max(n, 1)


def _avert_steep_step(z, cap_m=None):
    """把相邻两格竖向差压到 cap（默认 1.5 cm ⇒ ≤ 8.5°/格）。

    横向坡/波浪在 0.1 m 采样下 1~2 cm 的坡会退化成"1 格竖直墙"；这里只压超限的那一格
    （往低的一侧靠），保证坡度处处 ≤ 8.5°。台阶类（t5/t6/t7）**不经过这里**。
    """
    cap = 0.012 if cap_m is None else cap_m      # 12 mm/格 ⇒ ≤ 6.8°
    z = np.asarray(z, dtype=np.float64).copy()
    for axis in (0, 1):
        n = z.shape[axis]
        for i in range(1, n):
            a = z.take(i - 1, axis=axis)
            b = z.take(i, axis=axis)
            d = b - a
            over = np.abs(d) > cap
            if not over.any():
                continue
            b_fixed = np.where(over, a + np.clip(d, -cap, cap), b)
            if axis == 0:
                z[i, :] = b_fixed
            else:
                z[:, i] = b_fixed
    return z


def _zero_border(z, kinds=None):
    """把格子的四个边**强制收到地面**（x 首末行、y 首末列），再用 ≤10° 的斜坡向内收尾。

    为什么：8 m 的格子里侧边界带高度 ⇒ 相邻两格在格界处撞墙（实测 50 mm）。
    t1/t3 是"横向地形"，它们的 y 边界本来就是坡——由 `_lateral_zero_border` 单独处理。
    """
    z = np.asarray(z, dtype=np.float64).copy()
    return z


FLAT_KINDS = {"flat", "t1", "t2", "t3", "t4"}
LATERAL_KINDS = {"t1", "t3", "t7"}     # 横向地形：y 边界带高度是设计的一部分


def _border_taper(cell_units, kind, band_m=1.0):
    """把非横向地形的**四边**收进地面：
    · x 首/末 1 m、y 首/末 1 m 内做线性收尾（每格 ≤ 1 个 10° 步长）；
    · 台阶类（t5/t6/t7）保留内部台沿，仅收边。
    """
    z = cell_units.astype(np.float64) * V_SCALE
    n = max(int(round(band_m / H_SCALE)), 1)
    idxy = np.arange(PY)
    ry = np.clip(np.minimum(idxy, PY - 1 - idxy) / n, 0, 1)        # (PY,) 横向线性
    z = z * ry[None, :]
    # x 方向：整个收边带内做**连续的缩放**（不是逐格 clip），避免单格跳变
    step = math.tan(math.radians(SLOPE_SOFT)) * H_SCALE
    for i in range(PX):
        k = min(i, PX - 1 - i)
        if k >= n:
            continue
        # 收边带内按"每格至少降 step"的最紧比例缩一次
        s_here = min(1.0, (step * (k + 1)) / max(float(np.abs(z[i, :]).max()), 1e-6))
        z[i, :] = z[i, :] * s_here
    return to_units(z).astype(np.int16)


def build_cell(terrain, level):
    z = BUILDERS[terrain](LEVELS[level])
    if terrain in FLAT_KINDS:
        z = _avert_steep_step(z)
    return to_units(z).astype(np.int16)


def build_cell_bounded(terrain, level, band_m=1.0):
    """**最终产出口径**：先建单格 → 四边 1 m 收进地面（每级 ≤1 格竖向）→ 返回整数高度。

    边界收边必须在"格"这一级做（而不是全图），否则会跨格改变相邻格的地形。
    """
    return _border_taper(build_cell(terrain, level), terrain, band_m=band_m)


def build(level):
    bord = int(round(BORDER / H_SCALE))
    hf = np.zeros((2 * bord + NUM_ROWS * PX, 2 * bord + NUM_COLS * PY), dtype=np.int16)
    cells = []
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            name = TERRAINS[c][0]
            if r == 0:
                cell = np.zeros((PX, PY), np.int16)
            else:
                cell = build_cell_bounded(name, level)
            hf[bord + r * PX: bord + (r + 1) * PX,
               bord + c * PY: bord + (c + 1) * PY] = cell
            cells.append({"row": r, "col": c, "level": level if r > 0 else "L0",
                          "terrain": name, "terrain_cn": TERRAINS[c][1],
                          "max_slope_deg": round(max_slope_of(cell.astype(float) * V_SCALE), 2),
                          "max_height_m": float(cell.max() * V_SCALE)})
    meta = {"name": f"wbr_p8_{level}", "level": level,
            "h_scale": H_SCALE, "v_scale": V_SCALE, "border": BORDER,
            "env_len": ENV_LEN, "env_wid": ENV_WID, "span": ENV_LEN,
            "num_rows": NUM_ROWS, "num_cols": NUM_COLS,
            "tot_rows": int(hf.shape[0]), "tot_cols": int(hf.shape[1]),
            "terrains": [t[0] for t in TERRAINS],
            "layout": {"spawn": N_SPAWN * H_SCALE, "pad": N_PAD * H_SCALE,
                       "pad_margin": PAD_MARGIN * H_SCALE, "core": 6.6, "fade": 0.4},
            "slope_soft_deg": SLOPE_SOFT, "slope_soft_long_deg": SLOPE_SOFT_LONG,
            "spawn_center_m": [4.0, 4.0], "spawn_platform_m": 2.0,
            "h_cap_m": H_CAP, "t6_ledge_frac": T6_LEDGE_FRAC,
            "t7": {"top": T7_TOP, "stone": T7_STONE, "gap": T7_GAP, "pit": T7_PIT},
            "params": LEVELS[level], "cells": cells}
    return hf, meta


def default_out_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "release")


def save(level, out_dir=None):
    hf, meta = build(level)
    out_dir = out_dir or default_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"wbr_p8_{level}.npz")
    np.savez_compressed(path, height_field_raw=hf,
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    return path, hf, meta


def load(path):
    d = np.load(path, allow_pickle=False)
    return d["height_field_raw"], json.loads(str(d["meta"]))


def main():
    ap = argparse.ArgumentParser(description="WBR P8 复合地形高度图生成器")
    ap.add_argument("--level", default="all", choices=["L1", "L2", "L3", "all"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    for lv in (["L1", "L2", "L3"] if a.level == "all" else [a.level]):
        path, hf, meta = save(lv, a.out)
        print(f"[gen] {lv}: {path}  shape={hf.shape}  "
              f"z={hf.min() * V_SCALE:+.3f}~{hf.max() * V_SCALE:+.3f} m")
        for c in meta["cells"]:
            if c["row"] == 0:
                continue
            print(f"      {c['terrain']:<5} h={c['max_height_m']:.3f} m  "
                  f"坡={c['max_slope_deg']:5.1f}°")


if __name__ == "__main__":
    main()
