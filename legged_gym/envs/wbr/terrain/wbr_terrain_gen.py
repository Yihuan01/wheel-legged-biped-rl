# ==============================================================================
# Date      : 2026-11-02
# Author    : SYH
# File      : wbr_terrain_gen.py
# Purpose   : 地形元素库 + 单元素验收图生成器（纯 numpy，无 isaacgym 依赖）。
#
# 口径：
#   元素从 11 个精简到 7 个，按"能力维度"分四类：
#     基线      flat
#     连续起伏  wave（3~4 峰、峰高与波长逐峰变化）、rand（各向同性随机起伏）
#     横向差动  slope（侧向平面斜坡 + 高侧 30 cm 平台 + 断崖落地）、
#               speedbump（单边减速带，左右错开、高度随机、轴线过渡 5 cm）
#     离散断崖  stairs（斜坡上 20 cm → 10 cm → 平地）、
#               gap（10 cm 平台 | 10 cm 断开 | 10 cm 平台）
#   删除：t1 左右高低不平 / t3 单边横坡 / t4 纵坡（被 slope 取代）、
#         t2 旧波浪（被 wave 取代）、bumps/twave（被 speedbump 取代）、
#         obs 离散障碍 / stones 跳石 / t6 上台阶（不做）。
#
# 单元素验收图：4 行（行0 = 平地基线，行1~3 = L1/L2/L3 档）× 7 列（7 个元素），格 8 m × 8 m。
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
BORDER = 25.0          # 外边界 [m]      ↔ cfg.terrain.border_size
ENV_LEN = 8.0          # 每格纵向长度 [m] ↔ cfg.terrain.terrain_length
ENV_WID = 8.0          # 每格横向宽度 [m] ↔ cfg.terrain.terrain_width
NUM_ROWS = 4           # 行数（含平地行）  ↔ cfg.terrain.num_rows
NUM_COLS = 7           # 列数（元素种类）  ↔ cfg.terrain.num_cols

SLOPE_SOFT = 10.0      # 缓坡上限 [°]：一切**可通行衔接**（上台坡、包络收尾）都不超过它
EDGE_MARGIN = 1.0      # 每格四周留的平地 [m]（保证格界不撞墙、出生点平）

# 侧向斜坡：8°，峰高上限 0.28 m ⇒ 横向跨度 = 0.28/tan(8°) ≈ 2.0 m
SLOPE_DEG = 8.0
SLOPE_RAMP_W = 2.0     # 横向斜坡跨度 [m]（左侧一个宽 2 m 左右的横向斜坡）
SLOPE_H_MAX = SLOPE_RAMP_W * math.tan(math.radians(SLOPE_DEG))   # 28.1 cm（由 8° 定）
SLOPE_PLATFORM = 0.30  # 右侧高台宽 [m]，截面 = 直角梯形

# 单边减速带：轴线过渡带宽 0.05 m（横向、非行进方向）
BUMP_AXIS_BAND = 0.05

# 随机起伏的颗粒度（自相关半高宽）：0.3 m
RAND_KERNEL = 0.3

# 断崖：gap 的缺口挖到底的深度（掉下去算摔）
GAP_DEPTH = 0.60

PX = int(round(ENV_LEN / H_SCALE))     # 80
PY = int(round(ENV_WID / H_SCALE))     # 80

STEP = math.tan(math.radians(SLOPE_SOFT)) * H_SCALE    # 单格允许的最大竖向差 [m]（17.6 mm）

# ---------------------------------------------------------------- 元素清单
ELEMENTS = [
    ("flat", "平地"),
    ("wave", "波浪(3~4峰变幅)"),
    ("rand", "随机起伏"),
    ("slope", "侧向斜坡"),
    ("speedbump", "单边减速带"),
    ("stairs", "下台阶"),
    ("gap", "断开平台"),
]

# 三档参数。L3 = 目标档（stairs 20/10 cm、gap 平台 10 cm + 断开 10 cm）。
LEVELS = {
    "L1": dict(wave_amp=0.060, wave_lam=(1.0, 1.3), wave_peaks=3, rand_amp=0.02,
               slope_deg=5.0, bump_h=(0.025, 0.035), bump_per_side=3,
               stairs_h1=0.10, stairs_h2=0.05, gap_h=0.05, gap_w=0.06),
    "L2": dict(wave_amp=0.100, wave_lam=(1.2, 1.6), wave_peaks=3, rand_amp=0.04,
               slope_deg=6.5, bump_h=(0.035, 0.048), bump_per_side=3,
               stairs_h1=0.15, stairs_h2=0.075, gap_h=0.08, gap_w=0.08),
    "L3": dict(wave_amp=0.150, wave_lam=(1.0, 1.8), wave_peaks=4, rand_amp=0.08,
               slope_deg=8.0, bump_h=(0.030, 0.060), bump_per_side=4,
               stairs_h1=0.20, stairs_h2=0.10, gap_h=0.10, gap_w=0.10),
}


# ============================================================ 工具
def _n(m):
    """米 → 采样格数（≥1）。"""
    return max(int(round(m / H_SCALE)), 1)


def _ramp_up(n, h):
    """n 格从 0 线性升到 h（调用方保证 h/n ≤ STEP）。"""
    return h * np.arange(1, n + 1) / max(n, 1)


def _n_ramp(h):
    """走完高度 h 所需的格数（每格 ≤ STEP）。"""
    return max(int(math.ceil(h / STEP - 1e-9)), 1)


# ============================================================ 元素：波浪（3~4 峰，逐峰变幅变波长）
def elem_wave(nx, ny, p, seed=0):
    """沿 +x 的 3~4 个峰，**峰高与波长逐峰不同**；横向（y）完全一致。

    逐峰用 sin² 半波：起止高度与一阶导都为 0 ⇒ 峰间自然回 0。
    **优先保峰高**：每峰按目标幅度算出"不超 10° 所需的最小长度"，长度不够时
    **先减峰数**（3~4 峰都放不下就退回 2 峰），而不是压低幅度——否则"加高"永远无效。
    """
    rng = np.random.default_rng(seed)
    npk_t = int(p.get("wave_peaks", 3))
    amp0 = float(p.get("wave_amp", 0.03))
    m = max(_n(EDGE_MARGIN), 1)                 # 两端各留 1 m 平地
    usable = nx - 2 * m
    if usable < 4:
        return np.zeros((nx, ny))
    cap = math.tan(math.radians(SLOPE_SOFT)) * H_SCALE
    # 峰幅度：0.70~1.0 × amp0（逐峰不同，确定性 seed）
    amps_all = amp0 * rng.uniform(0.70, 1.0, npk_t)
    # sin² 半波峰值斜率 = a·π/L ⇒ 保该幅度所需最小格数
    need_all = np.array([max(int(math.ceil(a * math.pi / cap)), 3) for a in amps_all])
    # 放不下就减峰数（3~4 → 2 → 1），仍放不下才压幅度
    npk = npk_t
    while npk > 1 and need_all[:npk].sum() > usable:
        npk -= 1
    amps = amps_all[:npk].copy()
    need = need_all[:npk].copy()
    if need.sum() > usable:
        # 连 1 峰都放不下 ⇒ 压幅度（这是唯一的兜底）
        k = usable / float(max(need.sum(), 1))
        amps = np.minimum(amps, amps * k)
        need = np.array([max(int(math.ceil(a * math.pi / cap)), 3) for a in amps])
    # 分配长度：最小需求 + 随机富余（波长各不相同）
    extra = usable - int(need.sum())
    if extra > 0:
        share = rng.dirichlet(np.ones(npk)) * extra
        lens = (need + np.floor(share)).astype(int)
        lens[-1] += usable - int(lens.sum())
    else:
        lens = need.copy()
    lens = np.maximum(lens, 3)
    z = np.zeros(nx)
    x = m
    for i in range(npk):
        n = int(lens[i])
        a = float(amps[i])
        n = min(n, nx - x)                       # 防御：不越界
        if n < 2:
            break
        t = (np.arange(n) + 0.5) / n
        z[x:x + n] = a * np.sin(math.pi * t) ** 2
        x += n
    return np.repeat(z[:, None], ny, axis=1)


# ============================================================ 元素：随机起伏（各向同性）
def _smooth_noise_2d(rng, nrow, ncol, amp, k_px):
    raw = rng.standard_normal((nrow, ncol))
    rad = max(int(round(k_px)), 1)
    xs = np.arange(-3 * rad, 3 * rad + 1)
    k = np.exp(-0.5 * (xs / rad) ** 2)
    k /= k.sum()
    pad = rad * 3
    big = np.pad(raw, pad, mode="reflect")
    tmp = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 0, big)
    sm = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, tmp)
    sm = sm[pad:-pad, pad:-pad]
    mx = np.abs(sm).max()
    if mx > 0:
        sm = sm / mx * amp
    # 坡度软截断：颗粒变细（0.3 m）后局部单格坡度会超 10°。
    #   直接用"当前最大单格差 / 上限"一次算出缩放系数（线性关系，不用迭代）
    cap = math.tan(math.radians(SLOPE_SOFT)) * H_SCALE
    dz = max(np.abs(np.diff(sm, axis=0)).max(), np.abs(np.diff(sm, axis=1)).max())
    if dz > cap:
        sm *= cap / dz
    return sm


def elem_rand(nx, ny, p, seed=0):
    """各向同性随机起伏（与训练分布同一套算法），出生端 2 m × 全宽压平。"""
    rng = np.random.default_rng(seed)
    amp = float(p.get("rand_amp", 0.04))
    z = _smooth_noise_2d(rng, nx, ny, amp, _n(RAND_KERNEL))   # 颗粒度 = RAND_KERNEL
    # 出生端压平 + **平滑过渡**：硬切会在压平边界造 30 mm 台阶（颗粒越细越明显）。
    #   压平长度按段长自适应（≤ 段长 1/4），否则 3 m 的短段会整段被压平。
    L_m = nx * H_SCALE
    m = max(_n(min(2.0, L_m * 0.25)), 1)
    fade = max(_n(min(1.0, L_m * 0.15)), 1)
    env = np.ones(nx)
    env[:m] = 0.0
    for i in range(min(fade, nx - m)):
        env[m + i] = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / fade))
    z = z * env[:, None]
    return z


# ============================================================ 元素：侧向斜坡
def elem_slope(nx, ny, p, low_at=-1, ramp_w=2.0, plat=0.30, deg=8.0,
               width_m=None, y0_m=0.0, fill_region=False):
    """侧向斜坡（**直角梯形截面**）：左侧一段宽 `ramp_w` 的横向斜坡 → 右侧 `plat` 宽的高台 → 断崖落地。

    口径：
      · 截面形状 = **直角梯形**：缓坡（斜边）在上，高台（顶边）在下，短直角边贴外墙；
      · 斜坡与平面**平滑过渡**（坡脚、坡顶都做圆角，不是折线）；
      · **坡高由坡度 8° 和坡宽算出**：h = ramp_w · tan(8°)（默认 2.0 m → **28.1 cm**）；
      · 滚落线沿机器人 **y 轴**（横向），沿 x 只在段中间存在、两端 ≤10°/格 过渡回平地。

    参数
      low_at   : 低侧在**区域内**的哪一端：−1 = −y 端低（向 +y 爬升），+1 = +y 端低
      ramp_w   : 斜坡水平跨度 [m]（默认 2.0）
      plat     : 高台宽度 [m]（默认 0.30）
      deg      : 坡角 [°]（默认 8.0）⇒ 高度 = ramp_w·tan(deg)
      width_m  : 该元素可用的横向总宽 [m]（区域宽）
    """
    ramp_n = max(_n(ramp_w), 2)
    plat_n = max(_n(plat), 1)
    # 横向斜坡**填满该区域**（`fill_region=True` 时）：
    #   斜坡宽 = 区域宽 − 平台宽。元素图里保持固定斜坡宽（2 m），否则 8 m 格会算出 1 m 高的坡。
    if fill_region and width_m is not None:
        ramp_w = max(float(width_m) - float(plat), 0.3)
        ramp_n = max(_n(ramp_w), 2)
    h = float(ramp_w) * math.tan(math.radians(deg))
    # --- 一维横向剖面：低 → 平滑爬升 → 高台 → 竖直断崖，占满整幅 ny
    #   先在 [0, ramp_n+plat_n) 里排好梯形，再按 low_at 决定贴哪一端、是否镜像
    span = ramp_n + plat_n
    prof = np.zeros(span)
    t = (np.arange(ramp_n) + 0.5) / ramp_n
    prof[:ramp_n] = 0.5 * (1.0 - np.cos(math.pi * t)) * h     # cos 缓入缓出（坡脚/坡顶一阶导 0）
    prof[ramp_n:] = h                                          # 高台
    lat = np.zeros(ny)
    if low_at > 0:
        # 低侧在 +y 端：梯形整体贴在数组末尾（高台贴 col 0，斜坡向 col 增大方向降）
        prof = prof[::-1]
        lat[max(ny - span, 0):] = prof[:min(span, ny)]
    else:
        # 低侧在 −y 端：梯形贴数组开头（低 → 高）
        lat[:min(span, ny)] = prof[:min(span, ny)]
    # --- 纵向包络：两端各 EDGE_MARGIN 平地，再用 ≤10°/格 爬到满幅
    m = max(_n(EDGE_MARGIN), 1)
    # cos 包络的峰值斜率是 secant 的 π/2 倍 ⇒ 格数按 π/2 放大，保证每格 ≤ STEP
    taper = max(int(math.ceil(h / STEP * (math.pi / 2.0))), 1)
    env = np.ones(nx)
    env[:m] = 0.0
    env[nx - m:] = 0.0
    for i in range(min(taper, max(nx // 2 - m, 1))):
        w = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / taper))   # 平滑爬升
        env[m + i] = w
        env[nx - 1 - m - i] = w
    return env[:, None] * lat[None, :]


# ============================================================ 元素：单边减速带
def elem_speedbump(nx, ny, p, seed=0, width_m=None):
    """单边减速带：半宽横条，左右**错开布置**、高度随机、一侧 3~4 个。

    · 每条横条沿 +x 有 ≤10° 的升/降坡（可通行），沿 y 只覆盖**半宽**；
    · **轴线附近过渡带 y 向 0.05 m**：内侧边缘距轴线 0.025 m，爬满在 0.075 m 处；
    · 左右交替：沿 +x 直行时左右轮依次踩到带顶。

    这一项的 0.05 m 过渡是**横向**的、不是行进方向，故不受 10° 缓坡约束。
    """
    rng = np.random.default_rng(seed)
    w_m = float(width_m if width_m is not None else ENV_WID)
    npy = ny
    c = npy // 2                                    # 轴线（采样格）
    edges = np.arange(npy) + 0.5
    y_m = edges * H_SCALE
    y_c = w_m / 2.0
    # s(y) ∈ [0,1]：距轴线 0.025→0.075 m 内从 0 爬到 1；左右对称
    dy = np.abs(y_m - y_c)
    band_lo, band_hi = BUMP_AXIS_BAND / 2.0, BUMP_AXIS_BAND / 2.0 + BUMP_AXIS_BAND
    s = np.clip((dy - band_lo) / (band_hi - band_lo), 0.0, 1.0)
    h_lo, h_hi = p.get("bump_h", (0.03, 0.06))
    per = int(p.get("bump_per_side", 3))
    m = max(_n(0.2), 1)
    usable = nx - 2 * m
    total = per * 2                                 # 左右交替，各 per 条
    if usable < total * 3:
        return np.zeros((nx, ny))
    slot = usable // total                          # 每条占的 x 长度（含间隔）
    z = np.zeros((nx, ny))
    x = m
    for k in range(total):
        h = float(rng.uniform(h_lo, h_hi))
        n_up = _n_ramp(h)
        n_top = max(slot - 2 * n_up - max(_n(0.15), 1), 1)
        n_dn = n_up
        if n_top < 1:
            n_top = 1
        prof = np.concatenate([_ramp_up(n_up, h), np.full(n_top, h), h * (1.0 - np.arange(1, n_dn + 1) / n_dn)])
        n_len = len(prof)
        if x + n_len > nx - m:
            break
        # 偶数条只占 +y 半侧、奇数条只占 −y 半侧 ⇒ 沿 +x 直行时左右轮依次踩到带顶
        if k % 2 == 0:
            lat = np.where(y_m > y_c, s, 0.0)
        else:
            lat = np.where(y_m < y_c, s, 0.0)
        z[x:x + n_len, :] = np.maximum(z[x:x + n_len, :], prof[:, None] * lat[None, :])
        x += n_len + max(_n(0.12), 1)                # 条间间隔
    return z


# ============================================================ 元素：下台阶
def elem_stairs(nx, ny, p):
    """下台阶：斜坡驶上 **h1 平台** → 平台 1 m → 竖直下一级 **h2** → 平台 1 m → 竖直下一级回平地。

    L3 = 20 cm → 10 cm → 0。台阶之间的落差是**竖直面**（离散元素不需要
    高台到地面的过渡，竖直面即可）。
    """
    h1 = float(p.get("stairs_h1", 0.20))
    h2 = float(p.get("stairs_h2", 0.10))
    plat = max(_n(1.0), 1)
    up = _n_ramp(h1)
    m = max(_n(0.4), 1)
    if m + up + plat + plat > nx:
        plat = max((nx - m - up) // 2, 1)
    prof = np.zeros(nx)
    i = m
    prof[i:i + up] = _ramp_up(up, h1)
    i += up
    prof[i:i + plat] = h1
    i += plat
    prof[i:i + plat] = h2                            # 竖直落一级（一格的跳变）
    i += plat
    prof[i:] = 0.0
    return np.repeat(prof[:, None], ny, axis=1)


# ============================================================ 元素：断开平台
def elem_gap(nx, ny, p):
    """断开平台：斜坡上 **h 平台** → 平台 1 m → **断开 w**（真断崖：挖到 GAP_DEPTH）→ 平台 1 m → 斜坡回平地。

    L3 = 平台 10 cm、断开 10 cm。断裂处是**竖直面 + 深坑**，掉下去即判定摔。
    """
    h = float(p.get("gap_h", 0.10))
    w = max(_n(p.get("gap_w", 0.10)), 1)
    plat = max(_n(1.0), 1)
    up = _n_ramp(h)
    m = max(_n(0.4), 1)
    if m + up + plat + w + plat + up > nx:
        plat = max((nx - m - w - 2 * up) // 2, 1)
    prof = np.zeros(nx)
    i = m
    prof[i:i + up] = _ramp_up(up, h)
    i += up
    prof[i:i + plat] = h
    i += plat
    prof[i:i + w] = -GAP_DEPTH                       # 断口：挖到底
    i += w
    prof[i:i + plat] = h
    i += plat
    n_dn = min(up, nx - i)
    if n_dn > 0:
        prof[i:i + n_dn] = h * (1.0 - np.arange(1, n_dn + 1) / n_dn)
    return np.repeat(prof[:, None], ny, axis=1)


# ============================================================ 统一入口
def build_element(kind, nx, ny, params, seed=0, **kw):
    """按元素名生成 (nx, ny) 的**米制**高度场。"""
    if kind == "flat":
        return np.zeros((nx, ny))
    if kind == "wave":
        return elem_wave(nx, ny, params, seed=seed)
    if kind == "rand":
        return elem_rand(nx, ny, params, seed=seed)
    if kind == "slope":
        return elem_slope(nx, ny, params, **kw)
    if kind == "speedbump":
        return elem_speedbump(nx, ny, params, seed=seed, width_m=kw.get("width_m"))
    if kind == "stairs":
        return elem_stairs(nx, ny, params)
    if kind == "gap":
        return elem_gap(nx, ny, params)
    raise KeyError(f"未知元素 {kind}")


def build_cell(kind, level):
    """单元素验收图的一格（8 m × 8 m，单位 = V_SCALE 的整数）。"""
    p = LEVELS[level]
    nx, ny = _n(ENV_LEN), _n(ENV_WID)
    kw = dict(low_at=-1, ramp_w=SLOPE_RAMP_W, plat=SLOPE_PLATFORM,
              deg=float(p.get("slope_deg", SLOPE_DEG)), width_m=ENV_WID) if kind == "slope" else {}
    z = build_element(kind, nx, ny, p, seed=7, **kw)
    # 行首 1 m 压平作为出生带（元素本身从 EDGE_MARGIN 起算，这里只是兜底）
    z[:_n(EDGE_MARGIN), :] = 0.0
    return np.round(z / V_SCALE).astype(np.int16)


def build(level):
    """单元素验收图：4 行（L0 平地 + L1~L3）× 7 列（元素）。"""
    bord = _n(BORDER)
    npx, npy = _n(ENV_LEN), _n(ENV_WID)
    hf = np.zeros((2 * bord + NUM_ROWS * npx, 2 * bord + NUM_COLS * npy), dtype=np.int16)
    cells = []
    for r in range(NUM_ROWS):
        for c, (kind, cn) in enumerate(ELEMENTS):
            cell = np.zeros((npx, npy), np.int16) if r == 0 else build_cell(kind, level)
            hf[bord + r * npx: bord + (r + 1) * npx,
               bord + c * npy: bord + (c + 1) * npy] = cell
            zz = cell.astype(float) * V_SCALE
            cells.append({"row": r, "col": c, "terrain": kind, "terrain_cn": cn,
                          "level": "L0" if r == 0 else level,
                          "max_height_m": float(zz.max()),
                          "min_height_m": float(zz.min())})
    meta = {"name": f"wbr_p8_{level}", "level": level,
            "h_scale": H_SCALE, "v_scale": V_SCALE, "border": BORDER,
            "env_len": ENV_LEN, "env_wid": ENV_WID, "span": ENV_LEN,
            "num_rows": NUM_ROWS, "num_cols": NUM_COLS,
            "tot_rows": int(hf.shape[0]), "tot_cols": int(hf.shape[1]),
            "tes": [e[0] for e in ELEMENTS],
            "spawn_center_m": [ENV_LEN / 2.0, ENV_WID / 2.0],
            "spawn_platform_m": 2.0,
            "slope_soft_deg": SLOPE_SOFT, "slope_deg": SLOPE_DEG,
            "slope_h_max": SLOPE_H_MAX, "slope_platform": SLOPE_PLATFORM,
            "bump_axis_band": BUMP_AXIS_BAND, "gap_depth": GAP_DEPTH,
            "params": LEVELS[level], "cells": cells}
    return hf, meta


# ============================================================ 下台阶训练图（P8W9）
# 与验收图分开：验收图用 stairs 元素（斜坡上台 + 两级落差），默认出生点在落差之后，
# 评测要靠 `--spawn-dx` 挪；训练图必须让**默认出生点就在高台面上**，所以元素从格子起点
# 就是 h1 平台，落差放在出生点前方 1.0~2.5 m，且 meta.spawn_center_m 改到 2.0 m。
DROP_ROWS = (0.0, 0.20, 0.22, 0.24, 0.26, 0.30)  # 总落差档（两级各一半）= 每级 0/10/11/12/13/15 cm；P8W9 主目标＝15 cm
DROP_DISTS = (3.0, 3.5, 4.0, 4.5)         # 落差位置（x, m）；出生 2.0 ⇒ 接近 1.0~2.5 m
DROP15_ROWS = (0.30, 0.30, 0.30, 0.20)   # 15 cm 专项图：3/4 的行是每级 15 cm，1 行每级 10 cm 防遗忘
DROP_SPAWN_X = 2.0


def elem_drop(nx, ny, h1, dist_m, plat_m=1.0):
    """下台阶训练元素：起点即 h1 平台 → dist 处竖直落到 h2=h1/2 → 1 m → 竖直落到 0。"""
    h1 = float(h1)
    h2 = 0.5 * h1
    i = max(_n(dist_m), 1)
    p = max(_n(plat_m), 1)
    prof = np.zeros(nx)
    prof[:i] = h1
    prof[i:i + p] = h2
    prof[i + p:] = 0.0
    return np.repeat(prof[:, None], ny, axis=1)


def build_drop_grid(rows=None):
    """下台阶训练图：len(rows) 行（落差档）× 4 列（接近距离），8 m × 8 m 格。"""
    rows = tuple(rows) if rows is not None else DROP_ROWS
    bord = _n(BORDER)
    npx, npy = _n(ENV_LEN), _n(ENV_WID)
    hf = np.zeros((2 * bord + len(rows) * npx,
                   2 * bord + len(DROP_DISTS) * npy), dtype=np.int16)
    cells = []
    for r, h in enumerate(rows):
        for c, d in enumerate(DROP_DISTS):
            if h <= 0.0:
                cell = np.zeros((npx, npy), np.int16)
            else:
                cell = np.round(elem_drop(npx, npy, h, d) / V_SCALE).astype(np.int16)
            hf[bord + r * npx: bord + (r + 1) * npx,
               bord + c * npy: bord + (c + 1) * npy] = cell
            zz = cell.astype(float) * V_SCALE
            cells.append({"row": r, "col": c, "terrain": "drop", "terrain_cn": "下台阶",
                          "level": f"D{r}", "drop_m": h, "dist_m": d,
                          "max_height_m": float(zz.max())})
    meta = {"grid": "drop", "env_len_m": ENV_LEN, "env_wid_m": ENV_WID,
            "num_rows": len(rows), "num_cols": len(DROP_DISTS),
            "rows_drop_m": list(rows), "cols_dist_m": list(DROP_DISTS),
            "spawn_center_m": [DROP_SPAWN_X, ENV_WID / 2.0],
            "spawn_platform_m": 1.0, "cells": cells}
    return hf, meta


def save_drop(out_dir=None, rows=None, name="wbr_p8w9_drop.npz"):
    hf, meta = build_drop_grid(rows)
    out_dir = out_dir or default_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    np.savez_compressed(path, height_field_raw=hf,
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    return path, hf, meta


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
    ap = argparse.ArgumentParser(description="WBR 地形元素图生成器（7 元素 × 3 档；--drop 出下台阶训练图）")
    ap.add_argument("--level", default="all", choices=["L1", "L2", "L3", "all"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--drop", action="store_true", help="生成下台阶 6 行课程图 wbr_p8w9_drop.npz")
    ap.add_argument("--drop15v", action="store_true",
                    help="生成单行 15 cm 回放图 wbr_p8w9_drop15v.npz（1 行 × 4 列，观看着用）")
    ap.add_argument("--drop15", action="store_true",
                    help="生成 15 cm 专项图 wbr_p8w9_drop15.npz（4 行：15/15/15/10 cm）")
    a = ap.parse_args()
    if a.drop15v:
        path, hf, meta = save_drop(a.out, rows=(0.30,), name="wbr_p8w9_drop15v.npz")
        print(f"[gen] drop15v: {path}  shape={hf.shape}  rows={meta['rows_drop_m']}  "
              f"spawn={meta['spawn_center_m']}")
        return
    if a.drop15:
        path, hf, meta = save_drop(a.out, rows=DROP15_ROWS, name="wbr_p8w9_drop15.npz")
        print(f"[gen] drop15: {path}  shape={hf.shape}  z={hf.max() * V_SCALE:+.3f} m  "
              f"rows={meta['rows_drop_m']} spawn={meta['spawn_center_m']}")
        return
    if a.drop:
        path, hf, meta = save_drop(a.out)
        print(f"[gen] drop: {path}  shape={hf.shape}  "
              f"z={hf.min() * V_SCALE:+.3f}~{hf.max() * V_SCALE:+.3f} m  "
              f"spawn={meta['spawn_center_m']}")
        for c in meta["cells"]:
            print(f"      {c['level']} col{c['col']} 落差 {c['drop_m'] * 100:4.1f} cm  "
                  f"位置 {c['dist_m']:.1f} m")
        return
    for lv in (["L1", "L2", "L3"] if a.level == "all" else [a.level]):
        path, hf, meta = save(lv, a.out)
        print(f"[gen] {lv}: {path}  shape={hf.shape}  "
              f"z={hf.min() * V_SCALE:+.3f}~{hf.max() * V_SCALE:+.3f} m")
        for c in meta["cells"]:
            if c["row"] != 1:
                continue
            print(f"      {c['terrain']:<10} {c['terrain_cn']:<14} "
                  f"峰 {c['max_height_m'] * 100:6.1f} cm")


if __name__ == "__main__":
    main()
