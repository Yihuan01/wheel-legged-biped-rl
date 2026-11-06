# ==============================================================================
# Date      : 2026-10-29
# Author    : SYH
# File      : wbr_rand_gen.py
# Purpose   : 随机起伏地形生成器：无向起伏曲线 → 高度图 .npz，产出训练分布图与留出用的
#             种子变体。纯 numpy，不需要 GPU。
# ==============================================================================

import argparse
import json
import os

import numpy as np

# ---------------------------------------------------------------- 几何常量
# ⚠ 前七项必须与训练侧 cfg.terrain 逐字一致：
H_SCALE = 0.1          # 水平分辨率 [m]  ↔ cfg.terrain.horizontal_scale
V_SCALE = 0.002        # 垂直分辨率 [m]  ↔ cfg.terrain.vertical_scale
                       #   5 mm 步长下 ±2 cm 档只有 8 个高度级、剖面呈阶梯状 ⇒ 用 2.5 mm。
BORDER = 25.0          # 外边界 [m]      ↔ cfg.terrain.border_size
ENV_LEN = 5.0          # 每格纵向长度 [m] ↔ cfg.terrain.terrain_length
                       #   基座正速度约 0.2 m/s：5 m 的格子让起伏在约 2 s 内就出现。
ENV_WID = 5.0          # 每格横向宽度 [m] ↔ cfg.terrain.terrain_width
NUM_ROWS = 4           # 行数（含平地行）  ↔ cfg.terrain.num_rows
NUM_COLS = 6           # 列数（独立随机种子）↔ cfg.terrain.num_cols

# ---------------------------------------------------------------- 起伏参数
KERNEL = 0.6           # 起伏颗粒度 [m]：自相关半高宽的目标尺度
                       #   0.6 m 约轮径 0.16 m 的 3.7 倍：再小起伏会碎成小台阶。
STEP_M = V_SCALE       # 高度量化步长 [m] = vertical_scale（两者必须一致）
SLOPE_CAP_DEG = 10.0   # 起伏坡度上限（默认档）：与"缓坡衔接、无垂直断崖"同口径
SLOPE_CAP = float(np.tan(np.radians(SLOPE_CAP_DEG)))
# 分档坡度上限（P8W6 2026-10-31 阶梯真实化）：幅度要在 0.6 m 颗粒的半宽（0.3 m）内
# 打满标称，所需坡度 ≈ atan(amp/0.3)。行 1/2 在 10° 下实测已能打满标称，维持不变；
# 行 3（±8 cm 需 ~15°）在 10° 下被截断+回收被拒，实测只剩 85%（6.8 cm）⇒ 放到 15°。
ROW_SLOPE_CAP_DEG = {1: 10.0, 2: 10.0, 3: 15.0}

# 行 → 峰谷半幅 [m]，一张图含完整难度阶梯，由课程（走得远 ⇒ 升档）往上爬：
#   行 0 平地 / 行 1 ±2 cm / 行 2 ±4 cm / 行 3 ±8 cm。
ROW_AMP = {1: 0.02, 2: 0.04, 3: 0.08}

# 出生平台：复位抖动 ±1 m（基类硬编码），中心必须平整
SPAWN_PLATFORM = 3.0   # [m] 出生平台边长
                       #   基类复位在中心 ±1 m 内抖动 ⇒ 3.0 m 平台（半宽 1.5）留 0.5 m 余量，
                       #   机器人不会出生在坡上。
SPAWN_FADE = 0.6       # [m] 平台到起伏区的过渡带（格长缩短后同步收窄）


def _smooth_noise(rng, nrow, ncol, amp, k_px, cap_deg=SLOPE_CAP_DEG):
    """生成满足三个约束的二维起伏：

    1. **幅度**恰好打满 ±amp（生成后归一化，避免随机采样的偶然不足）；
    2. **空间尺度**由 `k_px` 决定（高斯核平滑白噪声，核宽即自相关长度）；
    3. **坡度**处处 <= cap_deg（按坡度上限做软截断 + 再平滑）。

    与 isaacgym 的 `random_uniform_terrain`（下采样 + 样条插值）不同，这里用
    高斯平滑白噪声：平滑后的场天然各向同性、无方向性，且尺度参数有明确物理含义。
    """
    # 1) 白噪声 → 高斯核平滑（可分离，避免 FFT 依赖）
    raw = rng.standard_normal((nrow, ncol))
    rad = max(int(round(k_px)), 1)
    xs = np.arange(-3 * rad, 3 * rad + 1)
    k = np.exp(-0.5 * (xs / rad) ** 2)
    k /= k.sum()
    pad = rad * 3
    big = np.pad(raw, pad, mode="reflect")
    tmp = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, big)
    sm = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 0, tmp)
    sm = sm[pad:-pad, pad:-pad]

    # 2) 归一化到 ±amp
    m = np.abs(sm).max()
    if m > 1e-12:
        sm = sm / m * amp

    # 3) 坡度软截断：反复压制超限的相邻差，直到处处守 cap
    cap = float(np.tan(np.radians(cap_deg))) * H_SCALE
    for _ in range(30):
        over = 0
        for axis in (0, 1):
            d = np.diff(sm, axis=axis)
            bad = np.abs(d) > cap
            if not bad.any():
                continue
            over += int(bad.sum())
            # 对称地削掉超出部分：把这一对像素各拉回一半
            idx = np.where(bad)
            scale = np.ones_like(d)
            scale[bad] = cap / np.abs(d[bad])
            if axis == 0:
                sm[:-1, :] = sm[:-1, :] - d * (1 - scale) * 0.5
                sm[1:, :] = sm[1:, :] + d * (1 - scale) * 0.5
            else:
                sm[:, :-1] = sm[:, :-1] - d * (1 - scale) * 0.5
                sm[:, 1:] = sm[:, 1:] + d * (1 - scale) * 0.5
        if over == 0:
            break
        # 每次截断后轻平滑一次，避免出现"台阶化"痕迹
        sm = 0.5 * sm + 0.25 * (np.roll(sm, 1, 0) + np.roll(sm, -1, 0))

    # 4) 截断会损失幅度：只做一次温和回收（放大到 ±amp），再验一次坡度；
    #    若回收后坡度超限，则放弃回收（宁可比标称幅度小，也不出陡坡）
    m = np.abs(sm).max()
    if m > 1e-12:
        cand = sm / m * amp
        if _max_step(cand) <= cap * 1.05:
            sm = cand
    return sm


def _quantize_guard(z, amp, lim=2):
    """量化到高度图整数单位，并**在量化后**再压一次单格跳变。

    为什么需要：量化本身会引入"每格跳 STEP_M"的伪台阶。连续域里坡度合格时，
    粗量化步长仍会在个别相邻格造成超坡度的跳变，剖面看上去是"积木阶梯"。
    这里在**整数域**上做最后的局部平滑，保证相邻像素差 <= lim 个量化步。

    【P8W6 2026-10-31】lim 改为**逐行自适应**（调用方按传入值）：取"该行连续场
    自身最大像素步 + 1 个量化步"（只削量化伪台阶、不削场本身的坡度），并以该档
    坡度上限对应值为天花板。此前固定 lim=2（4 mm/px ≈ 2.3°）会把行 3 连续场里
    6°+ 的真实坡度一并压平，实测把 ±8 cm 档吃到 6.8 cm（85%）。

    ⚠ 平移必须**对称**：用 `(fix + 1) // 2` / `fix // 2` 这类整除在 fix 为奇数时
    一侧减 1、另一侧加 0，每次迭代整体下移，实测把整行推到限幅值（58°/160 mm 断崖）。
    这里改成先对半取整、左右用同一幅度，并把余数交替分配，保证总量守恒。
    """
    q = np.round(z / STEP_M).astype(np.int64)
    for _ in range(400):
        changed = False
        for axis in (0, 1):
            d = np.diff(q, axis=axis)
            bad = np.abs(d) > lim
            if not bad.any():
                continue
            changed = True
            excess = np.abs(d) - lim
            half = excess // 2                  # 两侧各让一半（整数，向下取整）
            rest = excess - 2 * half            # 余下的 1（可能是 0 或 1）
            sgn = np.sign(d)
            # 左端向中间靠拢 half+rest，右端靠拢 half ⇒ 总量守恒、方向正确
            left_shift = (half + rest) * sgn
            right_shift = half * sgn
            if axis == 0:
                q[:-1, :] += np.where(bad, left_shift, 0)
                q[1:, :] -= np.where(bad, right_shift, 0)
            else:
                q[:, :-1] += np.where(bad, left_shift, 0)
                q[:, 1:] -= np.where(bad, right_shift, 0)
        if not changed:
            break
    lim_units = int(round(amp / STEP_M))
    return np.clip(q, -lim_units, lim_units)


def _max_step(z):
    return max(np.abs(np.diff(z, axis=0)).max(), np.abs(np.diff(z, axis=1)).max())


def spawn_center_px():
    """出生平台中心（像素坐标）：**放在格子的入口端**而不是正中。

    为什么（P8W1 2026-10-28）：格长缩到 5 m 后，3 m 平台若居中就占掉 60%，
    起伏区只剩 2 m。挪到入口端 ⇒ 起伏区从 2.0 m 增到 3.5 m，同样格长下起伏样本翻倍。
    返回 (cx, cy) 浮点像素坐标；y 仍居中。
    """
    return (SPAWN_PLATFORM / 2.0 / H_SCALE,
            (ENV_WID / H_SCALE - 1) / 2.0)


def _spawn_mask(nrow, ncol):
    """出生平台权重：1 = 完全平（保持 0 高度），0 = 允许起伏，中间 smoothstep 过渡。

    x 方向平台居中在**入口端**（见 spawn_center_px）；y 方向居中。
    """
    xs = np.arange(nrow)
    ys = np.arange(ncol)
    cx, cy = spawn_center_px()
    half_pf = SPAWN_PLATFORM / 2.0 / H_SCALE
    fade = max(SPAWN_FADE / H_SCALE, 1.0)
    d = np.maximum(np.abs(xs - cx)[:, None] - half_pf,
                   np.abs(ys - cy)[None, :] - half_pf)
    t = np.clip(d / fade, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build(seed=0):
    """整张图一次性生成，按行给不同幅度，再按格切开。

    为什么不在每格里独立生成：独立生成会让相邻格的边界高度对不上（实测接缝 150 mm），
    机器人横穿格界时等于撞上一堵墙。整图连续生成后，格界两侧高度天然一致。

    幅度按行取 ROW_AMP（行 0 平地 / 行 1..3 = ±2/±4/±8 cm）⇒ 课程"走得远就升档"
    真的对应"在更起伏的地面上走"。
    """
    nrow = int(round(ENV_LEN / H_SCALE))
    ncol = int(round(ENV_WID / H_SCALE))
    bord = int(round(BORDER / H_SCALE))
    k_px = KERNEL / H_SCALE

    rng = np.random.default_rng(seed)
    hf = np.zeros((2 * bord + NUM_ROWS * nrow, 2 * bord + NUM_COLS * ncol), dtype=np.int16)

    # 逐行生成：每行一个幅度，行内沿 y 连续（跨列无缝）
    for r in range(1, NUM_ROWS):
        amp = ROW_AMP[r]
        band = _smooth_noise(rng, nrow, NUM_COLS * ncol, amp, k_px,
                             cap_deg=ROW_SLOPE_CAP_DEG.get(r, SLOPE_CAP_DEG))
        # ⚠ 出生平台必须在量化平滑之后再强制压平一次：
        #   _quantize_guard 的局部平滑会把平台边缘的起伏带进平台内部、让机器人在出生点就歪着。
        w = _spawn_mask(nrow, ncol)
        w_row = np.tile(w, (1, NUM_COLS))          # 每格入口端都有平台 ⇒ 掩蔽整行平铺
        # 【P8W6 2026-10-31 阶梯真实化】按**可行走区域**（掩蔽后）重新归一化到 ±amp：
        #   平台+过渡带占每格 3.6/5 m，随机场的最高峰常落在被抹零的带里，
        #   实测把行 2 的可用幅度从 4.0 cm 吃到 2.4 cm（60%）。机器只走掩蔽外的区域，
        #   所以"标称 ±amp"应该指掩蔽后的场，而不是整行原始场。
        m = np.abs(band * w_row).max()
        if m > 1e-12:
            band = band / m * amp
        # 量化 + 单格跳变压制（整行一起做，行内跨列保持连续）。
        # lim 逐行自适应：连续场最大像素步 +1 个量化步，上限=该档坡度对应值（见 _quantize_guard）。
        cap_deg = ROW_SLOPE_CAP_DEG.get(r, SLOPE_CAP_DEG)
        lim = int(np.ceil(_max_step(band) / STEP_M)) + 1
        lim = max(2, min(lim, int(np.ceil(np.tan(np.radians(cap_deg)) * H_SCALE / STEP_M))))
        q = _quantize_guard(band, amp, lim=lim)
        for c in range(NUM_COLS):
            seg = q[:, c * ncol: (c + 1) * ncol]
            q[:, c * ncol: (c + 1) * ncol] = np.round(seg * w).astype(np.int64)
        r0 = bord + r * nrow
        hf[r0:r0 + nrow, bord: bord + NUM_COLS * ncol] = q.astype(np.int16)

    cells = []
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            cell = hf[bord + r * nrow: bord + (r + 1) * nrow,
                      bord + c * ncol: bord + (c + 1) * ncol]
            cells.append({"row": r, "col": c,
                          "kind": "flat" if r == 0 else "random",
                          "amp_m": 0.0 if r == 0 else ROW_AMP[r],
                          "max_height_m": float(cell.max() * V_SCALE),
                          "min_height_m": float(cell.min() * V_SCALE)})

    meta = {"name": "wbr_rand", "seed": seed,
            "h_scale": H_SCALE, "v_scale": V_SCALE, "border": BORDER,
            "env_len": ENV_LEN, "env_wid": ENV_WID,
            "num_rows": NUM_ROWS, "num_cols": NUM_COLS,
            "tot_rows": int(hf.shape[0]), "tot_cols": int(hf.shape[1]),
            "row_amp_m": {str(k): v for k, v in ROW_AMP.items()},
            "kernel_m": KERNEL, "step_m": STEP_M,
            "slope_cap_deg": SLOPE_CAP_DEG,
            "row_slope_cap_deg": {str(k): v for k, v in ROW_SLOPE_CAP_DEG.items()},
            "spawn_platform_m": SPAWN_PLATFORM, "spawn_fade_m": SPAWN_FADE,
            # 出生平台中心（米，格内坐标）；训练侧据此算 env_origins，不能写死"格中心"
            "spawn_center_m": [SPAWN_PLATFORM / 2.0, ENV_WID / 2.0],
            "cells": cells}
    return hf, meta


def default_out_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "release")


def save(out_dir=None, seed=0, name="wbr_rand_curve"):
    hf, meta = build(seed)
    out_dir = out_dir or default_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(path, height_field_raw=hf,
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    return path, hf, meta


def load(path):
    d = np.load(path, allow_pickle=False)
    return d["height_field_raw"], json.loads(str(d["meta"]))


def main():
    ap = argparse.ArgumentParser(description="WBR P8 无向随机起伏地形生成器")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    path, hf, meta = save(a.out, a.seed)
    print(f"[rand] {path}")
    print(f"       shape={hf.shape}  z={hf.min() * V_SCALE:+.3f}~{hf.max() * V_SCALE:+.3f} m")
    for r in range(NUM_ROWS):
        amp = 0.0 if r == 0 else ROW_AMP[r]
        cap = ROW_SLOPE_CAP_DEG.get(r, SLOPE_CAP_DEG)
        print(f"       row{r}: 峰谷半幅 ±{amp * 100:.0f} cm  坡度上限 {cap:.0f}°")


if __name__ == "__main__":
    main()
