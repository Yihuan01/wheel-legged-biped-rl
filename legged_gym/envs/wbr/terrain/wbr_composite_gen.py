# ==============================================================================
# Date      : 2026-11-02
# Author    : SYH
# File      : wbr_composite_gen.py
# Purpose   : 组合地形生成器（纯 numpy，无 isaacgym 依赖）。
#
# 产出两套：
#   1) wbr_p8_demo.npz —— **四象限演示场**
#        13.5 m (x) × 7.0 m (y)，出生点在正中，四向各留 1 m 平地走廊。
#          前·左 (+x,+y)：波浪 4 m | 平地 1 m | 随机起伏 4 m
#          前·右 (+x,−y)：侧向斜坡 4 m | 平地 1 m | 单边减速带 4 m
#          后·左 (−x,+y)：下台阶 3.5 m
#          后·右 (−x,−y)：断开平台 3.5 m
#        四个区域机器人**前进方向都沿 x 轴**；后区进入方向为 −x，元素沿 x 镜像排列。
#   2) wbr_p8_combo_L{1,2,3}.npz —— 3 档 × 3 种排列的组合走廊，每格 20 m × 8 m。
#
# 元素定义全部来自 wbr_terrain_gen.py（同一份口径，避免两套漂移）。
# ==============================================================================

import argparse
import json
import math
import os

import numpy as np

from wbr_terrain_gen import (H_SCALE, V_SCALE, BORDER, SLOPE_SOFT, STEP,
                             SLOPE_DEG, SLOPE_H_MAX, SLOPE_RAMP_W, SLOPE_PLATFORM,
                             BUMP_AXIS_BAND, GAP_DEPTH, LEVELS, ELEMENTS,
                             _n, elem_wave, elem_rand, elem_slope,
                             elem_speedbump, elem_stairs, elem_gap)

# ---------------------------------------------------------------- demo 布局（米，机器人原点 = 出生点）
X_BACK = -4.0          # 后区远端
X_COR_N = -0.5         # 中央走廊近边
X_COR_P = +0.5
X_FRONT = +25.5        # 前区远端（8 + 1 + 16）
Y_RIGHT = -3.5         # 右区远端
Y_COR_N = -0.5
Y_COR_P = +0.5
Y_LEFT = +3.5          # 左区远端

DEMO_LEN = X_FRONT - X_BACK      # 13.5 m
DEMO_WID = Y_LEFT - Y_RIGHT      # 7.0 m
CORRIDOR_W = X_COR_P - X_COR_N   # 1.0 m
# 前区每区 2 个元素（斜坡区与连续起伏区长度翻倍后的布局）：
#   wave/slope 4 → 8 m，rand/speedbump 8 → 16 m
FRONT_EL = 8.0
FRONT_EL_LONG = 16.0
FRONT_GAP = 1.0
FRONT_SPAN = FRONT_EL + FRONT_GAP + FRONT_EL_LONG     # 25.0 m
BACK_LEN = 3.5                   # 后区元素长 3.5 m

# combo 走廊
COMBO_LEN = 20.0
COMBO_WID = 8.0
COMBO_SPAWN = 2.0
END_PAD = 1.0


# ============================================================ 工具
def _mirror_x(z):
    return z[::-1, :].copy()


def _blit(Z, z, x0, y0):
    """把 z 贴到 Z 的 (x0, y0) 起点（索引坐标）。"""
    nx, ny = z.shape
    Z[x0:x0 + nx, y0:y0 + ny] = z


# ============================================================ ① demo 四象限
DEMO_REGIONS = [
    # (标签, x0_m, x1_m, y0_m, y1_m, 行进方向)
    ("front_left", X_COR_P, X_FRONT, Y_COR_P, Y_LEFT, +1),
    ("front_right", X_COR_P, X_FRONT, Y_RIGHT, Y_COR_N, +1),
    ("back_left", X_BACK, X_COR_N, Y_COR_P, Y_LEFT, -1),
    ("back_right", X_BACK, X_COR_N, Y_RIGHT, Y_COR_N, -1),
]


def build_demo(level="L3"):
    """四象限演示场。返回 (hf, meta)。"""
    p = LEVELS[level]
    npx, npy = _n(DEMO_LEN), _n(DEMO_WID)
    Z = np.zeros((npx, npy))
    ox, oy = -X_BACK, -Y_RIGHT          # 米 → 索引偏移（cell 原点 = 左下角）

    def ix(xm):
        return int(round((xm - X_BACK) / H_SCALE))

    def iy(ym):
        return int(round((ym - Y_RIGHT) / H_SCALE))

    spans = []

    # ---- 前区：左 = 波浪 + 随机起伏；右 = 侧向斜坡 + 单边减速带
    for region, kinds in (("front_left", ("wave", "rand")),
                          ("front_right", ("slope", "speedbump"))):
        if region == "front_left":
            y0m, y1m = Y_COR_P, Y_LEFT
        else:
            y0m, y1m = Y_RIGHT, Y_COR_N
        w_m = y1m - y0m
        ny_r = _n(w_m)
        y0i = iy(y0m)
        x_cursor = X_COR_P
        for i, kind in enumerate(kinds):
            # rand / speedbump 长度翻倍（8 m），wave / slope 保持 4 m
            el_len = FRONT_EL_LONG if kind in ("rand", "speedbump") else FRONT_EL
            nx_r = _n(el_len)
            if kind == "slope":
                # 前右区（**左低右高**）：y=−0.5（靠中央走廊）低、
                #   y=−3.5（外侧）高 ⇒ 区域数组 col0 是高侧 ⇒ low_at=+1
                kw = dict(low_at=+1, plat=SLOPE_PLATFORM,
                          deg=float(p.get("slope_deg", SLOPE_DEG)),
                          width_m=w_m, fill_region=True)
                z = elem_slope(nx_r, ny_r, p, **kw)
            elif kind == "speedbump":
                z = elem_speedbump(nx_r, ny_r, p, seed=13, width_m=w_m)
            elif kind == "wave":
                z = elem_wave(nx_r, ny_r, p, seed=11)
            else:
                z = elem_rand(nx_r, ny_r, p, seed=12)
            _blit(Z, z, ix(x_cursor), y0i)
            spans.append({"kind": kind, "region": region,
                          "x0_m": round(x_cursor, 2),
                          "x1_m": round(x_cursor + el_len, 2),
                          "y0_m": round(y0m, 2), "y1_m": round(y1m, 2)})
            x_cursor += el_len
            if i == 0:
                x_cursor += FRONT_GAP      # 两元素之间 1 m 平地

    # ---- 后区：左 = 下台阶；右 = 断开平台（行进方向 −x ⇒ 沿 x 镜像）
    for region, kind in (("back_left", "stairs"), ("back_right", "gap")):
        if region == "back_left":
            y0m, y1m = Y_COR_P, Y_LEFT
        else:
            y0m, y1m = Y_RIGHT, Y_COR_N
        w_m = y1m - y0m
        ny_r = _n(w_m)
        nx_r = _n(BACK_LEN)
        z = elem_stairs(nx_r, ny_r, p) if kind == "stairs" else elem_gap(nx_r, ny_r, p)
        z = _mirror_x(z)                   # 机器人从 −x 方向进入 ⇒ 元素沿 x 镜像
        spans.append({"kind": kind, "region": region,
                      "x0_m": X_BACK, "x1_m": X_COR_N,
                      "y0_m": round(y0m, 2), "y1_m": round(y1m, 2)})
        _blit(Z, z, ix(X_BACK), iy(y0m))

    # ---- 中央十字走廊 + 出生点：强制压平
    Z[ix(X_COR_N):ix(X_COR_P), :] = 0.0
    Z[:, iy(Y_COR_N):iy(Y_COR_P)] = 0.0

    hf_cell = np.round(Z / V_SCALE).astype(np.int16)
    bord = _n(BORDER)
    hf = np.zeros((2 * bord + npx, 2 * bord + npy), dtype=np.int16)
    hf[bord:bord + npx, bord:bord + npy] = hf_cell

    meta = {
        "name": "wbr_p8_demo", "kind": "demo_quadrant", "level": level,
        "h_scale": H_SCALE, "v_scale": V_SCALE, "border": BORDER,
        "env_len": DEMO_LEN, "env_wid": DEMO_WID,
        "num_rows": 1, "num_cols": 1,
        "tot_rows": int(hf.shape[0]), "tot_cols": int(hf.shape[1]),
        # 机器人出生点（世界原点）在 cell 内的位置
        "spawn_center_m": [-X_BACK, -Y_RIGHT],
        "spawn_platform_m": 2.0,
        "span": DEMO_LEN,
        "corridor_w": CORRIDOR_W,
        "params": p,
        "spans": spans,
    }
    return hf, meta


# ============================================================ ② 组合走廊（3 档 × 3 排列）
COMBO_ORDERS = [
    [("flat", 2.0), ("wave", 3.0), ("rand", 3.0), ("slope", 3.0),
     ("speedbump", 3.0), ("stairs", 3.0), ("gap", 3.0)],
    [("flat", 2.0), ("slope", 3.0), ("stairs", 3.0), ("wave", 3.0),
     ("speedbump", 3.0), ("gap", 3.0), ("rand", 3.0)],
    [("flat", 2.0), ("speedbump", 3.0), ("rand", 3.0), ("stairs", 3.0),
     ("slope", 3.0), ("wave", 3.0), ("gap", 3.0)],
]


def build_corridor(kinds, level, length_m, width_m, spawn_m):
    """把若干元素沿 +x 首尾相接，每段前后各留 0.5 m 平地。返回 (z_m, spans)。"""
    p = LEVELS[level]
    npx, npy = _n(length_m), _n(width_m)
    Z = np.zeros((npx, npy))
    x = _n(spawn_m)
    pad = _n(0.5)
    spans = []
    for i, (kind, L) in enumerate(kinds):
        n = _n(L)
        kw = {}
        if kind == "slope":
            kw = dict(low_at=-1, ramp_w=SLOPE_RAMP_W, plat=SLOPE_PLATFORM,
                      deg=float(p.get("slope_deg", SLOPE_DEG)), width_m=width_m)
        elif kind == "speedbump":
            kw = dict(width_m=width_m)
        z = elem_wave(n, npy, p, seed=20 + i) if kind == "wave" else \
            elem_rand(n, npy, p, seed=20 + i) if kind == "rand" else \
            elem_slope(n, npy, p, **kw) if kind == "slope" else \
            elem_speedbump(n, npy, p, seed=20 + i, width_m=width_m) if kind == "speedbump" else \
            elem_stairs(n, npy, p) if kind == "stairs" else \
            elem_gap(n, npy, p) if kind == "gap" else np.zeros((n, npy))
        # 段首/段末各压 0.5 m 平地，保证接缝连续
        z[:min(pad, n), :] = 0.0
        z[max(n - pad, 0):, :] = 0.0
        if x + n > npx:
            break
        Z[x:x + n, :] = z
        spans.append({"kind": kind, "x0_m": round((x) * H_SCALE, 2),
                      "x1_m": round((x + n) * H_SCALE, 2)})
        x += n
    return Z, spans


def build_combo(level):
    p = LEVELS[level]
    bord = _n(BORDER)
    npx, npy = _n(COMBO_LEN), _n(COMBO_WID)
    num_rows, num_cols = 3, len(COMBO_ORDERS)
    hf = np.zeros((2 * bord + num_rows * npx, 2 * bord + num_cols * npy), dtype=np.int16)
    cells = []
    for r, lv in enumerate(["L1", "L2", "L3"]):
        for c, order in enumerate(COMBO_ORDERS):
            Z, spans = build_corridor(order, lv, COMBO_LEN, COMBO_WID, COMBO_SPAWN)
            cell = np.round(Z / V_SCALE).astype(np.int16)
            hf[bord + r * npx: bord + (r + 1) * npx,
               bord + c * npy: bord + (c + 1) * npy] = cell
            zz = cell.astype(float) * V_SCALE
            cells.append({"row": r, "col": c, "level": lv,
                          "order": [k for k, _L in order],
                          "max_height_m": float(zz.max()),
                          "min_height_m": float(zz.min()),
                          "spans": spans})
    meta = {
        "name": "wbr_p8_combo", "kind": "combo", "level": level, "file_level": level,
        "h_scale": H_SCALE, "v_scale": V_SCALE, "border": BORDER,
        "env_len": COMBO_LEN, "env_wid": COMBO_WID,
        "num_rows": num_rows, "num_cols": num_cols,
        "tot_rows": int(hf.shape[0]), "tot_cols": int(hf.shape[1]),
        "spawn_center_m": [COMBO_SPAWN / 2.0, COMBO_WID / 2.0],
        "spawn_platform_m": COMBO_SPAWN,
        "levels": ["L1", "L2", "L3"], "params": LEVELS, "cells": cells,
    }
    return hf, meta


# ============================================================ 落盘
def default_out_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "release")


def save(obj, out_dir=None, name=None):
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


def main():
    ap = argparse.ArgumentParser(description="WBR 组合地形生成器（demo 四象限 + combo 走廊）")
    ap.add_argument("--what", default="all", choices=["demo", "combo", "all"])
    ap.add_argument("--level", default="all", choices=["L1", "L2", "L3", "all"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.what in ("demo", "all"):
        path, hf, meta = save(build_demo("L3"), a.out)
        print(f"[gen] {path}  shape={hf.shape}  "
              f"{meta['env_len']}×{meta['env_wid']} m  z={hf.min() * V_SCALE:+.3f}~{hf.max() * V_SCALE:+.3f}")
        for s in meta["spans"]:
            print(f"       {s['region']:<12} {s['kind']:<10} "
                  f"x {s['x0_m']:+.1f}..{s['x1_m']:+.1f}  y {s['y0_m']:+.1f}..{s['y1_m']:+.1f}")
    if a.what in ("combo", "all"):
        for lv in (["L1", "L2", "L3"] if a.level == "all" else [a.level]):
            path, hf, meta = save(build_combo(lv), a.out)
            print(f"[gen] {path}  shape={hf.shape}  "
                  f"z={hf.min() * V_SCALE:+.3f}~{hf.max() * V_SCALE:+.3f}")


if __name__ == "__main__":
    main()
