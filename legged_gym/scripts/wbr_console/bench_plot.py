# ==============================================================================
# Date      : 2026-10-26
# Author    : SYH
# File      : bench_plot.py
# Purpose   : 绘图数据层：复用 plot02_core 的数据整理，补上 switch 面板所需的曲线。
# ==============================================================================

import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from deps import plot02_core as plot02  # 纯标准库模块，import 无副作用

# 运行记录目录：控制台把运行写在 wbr_console/wbr_bench_store/runs（见 bench_store.STORE_DIR = HERE/wbr_bench_store）。
# 该目录用于从录像回补四槽模型配置（两轮/趴下）。
STORE_RUNS = os.path.join(HERE, "wbr_bench_store", "runs")

# 允许用环境变量把遥测目录指到别处（自测/换盘）；默认就是 play02 的 wbr_play_data
DATA_DIR = os.environ.get("WBR_PLAY_DATA") or plot02.DATA_DIR
# 随仓库附带的样例遥测：真实目录为空时也能预览绘图页（真实文件优先）
SAMPLES_DIR = os.path.join(HERE, "samples")
PATTERNS = ("wbr_tel_*.tsv", "wbr_switch_tel_*.tsv")


def is_switch(name):
    return name.startswith("wbr_switch_tel")


def _path(name):
    return os.path.join(DATA_DIR, os.path.basename(name))


def _resolve(name):
    """真实目录里没有这份文件时，回退到随仓库的 samples/（同名）。"""
    base = os.path.basename(name)
    p = os.path.join(DATA_DIR, base)
    if os.path.isfile(p):
        return p
    q = os.path.join(SAMPLES_DIR, base)
    return q if os.path.isfile(q) else p


def list_logs():
    """两个前缀都列；真实目录为空时列 samples/，并标注 sample=True。按 mtime 新→旧。"""
    seen, out = set(), []
    for base in (DATA_DIR, SAMPLES_DIR):
        if not os.path.isdir(base):
            continue
        for pat in PATTERNS:
            for full in glob.glob(os.path.join(base, pat)):
                name = os.path.basename(full)
                if name in seen:
                    continue
                seen.add(name)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                out.append({"name": name, "kind": "switch" if is_switch(name) else "single",
                            "sizeMB": round(st.st_size / 1e6, 2), "mtime": st.st_mtime,
                            "sample": base == SAMPLES_DIR})
    out.sort(key=lambda x: (-x["mtime"], x["sample"]))
    return out


def _config_from_meta(meta, name):
    """从日志 meta 还原“这套录像用的模型配置”，供绘图页顶部胶囊显示。"""
    items = []
    if is_switch(name):
        smode = str(meta.get("standup_mode", "p7s") or "p7s").lower()
        for role, key in (("四轮", "4w"), ("起立", "standup"), ("两轮", "3w"), ("趴下", "lie")):
            if role == "起立" and smode in ("trad", "script"):
                # 传统/相位脚本起立：不加载 P7S 学习策略，改用 P7T 脚本链（script＝env 内 RiseScript，无权重）
                _lbl = "P7T 相位脚本（无权重）" if smode == "script" else "P7T 传统脚本链"
                items.append({"role": "起立（传统）", "model": _lbl, "ckpt": "", "path": ""})
                continue
            p = meta.get(key)
            if not p:
                continue
            d = os.path.basename(os.path.dirname(p)) or "?"
            items.append({"role": role, "model": d, "ckpt": os.path.basename(p), "path": p})
        if not items and meta.get("task"):
            items.append({"role": "task", "model": meta.get("task"), "ckpt": "", "path": ""})
    else:
        md = meta.get("model_dir", "")
        if md and md != "?":
            items.append({"role": "单策略", "model": os.path.basename(md.rstrip("/")) or md,
                          "ckpt": "model_%s" % meta.get("model_iter", "?"),
                          "path": md})
        elif meta.get("model_cli"):
            # 【P7F4】旧录像的 model_dir 恒为 "?"（rsl_rl 1.0.2 无 model_file_path 属性）：
            #   从 model_cli="<load_run>|<ckpt>" 兜底解析（load_run 可能是 run 目录或 .pt 路径）。
            _p, _, _it = str(meta.get("model_cli")).partition("|")
            _p = _p.strip()
            if _p and _p != "?":
                if _p.endswith(".pt") and os.path.isfile(_p):
                    items.append({"role": "单策略", "model": os.path.basename(os.path.dirname(_p)),
                                  "ckpt": os.path.basename(_p), "path": _p})
                else:
                    _it = _it.strip()
                    items.append({"role": "单策略", "model": os.path.basename(_p.rstrip("/")) or _p,
                                  "ckpt": ("model_%s" % _it) if _it and _it != "?" else "",
                                  "path": _p})
        elif meta.get("task"):
            items.append({"role": "task", "model": meta.get("task"), "ckpt": "", "path": ""})
    return items


def _run_start_epoch(run_id):
    """'20261025-220010_switch' → epoch；对不上返回 None。"""
    m = re.match(r"(\d{8})-(\d{6})", run_id or "")
    if not m:
        return None
    try:
        return time.mktime(time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"))
    except ValueError:
        return None


def _config_from_run(tsv_name):
    """switch 日志 meta 只记了 4w/standup；两轮/趴下取自对应那次运行 spec.json 的四槽。"""
    m = re.match(r"wbr_switch_tel_(\d{8}-\d{6})", tsv_name)
    if not m:
        return None
    log_epoch = _run_start_epoch(m.group(1))
    if log_epoch is None or not os.path.isdir(STORE_RUNS):
        return None
    best, best_gap = None, 1e9
    for rid in os.listdir(STORE_RUNS):
        ep = _run_start_epoch(rid)
        if ep is None:
            continue
        gap = abs(ep - log_epoch)
        if gap < best_gap:
            best_gap, best = gap, rid
    if best is None or best_gap > 3600:
        return None
    try:
        with open(os.path.join(STORE_RUNS, best, "argv.json"), "r", encoding="utf-8") as fh:
            spec = json.load(fh).get("spec", {})
    except Exception:
        return None
    slots = spec.get("slots") or {}
    params = spec.get("params") or {}
    smode = str(params.get("standup_mode", "p7s") or "p7s").lower()
    out = []
    for role, key in (("四轮", "fourw"), ("起立", "standup"), ("两轮", "w3"), ("趴下", "lie")):
        if role == "起立" and smode in ("trad", "script"):
            _lbl = "P7T 相位脚本（无权重）" if smode == "script" else "P7T 传统脚本链"
            out.append({"role": "起立（传统）", "model": _lbl, "ckpt": "", "path": ""})
            continue
        p = slots.get(key)
        if not p or not isinstance(p, str):   # 起立槽可能写成 {"type":"script"}（相位脚本，无权重）
            continue
        out.append({"role": role, "model": os.path.basename(os.path.dirname(p)) or "?",
                    "ckpt": os.path.basename(p), "path": p})
    return out or None


def meta(name):
    path = _resolve(name)
    if not os.path.isfile(path):
        return None
    idx = plot02.get_index(path)
    if not idx.times:
        return None
    config = _config_from_meta(idx.meta, name)
    if is_switch(name) and len(config) < 4:
        config = _config_from_run(name) or config
    return {
        "file": os.path.basename(path),
        "kind": "switch" if is_switch(name) else "single",
        "n": len(idx.times), "t0": idx.times[0], "t1": idx.times[-1],
        "dt": float(idx.meta.get("dt", 0.01)),
        "sample": os.path.dirname(path) == SAMPLES_DIR,
        "config": config,
    }


def overview(name):
    return plot02.read_overview(_resolve(name), 1400)


# ---------------------------------------------------------------- 统一面板
# 单策略与四槽共用 plot02.build_panel_cols 的五列模板，两类日志都直接走 plot02.read_window。
# 历史四槽日志缺新列时，read_window 只画有数据的系列，缺的面板留空。


def window(name, t0, t1, thin):
    path = _resolve(name)
    if not os.path.isfile(path):
        return None
    return plot02.read_window(path, t0, t1, thin)


def kind_json():
    return plot02.KIND
