# ==============================================================================
# Date      : 2026-10-26
# Author    : SYH
# File      : bench_models.py
# Purpose   : checkpoint 扫描与 obs/act 维度探测：批量走子进程探测并用 mtime 做缓存。
# ==============================================================================

import json
import os
import subprocess
import sys

from bench_store import REPO_ROOT, get_cache, save_cache

SCAN_ROOTS = ("releases", "logs")
MODEL_GLOB_PREFIX = "model_"

_cache = None
_cache_dirty = False

# 探测 obs/act 需要 torch ⇒ 用带 torch 的解释器；后端本身可跑在系统 python。
# 查找顺序：WBR_PYTHON → 当前解释器 → $CONDA_PREFIX/bin/python
_PROBE_CANDIDATES = (
    os.environ.get("WBR_PYTHON"),
    sys.executable,
    os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin", "python") if os.environ.get("CONDA_PREFIX") else None,
)


def env_python():
    for p in _PROBE_CANDIDATES:
        if p and os.path.isfile(p):
            return p
    return sys.executable


_PROBE_SNIPPET = r"""
import sys, json, os
import torch
paths = json.loads(sys.argv[1])
out = {}
for p in paths:
    try:
        sd = torch.load(p, map_location="cpu", weights_only=True)
        sd = sd.get("model_state_dict", sd)
        first = sd.get("actor.0.weight")
        if first is None:
            out[p] = None; continue
        idxs = [int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")]
        out[p] = [int(first.shape[1]), int(sd["actor.%d.weight" % max(idxs)].shape[0])]
    except Exception:
        out[p] = None
print(json.dumps(out))
"""


def _batch_probe(paths):
    """一次子进程批量探测，返回 {path: (obs,act)|None}。"""
    if not paths:
        return {}
    env = dict(os.environ)
    lib = os.path.join(os.path.dirname(os.path.dirname(env_python())), "lib")
    env["LD_LIBRARY_PATH"] = lib + ":" + env.get("LD_LIBRARY_PATH", "")
    env.setdefault("PYTHONWARNINGS", "ignore")
    try:
        res = subprocess.run([env_python(), "-c", _PROBE_SNIPPET, json.dumps(paths)],
                             capture_output=True, text=True, timeout=180, env=env)
        if res.returncode != 0:
            sys.stderr.write("[bench_models] probe failed: " + res.stderr[-400:] + "\n")
            return {}
        raw = json.loads(res.stdout.strip().split("\n")[-1])
        return {k: (tuple(v) if v else None) for k, v in raw.items()}
    except Exception as exc:
        sys.stderr.write(f"[bench_models] probe exception: {exc}\n")
        return {}


def _load_cache():
    global _cache
    if _cache is None:
        _cache = get_cache()
    return _cache


def flush_cache():
    global _cache_dirty
    if _cache_dirty and _cache is not None:
        save_cache(_cache)
        _cache_dirty = False


def _iter_model_files():
    for root in SCAN_ROOTS:
        base = os.path.join(REPO_ROOT, root)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            for fn in filenames:
                if fn.startswith(MODEL_GLOB_PREFIX) and fn.endswith(".pt"):
                    yield root, dirpath, fn


def scan(probe=True, limit_probe=None):
    """返回 checkpoint 列表。probe=False 时只给 stat（不读权重，用于超快首屏）。"""
    global _cache_dirty
    cache = _load_cache()
    out = []
    need = []          # [(rel, full, key)]
    for root, dirpath, fn in _iter_model_files():
        full = os.path.join(dirpath, fn)
        try:
            st = os.stat(full)
        except OSError:
            continue
        rel = os.path.relpath(full, REPO_ROOT)
        key = f"{st.st_mtime_ns}:{st.st_size}"
        entry = cache.get(rel)
        obs = act = None
        if entry and entry.get("key") == key:
            obs, act = entry.get("obs"), entry.get("act")
        elif probe and (limit_probe is None or len(need) < limit_probe):
            need.append((rel, full, key))
        out.append({
            "path": rel, "model": os.path.basename(dirpath), "ckpt": fn,
            "obs": obs, "act": act, "sizeMB": round(st.st_size / 1e6, 2),
            "source": "releases" if root == "releases" else "logs", "mtime": st.st_mtime,
            "_rel": rel,
        })

    if need:
        dims = _batch_probe([n[1] for n in need])
        by_full = {n[1]: n for n in need}
        for full, d in dims.items():
            n = by_full.get(full)
            if not n or not d:
                continue
            cache[n[0]] = {"key": n[2], "obs": d[0], "act": d[1]}
            _cache_dirty = True
        # 回填到 out
        for c in out:
            e = cache.get(c["_rel"])
            if e and e.get("key") and c["obs"] is None:
                c["obs"], c["act"] = e.get("obs"), e.get("act")
        if _cache_dirty:
            flush_cache()
    for c in out:
        c.pop("_rel", None)
    out.sort(key=lambda c: (c["source"] != "releases", -c["mtime"]))
    return out


def probe_one(path):
    """按路径探测单个文件（选择器里点开时用）。"""
    full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    if not os.path.isfile(full):
        return None
    st = os.stat(full)
    rel = os.path.relpath(full, REPO_ROOT)
    key = f"{st.st_mtime_ns}:{st.st_size}"
    cache = _load_cache()
    entry = cache.get(rel)
    if entry and entry.get("key") == key:
        return entry.get("obs"), entry.get("act")
    d = _batch_probe([full]).get(full)
    if d:
        cache[rel] = {"key": key, "obs": d[0], "act": d[1]}
        flush_cache()
    return d
