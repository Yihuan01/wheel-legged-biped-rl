# ==============================================================================
# Date      : 2026-10-26
# Author    : SYH
# File      : bench_store.py
# Purpose   : 存储层：settings / scenarios / saved / runs 四类数据的落盘与读取，
#             目录固定在本目录的 wbr_bench_store/。
# ==============================================================================

import json
import os
import re
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)                       # legged_gym/scripts
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))  # 仓库根
STORE_DIR = os.path.join(HERE, 'wbr_bench_store')


def _ensure(*parts):
    p = os.path.join(STORE_DIR, *parts)
    os.makedirs(p if not os.path.splitext(p)[1] else os.path.dirname(p), exist_ok=True)
    return p


def store_path(*parts):
    return os.path.join(STORE_DIR, *parts)


def safe_name(text, limit=90):
    """只留中日韩、字母数字和 . _ - ⇒ 不可能带路径分隔符。"""
    cleaned = re.sub(r"[^\w .\-]", "", str(text), flags=re.UNICODE).strip().lstrip(".")
    return cleaned[:limit] or "untitled"


def _read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ------------------------------------------------------------------ settings
def get_settings():
    return _read_json(store_path("settings.json"), {})


def save_settings(patch):
    cur = get_settings()
    cur.update(patch or {})
    _write_json(store_path("settings.json"), cur)
    return cur


# ------------------------------------------------------------------ 场景
def list_scenarios():
    d = _ensure("scenarios")
    out = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        obj = _read_json(os.path.join(d, name), None)
        if obj is None:
            continue
        out.append({
            "id": name[:-5],
            "name": obj.get("name", name[:-5]),
            "mode": obj.get("mode", "switch"),
            "when": obj.get("when", ""),
            "slots": obj.get("slots"),
            "single": obj.get("single"),
            "params": obj.get("params", {}),
        })
    return out


def save_scenario(name, payload):
    name = safe_name(name)
    payload = dict(payload or {})
    payload.setdefault("name", name)
    payload["when"] = payload.get("when") or time.strftime("%m-%d %H:%M")
    _write_json(_ensure("scenarios", name + ".json"), payload)
    return payload


def load_scenario(name):
    return _read_json(store_path("scenarios", safe_name(name) + ".json"), None)


def delete_scenario(name):
    p = store_path("scenarios", safe_name(name) + ".json")
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


# ------------------------------------------------------------------ 收藏固化
def list_saved():
    d = _ensure("saved")
    out = []
    for name in sorted(os.listdir(d)):
        meta = _read_json(os.path.join(d, name, "meta.json"), None)
        if meta:
            meta.setdefault("name", name)
            out.append(meta)
    out.sort(key=lambda m: m.get("when", ""), reverse=True)
    return out


def solidify(model_path, extra=None, folder=None):
    """把 checkpoint 复制固化到 saved/<模型>__<ckpt>/，返回 meta。

    folder 指定时覆写到该条目（沿用旧目录名），否则按「模型__ckpt」新建。"""
    model_path = model_path if os.path.isabs(model_path) else os.path.join(REPO_ROOT, model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_dir) or "model"
    ckpt = os.path.basename(model_path)
    folder = safe_name(folder) if folder else safe_name(f"{model_name}__{ckpt.replace('.pt', '')}")
    dest = _ensure("saved", folder)

    shutil.copy2(model_path, os.path.join(dest, "model.pt"))
    cfg = os.path.join(model_dir, "config.json")
    if os.path.isfile(cfg):
        shutil.copy2(cfg, os.path.join(dest, "config.json"))
    readme = os.path.join(model_dir, "README.md")
    if os.path.isfile(readme):
        shutil.copy2(readme, os.path.join(dest, "README_from_source.md"))

    import hashlib
    h = hashlib.sha256()
    with open(model_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)

    meta = {
        "name": folder,
        "source": os.path.relpath(model_path, REPO_ROOT),
        "ckpt": ckpt,
        "model": model_name,
        "obs": (extra or {}).get("obs"),
        "act": (extra or {}).get("act"),
        "sizeMB": round(os.path.getsize(model_path) / 1e6, 2),
        "sha": h.hexdigest()[:16],
        "when": time.strftime("%m-%d %H:%M"),
    }
    _write_json(os.path.join(dest, "meta.json"), meta)
    with open(os.path.join(dest, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# 固化存档：{folder}\n\n"
                 f"- 源路径：`{meta['source']}`\n"
                 f"- obs/act：{meta['obs']} / {meta['act']}\n"
                 f"- 大小：{meta['sizeMB']} MB\n"
                 f"- sha256(前16)：`{meta['sha']}`\n"
                 f"- 固化时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                 "本目录由 WBR Console 生成：复制 checkpoint + config.json 留档，"
                 "防止源 run 被空间清理删掉。\n")
    return meta


def delete_saved(name):
    p = store_path("saved", safe_name(name))
    if os.path.isdir(p):
        shutil.rmtree(p)
        return True
    return False


# ------------------------------------------------------------------ 运行记录
def new_run_dir(tag=""):
    stamp = time.strftime("%Y%m%d-%H%M%S") + (("_" + safe_name(tag)) if tag else "")
    d = _ensure("runs", stamp)
    return d, stamp


def list_runs():
    d = _ensure("runs")
    out = []
    for name in sorted(os.listdir(d), reverse=True):
        full = os.path.join(d, name)
        if not os.path.isdir(full):
            continue
        argv = _read_json(os.path.join(full, "argv.json"), {})
        con = os.path.join(full, "console.log")
        out.append({
            "id": name,
            "when": argv.get("when", ""),
            "mode": argv.get("mode", "?"),
            "sizeKB": round(os.path.getsize(con) / 1024, 1) if os.path.isfile(con) else 0,
        })
    return out


def tail_console(run_id, max_bytes=20000):
    p = store_path("runs", safe_name(run_id), "console.log")
    if not os.path.isfile(p):
        return ""
    size = os.path.getsize(p)
    with open(p, "rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        data = fh.read()
    return data.decode("utf-8", "replace")


# ------------------------------------------------------------------ 缓存
def get_cache():
    return _read_json(store_path("checkpoint_cache.json"), {})


def save_cache(cache):
    _write_json(store_path("checkpoint_cache.json"), cache)
