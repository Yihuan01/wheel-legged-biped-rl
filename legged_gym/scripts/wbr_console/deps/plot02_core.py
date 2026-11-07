# ==============================================================================
# WBR Console 独立化抽取副本：plot02_core.py
# 用途：供 wbr_console/bench_plot.py 复用（Index / get_index / read_overview /
# read_window / KIND 等数据层）。原始 plot02.py 在 scripts/ 下保持一行不改。
# 与原版差异（仅 2 处，函数体逐字未动）：
# 1) DATA_DIR 改为绝对路径；
# 2) 顶部追加了这段出处说明（其余内容逐字节同 scripts/plot02.py）。
# ==============================================================================

# ==============================================================================
# Date      : 2026-10-12
# Author    : SYH
# File      : plot02.py
# Purpose   : play02 遥测日志的看图端。纯标准库，不 import isaacgym/torch。
# 起一个本地服务开浏览器：整宽缩略轴拖拽选时间窗 → 按「绘制」刷新 18 张子图。
#
# 用法:
# python plot02.py                      # 看图（Ctrl+C 关服务）
# python plot02.py --dir 别的目录 --thin 600 --port 9000 --no-browser
# python plot02.py --out dash.html      # 导出单文件 HTML（数据内嵌，窗口在前端切片，可拷走）
#
# 为什么不会卡死：日志是一个大文件（约 3 MB/分钟），但
# ① 服务端只扫一遍建 (字节偏移, 时间) 索引并缓存，之后按窗口 seek 只读需要的那几行；
# ② 前端永远只收"窗口内、抽稀到 ≤thin 点"的数据，所以传回来的体积与文件多大无关。
#
# 版面：5 列，每列 3/3/4/4/4 = 18 张，五列总高相等、列内按 weight 分摊
# 列1 线速度/偏航/滚转（各 1/3）  列2 竖直速度/跳跃/轮底（各 1/3；相位＝底色带，指令＝段内灰虚线）
# 列3 髋  列4 膝  列5 轮
# ==============================================================================

import argparse
import base64
import glob
import json
import math
import os
import re
import threading
import webbrowser
from bisect import bisect_left, bisect_right
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")), "legged_gym/scripts/wbr_play_data")
PATTERN = "wbr_tel_*.tsv"

KIND_CN = {'hip': '髋', 'knee': '膝', 'wheel': '轮'}
SIDE_CN = {'left': '左', 'right': '右'}
COL_ORDER = ('hip', 'knee', 'wheel')

# 线型在 Python 侧一次定死，前端只按 kind 查表 ⇒ 屏上、导出、图例三处不可能对不上
KIND = {
    'in':   dict(color="#94a3b8", w=1.3, a=0.95, dash=[2, 3]),
    'tgt':  dict(color="#f59e0b", w=5.0, a=0.38, dash=[]),
    'act':  dict(color="#1d4ed8", w=1.7, a=1.0, dash=[]),
    'alt':  dict(color="#0f766e", w=1.7, a=1.0, dash=[]),
    'alt2': dict(color="#7c3aed", w=1.7, a=1.0, dash=[]),
    'jump': dict(color="#6b7280", w=1.5, a=0.95, dash=[5, 3]),   # 跳跃指令：灰色虚线
    'wheel': dict(color="#7f1d1d", w=2.4, a=1.0, dash=[]),       # 轮底离地（真跳高·本事件峰值）：深红粗线
    'air':  dict(color="#ea580c", w=1.7, a=1.0, dash=[]),       # 双轮离地（瞬时轮底离地）：橙色实线
}

# 跳跃相位底色（相位号与 play02.py 的 JUMP_PHASES、wbr.py::_update_jump_phase 一致）：
# 0 待机**不涂色**（否则整页都是底色，反而看不出哪一段是跳）；1~5 各一色。
# 配色：色相沿"蓝→青→绿→黄→橙"走一段（HSL 色相 214°→167°→81°→48°→31°），明度
# 93/89/80/77/72% 递减 ⇒ 哪一档靠色相认、先后次序靠明度认。取 Tailwind 的 100/200/300
# 色阶（饱和度 85~97%），仍是浅底、不抢三条数据线。
PHASE_BANDS = [
    dict(name="dbg_ph", value=1, color="#dbeafe", cn="1预蹲", h=1, legend=False),
    dict(name="dbg_ph", value=2, color="#ccfbf1", cn="2爆发", h=1, legend=False),
    dict(name="dbg_ph", value=3, color="#d9f99d", cn="3腾空", h=1, legend=False),
    dict(name="dbg_ph", value=4, color="#fde68a", cn="4够地", h=1, legend=False),
    dict(name="dbg_ph", value=5, color="#fdba74", cn="5缓冲", h=1, legend=False),
]

CMD_PANELS = [
    dict(title="线速度 vx", short="vx", unit="m/s"),
    dict(title="偏航角速度 ω", short="yaw", unit="rad/s"),
    dict(title="滚转角 φ", short="roll", unit="rad"),
]


def series(names):
    """[(列名, 图例, kind), ...] → 前端用的 series 字典。"""
    return [dict(name=n, cn=c, kind=k) for n, c, k in names]


# 统一相位 0-11 的中文名（与 envs/wbr/uni/phase.py 的 PHASE_NAMES 一一对应）。
# 相位面板纵轴只印数字（0-11）；编号↔中文的对应关系放到「鼠标扫过图标题」的浮动窗口里显示。
PHASE_CN = {
    0: "两轮待机", 1: "预蹲", 2: "爆发伸腿", 3: "腾空收膝", 4: "落地前伸腿", 5: "缓冲",
    6: "四轮静稳", 7: "起立转起", 8: "起立交接", 9: "趴下降高", 10: "趴下探地", 11: "趴下吸能",
}


def phase_legend(lo, hi):
    """相位图标题浮动窗口用：lo..hi 逐号给出 [{'n': 编号, 'cn': 中文名}, ...]。"""
    return [dict(n=k, cn=PHASE_CN.get(k, "")) for k in range(int(lo), int(hi) + 1)]


# 统一相位 0-11（与 envs/wbr/uni/phase.py 一致）。控制层 env.ph 是重载的
# （四轮/起立/趴下=切换机 0..6，两轮=跳跃机 0..5），绘图侧把历史/旧模型「补丁映射」到 0-11。
_SP_TO_UNI = {0: 6, 1: 7, 2: 8, 3: 0, 4: 9, 5: 10, 6: 11}
_MODE_FOURW, _MODE_RISING, _MODE_TWOW, _MODE_LOWERING, _MODE_TRAD = 0, 1, 2, 3, 4


def uni_phase(mode, ph, jump):
    """内部相位 → 统一 0-11（两轮：可跳→ph本身 / 不可跳→0；其余：_SP_TO_UNI）。"""
    ph = int(round(ph)); mode = int(round(mode))
    if mode == _MODE_TWOW:
        return ph if jump else 0
    return _SP_TO_UNI.get(ph, 0)


def build_panel_cols(joints, leg_idx, torque_limits, ranges):
    """18 张子图按列组织。列内顺序＝从上到下；weight 决定该张占本列多高（同列按权重分摊）。

    ⚠ 面板归类两个坑（过滤规则缺一不可）：
    ① `left_knee_caster_joint` 名含 "knee" 会被归为膝类，生成「左膝 速度/扭矩」面板而
       实际数据是 caster（从动轮）——标题错、真右膝面板被挤掉 ⇒ caster（无执行器，
       画出来恒 0）必须排除；
    ② 若按 groups[kind][:2] 只取每类前 2 个，knee 组 4 个时右膝面板整个消失 ⇒
       不截断，左右各一对面板都生成。"""
    groups = {k: [] for k in KIND_CN}
    for i, name in enumerate(joints):
        low = name.lower()
        if 'caster' in low:          # 从动轮：无执行器，不建面板（避免误标"膝"+挤掉真面板）
            continue
        kind = next((k for k in KIND_CN if k in low), None)
        if kind is None:
            continue
        side = next((s for s in SIDE_CN if s in low), '')
        groups[kind].append(dict(dof=name, cn=f"{SIDE_CN[side]}{KIND_CN[kind]}",
                                 leg=i in leg_idx,
                                 limit=torque_limits[i] if torque_limits else None))

    col1 = [dict(title=p['title'], unit=p['unit'], sym=(p['short'] != 'h'), weight=1,
                 yrange=ranges.get(p['short']),
                 series=series([(f"{pre}_{p['short']}", cn, pre) for pre, cn in
                                (("in", "输入"), ("tgt", "目标"), ("act", "实际"))]))
            for p in CMD_PANELS]
    col1[1]['title'] = "偏航角速度 ω"

    # 高度模块换成竖直速度 vz（base 竖直速度，与躯干高度同源），相位底色同跳跃模块
    vz = dict(title="竖直速度 vz", unit="m/s", sym=True, weight=1,
              bands=PHASE_BANDS,
              series=series([("vz_body", "竖直速度", "act")]))
    # 高度跟随（与线速度 vx 同一格式：目标/实际/输入）。
    # **实际值用 `act_h`（关节角换算的机身高度，不含姿态偏置）** —— 这正是奖励
    # `_reward_tracking_base_height` 跟的那个量（body-frame、俯仰免疫）。
    # 注意：跳跃格里的"机身高度"用的是**姿态校正后**的 `act_h_corr`（世界系），
    # 两者刻意不同：这里＝奖励的视角，跳跃格＝与"绝对高度/轮底离地"自洽的视角。
    htrack = dict(title="高度跟随", unit="m", sym=False, weight=1,
                  series=series([("tgt_h", "目标", "tgt"), ("act_h", "实际", "act"),
                                 ("in_h", "输入", "in")]))
    # 跳跃总览：相位＝底色。
    # 绝对高度（base 原点的世界 z，实测，蓝）／机身高度（**带姿态偏移**的世界系值 ＝ 绝对高度 − 轮底离地，绿）
    # ＋ 双轮离地（橙实线＝瞬时轮底离地，只在双轮离地窗口内非 0）。
    # ⇒ **绝对高度 − 机身高度 ＝ 轮底离地(瞬时)**，蓝绿之间的落差就是真跳高，三个量同坐标系自洽。
    # act_h_corr ＝ body_z − wheel_clear ⇒ 轮底离地由蓝绿落差直接读出，不单画阶梯线。
    # 量程用 `ticks` 钉死 0~0.5（单轴）：高度占 0.2~0.4、轮底离地占底部 0~0.07。
    rise = dict(title="跳跃", unit="m", sym=False, yrange=None, weight=2,
                ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                bands=PHASE_BANDS,
                cmd_line=[dict(name="in_jump", over="dbg_ph", cn="跳跃指令", kind="jump")],
                lg_rows=[["s:body_z", "s:act_h_corr"],
                         ["c:in_jump", "s:wheel_clear"]],
                lg_cls="tight",
                series=series([("body_z", "绝对高度", "act"),
                               ("act_h_corr", "机身高度", "alt"),
                               ("wheel_clear", "双轮离地", "air")]))

    # 列1 = 线速度/偏航/滚转/**高度跟随**（高度跟随挪到列1最下面）；
    # 列2 = 竖直速度 vz / 跳跃 / **相位**（相位＝dbg_ph 阶梯线，直观显示相位转换）；
    # 列3~5 = 髋/膝/轮（沿用原关节定义）。单策略与四槽共用这一份 build_panel_cols。
    col1.append(htrack)
    # 相位面板：纵轴只印数字（0-11）；编号↔中文的对应放到「鼠标扫过图标题」的浮动窗口里（phase_legend）。
    phase = dict(title="相位", unit="", sym=False, step=True, weight=1, phase_axis=True,
                 series=series([("uni_ph", "相位", "alt2")]))
    cols = [col1, [vz, rise, phase]]
    for kind in COL_ORDER:
        col = []
        for j in groups[kind]:      # 不截断：knee 组 4 个时 [:2] 会挤掉右膝面板
            pre, qty, unit = ("pos", "位置", "rad") if j['leg'] else ("vel", "速度", "rad/s")
            col.append(dict(title=f"{j['cn']} {qty}", unit=unit, sym=False, limit=None, weight=1,
                            series=series([(f"{pre}_tgt__{j['dof']}", "目标", "tgt"),
                                           (f"{pre}_act__{j['dof']}", "实际", "act")])))
            col.append(dict(title=f"{j['cn']} 扭矩", unit="Nm", sym=True, limit=j['limit'], weight=1,
                            series=series([(f"tau_tgt__{j['dof']}", "目标", "tgt"),
                                           (f"tau_act__{j['dof']}", "实际", "act")])))
        cols.append(col)
    return cols


def parse_meta_and_cols(meta):
    joints = meta.get("joints", "").split(",")
    leg_idx = {int(v) for v in meta.get("leg_idx", "").split(",") if v}
    limits = [float(v) for v in meta.get("torque_limits", "").split(",") if v] or None
    ranges = {}
    for item in meta.get("ranges", "").split(";"):
        if ":" in item and "," in item:
            short, _, span = item.partition(":")
            lo, _, hi = span.partition(",")
            try:
                ranges[short] = [float(lo), float(hi)]
            except ValueError:
                pass
    return build_panel_cols(joints, leg_idx, limits, ranges)


def safe_name(text, limit=80):
    """只留中日韩、字母数字和 . _ - 空格 ⇒ 浏览器传来的名字不可能带路径分隔符。"""
    cleaned = re.sub(r"[^\w .\-]", "", str(text), flags=re.UNICODE).strip().lstrip(".")
    return cleaned[:limit] or "untitled"


class Index:
    """一个日志文件的行索引：每条数据行的字节偏移 + 时间 + body_z + done。
       扫过的部分不重复扫（日志在跑的过程中还会继续长），所以打开着看也不会越看越慢。"""

    def __init__(self, path):
        self.path, self.pos = path, 0
        self.meta, self.cols, self.ci = {}, [], {}
        self.offsets, self.times, self.zs, self.dones = [], [], [], []

    def scan(self):
        if os.path.getsize(self.path) <= self.pos:
            return
        with open(self.path, "rb") as fh:
            fh.seek(self.pos)
            off = self.pos
            for raw in fh:
                if not raw.endswith(b"\n"):
                    break                       # 半行（写盘被截断），留给下次
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if not self.cols:               # 还没读到列名行 ⇒ 现在都是表头注释
                    if line.startswith("#"):
                        body = line[1:]
                        if not body.startswith(" ") and "=" in body:
                            k, _, v = body.partition("=")
                            self.meta[k.strip()] = v.strip()
                    elif "\t" in line:          # 列名行一定带制表符；不带的杂行不当表头，免得整文件被认成空
                        self.cols = line.split("\t")
                        self.ci = {c: i for i, c in enumerate(self.cols)}
                        self.t_i = self.ci.get("time_s", 1)
                        self.z_i = self.ci.get("body_z")
                        self.d_i = self.ci.get("done")
                    off += len(raw)
                    continue
                fields = line.split("\t")
                if len(fields) == len(self.cols):
                    try:
                        self.offsets.append(off)
                        self.times.append(float(fields[self.t_i]))
                        self.zs.append(float(fields[self.z_i]) if self.z_i is not None else 0.0)
                        self.dones.append(float(fields[self.d_i]) if self.d_i is not None else 0.0)
                    except (ValueError, IndexError):
                        pass
                off += len(raw)
            self.pos = off


_CACHE = {}


def get_index(path):
    key = os.path.abspath(path)
    idx = _CACHE.get(key)
    if idx is None:
        idx = _CACHE[key] = Index(path)
    idx.scan()
    return idx


def read_window(path, t0, t1, thin):
    """按时间窗取数：seek 到窗口首行，只读那几行，再抽稀到 ≤thin 点。"""
    idx = get_index(path)
    i0 = bisect_left(idx.times, t0)
    i1 = bisect_right(idx.times, t1)
    if i1 - i0 < 2 or not idx.cols:
        return None
    with open(path, "rb") as fh:
        fh.seek(idx.offsets[i0])
        blob = fh.read(idx.offsets[i1 - 1] - idx.offsets[i0] + 1)
    lines = blob.split(b"\n")[:i1 - i0]
    rows = [ln.split(b"\t") for ln in lines if ln.count(b"\t") == len(idx.cols) - 1]
    if len(rows) < 2:
        return None

    stride = max(1, math.ceil(len(rows) / thin))
    picks = list(range(0, len(rows), stride))
    if picks[-1] != len(rows) - 1:
        picks.append(len(rows) - 1)

    panel_cols = parse_meta_and_cols(idx.meta)          # 只解析一次，下面两处都用它
    need = {s['name'] for col in panel_cols for p in col for s in p['series']}
    need |= {b['name'] for col in panel_cols for p in col for b in p.get('bands', [])}
    # 指令线要两列：画哪一列的值（name）＋ 按哪一列切段（over，连续线为 None 则跳过）
    need |= {cl[f] for col in panel_cols for p in col
             for cl in (p.get('cmd_line') or []) for f in ('name', 'over') if cl.get(f)}
    data, time_axis = {}, []
    # 统一相位兜底需要 mode/ph/dbg_ph（模板已不含 mode/ph 面板 ⇒ 不在 need 里，这里显式补读）
    _read = need | {"time_s", "done"}
    if 'uni_ph' in need:
        _read |= {"mode", "ph", "dbg_ph"}
    for name in _read:
        j = idx.ci.get(name)
        if j is None:
            continue
        try:
            vals = [float(rows[k][j]) for k in picks]
        except (ValueError, IndexError):
            continue
        if name == "time_s":
            time_axis = vals
        else:
            data[name] = vals

    # 旧 TSV 没有 `act_h_corr` 列（该列是新加的）⇒ 用定义 body_z − wheel_clear 现场补出来，
    # 这样任何历史录像都能画出跳跃格的绿色「机身高度」线（= 带姿态偏移的世界系机身高度），不会缺线。
    # （新录像 play02 会直接写这一列；此处仅作老数据兜底，二者定义完全一致。）
    if 'act_h_corr' in need and 'act_h_corr' not in data and 'body_z' in data and 'wheel_clear' in data:
        data['act_h_corr'] = [bz - wc for bz, wc in zip(data['body_z'], data['wheel_clear'])]

    # 统一相位 0-11 兜底：新四槽日志已有 `uni_ph` 列；旧四槽/单策略没有则现场算。
    # - 四槽旧日志：有 mode + ph，用 mode/ph + meta.jump 映射；
    # - 单策略：dbg_ph 即跳跃相位 0-5，本身就是统一 0-11 的子集，直接用。
    if 'uni_ph' in need and 'uni_ph' not in data:
        if 'mode' in data and 'ph' in data:
            _jm = idx.meta.get('jump')
            if _jm is None:                                   # 旧日志无 jump 标记 → 按 task 推断（wbr_uni 可跳）
                jump = str(idx.meta.get('task', '')).strip() == 'wbr_uni'
            else:
                jump = str(_jm).strip() in ('1', 'true', 'True')
            data['uni_ph'] = [float(uni_phase(m, p, jump)) for m, p in zip(data['mode'], data['ph'])]
        elif 'dbg_ph' in data:
            data['uni_ph'] = list(data['dbg_ph'])

    panels = []
    # 相位面板：纵轴只印数字；编号↔中文对应关系用 phase_legend 传给前端，鼠标扫过图标题时浮动显示。
    _uni = data.get('uni_ph')
    _ph_lo = _ph_hi = None
    if _uni:
        _vals = [int(round(v)) for v in _uni if v is not None]
        if _vals:
            _ph_lo, _ph_hi = min(_vals), max(_vals)
    for col in panel_cols:
        row = []
        for p in col:
            d = dict(title=p['title'], unit=p['unit'], sym=bool(p.get('sym', True)),
                     yrange=p.get('yrange'), ticks=p.get('ticks'), step=bool(p.get('step')),
                     note=p.get('note'), limit=p.get('limit'), bands=p.get('bands', []),
                     cmd_line=p.get('cmd_line'), weight=p.get('weight', 1),
                     lg_rows=p.get('lg_rows'), lg_cls=p.get('lg_cls'),
                     series=[s for s in p['series'] if s['name'] in data])
            if p.get('phase_axis'):
                d['phase_axis'] = True
                if _ph_lo is not None:
                    _hi = _ph_hi if _ph_hi > _ph_lo else _ph_lo + 1   # 防单相位窗口 hi-lo=0 除零
                    d['ticks'] = list(range(_ph_lo, _hi + 1))          # 纵轴只印整数相位号
                    d['phase_legend'] = phase_legend(_ph_lo, _hi)      # 图标题浮动窗口用的编号↔中文
            row.append(d)
        panels.append(row)

    total = sum(1 for k in range(len(rows)) if rows[k][idx.ci['done']] != b'0') if 'done' in idx.ci else 0
    return dict(file=os.path.basename(path), dt=float(idx.meta.get("dt", 0.01)),
                rows=len(rows), stride=stride, done=total, time=time_axis,
                series=data, columns=panels,
                npanel=sum(len(c) for c in panels),
                ncurve=sum(len(p['series']) for c in panels for p in c))


def read_overview(path, buckets=1400):
    """整宽缩略轴用的全程包络：分桶取 body_z 的 min/max 与重置次数，数据全在索引缓存里，不再读文件。"""
    idx = get_index(path)
    n = len(idx.times)
    if n < 2:
        return None
    size = max(1, math.ceil(n / buckets))
    zmin, zmax, ndone, tmid = [], [], [], []
    for k in range(0, n, size):
        seg_z = idx.zs[k:k + size]
        zmin.append(min(seg_z))
        zmax.append(max(seg_z))
        ndone.append(sum(1 for d in idx.dones[k:k + size] if d > 0))
        tmid.append(idx.times[min(k + size // 2, n - 1)])
    return dict(file=os.path.basename(path), t0=idx.times[0], t1=idx.times[-1], n=n,
                dt=float(idx.meta.get("dt", 0.01)), zmin=zmin, zmax=zmax, done=ndone, t=tmid)


def list_files(directory):
    return sorted(glob.glob(os.path.join(directory, PATTERN)))


# ============ 前端 ============
HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>WBR play02 遥测看图</title>
<style>
 body{margin:0;background:#fff;color:#1f2937;font:13px/1.5 system-ui,"Noto Sans CJK SC",sans-serif}
 header{position:sticky;top:0;z-index:9;background:#fbfcfe;border-bottom:1px solid #dfe4ea;padding:8px 14px}
 .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 select,input[type=number]{background:#fff;color:#1f2937;border:1px solid #c3cbd6;border-radius:6px;
        padding:3px 6px;font-size:13px}
 select{max-width:380px} input[type=number]{width:86px}
 b{color:#1d4ed8} .info{color:#6b7280} .warn{color:#b45309}
 button{background:#fff;color:#1f2937;border:1px solid #c3cbd6;border-radius:6px;padding:3px 10px;
        font-size:13px;cursor:pointer}
 button:hover{border-color:#1d4ed8;color:#1d4ed8}
 button.go{background:#1d4ed8;color:#fff;border-color:#1d4ed8}
 button.dirty{background:#b45309;color:#fff;border-color:#b45309}
 #ov{width:100%;height:52px;display:block;border:1px solid #e3e8ef;border-radius:6px;margin:6px 0;cursor:crosshair}
 #grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;padding:8px;align-items:start}
 .colg{display:flex;flex-direction:column;gap:8px;min-width:0}
 .cell{background:#fff;border:1px solid #e3e8ef;border-radius:8px;padding:6px 6px 4px}
 .cell h3{margin:0 0 1px;font-size:12px;font-weight:600;color:#111827;text-align:center}
 .lg{font-size:10px;color:#6b7280;display:flex;gap:9px;justify-content:flex-end;flex-wrap:wrap;min-height:13px}
 /* 紧凑图例：给"每行 3 项"的面板用（跳跃格）——默认 10px/9px 间距装不下三个长标签 */
 .lg.tight{font-size:9px;gap:6px}
 .lg.tight span i{width:12px;margin-right:3px}
 .lg span i{display:inline-block;width:16px;vertical-align:middle;margin-right:4px}
 /* 基准高度；真正的高度由 JS 按"五列总高相等 + 列内 weight 分摊"算（render 里） */
 canvas{width:100%;height:158px;display:block}
 @media(max-width:1280px){#grid{grid-template-columns:repeat(3,1fr)}}
 @media(max-width:820px){#grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header>
 <div class="row">
  <b>日志</b><select id="sel"></select>
  <span class="info">窗</span>
  <input id="t0" type="number" step="0.5"> <span class="info">→</span>
  <input id="t1" type="number" step="0.5"> <span class="info">s</span>
  <input id="wid" type="number" step="0.5" min="0.2"> <span class="info">s 宽</span>
  <button id="q10">10s</button><button id="q30">30s</button><button id="qall">全部</button>
  <button id="draw" class="go">绘制</button>
  <span style="flex:1"></span>
  <button id="b1">保存整页</button><button id="b2">保存分图</button>
 </div>
 <div class="row"><span id="seg" class="info"></span><span id="save" class="info" style="margin-left:auto"></span></div>
 <canvas id="ov"></canvas>
</header>
<div id="grid"></div>
<script id="kindjson" type="application/json">__KIND__</script>
<script>
const PAYLOAD = window.PAYLOAD || null;          // --out 内嵌模式；服务器模式走 fetch
const KIND = JSON.parse(document.getElementById('kindjson').textContent);
const EXP_W = 660, EXP_H = 360, EXP_SCALE = 2, EXP_FS = 2, EXP_TS = 1.5;
const HEAD = 34;
const headH = fs => Math.round(20*EXP_TS) + Math.round(18*fs);
const cache = {}; let redraw=[], REG=[], CUR=null, OV=null, SDATA=null, DRAG=null;

function el(t,a){const e=document.createElement(t); for(const k in a) e[k]=a[k]; return e}
function say(m,warn){ document.getElementById("save").innerHTML = warn?`<span class="warn">${m}</span>`:m; }

// ---- 纵轴：6 等分 7 个刻度，步长只取 1/2/2.5/5×10^k 且 ≤2 位小数 ⇒ 刻度值没有舍入尾巴 ----
const STEPS=(()=>{const a=[];for(let e=-2;e<=6;e++)for(const m of [1,2,2.5,5]){const s=m*Math.pow(10,e);
  if(Math.abs(s*100-Math.round(s*100))<1e-9) a.push(+s.toFixed(4));}
  return [...new Set(a)].sort((x,y)=>x-y);})();
function step6(sp){for(const s of STEPS) if(6*s>=sp-1e-12) return s; return STEPS[STEPS.length-1];}
function stepUp(s){const i=STEPS.findIndex(x=>Math.abs(x-s)<1e-12); return STEPS[Math.min(i+1,STEPS.length-1)];}
function niceAxis(lo,hi,sym){
  if(sym){ const s=step6(2*Math.max(Math.abs(lo),Math.abs(hi))); return [-3*s,3*s,s]; }
  let s=step6(hi-lo), L=Math.floor(lo/s+1e-9)*s;
  while(L+6*s < hi-1e-12){ s=stepUp(s); L=Math.floor(lo/s+1e-9)*s; }
  return [L, L+6*s, s];
}
function decimals(st){for(let d=0;d<=2;d++) if(Math.abs(st*Math.pow(10,d)-Math.round(st*Math.pow(10,d)))<1e-9) return d; return 2;}
function fmt(v,d){const t=v.toFixed(d); return /^-0(\.0+)?$/.test(t)? t.slice(1):t;}

function draw(cv, p, seg, W, H, scale, fs){
  const dpr=scale||window.devicePixelRatio||1; fs=fs||1;
  W=W||cv.clientWidth; H=H||cv.clientHeight; if(!W||!H) return;
  cv.width=W*dpr; cv.height=H*dpr;
  const g=cv.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  const M={l:Math.round(48*fs), r:8, t:Math.round(9*fs), b:Math.round(16*fs)}, iw=W-M.l-M.r, ih=H-M.t-M.b;
  let lo=Infinity, hi=-Infinity;
  p.series.forEach(s=>(seg.series[s.name]||[]).forEach(v=>{if(isFinite(v)){if(v<lo)lo=v;if(v>hi)hi=v;}}));
  // 指令线（可多条）：连续线（over=null）按实际指令值画阶梯虚线；事件线（over=相位列）取每段首步的值、段外不画。
  // 值先并进纵轴范围，否则指令常落在曲线区间外、被裁掉。
  const cmdLines=(p.cmd_line||[]).map(cl=>{
    const V=seg.series[cl.name]; if(!V) return null;
    if(!cl.over){ V.forEach(v=>{ if(isFinite(v)){ if(v<lo)lo=v; if(v>hi)hi=v; } });
      return {st:KIND[cl.kind]||KIND.in, segs:null, vals:V}; }
    const O=seg.series[cl.over], segs=[];
    if(O) for(let i=0,n=O.length;i<n;){ if(!O[i]){ i++; continue; } let j=i; while(j<n&&O[j]) j++;
      const v=V[i]; segs.push([i,j,v]); if(isFinite(v)){ if(v<lo)lo=v; if(v>hi)hi=v; } i=j; }
    return {st:KIND[cl.kind]||KIND.in, segs};
  }).filter(Boolean);
  g.font=`${Math.round(10*fs)}px system-ui`;
  if(!isFinite(lo)){ g.fillStyle="#9ca3af"; g.fillText("无该列数据",M.l,M.t+Math.round(12*fs)); return; }
  let dec, ticks=null;
  if(p.ticks){ lo=p.ticks[0]; hi=p.ticks[p.ticks.length-1]; ticks=p.ticks;
    // 小数位按**刻度步长**推（原来硬写 dec=0 ⇒ 0.1 步长会把 0.1/0.2 全印成 "0"）
    dec=decimals(ticks.length>1 ? (hi-lo)/(ticks.length-1) : 1); }
  else{
    if(p.yrange){ lo=Math.min(p.yrange[0],lo); hi=Math.max(p.yrange[1],hi); }
    if(p.limit && Math.max(Math.abs(lo),Math.abs(hi))>0.5*p.limit){ lo=Math.min(lo,-p.limit); hi=Math.max(hi,p.limit); }
    if(hi-lo<1e-9) hi=lo+1e-6;
    const ax=niceAxis(lo,hi,p.sym!==false); lo=ax[0]; hi=ax[1]; dec=decimals(ax[2]);
  }
  const last=seg.time.length-1||1;
  const X=i=>M.l+iw*i/last, Y=v=>M.t+ih*(1-(v-lo)/(hi-lo));

  // 底色带：画在网格线**之前**，否则满高的相位底色会把横线盖掉。
  // h=1 → 涂满绘图区（跳跃相位）；h<1 → 贴着横轴的一条窄带（双轮离地）。
  // 满高的带子只要够宽就把相位名写在带顶 —— 10 s 窗口里相位只有几像素、自动不写，
  // 放大到 1~2 s 才出现，等于就地图例，省得去数颜色。
  (p.bands||[]).forEach(bd=>{ const b=seg.series[bd.name]; if(!b) return;
    const bh=ih*(bd.h==null?1:bd.h), by=M.t+ih-bh, n=b.length;
    for(let i=0;i<n;){
      if(b[i]!==bd.value){ i++; continue; }
      let j=i; while(j<n && b[j]===bd.value) j++;
      const x0=X(i), w=Math.max(X(Math.min(j,n-1))-x0,0.6);
      g.fillStyle=bd.color; g.fillRect(x0,by,w,bh);
      if(bh>=ih*0.98){ g.font=`${Math.round(9*fs)}px system-ui`; g.textAlign="left";
        if(w-6>=g.measureText(bd.cn).width){ g.fillStyle="#6b7280";
          g.fillText(bd.cn,x0+3,M.t+Math.round(9*fs)); } }
      i=j; } });

  g.lineWidth=1; g.fillStyle="#6b7280";
  const gl = ticks || Array.from({length:7},(_,k)=>lo+(hi-lo)*k/6);
  gl.forEach(v=>{ const y=Y(v);
    g.strokeStyle="#eef1f5"; g.beginPath(); g.moveTo(M.l,y); g.lineTo(W-M.r,y); g.stroke();
    g.textAlign="right"; g.fillText(fmt(v,dec),M.l-5,y+Math.round(3*fs)); });
  g.textAlign="center"; g.fillStyle="#6b7280";
  const nx=Math.max(2,Math.min(6,Math.floor(iw/(56*fs))));
  for(let k=0;k<=nx;k++){ const i=Math.round(last*k/nx);
    const txt=seg.time[i].toFixed(1), half=g.measureText(txt).width/2;
    g.fillText(txt, Math.max(M.l+half,Math.min(X(i),W-M.r-half)), H-Math.round(4*fs)); }
  g.textAlign="right"; g.fillStyle="#9ca3af"; g.fillText(p.unit||"",W-M.r-2,M.t+Math.round(9*fs));

  if(p.limit){ g.strokeStyle="#c0c8d3"; g.setLineDash([4,4]);
    [p.limit,-p.limit].forEach(L=>{ if(L<lo||L>hi) return; g.beginPath(); g.moveTo(M.l,Y(L)); g.lineTo(W-M.r,Y(L)); g.stroke(); });
    g.setLineDash([]); }

  // 画指令线
  cmdLines.forEach(cl=>{ const st=cl.st;
    g.save(); g.globalAlpha=st.a; g.strokeStyle=st.color; g.lineWidth=st.w; g.setLineDash(st.dash||[]);
    if(cl.segs){ cl.segs.forEach(q=>{ const y=Y(q[2]); if(!(y>=M.t-1 && y<=M.t+ih+1)) return;
        const x0=X(q[0]), x1=Math.max(X(Math.min(q[1],seg.time.length-1)), x0+1);
        g.beginPath(); g.moveTo(x0,y); g.lineTo(x1,y); g.stroke(); }); }
    else{ g.beginPath(); cl.vals.forEach((v,j)=>{ const x=X(j), y=Y(v);
        if(!j) g.moveTo(x,y); else { g.lineTo(x, Y(cl.vals[j-1])); g.lineTo(x, y); } });
      g.stroke(); }
    g.restore(); });

  [...p.series].sort((a,b)=>(KIND[a.kind]||{w:0}).w-(KIND[b.kind]||{w:0}).w).forEach(s=>{
    const d=seg.series[s.name]; if(!d) return; const st=KIND[s.kind]||KIND.act;
    g.save(); g.globalAlpha=st.a; g.strokeStyle=st.color; g.lineWidth=st.w; g.setLineDash(st.dash||[]);
    g.lineJoin="round"; g.beginPath();
    d.forEach((v,j)=>{ const x=X(j), y=Y(v);
      if(!j) g.moveTo(x,y);
      else if(p.step){ g.lineTo(x, Y(d[j-1])); g.lineTo(x, y); }        // 阶梯线：先水平再垂直
      else g.lineTo(x,y); });
    g.stroke(); g.restore(); });
}

function render(seg){
  const grid=document.getElementById("grid");
  grid.innerHTML=""; redraw=[]; REG=[];
  seg.columns.forEach((col,ci)=>{
    const box=el("div",{className:"colg"});
    col.forEach((p,ri)=>{
      const c=el("div",{className:"cell"});
      c.appendChild(el("h3",{textContent:p.title}));
      const lg=el("div",{className:"lg"+(p.lg_cls?" "+p.lg_cls:"")});
      if(p.note) lg.appendChild(el("span",{textContent:p.note, style:"margin-right:auto;color:#9ca3af"}));
      // 图例项先收进 map（key: s=曲线 / b=底色带 / c=指令线），再按面板给的 `lg_rows` 出行。
      // 不给 `lg_rows` 时保持老顺序（曲线→底色→指令线）。
      const _lg={};
      [...p.series].sort((a,b)=>(KIND[b.kind]||{w:0}).w-(KIND[a.kind]||{w:0}).w).forEach(s=>{
        const st=KIND[s.kind]||KIND.act, sp=el("span",{});
        const bg = (st.dash&&st.dash.length)
          ? `repeating-linear-gradient(90deg,${st.color} 0 3px,transparent 3px 6px)` : st.color;
        sp.append(el("i",{style:`background:${bg};height:${st.w}px;opacity:${st.a}`}),
                  document.createTextNode(s.cn));
        _lg["s:"+s.name]=sp; });
      (p.bands||[]).forEach(bd=>{ if(bd.legend===false) return;   // 相位底色只涂在图里，不进图例
        const sp=el("span",{});
        sp.append(el("i",{style:`background:${bd.color};height:9px;border:1px solid ${bd.border||"#c9d3de"}`}),
                  document.createTextNode(bd.cn)); _lg["b:"+bd.name]=sp; });
      (p.cmd_line||[]).forEach(cl=>{ const st=KIND[cl.kind]||KIND.in, sp=el("span",{});
        sp.append(el("i",{style:`background:repeating-linear-gradient(90deg,${st.color} 0 3px,transparent 3px 6px);height:${st.w}px`}),
                  document.createTextNode(cl.cn)); _lg["c:"+cl.name]=sp; });
      if(p.lg_rows){ p.lg_rows.forEach((row,i)=>{
          row.forEach(k=>{ if(_lg[k]) lg.appendChild(_lg[k]); });
          if(i<p.lg_rows.length-1) lg.appendChild(el("span",{style:"flex-basis:100%;height:0"}));  // 强制换行
        }); }
      else { Object.keys(_lg).forEach(k=>lg.appendChild(_lg[k])); }
      if(p.limit) lg.appendChild(el("span",{className:"warn",textContent:"扭矩上限 ±"+p.limit.toFixed(1)}));
      c.appendChild(lg);
      const cv=el("canvas",{}); c.appendChild(cv); box.appendChild(c);
      c._w = p.weight || 1;                      // 本张占本列多高（同列按权重分摊，见下面）
      redraw.push(()=>draw(cv,p,seg)); REG.push({cv,p,ci,ri});
    });
    grid.appendChild(box);
  });
  // 五列**总高相等**，列内再按各自 weight 分摊 ⇒ 3 张的列每张 1/3、2 张的列 1/3＋2/3、
  // 4 张的列每张 1/4，于是"躯干高度"下缘与"线速度"下缘落在同一条横线附近。
  // 分摊的是**卡片外框**高度而不是画布高度：标题/图例那点像素差由画布自己吸收，
  // 卡片边界才对得齐。chrome（＝卡片高−画布高）与画布高度无关，所以先量后改。
  // 同步做（读 offsetHeight 会强制布局）：挂 rAF 的话后台标签页被节流就一直不齐。
  // 只改屏上高度：导出走 exportCell 的固定尺寸，每张 PNG 永远一样大。
  const cols=[...document.querySelectorAll(".colg")];
  if(cols.length){
    const H=Math.max(...cols.map(c=>c.getBoundingClientRect().height));
    cols.forEach(col=>{
      const cells=[...col.children], W=cells.reduce((s,c)=>s+c._w,0);
      const gap=parseFloat(getComputedStyle(col).rowGap)||8;
      const avail=H-gap*(cells.length-1);
      cells.forEach(c=>{ const cv=c.querySelector("canvas");
        const want=avail*c._w/W-(c.offsetHeight-cv.clientHeight);
        if(want>60) cv.style.height=want.toFixed(1)+"px"; });
    });
  }
  redraw.forEach(f=>f()); drawOv();
  const span=seg.time[seg.time.length-1]-seg.time[0];
  document.getElementById("seg").innerHTML=
    `${seg.file} &nbsp; 窗口 ${seg.time[0].toFixed(1)}–${seg.time[seg.time.length-1].toFixed(1)} s`
    +` · ${seg.rows} 步 · 抽稀 ${seg.stride} 步/点 · ${seg.npanel} 图/${seg.ncurve} 曲线`
    +(seg.done?` · <span class="warn">窗口内 ${seg.done} 步重置</span>`:"")
    +` · <span class="info">全程 ${(OV? (OV.t1-OV.t0):span).toFixed(0)} s</span>`;
}

// ---- 缩略轴：全程包络 + 可拖拽窗口框 ----
function drawOv(){
  const cv=document.getElementById("ov"), dpr=window.devicePixelRatio||1, W=cv.clientWidth, H=cv.clientHeight;
  if(!W) return; cv.width=W*dpr; cv.height=H*dpr;
  const g=cv.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  if(!OV || !OV.zmin) return;                     // 概览数据没到位就空着，别抛异常打断整页渲染
  const M={l:8,r:8,t:6,b:12}, iw=W-M.l-M.r, ih=H-M.t-M.b;
  const t0=OV.t0, t1=OV.t1||t0+1, T=v=>M.l+iw*(v-t0)/(t1-t0);
  let zlo=Math.min(...OV.zmin), zhi=Math.max(...OV.zmax); if(zhi-zlo<1e-9) zhi=zlo+1;
  const Z=v=>M.t+ih*(1-(v-zlo)/(zhi-zlo));
  g.strokeStyle="#cbd5e1"; g.lineWidth=1; g.beginPath();                 // 包络带（min/max 填成实体）
  OV.t.forEach((t,i)=>{ const x=T(t); i?g.lineTo(x,Z(OV.zmax[i])):g.moveTo(x,Z(OV.zmax[i])); });
  for(let i=OV.t.length-1;i>=0;i--) g.lineTo(T(OV.t[i]),Z(OV.zmin[i]));
  g.closePath(); g.fillStyle="#dbe6f3"; g.fill();
  g.fillStyle="#c0392b"; OV.done.forEach((d,i)=>{ if(d>0) g.fillRect(T(OV.t[i])-0.6,M.t,1.6,ih); });
  const a=T(+document.getElementById("t0").value), b=T(+document.getElementById("t1").value);
  g.fillStyle="rgba(29,78,216,.14)"; g.fillRect(a,M.t,Math.max(b-a,1),ih);
  g.strokeStyle="#1d4ed8"; g.beginPath(); g.moveTo(a,M.t); g.lineTo(a,M.t+ih); g.moveTo(b,M.t); g.lineTo(b,M.t+ih); g.stroke();
  g.fillStyle="#6b7280"; g.font="10px system-ui"; g.textAlign="left";
  g.fillText("body_z 全程包络 · 红＝重置 · 蓝框＝当前窗口（拖动/拉边界改窗口）",M.l+2,M.t+9);
}
function ovTime(ev){ const cv=document.getElementById("ov"), r=cv.getBoundingClientRect(), M=8, iw=r.width-16;
  return OV.t0 + (ev.clientX-r.left-M)/iw*(OV.t1-OV.t0); }
function bindOv(){
  const cv=document.getElementById("ov");
  cv.addEventListener("mousedown",e=>{ if(!OV) return;
    const t=ovTime(e), a=+t0el.value, b=+t1el.value, w=b-a, edge=0.02*(OV.t1-OV.t0);
    DRAG = Math.abs(t-a)<edge ? 'l' : Math.abs(t-b)<edge ? 'r' : (t>a&&t<b) ? 'm' : ['n',t];
    if(DRAG[0]==='n'){ setWin(t, t+w); DRAG=['m',t]; } });
  addEventListener("mousemove",e=>{ if(!DRAG||!OV) return; const t=ovTime(e);
    if(DRAG==='m'){ const w=+t1el.value-+t0el.value; setWin(t-w/2, t+w/2); }
    else if(DRAG==='l') setWin(Math.min(t, +t1el.value-0.2), +t1el.value);
    else if(DRAG==='r') setWin(+t0el.value, Math.max(t, +t0el.value+0.2));
    drawOv(); });
  addEventListener("mouseup",()=>{ DRAG=null; });
}
const t0el=document.getElementById("t0"), t1el=document.getElementById("t1"), widel=document.getElementById("wid");
function setWin(a,b){ const lo=OV?OV.t0:+t0el.value, hi=OV?OV.t1:+t1el.value;
  a=Math.max(lo,a); b=Math.min(hi,b);
  t0el.value=a.toFixed(1); t1el.value=b.toFixed(1); widel.value=(b-a).toFixed(1); dirty(true); }
function dirty(on){ document.getElementById("draw").className = on? "go dirty":"go"; }

// ---- 取数：服务器按窗口读，静态模式切内嵌数组 ----
async function loadWindow(){
  const a=+t0el.value, b=+t1el.value;
  if(!(b-a>0)){ say("窗口无效",true); return; }
  let seg;
  if(PAYLOAD){ seg=slice(SDATA.window, a, b); }
  else{
    say("读取窗口…");
    const r=await fetch(`/api/window?name=${encodeURIComponent(sel.value)}&t0=${a}&t1=${b}`);
    seg=await r.json(); if(!seg){ say("窗口内没有数据",true); return; }
  }
  CUR=seg; render(seg); dirty(false);
}
function slice(full,a,b){
  const i0=Math.max(0,full.time.findIndex(t=>t>=a)), i1=full.time.length;
  let j=i0; while(j<full.time.length && full.time[j]<=b) j++;
  const t=full.time.slice(i0,j), series={};
  for(const k in full.series) series[k]=full.series[k].slice(i0,j);
  const cols=full.columns.map(c=>c.map(p=>({...p, series:p.series.filter(s=>series[s.name])})));
  return {file:full.file, dt:full.dt, rows:j-i0, stride:full.stride,
          done:(series.done||[]).filter(v=>v>0).length, time:t, series,
          columns:cols, npanel:full.npanel, ncurve:full.ncurve};
}

// ---- 导出 ----
function tagOf(f){ const m=/^wbr_tel_(.+?)(?:_seg\d+)?\.tsv$/.exec(f); return m?m[1]:f.replace(/\.tsv$/,""); }
function stamp(){ return tagOf(CUR.file)+"_"+CUR.time[0].toFixed(1)+"-"+CUR.time[CUR.time.length-1].toFixed(1)+"s"; }
function fname(t){ return t.replace(/[^\p{L}\p{N} _.-]/gu,"").trim(); }
function drawHeader(g,p,W,fs,ts){
  const l1=Math.round(20*ts);
  g.fillStyle="#111827"; g.font=`600 ${Math.round(15*ts)}px system-ui`; g.textAlign="center";
  g.fillText(p.title, W/2, l1);
  const y2=l1+Math.round(13*fs); let x=W-Math.round(12*fs);
  g.font=`${Math.round(12*fs)}px system-ui`;
  const items=[...p.series];
  (p.cmd_line||[]).forEach(cl=>items.push({cn:cl.cn, kind:cl.kind||'in'}));   // 指令线也要有线例
  items.sort((a,b)=>(KIND[b.kind]||{w:0}).w-(KIND[a.kind]||{w:0}).w).forEach(s=>{
    const st=KIND[s.kind]||KIND.act;
    g.fillStyle="#374151"; g.textAlign="right"; g.fillText(s.cn,x,y2);
    x-=g.measureText(s.cn).width+Math.round(5*fs);
    g.fillStyle=st.color; g.fillRect(x-Math.round(18*fs), y2-Math.round(5*fs), Math.round(18*fs), Math.max(3,Math.round(3*fs)));
    x-=Math.round(23*fs); });
  if(p.limit){ g.fillStyle="#b45309"; g.textAlign="left"; g.fillText("扭矩上限 ±"+p.limit.toFixed(1), Math.round(12*fs), y2); }
  // 底色带的说明只能挤在第二行左侧（导出尺寸固定、表头不许长高）：放不下就截断，绝不与右对齐的线例叠字。
  else if(p.note){ const avail=x-Math.round(28*fs);
    g.font=`${Math.round(10*fs)}px system-ui`; g.fillStyle="#9ca3af"; g.textAlign="left";
    let s=p.note;
    if(g.measureText(s).width>avail){ while(s.length>1 && g.measureText(s+"…").width>avail) s=s.slice(0,-1);
      s+="…"; }
    if(g.measureText(s).width<=avail) g.fillText(s, Math.round(12*fs), y2); }
}
function exportCell(e,W,H){
  W=W||EXP_W; H=H||EXP_H; const hh=headH(EXP_FS);
  const out=el("canvas",{}); out.width=W*EXP_SCALE; out.height=(H+hh)*EXP_SCALE;
  const g=out.getContext("2d"); g.setTransform(EXP_SCALE,0,0,EXP_SCALE,0,0);
  g.fillStyle="#fff"; g.fillRect(0,0,W,H+hh); drawHeader(g,e.p,W,EXP_FS,EXP_TS);
  const plot=el("canvas",{}); draw(plot,e.p,CUR,W,H,EXP_SCALE,EXP_FS);
  g.drawImage(plot,0,hh,W,H); return out;
}
function compositeCanvas(){
  const NC=Math.max(...REG.map(e=>e.ci))+1, NR=Math.max(...REG.map(e=>e.ri))+1;
  const cw=EXP_W*EXP_SCALE, ch=(EXP_H+headH(EXP_FS))*EXP_SCALE, gap=30, bar=60;
  const out=el("canvas",{}); out.width=NC*cw+(NC+1)*gap; out.height=bar+NR*ch+(NR+1)*gap;
  const g=out.getContext("2d"); g.setTransform(1,0,0,1,0,0);
  g.fillStyle="#fff"; g.fillRect(0,0,out.width,out.height);
  g.fillStyle="#111827"; g.font="600 28px system-ui"; g.textAlign="left";
  g.fillText(`WBR 遥测全景  ${stamp()}`, gap, bar-20);
  REG.forEach(e=>g.drawImage(exportCell(e), gap+e.ci*(cw+gap), bar+gap+e.ri*(ch+gap)));
  return out;
}
function download(canvas,name){
  return new Promise(res=>canvas.toBlob(b=>{ const a=el("a",{href:URL.createObjectURL(b),download:name});
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(()=>{URL.revokeObjectURL(a.href); res();},300); },"image/png"));
}
function uniqueNames(){ const seen={};
  return REG.map((e,i)=>{ let base=(fname(e.p.title)||("panel"+i))+"_"+stamp();
    if(!seen[base]) seen[base]=1; else { seen[base]++; base+="_"+seen[base]; }
    return {e, name:base+".png"}; });
}
async function saveComposite(){
  say("正在拼整页图…");
  try{ await download(compositeCanvas(), `WBR遥测全景_${stamp()}.png`); say("✅ 整页已下载到浏览器下载目录"); }
  catch(err){ say("导出失败："+err.message,true); }
}
async function saveAll(){
  const cells=uniqueNames().map(u=>({name:u.name, cv:exportCell(u.e)}));
  say(`正在重画 ${cells.length} 张…`);
  try{
    const r=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({dir:`WBR遥测_${stamp()}`,
        files:cells.map(c=>({name:c.name,b64:c.cv.toDataURL("image/png").split(",")[1]}))})});
    const d=await r.json().catch(()=>({}));
    if(!r.ok||d.error) throw new Error(d.error||("HTTP "+r.status));
    say(d.n===cells.length?`✅ ${d.n} 张已存到 ${d.dir}`:`⚠️ 只写入 ${d.n}/${cells.length} 张到 ${d.dir}`, d.n!==cells.length);
  }catch(err){
    say(`无本地后端（${err.message}），改为逐个下载…`,true);
    for(const c of cells) await download(c.cv,c.name);
    say(`✅ 已触发 ${cells.length} 次下载`);
  }
}

// ---- 启动 ----
const sel=document.getElementById("sel");
async function api(u){ return (await fetch(u)).json(); }
function fill(files){
  sel.innerHTML="";
  files.forEach(f=>sel.appendChild(el("option",{value:f,text:f})));
  if(!files.length){ say("目录里没有 wbr_tel_*.tsv，先跑 play02.py 采数据",true); return; }
  openFile(files[files.length-1]);
}
async function openFile(name){
  sel.value=name;                             // 下拉必须跟着实际打开的文件走，否则窗口请求会串到别的文件上
  if(PAYLOAD){ SDATA=PAYLOAD.data[name]; OV=SDATA.overview;
    t0el.value=OV.t0.toFixed(1); t1el.value=Math.min(OV.t1, OV.t0+10).toFixed(1);
    widel.value=((+t1el.value)-(+t0el.value)).toFixed(1); await loadWindow(); return; }
  const m=await api(`/api/meta?name=${encodeURIComponent(name)}`);
  if(!m||!m.n){ say("这个文件里没有数据行",true); return; }
  OV=await api(`/api/overview?name=${encodeURIComponent(name)}&buckets=1400`) || {t0:m.t0,t1:m.t1};
  const a=m.t0, b=Math.min(m.t1, m.t0+10);
  t0el.value=a.toFixed(1); t1el.value=b.toFixed(1); widel.value=(b-a).toFixed(1);
  await loadWindow();
}
sel.onchange=()=>openFile(sel.value);
document.getElementById("draw").onclick=loadWindow;
document.getElementById("b1").onclick=saveComposite;
document.getElementById("b2").onclick=saveAll;
[t0el,t1el,widel].forEach(inp=>{
  inp.oninput=()=>dirty(true);
  inp.onkeydown=e=>{ if(e.key==="Enter") loadWindow(); };
});
t0el.onchange=()=>setWin(+t0el.value, +t1el.value);
t1el.onchange=()=>setWin(+t0el.value, +t1el.value);
widel.onchange=()=>setWin(+t1el.value-+widel.value, +t1el.value);
document.getElementById("q10").onclick=()=>setWin(+t1el.value-10, +t1el.value);
document.getElementById("q30").onclick=()=>setWin(+t1el.value-30, +t1el.value);
document.getElementById("qall").onclick=()=>{ if(OV) setWin(OV.t0, OV.t1); };
addEventListener("resize",()=>{redraw.forEach(f=>f()); drawOv();});
bindOv();
if(PAYLOAD) fill(PAYLOAD.files); else api("/api/list").then(fill);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    directory, thin = DATA_DIR, 600

    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _path_of(self, query):
        name = os.path.basename(query.split("name=", 1)[-1].split("&")[0])
        return os.path.join(self.directory, name)

    def do_GET(self):
        try:
            if self.path.startswith("/api/list"):
                self._send(200, json.dumps([os.path.basename(f) for f in list_files(self.directory)]),
                           "application/json; charset=utf-8")
            elif self.path.startswith("/api/meta"):
                path = self._path_of(self.path)
                idx = get_index(path) if os.path.isfile(path) else None
                self._send(200, json.dumps({"n": len(idx.times), "t0": idx.times[0], "t1": idx.times[-1],
                                            "file": os.path.basename(path)} if idx and idx.times else None,
                                           ensure_ascii=False), "application/json; charset=utf-8")
            elif self.path.startswith("/api/window"):
                q = self.path
                t0, t1 = (float(q.split(k + "=")[1].split("&")[0]) for k in ("t0", "t1"))
                seg = read_window(self._path_of(q), t0, t1, self.thin)
                self._send(200 if seg else 404, json.dumps(seg, ensure_ascii=False),
                           "application/json; charset=utf-8")
            elif self.path.startswith("/api/overview"):
                ov = read_overview(self._path_of(self.path), 1400)
                self._send(200, json.dumps(ov, ensure_ascii=False), "application/json; charset=utf-8")
            elif self.path in ("/", "/index.html"):
                self._send(200, HTML.replace("__KIND__", json.dumps(KIND)),
                           "text/html; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")
        except Exception as exc:
            self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False),
                       "application/json; charset=utf-8")

    def do_POST(self):
        if not self.path.startswith("/api/save"):
            self._send(404, "not found", "text/plain")
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            out_dir = os.path.join(os.path.expanduser("~/Downloads"), safe_name(req["dir"]))
            os.makedirs(out_dir, exist_ok=True)
            n = 0
            for item in req["files"]:
                name = safe_name(item["name"])
                if not name.lower().endswith(".png"):
                    name += ".png"
                with open(os.path.join(out_dir, name), "wb") as fh:
                    fh.write(base64.b64decode(item["b64"]))
                n += 1
            self._send(200, json.dumps({"dir": out_dir, "n": n}, ensure_ascii=False),
                       "application/json; charset=utf-8")
        except Exception as exc:
            self._send(400, json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False),
                       "application/json; charset=utf-8")

    def log_message(self, *args):
        pass                                    # 浏览器还会请求 /favicon.ico，别刷屏


def serve(httpd, directory, open_browser):
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    files = list_files(directory)
    print(f"📂 数据目录: {directory}  发现 {len(files)} 个日志文件")
    if not files:
        print(f"⚠️ 没找到 {PATTERN}，先跑 play02.py 采数据。")
    print(f"🌐 看图地址 {url}   （Ctrl+C 关闭服务）")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️ 已关闭看图服务")


def export_static(directory, thin, out_path, open_browser):
    """单文件模式：把每个日志的全程数据抽稀到 thin 点内嵌进来，窗口在前端切片。
       代价是窗口内分辨率受 thin 限制 —— 要边放大边看细节就用服务模式。"""
    files, data = [], {}
    for path in list_files(directory):
        idx = get_index(path)
        if len(idx.times) < 2:
            continue
        seg = read_window(path, idx.times[0], idx.times[-1], thin)
        name = os.path.basename(path)
        files.append(name)
        data[name] = dict(overview=read_overview(path, 1400), window=seg)
    inject = "window.PAYLOAD=" + json.dumps({"files": files, "data": data}, ensure_ascii=False) + ";"
    body = HTML.replace("__KIND__", json.dumps(KIND)).replace("<script>", "<script>" + inject, 1)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"✅ 单文件看图已导出 {out_path}  ({len(files)} 个日志, {os.path.getsize(out_path)/1e6:.2f} MB)")
    print("   注意：内嵌数据的窗口分辨率受 --thin 限制，要放大看细节请用服务模式")
    if open_browser:
        webbrowser.open("file://" + os.path.abspath(out_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="play02 遥测日志看图（本地服务 + 时间窗 + 18 张子图）")
    ap.add_argument("--dir", default=DATA_DIR, help=f"数据目录，默认 {DATA_DIR}")
    ap.add_argument("--thin", type=int, default=600, help="每个窗口内每条曲线最多画多少个点")
    ap.add_argument("--port", type=int, default=8765, help="本地端口，默认 8765，占用时自动顺延")
    ap.add_argument("--out", default="", help="给了文件名就导出单文件 HTML（数据内嵌），不起服务")
    ap.add_argument("--no-browser", action="store_true", help="不自动开浏览器，只打印地址")
    a = ap.parse_args()

    if a.out:
        export_static(a.dir, a.thin, a.out, not a.no_browser)
    else:
        Handler.directory, Handler.thin = a.dir, a.thin
        httpd = None
        for p in range(a.port, a.port + 20):
            try:
                httpd = HTTPServer(("127.0.0.1", p), Handler)
                break
            except OSError as e:
                print(f"端口 {p} 不可用（{e.strerror}），顺延")
        if httpd is None:
            raise SystemExit(f"❌ {a.port}~{a.port + 19} 端口都不可用，用 --port 换一个")
        serve(httpd, a.dir, not a.no_browser)
