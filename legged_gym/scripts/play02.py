# ==============================================================================
# Date      : 2026-10-11
# Author    : SYH
# File      : play02.py
# Purpose   : 遥测采集端：面板下发指令（支持键盘步进）→ 每控制步采样 → 每 10 s 落一个 TSV
#             数据段；画图交给 plot02.py。--play_mode single 跑单策略，switch 用单个 wbr_switch
#             env 承载四轮 / 起立 / 两轮站立三个策略。
# ==============================================================================

#   已加（本轮）：wbr.py::dbg_snapshot(env_idx) —— 目标层只读快照出口，纯读、不进任何计算路径。
#     本脚本据此多存 16 列 dbg_*，把"指令 → 目标"被谁改了口拆开记：
#       dbg_yaw_raw / dbg_yaw_tgt / dbg_yaw_clipped / dbg_yaw_cap  包络（ω ≤ expand·(b − k·|v|)）
#       dbg_roll_base / dbg_roll_tilt / dbg_roll_tgt               压弯前馈合成
#       dbg_h_ik_in / dbg_h_ceiling                                喂给 IK 的高度与几何上限
#       dbg_knee_tgt_l / _r, dbg_roll_delta, dbg_ff_gain           解算器膝角目标与差动量
#       dbg_hip_allow_l / _r, dbg_hip_over                         髋可达门限与是否越限
#     ⚠️ 陈旧零陷阱：dbg 里约一半字段由奖励通路里的 _compute_target_knee_angles / _reward_hip_reach 写，
#     开关（flags.enable_leg_adjust、scales.z_adjust_leg、scales.hip_reach）一关就返回**没算过的 0**，
#     读到 0 先确认开关。当前配置三项都开 ⇒ play 里是新鲜的。
#   未加（够用，暂不动）：
#     B) debug_* 每个物理子步写一次，一个控制步（decimation=2）被覆盖两次 ⇒ 落盘的是最后一个子步的
#        dof_pos/dof_vel；目标量（actions_scaled/raw_tau）子步间不变，无影响。
#     C) 腿部 PD 无速度目标项 ⇒ vel_tgt__<腿> 恒为 0，是结构常量不是采样值；腿的有效数据在
#        pos_tgt__/pos_act__ 与 tau_tgt__/tau_act__，轮子的在 vel_tgt__/vel_act__ 与 tau_*。
#     D) act_h 是脚本内按奖励公式重算的（纯 dof_pos 闭式，无历史依赖 ⇒ 重算安全）。解算器一旦改成
#        依赖历史状态，就该由 env 出 dbg["leg_extension"]。
#     A) debug_* 只导出 0 号机器人 —— 按用户决定保留一台，不加接口。
# ==============================================================================

import os
import time
import math
import threading
import traceback
import unicodedata
import tkinter as tk
from tkinter import font as tkfont

import numpy as np

import isaacgym  # 必须在 legged_gym / torch 之前导入
import torch
from isaacgym import gymtorch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wbr_play_data")
SEGMENT_SECONDS = 10.0   # 每 10 s flush 一次（整轮跑只写一个文件，画图脚本一屏正好放得下）
PRINT_EVERY = 50         # 终端打印节奏：每 50 个控制步（= 0.5 s）

# ---- 面板五路指令：区间来自 cfg，键盘步进与按键写死在这里 ----
# 按键标记「减 / 加」＝左边键让数值变小、右边键变大，与滑块方向一致。
# roll 正方向是左侧抬高（机身向右倒）；yaw 正方向 = 右转。
CHANNELS = (
    dict(key='lin_vel_x',   short='vx',   name='线速度', unit='m/s',
         label='前进线速度 (m/s)',      rng='lin_vel_x',   step=0.05,
         inc=('w',),        dec=('s',),        hint='S / W'),
    dict(key='ang_vel_yaw', short='yaw',  name='偏航',   unit='rad/s',
         label='转向角速度 (rad/s)',    rng='ang_vel_yaw', step=0.05,
         inc=('d',),        dec=('a',),        hint='A / D'),
    dict(key='base_height', short='h',    name='高度',   unit='m',
         label='躯干目标高度 (m)',      rng='base_height', step=0.01, fmt='+.3f', res=0.005,
         inc=('i',),        dec=('k',),        hint='K / I'),
    dict(key='ang_roll',    short='roll', name='侧倾',   unit='rad',
         label='滚转压弯角 (rad)',      rng='ang_roll',    step=0.02,
         inc=('l',),        dec=('j',),        hint='J / L'),
    # 第五路 = 跳跃指令（离地后机身上升量，0 = 不跳）。
    #   滑块只管"要多高"，跳不跳由空格 / 跳跃按钮触发；env 相位机只在装填状态下才吃这条指令。
    #   上限取 cfg.jump_height，下限放到 0 ⇒ 面板量程 [0, 0.15]。
    dict(key='jump_height', short='jump', name='跳跃升高', unit='m',
         label='跳跃指令·升高量 (m)', rng='jump_height', step=0.01, zero_ok=True,
         inc=('o',),        dec=('u',),        hint='U / O'),
)

JUMP_PHASES = ('待机', '预蹲', '爆发伸腿', '腾空收膝', '落地前伸腿', '缓冲')
BY_SHORT = {c['short']: c for c in CHANNELS}

# ---- 模式切换常量（与 wbr_switch_config.py 的 SP_* 对齐）----
SP_HOLD4W, SP_RISE, SP_HANDOVER, SP_HOLD2W, SP_LOWER, SP_TILT, SP_ABSORB = range(7)
MODE_FOURW, MODE_RISING, MODE_TWOW, MODE_LOWERING = 'FOURW', 'RISING', 'TWOW', 'LOWERING'
MODE_NUM = {MODE_FOURW: 0, MODE_RISING: 1, MODE_TWOW: 2, MODE_LOWERING: 3}
MODE_CN = {MODE_FOURW: '四轮', MODE_RISING: '起立中', MODE_TWOW: '两轮站立', MODE_LOWERING: '趴下中'}
# 切换按钮标签＝"按下去会切到哪"；起立中/趴下中为过渡态，按钮禁用
TOGGLE_LABEL = {MODE_FOURW: '切换两轮模式', MODE_TWOW: '切换四轮模式',
                MODE_RISING: '起立中…', MODE_LOWERING: '趴下中…'}


def span_of(ch, cfg_span):
    """面板与看图表头共用的指令区间：上限永远取 cfg，只有 zero_ok 的通道把下限放到 0。"""
    lo, hi = cfg_span
    return (min(lo, 0.0) if ch.get('zero_ok') else lo), hi

# 面板与采集线程之间唯一的共享状态（GIL 下单个 dict 项读写是原子的）
GUI_DEFAULTS = {'lin_vel_x': 0.0, 'ang_vel_yaw': 0.0, 'base_height': 0.25, 'ang_roll': 0.0,
                'jump_height': 0.0}
gui_cmds = dict(GUI_DEFAULTS)

# 跳跃触发状态（面板线程写、仿真线程读；同样只靠 GIL 下单个 dict 项的原子读写）
#   arm_once：空格/跳跃按钮按一次＝装填一次跳；auto：按住 P 打开"每 1.2 s 自动再装填"的连跳
jump_flags = {'arm_once': False, 'auto': False}

# 模式切换的共享状态（面板线程写、仿真线程读）：
#   mode    —— 当前模式（面板显示用）
#   request —— 面板按下的切换请求（'toggle'），仿真线程消费后清 None
#   reset   —— 面板请求复位（R / 复位按钮），仿真线程消费后清 False
mode_shared = {'mode': MODE_FOURW, 'request': None, 'reset': False}

# 主循环每步整体替换、面板线程只读 ⇒ 两边不共享同一个可变对象，不需要加锁
readout = {}
PANEL_MS = 100        # 面板回显刷新周期（10 Hz 足够，且不与仿真线程抢事做）

# 三个量的配色与网页看图一致：指令=灰、目标=橙、实际=蓝（plot02.py 里是同一组值）。
C_IN, C_TGT, C_ACT = "#475569", "#b45309", "#1d4ed8"


def dwidth(text):
    """按终端显示宽度量一串字：中日韩字符占 2 列。面板按显示宽度对齐，靠的就是这个而不是 len()。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text, width):
    """补空格到指定显示宽度（面板用它把各行名称拉齐，终端靠它对齐两列打印）。"""
    return text + " " * max(0, width - dwidth(text))


def cmd_cell(ch, v_in, v_tgt, v_act):
    """一路指令的打印单元，宽度固定 ⇒ 拼成两列后仍然竖向对齐。"""
    return (f"{pad(ch['name'], 11)}指令 {v_in:+6.2f}  目标 {v_tgt:+6.2f}  实际 {v_act:+6.2f} {pad(ch['unit'], 5)}")


def _dbg_seg_yaw(d):
    """目标层第一行的前半段：偏航指令→目标、包络限幅与是否被削。"""
    return (f"ω {d['yaw_raw']:+6.2f}→{d['yaw_tgt']:+6.2f} 限幅 {d['yaw_cap']:5.2f}"
            f" 削:{'是' if d['yaw_clipped'] > 0.5 else '否'}")


def _dbg_seg_h(d):
    """目标层第二行的前半段：喂给 IK 的高度与几何可达上限。"""
    return f"h IK {d['h_ik_in']:.3f}/可达上限 {d['h_ceiling']:.3f}"


def fmt_dbg_a(d):
    """目标层分解**第一行**（偏航包络 ＋ 侧倾压弯）。"""
    return (pad(_dbg_seg_yaw(d), DBG_SEG_W)
            + f" | φ {d['roll_base']:+6.2f}→{d['roll_tgt']:+6.2f} 压弯 {d['roll_tilt']:+6.3f}")


def fmt_dbg_b(d):
    """目标层分解**第二行**（高度几何上限 ＋ 髋可达门限）。"""
    return (pad(_dbg_seg_h(d), DBG_SEG_W)
            + f" | 髋限 {d['hip_allow_l']:.2f}/{d['hip_allow_r']:.2f}"
              f" 越限:{'是' if d['hip_over'] > 0.5 else '否'}")


def fmt_jump(d, in_cmd, armed, auto):
    """跳跃诊断行（**不含行首标签**，标签由调用方按 pad 补齐，与 fmt_dbg_a/b 同一约定）。"""
    ph = int(round(d.get('ph', 0.0)))
    return (f"指令 {in_cmd:5.3f}→{d.get('h_cmd', 0.0):5.3f}m"
            f" 相位 {JUMP_PHASES[min(ph, 5)]:<5} 模板偏移 {d.get('off_knee', 0.0):+.3f}rad"
            f" 机身升 {d.get('h_body_rise', 0.0)*1000:5.1f}mm"
            f" 机身净升 {d.get('h_body_lift', 0.0)*1000:5.1f}mm"
            f" 轮底 {d.get('wheel_clear', 0.0)*1000:+5.1f}mm"
            f" 轮峰 {d.get('wheel_peak_min', 0.0)*1000:5.1f}mm"
            f" 滞空 {d.get('t_air', 0.0)*1000:4.0f}ms"
            f" 装填 {'已装' if armed else '空'}{' 连跳' if auto else ''}")


def fmt_mode(mode, ph, pitch_deg, z, ev_up):
    """switch 档的模式状态行正文。"""
    return (f"{MODE_CN.get(mode, mode)}  相位 {ph}  pitch {pitch_deg:+5.1f}°  z {z:.3f}m"
            f"  ev_upright {'✓' if ev_up else '·'}")


# 面板状态文字的最坏情况探针：数字段定宽，只有相位名长度会变（最长「落地前伸腿」）。
# 启动时用它量一次窗口宽度，否则真数据一上来会超出窗口、右半截被切掉。
PROBE_DBG = {'ph': 4.0, 'h_cmd': 0.0, 'off_knee': 0.0, 'h_body_rise': 0.0, 'h_body_lift': 0.0,
             'wheel_clear': 0.0, 'wheel_peak_min': 0.0, 't_air': 0.0,
             'yaw_raw': 0.0, 'yaw_tgt': 0.0, 'yaw_cap': 0.0, 'yaw_clipped': 0.0,
             'roll_base': 0.0, 'roll_tgt': 0.0, 'roll_tilt': 0.0,
             'h_ik_in': 0.0, 'h_ceiling': 0.0, 'hip_allow_l': 0.0, 'hip_allow_r': 0.0, 'hip_over': 0.0}


def row_text(ch):
    """面板一行的名称栏文字。"""
    return f"{ch['label']}   {ch['hint']} ±{ch['step']:g}"


# 目标层两行用同一条分隔竖线「|」：两段补到同宽，竖线才落在同一列。
DBG_SEG_W = max(dwidth(_dbg_seg_yaw({'yaw_raw': 0.0, 'yaw_tgt': 0.0, 'yaw_cap': 0.0, 'yaw_clipped': 0.0})),
                dwidth(_dbg_seg_h({'h_ik_in': 0.0, 'h_ceiling': 0.0}))) + 1


def fmt_in(ch, v):
    """指令列的显示格式。默认五路统一 m/rad + 两位小数（作者要求同口径），跳跃那路也不例外；
    高度那路用 fmt='+.3f'（0.005 步进需要三位才不丢精度、也避免 0.175 被显示成 0.17）。"""
    return f"{v:{ch.get('fmt', '+.2f')}}"


ROW_LABEL_W = max(dwidth(row_text(c)) for c in CHANNELS)
# 通用循环只覆盖这四路：CHANNELS 里含跳跃，而 readout['cmd'] 里没有 'jump' 键（它在 'cmd_jump'）
DASH_CHANNELS = tuple(c for c in CHANNELS if c['key'] != 'jump_height')

# ---- 面板栅格常量（多处共用，集中定义免得各写一个魔法数）----
VAL_W = 7
VAL_PADX = 16
ROW_PAD = 10
STATUS_INDENT = 28
STATUS_GAP = 18


# 策略加载（switch 档直接构造 ActorCritic；与训练同结构 [512, 256, 128] elu）
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLICY_HIDDEN = [512, 256, 128]


def resolve_path(p):
    """把相对路径解析到仓库根（脚本可能在任意 cwd 下被调用）。"""
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def load_actor_critic(path, obs_dim, device):
    """构造与训练一致的 ActorCritic 并载入权重，返回 eval 模式网络（用 act_inference）。"""
    from rsl_rl.modules import ActorCritic
    net = ActorCritic(obs_dim, obs_dim, 6, actor_hidden_dims=POLICY_HIDDEN,
                      critic_hidden_dims=POLICY_HIDDEN, activation='elu').to(device)
    _p = resolve_path(path)
    if not os.path.exists(_p):
        raise FileNotFoundError(f"策略权重不存在：{_p}")
    sd = torch.load(_p, map_location=device)
    sd = sd.get('model_state_dict', sd)
    net.load_state_dict(sd)
    net.eval()
    print(f"[play02] 载入策略 {os.path.relpath(_p, REPO_ROOT)}  obs={obs_dim}")
    return net


# 回填白名单：地形几何不从历史 config 回填。
#   --from_run_config 指向平地年代的 run 时，其 terrain.mesh_type = "plane"、
#   height_field_file = None 等会把当前任务的地形图静默覆盖成平地。地形属于当前任务，不属于历史 run。
_TERRAIN_GEOM_KEYS = (
    "mesh_type", "height_field_file", "num_rows", "num_cols", "terrain_length",
    "terrain_width", "horizontal_scale", "vertical_scale", "border_size",
    "slope_treshold", "add_perlin_noise", "curriculum", "curriculum_max_level",
    "max_init_terrain_level", "track_test", "selected", "terrain_proportions",
    "terrain_kwargs", "origin_z_use_max",
)


def _fill_from_json(obj, d, _path=""):
    """把 config.json 的标量叶子递归回填到配置实例（--from_run_config 用）。

    ⚠ 两处例外（2026-10-29）：
      1. `terrain.*` 的**几何**键不再回填（见 `_TERRAIN_GEOM_KEYS`）；
      2. 值为 `None` 时不覆盖（旧 config 里大量 `height_field_file: null` 之类）。
    """
    for k, v in d.items():
        a = getattr(obj, k, None)
        if isinstance(v, dict) and a is not None and not isinstance(a, (int, float, bool, str, list, tuple)):
            _fill_from_json(a, v, _path + k + ".")
            continue
        if a is None:
            continue
        if _path == "terrain." and k in _TERRAIN_GEOM_KEYS:
            continue
        if v is None and not isinstance(a, type(None)):
            continue
        setattr(obj, k, v)


# 控制面板
def start_gui_thread(env, ranges=None, switch=False):
    """控制面板：每行 名称-滑块-指令-目标-实际，最下面几行是跳跃诊断与目标层分解。
       三个数值由主循环每步发布的 readout 驱动（after 轮询，绝不在仿真线程里碰 tkinter 控件）。
       switch=True 时顶部多一行"模式"：当前模式 + 切换按钮 + 跳跃按钮。
       ranges：{rng: (lo,hi)} 覆盖滑块量程（switch 档需覆盖 4W/2W 两套）；None ⇒ 取 env.cfg。"""

    def run_gui():
        if ranges is not None:
            _ranges = dict(ranges)
        else:
            _ranges = {c['rng']: span_of(c, getattr(env.cfg.commands.ranges, c['rng'])) for c in CHANNELS}

        root = tk.Tk()
        root.title("WBR 控制终端 (play02·切换)" if switch else "WBR 控制终端 (play02)")
        root.attributes('-topmost', True)

        BG_COLOR, FG_COLOR = "#f5f7fa", "#2d3436"
        ACCENT_COLOR, TROUGH_COLOR, MUTED_COLOR = "#0984e3", "#dfe6e9", "#636e72"
        C_STATUS = "#1f252b"
        CN_FONT = "fangsong ti"
        FONT_TITLE = (CN_FONT, 22, "bold")
        FONT_LABEL = (CN_FONT, 17)
        FONT_NUM = (CN_FONT, 16, "bold")
        FONT_HEAD = (CN_FONT, 14)
        FONT_STATUS = (CN_FONT, 17)
        f_status = tkfont.Font(font=FONT_STATUS)

        root.configure(bg=BG_COLOR)
        root.grid_columnconfigure(2, weight=1)
        R = 1 if switch else 0                     # 顶部模式行占 1 行，其余整体下移
        for r in range(2 + R, len(CHANNELS) + 2 + R):
            root.grid_rowconfigure(r, weight=1)

        tk.Label(root, text="WBR 实时指令下发", font=FONT_TITLE, bg=BG_COLOR, fg=ACCENT_COLOR).grid(
            row=0, column=0, columnspan=7, pady=(16, 4))

        widgets = {}
        set_state_map = {}       # key -> set_state(disabled,text) 回调（切模式时置灰/恢复某路滑块）

        # ---- 顶部模式行（仅 switch）----
        mode_state_lbl = switch_btn = None
        if switch:
            mode_frame = tk.Frame(root, bg=BG_COLOR)
            mode_frame.grid(row=1, column=0, columnspan=7, sticky='we',
                            padx=(STATUS_INDENT, VAL_PADX), pady=(2, 8))
            tk.Label(mode_frame, text="模式", font=FONT_LABEL, bg=BG_COLOR, fg=FG_COLOR).grid(
                row=0, column=0, sticky='w')
            mode_state_lbl = tk.Label(mode_frame, text=MODE_CN[MODE_FOURW], font=FONT_NUM,
                                      bg=BG_COLOR, fg=ACCENT_COLOR, width=9, anchor='w')
            mode_state_lbl.grid(row=0, column=1, sticky='w', padx=(8, 20))

            def _req_toggle():
                mode_shared['request'] = 'toggle'

            switch_btn = tk.Button(mode_frame, text=TOGGLE_LABEL[MODE_FOURW], font=FONT_HEAD,
                                   command=_req_toggle, takefocus=0)
            switch_btn.grid(row=0, column=2, sticky='w', padx=(0, 16))
            tk.Button(mode_frame, text="跳跃", font=FONT_HEAD, takefocus=0,
                      command=lambda: jump_flags.update(arm_once=True)).grid(
                row=0, column=3, sticky='w', padx=(0, 16))
            tk.Button(mode_frame, text="复位", font=FONT_HEAD, takefocus=0,
                      command=lambda: mode_shared.update(reset=True)).grid(
                row=0, column=4, sticky='w')
            root.bind("m", lambda _e: _req_toggle())
            root.bind("M", lambda _e: _req_toggle())

        for col, (title, color) in enumerate((("指令", C_IN), ("目标", C_TGT), ("实际", C_ACT)), start=4):
            tk.Label(root, text=title, font=FONT_HEAD, bg=BG_COLOR, fg=color,
                     width=VAL_W, anchor='e').grid(
                row=1 + R, column=col, sticky='se', padx=(0, VAL_PADX), pady=(0, 2))

        def push(ch, value):
            """写指令的唯一出口：夹进区间 → 按该通道 resolution 吸附到整数倍 → 同步滑块与指令列。"""
            lo, hi = _ranges[ch['rng']]
            res = ch.get('res', 0.01)
            v = round(min(max(value, lo), hi) / res) * res      # 吸附到 res 整数倍（from_ 非整倍时端点才取得到）
            v = round(v, 6)
            gui_cmds[ch['key']] = v
            scale, lbl_in = widgets[ch['key']][:2]
            scale.set(v)
            lbl_in.config(text=fmt_in(ch, v))

        def nudge(ch, sign):
            push(ch, gui_cmds[ch['key']] + sign * ch['step'])

        def reset_all():
            """回车＝五路全部回到 GUI_DEFAULTS（跳跃回 0＝不跳），走 push 所以滑块/数值/下发三者同步。"""
            for ch in CHANNELS:
                push(ch, GUI_DEFAULTS[ch['key']])

        for i, ch in enumerate(CHANNELS, start=2 + R):
            lo, hi = _ranges[ch['rng']]
            tk.Label(root, text=pad(row_text(ch), ROW_LABEL_W),
                     font=FONT_LABEL, bg=BG_COLOR, fg=FG_COLOR).grid(
                row=i, column=0, sticky='w', padx=(28, 18), pady=(0, ROW_PAD))
            tk.Label(root, text=f"{lo:.2f}", font=FONT_NUM, bg=BG_COLOR, fg=MUTED_COLOR,
                     width=6, anchor='e').grid(
                row=i, column=1, sticky='e', padx=(0, 10), pady=(0, ROW_PAD))

            res = ch.get('res', 0.01)
            scale = tk.Scale(root, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                             font=FONT_NUM, bg=ACCENT_COLOR, fg=BG_COLOR, troughcolor=TROUGH_COLOR,
                             activebackground=ACCENT_COLOR, highlightthickness=0, bd=0,
                             sliderrelief='raised',
                             length=300, width=26, showvalue=False, takefocus=0)
            scale.set(round(min(max(gui_cmds[ch['key']], lo), hi), 6))
            scale.configure(command=lambda val, c=ch: push(c, float(val)))
            scale.grid(row=i, column=2, sticky='we', pady=(0, ROW_PAD))

            tk.Label(root, text=f"{hi:.2f}", font=FONT_NUM, bg=BG_COLOR, fg=MUTED_COLOR,
                     width=6, anchor='w').grid(
                row=i, column=3, sticky='w', padx=(10, VAL_PADX), pady=(0, ROW_PAD))

            lbls = []
            for col, first, color in ((4, fmt_in(ch, float(scale.get())), C_IN),
                                      (5, "—", C_TGT), (6, "—", C_ACT)):
                lbl = tk.Label(root, text=first, font=FONT_NUM, bg=BG_COLOR,
                               fg=color, width=VAL_W, anchor='e')
                lbl.grid(row=i, column=col, sticky='e', padx=(0, VAL_PADX), pady=(0, ROW_PAD))
                lbls.append(lbl)

            widgets[ch['key']] = (scale, *lbls)
            gui_cmds[ch['key']] = float(scale.get())

            def set_state(disabled, text=None, _key=ch['key'], _ch=ch):
                """把某路滑块/指令列置灰（disabled）或恢复，并可临时改写指令列文字（高度在起立/两轮模式被策略固定时用）。"""
                sc, lin, ltgt, lact = widgets[_key]
                st = 'disabled' if disabled else 'normal'
                sc.config(state=st)
                col = MUTED_COLOR if disabled else C_IN
                lin.config(fg=col)
                if text is not None:
                    lin.config(text=text)
            set_state_map[ch['key']] = set_state

            for seq, sign in [(s, +1) for s in ch['inc']] + [(s, -1) for s in ch['dec']]:
                keysyms = (seq, seq.upper()) if len(seq) == 1 else (seq,)
                for keysym in keysyms:
                    root.bind(f"<{keysym}>", lambda _e, c=ch, k=sign: nudge(c, k))

        # 状态区（switch 档多一行模式状态）
        status_frame = tk.Frame(root, bg=BG_COLOR)
        status_frame.grid(row=len(CHANNELS) + 2 + R, column=0, columnspan=7, sticky='we',
                          padx=(STATUS_INDENT, VAL_PADX), pady=(16, 18))
        status_frame.grid_columnconfigure(1, weight=1)
        if switch:
            STATUS_TAGS = ("模式", "目标层", "")
        else:
            STATUS_TAGS = ("跳跃", "目标层", "")
        jump_lbl = dbg_lbl = dbg_lbl2 = None
        for k, tag in enumerate(STATUS_TAGS):
            tk.Label(status_frame, text=tag, font=FONT_STATUS, bg=BG_COLOR,
                     fg=C_STATUS, anchor='w').grid(row=k, column=0, sticky='w', pady=(0, ROW_PAD))
            body = tk.Label(status_frame, text="等第一步数据…", font=FONT_STATUS, bg=BG_COLOR,
                            fg=C_STATUS, anchor='w', justify='left')
            body.grid(row=k, column=1, sticky='we', padx=(STATUS_GAP, 0), pady=(0, ROW_PAD))
            if k == 0:
                jump_lbl = body
            elif k == 1:
                dbg_lbl = body
            else:
                dbg_lbl2 = body
        STATUS_LABELS = (jump_lbl, dbg_lbl, dbg_lbl2)

        # 触发键：空格＝装填一次（跳一下）；P＝连跳开关；回车＝五路指令回默认
        root.bind("<space>", lambda _e: jump_flags.update(arm_once=True))
        root.bind("<Return>", lambda _e: reset_all())
        root.bind("<KP_Enter>", lambda _e: reset_all())
        root.bind("p", lambda _e: jump_flags.update(auto=not jump_flags['auto']))
        root.bind("P", lambda _e: jump_flags.update(auto=not jump_flags['auto']))
        if switch:
            root.bind("r", lambda _e: mode_shared.update(reset=True))
            root.bind("R", lambda _e: mode_shared.update(reset=True))

        _last_mode = {'v': None}

        def tick():
            r = readout
            if r:
                m = mode_shared['mode'] if switch else None
                h_fixed = switch and m in (MODE_RISING, MODE_TWOW, MODE_LOWERING)
                for ch in DASH_CHANNELS:
                    v_in, v_tgt, v_act = r['cmd'][ch['short']]
                    _, lbl_in, lbl_tgt, lbl_act = widgets[ch['key']]
                    if ch['key'] == 'base_height' and h_fixed:
                        lbl_in.config(text='固定·两轮站立', fg=MUTED_COLOR)
                    else:
                        lbl_in.config(text=fmt_in(ch, v_in))
                    lbl_tgt.config(text=fmt_in(ch, v_tgt))
                    lbl_act.config(text=fmt_in(ch, v_act))
                jump_ch = BY_SHORT['jump']
                j_in, j_tgt, j_act = r['cmd_jump']
                _, lbl_in, lbl_tgt, lbl_act = widgets['jump_height']
                lbl_in.config(text=fmt_in(jump_ch, j_in))
                lbl_tgt.config(text=fmt_in(jump_ch, j_tgt))
                lbl_act.config(text=fmt_in(jump_ch, j_act))
                if switch:
                    if mode_state_lbl is not None:
                        mode_state_lbl.config(text=MODE_CN.get(m, m))
                    if switch_btn is not None:
                        switch_btn.config(text=TOGGLE_LABEL.get(m, '切换'),
                                          state=('disabled' if m in (MODE_RISING, MODE_LOWERING) else 'normal'))
                    # 高度滑块在起立/两轮模式置灰（P7S 不吃高度指令），四轮模式恢复可调
                    if _last_mode['v'] != m:
                        set_state_map['base_height'](h_fixed)
                        _last_mode['v'] = m
                    jump_lbl.config(text=r.get('mode_line', ''))
                else:
                    jump_lbl.config(text=fmt_jump(r['dbg'], j_in, r['armed'], r['auto']))
                dbg_lbl.config(text=fmt_dbg_a(r['dbg']))
                dbg_lbl2.config(text=fmt_dbg_b(r['dbg']))
            root.after(PANEL_MS, tick)

        root.after(PANEL_MS, tick)
        root.update_idletasks()
        if switch:
            probes = ("四轮  相位 0  pitch +00.0°  z 0.000m  ev_upright ·",
                      fmt_dbg_a(PROBE_DBG), fmt_dbg_b(PROBE_DBG))
        else:
            probes = (fmt_jump(PROBE_DBG, 0.0, True, True), fmt_dbg_a(PROBE_DBG), fmt_dbg_b(PROBE_DBG))
        tag_px = max(f_status.measure(t) for t in STATUS_TAGS)
        status_indent = STATUS_INDENT + tag_px + STATUS_GAP
        want_w = max(root.winfo_reqwidth(), status_indent + VAL_PADX
                     + max(f_status.measure(s) for s in probes))
        win_w = min(want_w, root.winfo_screenwidth() - 40)
        wrap = max(600, win_w - status_indent - VAL_PADX)
        placeholders = [lb.cget('text') for lb in STATUS_LABELS]
        for lb, s in zip(STATUS_LABELS, probes):
            lb.configure(wraplength=wrap, text=s)
        root.update_idletasks()
        need_h = root.winfo_reqheight()
        for lb, s in zip(STATUS_LABELS, placeholders):
            lb.config(text=s)
        root.geometry(f"{win_w}x{max(need_h, 470 + 40 * R)}")
        root.minsize(win_w, need_h)
        root.focus_force()
        root.mainloop()

    def boot():
        try:
            run_gui()
        except Exception:
            traceback.print_exc()
            print("❌ 控制面板线程异常退出 ⇒ 面板不会显示，但仿真与记录仍在跑（Ctrl+C 停止）")

    threading.Thread(target=boot, daemon=True).start()


# 采样 / 落盘的公共小工具
def _leg_extension(env, dpos):
    thigh = env.cfg.rewards.thigh_length
    shank = env.cfg.rewards.shank_length
    hl, kl = env.hip_l_idx, env.knee_l_idx
    hr, kr = env.hip_r_idx, env.knee_r_idx
    return 0.5 * (thigh * math.cos(dpos[hl]) + shank * math.cos(dpos[hl] + dpos[kl])
                  + thigh * math.cos(dpos[hr]) + shank * math.cos(dpos[hr] + dpos[kr]))


def _open_log(prefix, meta, notes, columns):
    os.makedirs(DATA_DIR, exist_ok=True)
    run_tag = time.strftime('%Y%m%d-%H%M%S')
    log_path = os.path.join(DATA_DIR, f"{prefix}_{run_tag}.tsv")
    log_fh = open(log_path, "w", encoding="utf-8")
    log_fh.write("\n".join(
        [f"# WBR play02 遥测日志  起于 {time.strftime('%Y-%m-%d %H:%M:%S')}"]
        + [f"#{m}" for m in meta] + [f"# 注: {n}" for n in notes] + ["\t".join(columns)]) + "\n")
    log_fh.flush()
    return log_fh, log_path


# single 档：单个 env + 一个策略
def _build_single(args):
    """建单策略 env + 策略（等价原 play02 第 1 步）。返回 env, policy, ppo_runner。"""
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _frc = getattr(args, "from_run_config", None)
    if _frc:
        import json as _json
        _p = _frc if os.path.isfile(_frc) else os.path.join(_frc, "config.json")
        _fill_from_json(env_cfg, _json.load(open(_p)))
        print(f"[from_run_config] 已回填 {_p} → env_cfg（复现该 run 参数；回放覆盖仍在其后生效）")
    env_cfg.env.num_envs = 3
    env_cfg.env.episode_length_s = 3600
    # 只有平地任务才缩小地形网格。wbr_terrain 的高度图是 4 行 × 6 列，
    #   强行改成 2×2 会让 Terrain._load_height_field 的尺寸校验报错，也看不到完整地形。
    if getattr(env_cfg.terrain, "height_field_file", None) is None:
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = 3600
    env_cfg.rewards.jump.arm_delay = 0.1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    return env, policy, ppo_runner


def run_single(args):
    global readout
    env, policy, ppo_runner = _build_single(args)

    _rb_t = env.gym.acquire_rigid_body_state_tensor(env.sim)
    rb_state = gymtorch.wrap_tensor(_rb_t).view(env.num_envs, env.num_bodies, 13)
    obs = env.get_observations()

    ROBOT = 0
    dof_names = list(env.dof_names)
    num_j = len(dof_names)
    leg_set = set(env.leg_indices.tolist())
    wheel_set = set(env.wheel_indices.tolist())

    thigh = env.cfg.rewards.thigh_length
    shank = env.cfg.rewards.shank_length
    wheel_r = env.cfg.rewards.wheel_radius
    default_pos = env.default_dof_pos[ROBOT].detach().cpu().numpy()
    torque_limits = env.torque_limits.detach().cpu().numpy()
    hip_l, knee_l = env.hip_l_idx, env.knee_l_idx
    hip_r, knee_r = env.hip_r_idx, env.knee_r_idx

    dt = env.dt
    seg_steps = int(round(SEGMENT_SECONDS / dt))

    dbg_keys = list(env.dbg_snapshot(ROBOT).keys())
    cmd_cols = [('lin_vel_x', 'vx'), ('ang_vel_yaw', 'yaw'), ('base_height', 'h'), ('ang_roll', 'roll'),
                ('jump_height', 'jump')]
    columns = ['step', 'time_s', 'done']
    for _, short in cmd_cols:
        columns += [f'in_{short}', f'tgt_{short}', f'act_{short}']
    columns.append('body_z')
    columns.append('vz_body')
    columns.append('n_ground')
    columns.append('body_pitch')
    columns.append('com_z')
    columns += ['wheel_bot_l', 'wheel_bot_r', 'wheel_clear']
    columns += ['act_h_corr']
    columns += ['wheel_peak_min', 'wheel_peak_max', 'wheel_at_air', 't_air']
    for i, name in enumerate(dof_names):
        columns += [f'vel_tgt__{name}', f'vel_act__{name}', f'tau_tgt__{name}', f'tau_act__{name}']
        if i in leg_set:
            columns += [f'pos_tgt__{name}', f'pos_act__{name}']
    columns += [f'dbg_{k}' for k in dbg_keys]
    formats = [('%d' if c in ('step', 'done') else '%0.4f') for c in columns]

    meta = [
        f"joints={','.join(dof_names)}",
        f"leg_idx={','.join(str(i) for i in sorted(leg_set))}",
        f"wheel_idx={','.join(str(i) for i in sorted(wheel_set))}",
        f"torque_limits={','.join(f'{v:.4f}' for v in torque_limits)}",
        f"dt={dt:.6f}",
        # 【P7F4】model_file_path/load_num_steps 属性在 rsl_rl 1.0.2 不存在（恒写 "?"），
        #   绘图页因此解析不出模型——改用真实来源：load_run 目录 + runner 已加载的迭代号。
        "model_dir=" + str(args.load_run if (getattr(args, "load_run", None) and os.path.isdir(str(args.load_run))) else "?"),
        "model_iter=" + str(getattr(ppo_runner, "current_learning_iteration", None) or getattr(args, "checkpoint", "?")),
        "model_cli=" + "%s|%s" % (getattr(args, "load_run", "?"), getattr(args, "checkpoint", "?")),
        "cmd_cols=" + ",".join(short for _, short in cmd_cols),
        "ranges=" + ";".join(f"{short}:{span_of(BY_SHORT[short], getattr(env.cfg.commands.ranges, attr))[0]:.4f},"
                             f"{span_of(BY_SHORT[short], getattr(env.cfg.commands.ranges, attr))[1]:.4f}"
                             for attr, short in cmd_cols),
    ]
    notes = [
        "in_* = 面板原始输入(2位小数)  tgt_* = env 目标层(包络+压弯+限速)输出，即策略观测里的指令  act_* = 实测量",
        "act_vx = base_lin_vel.x      act_yaw = −base_ang_vel.z(取负：对外口径正数=右转)      act_roll = asin(-projected_gravity.y)",
        "in_yaw/tgt_yaw/act_yaw/dbg_yaw_* 一律正数=右转(2026-10-12 起)；base_ang_vel 本身正数=左转",
        "act_h  = wheel_radius + 腿伸量(与 _reward_tracking_base_height 同口径)；body_z = root_states.z 为 base(躯干)世界高度(非质心)，仅参考",
        "act_jump = 本事件已达到的最大机身升高(m)，触发时清零并保持到下一跳触发",
        "n_ground = contact_forces[0, feet_indices, 2] > 0.1 的轮数(0/1/2)，与 env 相位机的 grounded/airborne 同一口径",
        "wheel_bot_l/r = 轮心世界 z − 轮半径（平地地面 z=0，左/右按 URDF 顺序）；wheel_clear = 较低那只＝真离地高度",
        "vel_tgt__<轮> = debug_actions_scaled(已含轮速加速度限幅)；腿部 PD 无速度目标 ⇒ vel_tgt__<腿> 恒为 0",
        "tau_tgt = 控制器算出的未裁剪力矩 raw_torques；tau_act = 夹进扭矩上限后真正下发的 torques_cliped",
        "done=1 表示该步触发过重置(摔倒/超时)，该数据段不可当作连续轨迹",
    ]

    log_fh, log_path = _open_log("wbr_tel", meta, notes, columns)
    written = 0
    seg = []

    def flush_rows():
        nonlocal written
        if not seg:
            return 0
        n = len(seg)
        log_fh.write("".join("\t".join(f % v for f, v in zip(formats, row)) + "\n" for row in seg))
        log_fh.flush()
        written += n
        seg.clear()
        return n

    print("\n" + "=" * 78)
    print("🚀 [play02 遥测采集·single] 已启动")
    for c in CHANNELS:
        print("🎮 " + pad(row_text(c), ROW_LABEL_W))
    print("🎮 空格＝装填跳一次   P＝连跳开关   回车＝五路全部回默认（跳跃回 0＝不跳）")
    print(f"👁️ 监控机器人: {ROBOT} / 关节顺序: {', '.join(dof_names)}")
    print(f" 日志文件: {log_path}")
    print("🛑 Ctrl+C 退出；退出时会把不足一段的零头也落盘")
    print("=" * 78 + "\n")

    start_gui_thread(env, ranges=None, switch=False)

    seg_no, step_counter, current_time = 0, 0, 0.0
    w_peak, w_peak_max, w_at_air, w_air, w_run, prev_ph = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    w_latched = False
    last_arm_at = -99.0
    try:
        while True:
            in_vx = gui_cmds['lin_vel_x']
            in_yaw = gui_cmds['ang_vel_yaw']
            in_h = gui_cmds['base_height']
            in_roll = gui_cmds['ang_roll']

            env.command_targets[:, 0] = in_vx
            env.command_targets[:, 1] = 0.0
            env.command_targets[:, 2] = in_yaw
            env.command_raw_yaw[:] = in_yaw
            env.command_targets[:, 3] = 0.0
            in_jump = gui_cmds['jump_height']
            env.command_targets[:, 5] = in_h
            env.base_roll_targets[:] = in_roll

            armed_now = bool(env.is_armed[ROBOT].item())
            if jump_flags['arm_once'] or (jump_flags['auto'] and not armed_now
                                          and current_time - last_arm_at > 1.2):
                env.h_request[:] = in_jump
                env.arm_at[:] = env.episode_length_buf
                env.is_armed[:] = True
                jump_flags['arm_once'] = False
                last_arm_at = current_time

            actions = policy(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())

            if step_counter == 1 and env.viewer is not None:
                robot_pos = env.root_states[ROBOT, 0:3].cpu().numpy()
                env.set_camera(robot_pos + np.array([-1.5, 1.0, 1.34]), robot_pos + np.array([0.6, 0.0, 0.0]))

            step_counter += 1
            current_time += dt

            cmd = env.commands[ROBOT].detach().cpu().numpy()
            dpos = env.debug_dof_pos.cpu().numpy()
            dvel = env.debug_dof_vel.cpu().numpy()
            scaled = env.debug_actions_scaled.cpu().numpy()
            raw_tau = env.debug_raw_tau.cpu().numpy()
            fin_tau = env.debug_final_tau.cpu().numpy()
            grav = env.projected_gravity[ROBOT].cpu().numpy()
            snap = env.dbg_snapshot(ROBOT)

            leg_ext = _leg_extension(env, dpos)
            act_vx = env.base_lin_vel[ROBOT, 0].item()
            act_yaw = -env.base_ang_vel[ROBOT, 2].item()
            act_h = wheel_r + leg_ext
            act_roll = math.asin(float(np.clip(-grav[1], -1.0, 1.0)))
            body_z = env.root_states[ROBOT, 2].item()
            vz_body = env.root_states[ROBOT, 9].item()
            n_ground = int((env.contact_forces[ROBOT, env.feet_indices, 2] > 0.1).sum().item())
            body_pitch = math.asin(float(np.clip(grav[0], -1.0, 1.0)))
            com_z = float(env._system_com_z()[ROBOT].item())
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            w_bot = (rb_state[ROBOT, env.feet_indices, 2] - wheel_r).cpu().numpy()
            wheel_bot_l, wheel_bot_r = float(w_bot[0]), float(w_bot[1])
            wheel_clear = min(wheel_bot_l, wheel_bot_r)
            h_bias = body_z - act_h - wheel_clear
            act_h_corr = act_h + h_bias
            ph_now = int(round(snap['ph']))
            if ph_now == 1 and prev_ph == 0:
                w_peak, w_peak_max, w_at_air, w_air, w_run = 0.0, 0.0, 0.0, 0.0, 0.0
                w_latched = False
            if n_ground == 0:
                if not w_latched:
                    w_at_air, w_latched = wheel_clear, True
                w_peak = max(w_peak, wheel_clear)
                w_peak_max = max(w_peak_max, max(wheel_bot_l, wheel_bot_r))
                w_run += dt
                w_air = max(w_air, w_run)
            else:
                w_run = 0.0
            prev_ph = ph_now
            snap['wheel_clear'] = wheel_clear
            snap['wheel_peak_min'] = w_peak
            snap['t_air'] = w_air

            row = [step_counter, current_time, int(dones[ROBOT].item()),
                   in_vx, cmd[0], act_vx,
                   in_yaw, cmd[2], act_yaw,
                   in_h, cmd[5], act_h,
                   in_roll, cmd[6], act_roll,
                   in_jump, cmd[4], snap['h_body_rise'],
                   body_z, vz_body, n_ground, body_pitch, com_z,
                   wheel_bot_l, wheel_bot_r, wheel_clear, act_h_corr,
                   w_peak, w_peak_max, w_at_air, w_air]
            for i in range(num_j):
                row += [scaled[i] if i in wheel_set else 0.0, dvel[i], raw_tau[i], fin_tau[i]]
                if i in leg_set:
                    row += [default_pos[i] + scaled[i], dpos[i]]
            row += [snap[k] for k in dbg_keys]
            assert len(row) == len(columns), "采样值数量与表头列数不一致"
            seg.append(row)

            readout = {'cmd': {'vx': (in_vx, float(cmd[0]), act_vx),
                               'yaw': (in_yaw, float(cmd[2]), act_yaw),
                               'h': (in_h, float(cmd[5]), act_h),
                               'roll': (in_roll, float(cmd[6]), act_roll)},
                       'dbg': snap,
                       'cmd_jump': (in_jump, float(cmd[4]), snap['h_body_rise']),
                       'armed': bool(env.is_armed[ROBOT].item()),
                       'auto': jump_flags['auto']}

            if step_counter % PRINT_EVERY == 0:
                print(f"⏱ {current_time:7.2f}s  窗{seg_no:03d}  第{step_counter:06d}步")
                print("   " + cmd_cell(BY_SHORT['vx'], in_vx, cmd[0], act_vx) + " | "
                      + cmd_cell(BY_SHORT['h'], in_h, cmd[5], act_h))
                print("   " + cmd_cell(BY_SHORT['yaw'], in_yaw, cmd[2], act_yaw) + " | "
                      + cmd_cell(BY_SHORT['roll'], in_roll, cmd[6], act_roll))
                print("   " + pad("跳跃", 11) + fmt_jump(snap, in_jump,
                                                             bool(env.is_armed[ROBOT].item()),
                                                             jump_flags['auto']))
                print("   " + pad("目标层", 11) + fmt_dbg_a(snap))
                print("   " + pad("", 11) + fmt_dbg_b(snap))
                print("-" * 118)

            if step_counter % seg_steps == 0:
                n = flush_rows()
                seg_no += 1
                print(f"📝 已写 {n} 行（累计 {written} 行 / {written * dt:.1f} s）→ {os.path.basename(log_path)}")

    except KeyboardInterrupt:
        print("\n⏹️ 收到退出信号，正在收尾...")
    finally:
        n = flush_rows()
        if n:
            print(f"📝 零头 {n} 行已追加")
        log_fh.close()
        print(f"✅ 共 {written} 行 → {log_path}，跑 python plot02.py 出图")


# switch 档：单 wbr_switch env + F3e / P7S 两策略 + 模式机
def _build_switch(args):
    env_cfg, _ = task_registry.get_cfgs(name="wbr_switch")
    env_cfg.env.num_envs = 3
    env_cfg.env.episode_length_s = 3600
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = 3600
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.obs_delay_max_steps = 0
    env_cfg.rewards.switch.residual_base = False      # 纯策略回放（不叠加 W3 基线）

    env, _ = task_registry.make_env(name="wbr_switch", args=args, env_cfg=env_cfg)
    env.playback_hold = True
    dev = env.device
    p_4w = load_actor_critic(args.fourw_checkpoint, 164, dev)
    p_stand = load_actor_critic(args.standup_checkpoint, 173, dev)
    p_w3 = load_actor_critic(args.w3_checkpoint, 161, dev)   # 两轮站立/行走平衡网（交接目标）
    p_lie = load_actor_critic(args.lie_checkpoint, 173, dev)  # 趴下（2W→4W）专用策略
    return env, p_4w, p_stand, p_w3, p_lie


def _set_cmd_ranges(env, lo, hi):
    """切模式时改体高指令量程（只影响观测里体高归一化那一列）。"""
    env.cfg.commands.ranges.base_height = [lo, hi]
    cr = getattr(env, "command_ranges", None)
    if isinstance(cr, dict) and "base_height" in cr:
        cr["base_height"] = [lo, hi]


def run_switch(args):
    global readout
    env, p_4w, p_stand, p_w3, p_lie = _build_switch(args)
    n = env.num_envs
    dev = env.device
    all_ids = torch.arange(n, dtype=torch.long, device=dev)

    ROBOT = 0
    dof_names = list(env.dof_names)
    wheel_r = env.cfg.rewards.wheel_radius
    sw = env.sw
    dt = env.dt
    seg_steps = int(round(SEGMENT_SECONDS / dt))

    # 两轮站立/行走用 W3 平衡网：动作 W3 序→wbr4 序重排；观测用 env._build_w3_obs()；
    #   两轮模式下把 default_dof_pos 切成 W3 的 2W 站姿（W3 的 dof_err/PD 都相对它）。
    W3_ACT_PERM = [0, 1, 3, 4, 2, 5]
    _dflt_4w = env.default_dof_pos.clone()
    _dflt_2w = _dflt_4w.clone()
    _dflt_2w[:, env.leg_dof] = torch.tensor([1.0, -2.1, -1.0, 2.1], device=dev)

    def _set_default(dflt):
        env.default_dof_pos.copy_(dflt)

    def _obs_for(mode):
        """按当前模式取该策略的观测（obs_buf 已在 compute_observations 里按当前 ranges/ph 算好）。"""
        if mode == MODE_FOURW:
            return env.obs_buf[:, :164]
        if mode == MODE_TWOW:
            return env._build_w3_obs()      # 两轮站立/行走：W3 的 161 维（6 通道重排、丢 caster2/tip1）
        # 起立：P7S 的 173 维（相位 one-hot 已反映 ph）
        return env.obs_buf[:]

    def _act(policy, obs):
        with torch.no_grad():
            return policy.act_inference(obs)

    def _apply_mode(mode, ph=SP_HOLD4W, ph_t=0.0):
        """进入某模式：设体高量程 + 默认姿态 + 相位，刷新观测。"""
        if mode == MODE_FOURW:
            _set_cmd_ranges(env, 0.175, 0.245); _set_default(_dflt_4w)
            env.playback_set_mode(all_ids, SP_HOLD4W, task=1, ph_t=0.0)
        elif mode == MODE_TWOW:
            _set_cmd_ranges(env, 0.2, 0.35); _set_default(_dflt_2w)   # W3 体高量程 + 2W 默认姿态
            env.playback_set_mode(all_ids, SP_HOLD2W, task=1, ph_t=ph_t)
        elif mode == MODE_LOWERING:
            _set_cmd_ranges(env, 0.19, 0.19); _set_default(_dflt_4w)  # 趴下：4W 默认姿态
            env.playback_set_mode(all_ids, ph, task=0, ph_t=ph_t)
        else:  # RISING：P7S 起立（4W 默认姿态，相位机推进到 RISE 末）
            _set_cmd_ranges(env, 0.19, 0.19); _set_default(_dflt_4w)
            env.playback_set_mode(all_ids, ph, task=1, ph_t=ph_t)
        env.compute_observations()

    columns = ['step', 'time_s', 'done', 'mode', 'ph', 'ev_up', 'n_ground', 'caster_off',
               'pitch', 'body_z', 'act_h', 'in_vx', 'tgt_vx', 'act_vx',
               'in_yaw', 'tgt_yaw', 'act_yaw', 'in_h', 'tgt_h']
    formats = [('%d' if c in ('step', 'done', 'mode', 'ph', 'ev_up', 'n_ground', 'caster_off') else '%0.4f')
               for c in columns]
    meta = [f"joints={','.join(dof_names)}", f"dt={dt:.6f}", "task=wbr_switch",
            f"4w={args.fourw_checkpoint}", f"standup={args.standup_checkpoint}"]
    notes = [
        "mode: 0=四轮 1=起立中 2=两轮站立 3=趴下中；ph=switch 相位(0..6)；ev_up=起立完成锁存",
        "caster_off = 两从动轮都离地(1/0)；n_ground = 驱动轮触地数；pitch = asin(projected_gravity.x)",
        "四轮模式用 F3e(164)，起立/两轮站立用 P7S(173)；切换为瞬时摆位（非物理过渡）",
    ]
    log_fh, log_path = _open_log("wbr_switch_tel", meta, notes, columns)
    written = 0
    seg = []

    def flush_rows():
        nonlocal written
        if not seg:
            return 0
        k = len(seg)
        log_fh.write("".join("\t".join(f % v for f, v in zip(formats, row)) + "\n" for row in seg))
        log_fh.flush()
        written += k
        seg.clear()
        return k

    # 初始：完整 4W 起点，进四轮模式
    env.playback_reset()
    mode = MODE_FOURW
    mode_shared['mode'] = mode
    _apply_mode(MODE_FOURW)
    rise_t = 0.0
    rise_up_cnt = 0
    lie_t = 0.0
    lie_leg0 = _dflt_2w[0, env.leg_dof].clone()   # 进入趴下瞬间腿姿态
    TWOW_BLEND_SEC = 0.30
    tw_leg0 = _dflt_2w[0, env.leg_dof].clone()   # 起立末腿姿态(进入两轮瞬间更新)，[4]
    tw_elapsed = TWOW_BLEND_SEC                  # 初值=完成，未进入两轮前不生效

    print("\n" + "=" * 78)
    print("🚀 [play02 遥测采集·switch] 已启动（四轮 ⇄ 起立 ⇄ 两轮站立）")
    print("🎮 五路滑块/键盘同 single；按钮「切换两轮/四轮模式」或 M 键切换；空格/「跳跃」=装填跳；R/「复位」复位")
    print("   两轮站立/行走由 W3 平衡网驱动（vx/yaw 可走；高度滑块在两轮模式固定）")
    print(f"👁️ 监控机器人: {ROBOT} / 关节顺序: {', '.join(dof_names)}")
    print(f" 日志文件: {log_path}")
    print("🛑 Ctrl+C 退出")
    print("=" * 78 + "\n")

    gui_cmds['base_height'] = 0.19   # switch 档初始体高取四轮工作高（面板滑块同显 0.19），四轮起始水平
    # vx/yaw 滑条量程跟随所加载四轮模型（ckpt 同目录 config.json 的训练量程，提速线 ±1.5 / 未来 ±2.0 自动跟上；
    # 读不到回退 ±1.0），并与两轮侧 ±1.0 取并集——否则 P7F4+ 模型在面板/控制台滑不出新量程。
    _vx4 = _yw4 = (-1.0, 1.0)
    try:
        import json as _json
        with open(os.path.join(os.path.dirname(os.path.abspath(args.fourw_checkpoint)), 'config.json')) as _f:
            _r = _json.load(_f)['commands']['ranges']
        _vx4 = (float(_r['lin_vel_x'][0]), float(_r['lin_vel_x'][1]))
        _yw4 = (float(_r['ang_vel_yaw'][0]), float(_r['ang_vel_yaw'][1]))
    except Exception:
        pass
    start_gui_thread(env, ranges={
        'lin_vel_x': (min(-1.0, _vx4[0]), max(1.0, _vx4[1])),
        'ang_vel_yaw': (min(-1.0, _yw4[0]), max(1.0, _yw4[1])),
        'base_height': (0.175, 0.245), 'ang_roll': (-0.2, 0.2),
        'jump_height': (0.0, 0.15),
    }, switch=True)

    step_counter, current_time, seg_no = 0, 0.0, 0
    ev_up_latch = False
    _max = int(os.environ.get('PLAY02_MAX_STEPS', '0') or 0)
    _togs = [int(x) for x in os.environ.get('PLAY02_TEST_TOGGLE_STEP', '').split(',') if x.strip()]
    _tog_fired = set()
    try:
        while True:
            if _max and step_counter >= _max:
                break
            for _ts in _togs:
                if _ts not in _tog_fired and step_counter >= _ts:
                    mode_shared['request'] = 'toggle'; _tog_fired.add(_ts); break
            # ---- 面板请求：复位 / 切换 ----
            if mode_shared.get('reset'):
                env.playback_reset()
                mode = MODE_FOURW
                mode_shared['mode'] = mode
                rise_t = 0.0
                ev_up_latch = False
                _apply_mode(MODE_FOURW)
                mode_shared['reset'] = False

            req = mode_shared.get('request')
            if req == 'toggle':
                if mode == MODE_FOURW:
                    mode = MODE_RISING
                    rise_t = 0.0
                    rise_up_cnt = 0
                    ev_up_latch = False
                    _apply_mode(MODE_RISING, ph=SP_HOLD4W, ph_t=0.0)
                elif mode == MODE_TWOW:
                    # 两轮站立 → 趴下（2W→4W：趴下策略驱动，腿折叠落回四轮）
                    mode = MODE_LOWERING
                    lie_t = 0.0
                    lie_leg0 = env.dof_pos[0, env.leg_dof].clone()   # 两轮末腿姿态 [4]
                    _apply_mode(MODE_LOWERING, ph=SP_HOLD2W, ph_t=0.0)
                mode_shared['request'] = None
                mode_shared['mode'] = mode

            # ---- 按模式注入指令 + 取观测 ----
            if mode == MODE_FOURW:
                in_vx = gui_cmds['lin_vel_x']
                in_yaw = gui_cmds['ang_vel_yaw']
                in_h = min(max(gui_cmds['base_height'], 0.175), 0.245)   # 夹进 F3e 量程，避免超范围
                env.command_targets[:, 0] = in_vx
                env.command_targets[:, 1] = 0.0
                env.command_targets[:, 2] = in_yaw
                env.command_raw_yaw[:] = in_yaw
                env.command_targets[:, 3] = 0.0
                env.command_targets[:, 5] = in_h
                env.base_roll_targets[:] = 0.0
                _apply_mode(MODE_FOURW)                     # ph=0，体高量程=F3e
                obs_in = _obs_for(MODE_FOURW)
                act = _act(p_4w, obs_in)

            elif mode == MODE_RISING:
                # 起立（方案②）：P7S 完整相位机起立（HOLD4W→RISE→HANDOVER→HOLD2W），到交接点切 W3 对称站定。
                #   P7S 负责抬起并站到两轮，W3 负责对称站定/平衡（之后 vx/yaw 可走）。
                in_vx = in_yaw = 0.0
                in_h = 0.19
                env.command_targets[:, 0] = 0.0
                env.command_targets[:, 2] = 0.0
                env.command_raw_yaw[:] = 0.0
                env.command_targets[:, 5] = 0.19
                env.base_roll_targets[:] = 0.0
                if rise_t < sw.t_align:
                    ph = SP_HOLD4W
                elif rise_t < sw.t_align + sw.t_rise:
                    ph = SP_RISE
                elif rise_t < sw.t_align + sw.t_rise + sw.t_handover:
                    ph = SP_HANDOVER
                else:
                    ph = SP_HOLD2W
                _apply_mode(MODE_RISING, ph=ph, ph_t=rise_t)
                obs_in = _obs_for(MODE_RISING)
                act = _act(p_stand, obs_in)
                # 到交接点 → 切两轮站立（交给 W3，对称站定；腿目标 ~0.3s 平滑并入 W3，轮由 W3 平衡）
                if rise_t >= sw.t_align + sw.t_rise + sw.t_handover:
                    mode = MODE_TWOW
                    ev_up_latch = True
                    mode_shared['mode'] = mode
                    tw_leg0 = env.dof_pos[0, env.leg_dof].clone()
                    tw_elapsed = 0.0
                    _apply_mode(MODE_TWOW, ph=SP_HOLD2W)
                rise_t += dt

            elif mode == MODE_TWOW:  # 两轮站立/行走：W3 平衡网，vx/yaw 可走
                in_vx = gui_cmds['lin_vel_x']
                in_yaw = gui_cmds['ang_vel_yaw']
                in_h = 0.28                                  # W3 站高（进观测归一化；实际高度由策略定）
                env.command_targets[:, 0] = in_vx
                env.command_targets[:, 1] = 0.0
                env.command_targets[:, 2] = in_yaw
                env.command_raw_yaw[:] = in_yaw
                env.command_targets[:, 3] = 0.0
                env.command_targets[:, 5] = in_h
                env.base_roll_targets[:] = 0.0
                _apply_mode(MODE_TWOW, ph=SP_HOLD2W)         # W3 体高量程 + 2W 默认姿态
                obs_in = _obs_for(MODE_TWOW)
                act = _act(p_w3, obs_in)[:, W3_ACT_PERM]      # W3 序 -> wbr4 序
                # 腿目标平滑并入：alpha 0→1 把腿 PD 目标从起立末姿态缓到 W3 目标（轮仍由 W3 平衡）
                if tw_elapsed < TWOW_BLEND_SEC:
                    alpha = min(1.0, tw_elapsed / TWOW_BLEND_SEC)
                    dflt2 = _dflt_2w[0, env.leg_dof]                       # [4]
                    w3_tgt = act[:, :4] * env.leg_scale + dflt2             # W3 腿目标关节角 [n,4]
                    tgt = (1.0 - alpha) * tw_leg0 + alpha * w3_tgt          # 平滑目标 [n,4]
                    act = act.clone()
                    act[:, :4] = (tgt - dflt2) / env.leg_scale
                    tw_elapsed += dt

            elif mode == MODE_LOWERING:
                # 趴下：2W→4W。腿折叠落回四轮（趴下策略驱动），指令清零；完成(到 HOLD4W)后切回四轮行驶。
                in_vx = in_yaw = 0.0
                in_h = 0.19
                env.command_targets[:, 0] = 0.0
                env.command_targets[:, 2] = 0.0
                env.command_raw_yaw[:] = 0.0
                env.command_targets[:, 5] = 0.19
                env.base_roll_targets[:] = 0.0
                # 相位按时刻推进 D0(HOLD2W)→D1(LOWER)→D2(TILT)→D3(ABSORB)→HOLD4W
                if lie_t < sw.t_align:
                    lph = SP_HOLD2W
                elif lie_t < sw.t_align + sw.t_lower:
                    lph = SP_LOWER
                elif lie_t < sw.t_align + sw.t_lower + sw.t_tilt:
                    lph = SP_TILT
                elif lie_t < sw.t_align + sw.t_lower + sw.t_tilt + sw.t_absorb:
                    lph = SP_ABSORB
                else:
                    lph = SP_HOLD4W
                _apply_mode(MODE_LOWERING, ph=lph, ph_t=lie_t)
                obs_in = _obs_for(MODE_LOWERING)
                act = _act(p_lie, obs_in)
                # 腿目标平滑并入：把腿 PD 目标从两轮末姿态缓到趴下策略目标（避免默认切换瞬间腿突跳）
                if lie_t < TWOW_BLEND_SEC:
                    alpha = min(1.0, lie_t / TWOW_BLEND_SEC)
                    dflt4 = _dflt_4w[0, env.leg_dof]                       # [4]
                    lie_tgt = act[:, :4] * env.leg_scale + dflt4            # 趴下策略腿目标 [n,4]
                    tgt = (1.0 - alpha) * lie_leg0 + alpha * lie_tgt
                    act = act.clone()
                    act[:, :4] = (tgt - dflt4) / env.leg_scale
                # 完成 → 切回四轮行驶
                if lie_t >= sw.t_align + sw.t_lower + sw.t_tilt + sw.t_absorb:
                    mode = MODE_FOURW
                    mode_shared['mode'] = mode
                    _apply_mode(MODE_FOURW)
                lie_t += dt

            # 跳跃按钮/空格：接 env 装填通路（当前 P7S 无跳跃、enable_jump=False 惰性；接含跳跃的 W3 时即生效）
            if jump_flags['arm_once'] and hasattr(env, 'is_armed'):
                env.h_request[:] = gui_cmds['jump_height']
                env.arm_at[:] = env.episode_length_buf
                env.is_armed[:] = True
                jump_flags['arm_once'] = False

            obs, _, _, dones, _ = env.step(act.detach())

            if step_counter == 1 and env.viewer is not None:
                robot_pos = env.root_states[ROBOT, 0:3].cpu().numpy()
                env.set_camera(robot_pos + np.array([-1.5, 1.0, 1.34]), robot_pos + np.array([0.6, 0.0, 0.0]))

            step_counter += 1
            current_time += dt

            # ---- 采样 ----
            cmd = env.commands[ROBOT].detach().cpu().numpy()
            dpos = env.debug_dof_pos.cpu().numpy()
            grav = env.projected_gravity[ROBOT].cpu().numpy()
            body_z = env.root_states[ROBOT, 2].item()
            act_h = wheel_r + _leg_extension(env, dpos)
            act_vx = env.base_lin_vel[ROBOT, 0].item()
            act_yaw = -env.base_ang_vel[ROBOT, 2].item()
            pitch = math.asin(float(np.clip(grav[0], -1.0, 1.0)))
            n_ground = int((env.contact_forces[ROBOT, env.feet_indices, 2] > 0.1).sum().item())
            caster_off = int((~env._caster_contacts()[ROBOT]).all().item())
            ph_now = int(env.ph[ROBOT].item())

            row = [step_counter, current_time, int(dones[ROBOT].item()), MODE_NUM[mode], ph_now,
                   int(ev_up_latch), n_ground, caster_off, pitch, body_z, act_h,
                   in_vx, cmd[0], act_vx, in_yaw, cmd[2], act_yaw, in_h, cmd[5]]
            assert len(row) == len(columns), "采样值数量与表头列数不一致"
            seg.append(row)

            snap = env.dbg_snapshot(ROBOT)
            readout = {'cmd': {'vx': (in_vx, float(cmd[0]), act_vx),
                               'yaw': (in_yaw, float(cmd[2]), act_yaw),
                               'h': (in_h, float(cmd[5]), act_h),
                               'roll': (0.0, 0.0, 0.0)},
                       'dbg': snap,
                       'cmd_jump': (gui_cmds['jump_height'], float(cmd[4]), 0.0),
                       'armed': bool(env.is_armed[ROBOT].item()),
                       'auto': jump_flags['auto'],
                       'mode_line': fmt_mode(mode, ph_now, pitch * 180.0 / math.pi, body_z, ev_up_latch)}

            if step_counter % PRINT_EVERY == 0:
                print(f"⏱ {current_time:7.2f}s  模式 {MODE_CN[mode]:<4} ph {ph_now}  "
                      f"pitch {pitch*180/math.pi:+5.1f}°  z {body_z:.3f}  casterOff {caster_off}  "
                      f"ev {'✓' if ev_up_latch else '·'}")

            if step_counter % seg_steps == 0:
                k = flush_rows()
                seg_no += 1
                print(f"📝 已写 {k} 行（累计 {written} 行）→ {os.path.basename(log_path)}")

    except KeyboardInterrupt:
        print("\n⏹️ 收到退出信号，正在收尾...")
    finally:
        k = flush_rows()
        if k:
            print(f"📝 零头 {k} 行已追加")
        log_fh.close()
        print(f"✅ 共 {written} 行 → {log_path}")


# ==============================================================================
def play02(args):
    if getattr(args, "play_mode", "single") == "switch":
        run_switch(args)
    else:
        run_single(args)


if __name__ == '__main__':
    play02(get_args())
