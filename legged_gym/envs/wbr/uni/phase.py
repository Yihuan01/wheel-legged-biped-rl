# ==============================================================================
# Date      : 2026-10-26
# Author    : SYH
# File      : phase.py
# Purpose   : 统一相位枚举（0-11）与相位分组常量。纯 Python 常量，不依赖 torch / isaacgym，
#             仿真与真机共用同一套编号。
# ==============================================================================

# ── 相位编号 ──
PH_IDLE2W = 0      # 两轮：站立 / 行走 / 跳跃待触发
PH_JUMP1 = 1       # 跳跃五相（按顺序推进，见 JUMP_PHASES）
PH_JUMP2 = 2
PH_JUMP3 = 3
PH_JUMP4 = 4
PH_JUMP5 = 5
PH_HOLD4W = 6      # 四轮静稳
PH_RISE = 7        # 起立过渡：转起
PH_HANDOVER = 8    # 起立过渡：交接给两轮策略
PH_LOWER = 9       # 趴下过渡：降高
PH_TILT = 10       # 趴下过渡：探地
PH_ABSORB = 11     # 趴下过渡：吸能定住

NUM_PHASES = 12

# 跳跃相位集合（FSM 不主动进入；外部可手动置入做结构测试）
JUMP_PHASES = (PH_JUMP1, PH_JUMP2, PH_JUMP3, PH_JUMP4, PH_JUMP5)

# ── 相位 → 控制模式（决定"这一拍由谁驱动"）──
#   TWOW  : 两轮（行走/站立/跳跃子状态）→ 两轮行走模型（W12）
#   FOURW : 四轮静稳                  → 四轮模型（F3e）
#   RISE  : 起立过渡                  → 起立脚本（P7T）
#   LOWER : 趴下过渡                  → 趴下模型（P7L）
MODE_TWOW = "TWOW"
MODE_FOURW = "FOURW"
MODE_RISE = "RISE"
MODE_LOWER = "LOWER"

PHASE_TO_MODE = {
    PH_IDLE2W: MODE_TWOW,
    PH_JUMP1: MODE_TWOW, PH_JUMP2: MODE_TWOW, PH_JUMP3: MODE_TWOW,
    PH_JUMP4: MODE_TWOW, PH_JUMP5: MODE_TWOW,
    PH_HOLD4W: MODE_FOURW,
    PH_RISE: MODE_RISE, PH_HANDOVER: MODE_RISE,
    PH_LOWER: MODE_LOWER, PH_TILT: MODE_LOWER, PH_ABSORB: MODE_LOWER,
}

PHASE_NAMES = {
    PH_IDLE2W: "IDLE2W", PH_JUMP1: "JUMP1", PH_JUMP2: "JUMP2", PH_JUMP3: "JUMP3",
    PH_JUMP4: "JUMP4", PH_JUMP5: "JUMP5", PH_HOLD4W: "HOLD4W", PH_RISE: "RISE",
    PH_HANDOVER: "HANDOVER", PH_LOWER: "LOWER", PH_TILT: "TILT", PH_ABSORB: "ABSORB",
}


def mode_of(phase):
    """相位 → 控制模式（标量或逐元素均可）。"""
    return PHASE_TO_MODE[phase]


def is_jumping(phase):
    return phase in JUMP_PHASES
