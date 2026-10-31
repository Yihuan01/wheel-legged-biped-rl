# ==============================================================================
# Date      : 2026-10-26
# Author    : SYH
# File      : obs_builder.py
# Purpose   : 跨策略的观测与动作重排：wbr_v1（161 维）与 wbr_v2（164 维）的关节块顺序互换，
#             以及 164 维观测的分段偏移常量。
# ==============================================================================

import torch

# 关节序不同：wbr_v2(wbr4) = [l_hip, l_knee, r_hip, r_knee, l_wheel, r_wheel]
#             wbr_v1(W12)  = [l_hip, l_knee, l_wheel, r_hip, r_knee, r_wheel]
W12_OBS_PERM = [0, 1, 4, 2, 3, 5]
# 动作重排：W12 输出（wbr_v1 序）→ env 输入（wbr4 序）
W12_ACT_PERM = [0, 1, 3, 4, 2, 5]

# 164 布局常量（wbr4 / wbr_uni 一致）
BASE_DIM = 15          # 基础 15（lin_vel3/ang_vel3/grav3/command3/jump1/aux2 → 实为 15? 见下）
# 164 维观测的分段：[0,15) 基础量 → [15,21) dof_err → [21,27) dof_vel →
# [27,33) last_action → [33,153) 历史 → [153,161) 跳跃相位 → [161,164) 从动轮与倾翻裕度。
_SEG_BASE = (0, 15)
_SEG_DOF_ERR = (15, 21)
_SEG_DOF_VEL = (21, 27)
_SEG_LAST_ACT = (27, 33)
_SEG_POS_HIST = (33, 93)
_SEG_VEL_HIST = (93, 153)
_SEG_PHASE = (153, 161)


def _reindex_6(x, perm):
    """把最后一维按 6 一组重排（x[..., :6] 视为一块）。"""
    return x[..., perm]


def _reindex_hist(x, perm):
    """历史块 [n, 10*6] → 按帧重排每个 6 维。"""
    s = x.shape
    return x.view(*s[:-1], s[-1] // 6, 6)[..., perm].reshape(*s)


def build_w12_obs(obs164):
    """wbr_v2 的 164 维观测 → wbr_v1(W12) 的 161 维（6 维关节块按名字重排）。

    输入：obs164 [n, 164]（wbr4 / wbr_uni 布局）
    输出：obs161 [n, 161]（W12 训练布局）
    """
    perm = W12_OBS_PERM
    return torch.cat([
        obs164[:, _SEG_BASE[0]:_SEG_BASE[1]],
        _reindex_6(obs164[:, _SEG_DOF_ERR[0]:_SEG_DOF_ERR[1]], perm),
        _reindex_6(obs164[:, _SEG_DOF_VEL[0]:_SEG_DOF_VEL[1]], perm),
        _reindex_6(obs164[:, _SEG_LAST_ACT[0]:_SEG_LAST_ACT[1]], perm),
        _reindex_hist(obs164[:, _SEG_POS_HIST[0]:_SEG_POS_HIST[1]], perm),
        _reindex_hist(obs164[:, _SEG_VEL_HIST[0]:_SEG_VEL_HIST[1]], perm),
        obs164[:, _SEG_PHASE[0]:_SEG_PHASE[1]],
    ], dim=-1)


def build_4w_obs(obs164):
    """四轮 F3e：直接 164（wbr_v2 布局，无需重排）。"""
    return obs164


def w12_action_to_env(act_w12):
    """W12 动作（wbr_v1 序）→ env 动作（wbr4 序）。"""
    return act_w12[:, W12_ACT_PERM]
