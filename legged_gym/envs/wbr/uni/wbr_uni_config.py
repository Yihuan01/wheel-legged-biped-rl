# ==============================================================================
# Date      : 2026-10-26
# Author    : SYH
# File      : wbr_uni_config.py
# Purpose   : 统一多模式任务 wbr_uni 的配置：173 维观测，跳跃只允许在 PH_IDLE2W 触发。
# ==============================================================================

from legged_gym.envs.wbr.switch.wbr_switch_config import WBRSwitchCfg, WBRSwitchCfgPPO


class WBRUniCfg(WBRSwitchCfg):
    class env(WBRSwitchCfg.env):
        num_observations = 173   # 164 基础 + 9 switch 块（与 wbr_switch 同）
        num_actions = 6
        caster_air_timeout_s = 1.0e9  # 两轮/过渡必然从动轮离地，离地终止关死
        # 跳跃只允许在 ph == 0（两轮 idle）触发，清掉其它相位残留的 is_armed。
        jump_gate_phase0 = True

    # 跳跃机继承自 wbr4，但默认关闭：demo_cycle / smoke_uni 直接把相位写成 SP_RISE / SP_LOWER，
    # 全局开跳会与跳跃相位撞号。开与关由 play02 按当前模式在运行时切 flags.enable_jump。


class WBRUniCfgPPO(WBRSwitchCfgPPO):
    class runner(WBRSwitchCfgPPO.runner):
        experiment_name = 'wbr_uni'
        run_name = 'uni_w12'
