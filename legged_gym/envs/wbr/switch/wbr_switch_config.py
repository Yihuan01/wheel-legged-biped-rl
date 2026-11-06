# ==============================================================================
# Date      : 2026-10-25
# Author    : SYH
# File      : wbr_switch_config.py
# Purpose   : 切换任务 wbr_switch 的配置，并定义 7 个切换相位号与观测维度（164 + 9 = 173）。
# ==============================================================================

from legged_gym.envs.wbr.flat4.wbr_flat4_config import WBRFlat4Cfg, WBRFlat4CfgPPO


# ── 相位号（与 wbr_switch.py::WBRSwitchMob 保持一致）──
# 起立：0=U0 对齐4W  → 1=U1 转起(残差) → 2=U2 交接 → 3=2W 保持
# 趴下：3=D0 对齐2W(站定) → 4=D1 降高 → 5=D2 探地 → 6=D3 吸能定住 → 0=4W 保持
SP_HOLD4W, SP_RISE, SP_HANDOVER, SP_HOLD2W, SP_LOWER, SP_TILT, SP_ABSORB = range(7)
NUM_SW_PHASES = 7


class WBRSwitchCfg(WBRFlat4Cfg):
    class env(WBRFlat4Cfg.env):
        # 164（四轮基础） + 9（switch 块） = 173
        num_observations = 173
        num_actions = 6
        num_envs = 4096
        episode_length_s = 8    # 切换是短时任务；8s 足够"对齐+转起+保持"整段
        caster_air_timeout_s = 1.0e9  # 切换中从动轮必然离地（2W 段），离地终止**关死**

    class init_state(WBRFlat4Cfg.init_state):
        # 复位姿态由 wbr_switch.py::_reset_root_states/_reset_dofs 按模式动态设置；
        # 此处保留 4W 默认作基类占位（两默认姿态见 rewards.switch）。
        pos = [0.0, 0.0, 0.19]
        rot = [0.0, 0.0, 0.0, 1.0]

    class domain_rand(WBRFlat4Cfg.domain_rand):
        # 切换任务：动作延迟保留（真实执行滞后），**观测延迟置 0**（switch 块 9 维不加噪/不延迟）。
        obs_delay_max_steps = 0
        action_delay_max_steps = 2

    class rewards(WBRFlat4Cfg.rewards):
        class flags(WBRFlat4Cfg.rewards.flags):
            enable_jump = False        # 相位机由 switch 自己的 _update_switch_phase 驱动
            enable_turning_tilt = False
            enable_leg_adjust = False

        class scales(WBRFlat4Cfg.rewards.scales):
            # ── 四轮行驶专用项在切换任务里全关（不学行驶）──
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            tracking_wheel_lin_vel = 0.0
            turning_wheel = 0.0
            tracking_base_height = 0.0
            no_fly = 0.0
            caster_contact = 0.0
            wrong_way = 0.0
            no_moonwalk = 0.0
            # ── 起立密集塑形（P7S，站立阶段；目标态密集奖励）──
            sw_goal_state = 12.0       # 【核心】目标态：双从动轮离地 ∧ 双驱动轮触地 ∧ |pitch|小 ∧ 低速
            sw_height = 3.0            # 升体：base_z 从 0.19→0.28，正向进度
            sw_caster_lift = 3.0       # 抬从动轮：从动轮离地
            sw_keep_wheel = 3.0        # 保持双驱动轮触地（不跳起/不打滑悬空）
            sw_pitch_zero = 5.0        # 俯仰归零 exp(−pitch²/2σ²)（σ 温和，允许瞬态）
            sw_align = 1.0             # U0 对齐起点姿态（起点保持）
            sw_symmetry = 4.0          # 对称奖励（P7S 仅能~45°；真正对称站定交 W3 交接）
            sw_still = 4.0             # 【甲】站定静止（HANDOVER+HOLD2W 抑残余线速/偏航角速度，σ宽）
            # ── 趴下密集塑形（P7L，趴下阶段；仅 sw_task==0 生效）──
            sw_lower = 3.0             # 降体：base_z 从 z2w→z4w 的归一化进度
            sw_caster_touch = 3.0      # 探地：双从动轮触地
            sw_goal_4w = 12.0          # 【核心】四轮目标态：双从动轮触地 ∧ 双驱动轮触地 ∧ |pitch|小 ∧ 低速
            sw_ev_upright = 0.0        # 起立成功事件（关：靠密集奖励）
            sw_ev_settled4 = 8.0       # 趴下成功一次性结算（到 HOLD4W 确认那一拍给 1）
            sw_impact = -1.0           # 从动轮触地冲击速度 hinge 罚（趴下触地冲击）
            # ── 通用姿态/平顺（轻罚；不罚必要的前冲/俯仰瞬态）──
            orientation = -1.0
            lin_vel_z = -0.5
            ang_vel_xy = -0.3
            leg_action_rate = -0.15
            wheel_action_rate = -0.05
            dof_pos_limits = -1.0
            dof_vel_limits = -0.2
            torque_limits = -0.001
            torques = -0.002
            dof_acc = -1e-7

        # ── 两默认姿态与切换参数（wbr_switch.py 直读）──
        class switch:
            # 【方案1 低姿态】4W 低默认（z=0.18，几何 IK：hip56.3°/knee−149.8°）
            q4w_hip = 0.9833
            q4w_knee = -2.5589
            z4w = 0.19
            # 【方案1 低姿态】2W 低默认（z=0.22，几何 IK：hip57.3°/knee−137.1°；W3 控轮平衡）
            q2w_hip = 1.0
            q2w_knee = -2.1
            z2w = 0.2697
            # 相位时长（保险丝；推进靠事件）
            t_align = 0.30
            t_rise = 0.60
            t_handover = 0.50
            t_lower = 0.50
            t_tilt = 0.30
            t_absorb = 0.40
            # 事件确认与阈值
            ev_confirm_steps = 3      # ev_upright / ev_settled4 需连续确认拍数
            upright_pitch_tol = 0.15
            settled_v_tol = 0.12
            settled_w_tol = 0.10
            align_tol = 0.15          # 对齐判定：|q−q_def| 均值 < 此值
            # 残差基线（起立：腿沿走廊伸 + 轮=W3平衡输出 + pitch-PD 辅助；策略学 Δ）
            residual_base = False
            wheel_ff_scale = 0.0      # （保留）额外轮前馈
            w3_model = "releases/W3_vlim085/model_2200.pt"  # 起立基线用的 2W 平衡网（W3）
            base_kp_pitch = 0.5       # 基线 pitch-PD 辅助增益
            base_kd_pitch = 0.15
            # 反向课程：起点采样 frac ∈ [0, cur_frac]（混合稳定 4W 起点与近 2W 起点）
            reverse_curriculum_frac = 0.90
            curriculum = False
            cur_step_down = 0.10
            cur_step_up = 0.05
            cur_rate_hi = 0.50
            cur_rate_lo = 0.25
            cur_window = 400
            # 采样：standup vs liedown 回合比例（本轮专训趴下）
            standup_ratio = 1.0
            # 起立完成(到 HOLD2W)后**继续保持**该时长(s)再结算 reset —— 让策略学"站住"(对称+静止)，
            #   而不是一到 HOLD2W 立刻 reset（那样 still/对称根本来不及学，也是"站不稳/不对称"的根因）。
            hold_after_upright = 1.0
            # 【对称起立】强制左右腿动作镜像：a_hip_r = −a_hip_l, a_knee_r = −a_knee_l。
            #   起立全程双腿结构上必然对称（治"一前一后"）；轮/平衡留给策略（W3 已证对称平衡可行）。
            force_symmetry = False
            # 起立密集塑形参数
            vel_tol = 0.5             # （保留）速度尺度
            pitch_sigma = 0.20        # sw_pitch_zero 的 σ（rad，温和）

        # 复用阈值（切换不跟速度/高度，仅占位）
        velocity_tracking_error_limit = 0.8
        yaw_tracking_error_limit = 0.5
        height_tracking_error_limit = 0.05
        roll_tracking_gain = 60.0
        tracking_sigma = 0.1

    class commands(WBRFlat4Cfg.commands):
        # 切换任务不发速度指令；指令通道保留占位（wbr4 的 resample 仍会填，但奖励不用）
        turning_command_probability = 0.0
        class ranges(WBRFlat4Cfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            base_height = [0.19, 0.19]


class WBRSwitchCfgPPO(WBRFlat4CfgPPO):
    class runner(WBRFlat4CfgPPO.runner):
        experiment_name = 'wbr_switch'
        run_name = 'P7S1b'
        max_iterations = 1300
        learning_rate = 3.0e-4
        init_noise_std = 0.25
