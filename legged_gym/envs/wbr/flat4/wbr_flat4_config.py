# ==============================================================================
# Date      : 2026-10-25
# Author    : SYH
# File      : wbr_flat4_config.py
# Purpose   : 四轮任务 wbr_flat4 的配置：观测 164 维（基础 33 + 历史 120 + 跳跃相位 8 +
#             从动轮接触 2 + 倾翻裕度 1），动作 6 维（2 轮 + 4 腿，从动轮被动）。
# ==============================================================================

from legged_gym.envs.wbr.flat.wbr_flat_config import WBRFlatCfg, WBRFlatCfgPPO


class WBRFlat4Cfg(WBRFlatCfg):
    class env(WBRFlatCfg.env):
        num_observations = 164           # 见文件头
        num_actions = 6                  # 2 轮 + 4 腿（从动轮被动）
        num_envs = 4096
        env_spacing = 3.0
        episode_length_s = 20
        enable_joint_state_history = True
        joint_state_history_length = 10
        caster_air_timeout_s = 1.0e9     # 从动轮离地终止关闭（1e9 s 永不触发）；转向时需要抬起后从动轮

    class terrain(WBRFlatCfg.terrain):
        mesh_type = "plane"  # P7 先平地；地形留 P8
        measure_heights = False
        track_test = False

    class init_state(WBRFlatCfg.init_state):
        # 站姿：躯干水平（pitch = 0），四轮着地，默认工作高 0.19 m。
        #   姿态解（对称双腿，四点全触 z=0，见 geo_check 精解）：
        #     hip=±56.34° (0.9833 rad)、knee=∓146.61° (−2.5589 rad)，base_z=0.1900 m
        #   关系：knee = −90° − hip（同时解膝+高度使轮与从动轮都触地）。
        pos = [0.0, 0.0, 0.19]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w；躯干水平（pitch=0）
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        # ★必须列出全部 8 个关节（基类 _init_buffers 逐 dof 名查此字典，缺键即崩）
        #   顺序与 URDF 一致：l_hip,l_knee,l_wheel,r_hip,r_knee,r_wheel,l_caster,r_caster
        default_joint_angles = {
            'left_hip_joint': 0.9833,           # +56.34°
            'left_knee_joint': -2.5589,         # −146.61°
            'left_wheel_joint': 0.0,
            'right_hip_joint': -0.9833,         # −56.34°（镜像）
            'right_knee_joint': 2.5589,         # +146.61°
            'right_wheel_joint': 0.0,
            'left_knee_caster_joint': 0.0,      # 被动，生成即触地
            'right_knee_caster_joint': 0.0,
        }

    class control(WBRFlatCfg.control):
        # 扩到 8 dof：从动轮被动关节 stiffness 0 / damping 0.02（基类按名子串匹配 'caster'）
        stiffness = {
            'hip_joint': 20.0,
            'knee_joint': 40.0,
            'wheel_joint': 0.4,
            'caster': 0.0,
        }
        damping = {
            'hip_joint': 1.20,
            'knee_joint': 1.20,
            'wheel_joint': 0.005001,
            'caster': 0.02,
        }
        leg_action_scale = 0.5
        wheel_action_scale = 3.0
        wheel_target_accel_limit = 0.0
        wheel_acceleration_filter_alpha = 0.8
        decimation = 2

    class asset(WBRFlatCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/wbr/urdf/wbr_v2.urdf"
        name = "wbr"
        foot_name = "wheel"                                 # 驱动轮＝"脚"（feet_indices = 2 轮）
        # ★去掉 shank：四轮姿态小腿盒离地仅 22.9 mm，留着会误触发终止
        penalize_contacts_on = ["base_link", "thigh"]
        terminate_after_contacts_on = ["base_link", "thigh"]
        flip_visual_attachments = False
        # 必须关自碰撞：pitch = 0 的深折姿态下 thigh 盒与 wheel 盘间隙仅约 10 mm
        #   （0.19m 处解析间隙 +9.7mm，落在 10mm contact_offset 内；0.17m 处真实重叠 ~1mm），
        #   开自碰撞会产生假接触力→假 base_term→每步终止。原 −48° 姿态无重叠对，故此前为 0 也无碍。
        self_collisions = 1

    class domain_rand(WBRFlatCfg.domain_rand):
        # 四轮域随机化，共 15 项：sim 侧 10 + 传感器侧 5。
        #   驱动轮摩擦/基座质量/水平推扰（基类）；从动轮质量/摩擦（wbr4）；
        #   pd/motor/com/damp/pitch + 陀螺/加计/重力 bias + obs/act 延迟（wbr4 内实现，未改 wbr.py）。
        enable_dr_sim = True                  # sim 侧总闸（启用从动轮质量/摩擦随机化）
        randomize_friction = True             # 驱动轮地摩擦随机化
        friction_range = [0.8, 1.2]           # 驱动轮地面摩擦区间
        randomize_base_mass = True            # 基座质量随机化
        added_mass_range = [-0.1, 0.1]        # 基座附加质量区间（kg）
        push_robots = True                    # 水平推扰
        max_push_vel_xy = 0.2                 # 推扰的最大速度（m/s）
        push_interval_s = 10.0                # 每 10s 推一次
        # 从动轮（wbr4 实现，enable_dr_sim=True 时按区间随机）
        caster_mass_range = [0.04, 0.07]      # 从动轮质量区间（kg）
        caster_friction_range = [0.02, 0.04]  # 从动轮摩擦区间（真实万向轮侧向 μ 上限 0.04）
        # ── wbr4 内实现的随机化（保守值；退化值 = 关）──
        nominal_env_fraction = 1.0            # 1.0=不做名义覆盖（全部env参与DR）
        pd_gain_range = [0.9, 1.1]            # 腿部 PD 增益乘子
        motor_strength_range = [0.9, 1.1]     # 电机力矩乘子
        com_offset_range = [-0.01, 0.01]      # 基座质心三轴偏移（m）
        caster_joint_damp_range = [0.015, 0.03]  # 从动轮关节阻尼区间
        push_pitch_vel_max = 0.2              # 推扰叠加的俯仰角速度脉冲
        imu_gyro_bias_range = [-0.03, 0.03]   # 陀螺零偏（rad/s）
        imu_acc_bias_range = [-0.1, 0.1]      # 加计零偏（m/s²）
        gravity_offset_range = [-0.02, 0.02]  # 重力方向偏移（rad）
        obs_delay_max_steps = 2               # 观测延迟上限（控制步）
        action_delay_max_steps = 2            # 动作延迟上限（控制步）

    class rewards(WBRFlatCfg.rewards):
        class flags:
            enable_jump = False        # ★关跳跃整条通道（相位机/跳跃奖励全惰性）
            enable_turning_tilt = False  # 四轮不压弯
            enable_leg_adjust = False

        class scales(WBRFlatCfg.rewards.scales):
            # ── 终止与安全 ──
            termination = -20.0
            collision = -10.0
            no_fly = -3.0              # 两驱动轮须都贴地，缺则软罚（同两轮）
            # ── 指令跟踪（四轮用水平前向速度，见 wbr_flat4.py::_reward_tracking_lin_vel）──
            tracking_lin_vel = 8.0     # 线速度跟踪权重（与两轮对齐）
            tracking_ang_vel = 6.0     # 角速度跟踪权重（与两轮对齐）
            tracking_wheel_lin_vel = 4.0  # 轮速跟踪权重（同两轮）：直接奖励轮速，避开"缩腿 + 低速"局部最优
            turning_wheel = 2.0        # 转向轮形塑：yaw 时奖励同向轮转 ωL + ωR ≈ 4.5·yaw
            tracking_base_height = 5.0  # 体高跟踪权重；顶端物理上限 0.246 m
            # ── 姿态与平顺 ──
            orientation = -5.0         # 姿态惩罚：pitch → 0、roll → 0（见 wbr_flat4.py 覆写；负号 = 惩罚）
            lin_vel_z = -4.0
            ang_vel_xy = -1.2          # 俯仰 / 滚转角速度惩罚，抑制抖动
            wrong_way = -2.0
            no_moonwalk = -20.0
            leg_action_rate = -0.20
            wheel_action_rate = -0.08
            # ── 关节与执行器 ──
            z_adjust_leg = 0.0
            hip_reach = 0.0
            dof_pos_limits = -1.0
            dof_vel_limits = -0.2
            torque_limits = -0.001
            torques = -0.002           # 能耗项（四轮卖点：省能）
            dof_acc = -1e-7
            # ── 四轮专属（wbr_flat4.py 新函数）──
            leg_pose_hold = 0.0        # 不锁死折叠姿态，体高由 tracking_base_height 驱动
            caster_contact = 1.0       # 从动轮着地奖励：2 个 = 100%、1 个 = 60%、0 个 = 0
            # ── 跳跃项全 0（enable_jump=False 已惰性，权重也清零省开销）──
            jump_lift = 0.0
            jump_landing = 0.0
            jump_top_height = 0.0
            jump_shape_track = 0.0
            jump_push_lead = 0.0
            jump_crouch_track = 0.0
            jump_leg_apex = 0.0

        # 复用基类阈值
        velocity_tracking_error_limit = 0.8  # 线速度 hinge 带宽；给倒退 −1.0 留余量
        yaw_tracking_error_limit = 0.5       # 偏航 hinge 带宽；cmd ±1.0 时误差约 0.1 < 0.5，梯度充足
        height_tracking_error_limit = 0.05   # 体高 hinge 带宽（收窄后顶端仍有梯度）
        roll_tracking_gain = 60.0
        tracking_sigma = 0.1

    class commands(WBRFlatCfg.commands):
        curriculum = False
        num_commands = 7
        resampling_time = 10
        heading_command = False
        smooth_commands = True
        turning_command_probability = 0.5  # 转向指令占比，vx / yaw 均衡采样
        yaw_command_deadzone = 0.1
        roll_command_probability = 0.0  # 四轮不发侧倾
        roll_command_deadzone = 0.02
        jump_command_probability = 0.0  # 四轮不发跳
        jump_stand_ratio = 0.0
        turning_roll_coeff = 0.0        # 不压弯

        class ranges:
            lin_vel_x = [-1.0, 1.0]     # vx 指令量程（m/s）
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-1.0, 1.0]   # 偏航指令量程（rad/s），实测可达约 0.95
            heading = [0, 0]
            jump_height = [0.05, 0.05]  # 闸门关死
            # 高度固定 0.19 m（区间退化）。
            #   半宽地板已在 wbr_flat4.py 观测归一化处理（0.05），退化不再爆炸。
            base_height = [0.175, 0.245]  # 体高指令量程（m）；可达范围 0.171~0.246，两端留余量
            ang_roll = [0.0, 0.0]

    class normalization(WBRFlatCfg.normalization):
        class obs_scales(WBRFlatCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            yaw_command = 1.0
            roll_command = 4.0
            dof_pos = 1.0
            dof_vel = 0.05
            joint_state_history = 0.25
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 5.


class WBRFlat4CfgPPO(WBRFlatCfgPPO):
    class runner(WBRFlatCfgPPO.runner):
        experiment_name = 'wbr_flat4'
        run_name = 'P7F3e'
        max_iterations = 500  # 转向 ±1 + 15 项随机化 + 延迟；交付权重 model_11800
