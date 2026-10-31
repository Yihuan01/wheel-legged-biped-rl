# ==============================================================================
# Date      : 2026-04-08
# Author    : SYH
# File      : wbr_flat_config.py
# Purpose   : 两轮平地任务 wbr_flat 的训练配置：观测 161 维、动作 6 维、地形 plane、
#             奖励权重与指令范围。其余 WBR 任务的配置都从这里继承。
# ==============================================================================

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class WBRFlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 161
        num_actions = 6
        env_spacing = 3.0
        episode_length_s = 20
        enable_joint_state_history = True
        joint_state_history_length = 10

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        track_test = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        # ===== 随机化接口：2 个类别总闸 + 逐项数值；退化值 = 关闭，非退化即生效 =====
        # 训练侧默认全关（W 线四连败，见 WBR_行走随机化.md §2.5/§6.5）；名义闸永远全关，鲁棒性用 --robust 探针另测。
        enable_dr_sim = False                 # 模拟器侧总闸 关闭=False 建议训练侧False
        # --- 继承：基类已接线，保留各自布尔 ---
        randomize_friction = False            # 轮地摩擦开关 关闭=False 建议[0.8,1.2]
        friction_range = [1.0, 1.0]           # 摩擦区间 关闭[1,1] 建议[0.8,1.2]
        randomize_base_mass = False           # 基座质量开关 关闭=False 建议±0.1kg
        added_mass_range = [0.0, 0.0]         # 质量增量kg 关闭[0,0] 建议[-0.1,0.1]⚠勿用基类[-1,1]
        push_robots = False                   # 水平推扰开关 关闭=False 建议0.2m/s@10s
        max_push_vel_xy = 0.0                 # 推扰速度m/s 关闭0 建议0.2
        push_interval_s = 10.0                # 推扰间隔s 关闭随推扰 建议10
        # --- 新建：待补实现，键先占位、退化值即关闭 ---
        com_offset_range = [0.0, 0.0]         # 基座质心三轴偏移m 关闭[0,0] 建议±0.02
        pd_gain_range = [1.0, 1.0]            # 腿部PD增益乘子 关闭[1,1] 建议[0.8,1.2]
        motor_strength_range = [1.0, 1.0]     # 电机力矩乘子 关闭[1,1] 建议[0.8,1.2]
        push_pitch_vel_max = 0.0              # 推扰俯仰角速度rad/s 关闭0 建议0.2~0.5
        nominal_env_fraction = 1.0            # 名义环境占比 关闭1(全名义) 建议0.5~0.8
        caster_mass_range = [0.05, 0.05]      # 从动轮质量kg 关闭[0.05,0.05] 建议[0.04,0.07]
        caster_friction_range = [0.10, 0.10]  # 从动轮摩擦μ 关闭[0.10,0.10] 建议[0.05,0.15]
        caster_joint_damp_range = [0.02, 0.02]  # 从动轮关节阻尼 关闭[0.02,0.02] 建议[0.01,0.05]

    class noise(LeggedRobotCfg.noise):
        # --- 继承：基础观测白噪声（W10~12 已证可训，交付态常开）---
        add_noise = True                  # 观测白噪声开关 关闭=False 建议=True
        noise_level = 1.0                 # 噪声档位 关闭0 建议1.0(真机量级)
        # ===== 传感器侧总闸（管恒定不可观测差距，先小探针再开）=====
        enable_dr_sensor = False          # 传感器侧总闸 关闭=False 建议=False
        # --- 新建：待补实现 ---
        imu_gyro_bias_range = [0.0, 0.0]  # 陀螺零偏rad/s 关闭[0,0] 建议±0.05
        imu_acc_bias_range = [0.0, 0.0]   # 加计零偏m/s² 关闭[0,0] 建议±0.2
        gravity_offset_range = [0.0, 0.0]  # 重力方向偏移rad 关闭[0,0] 建议±0.05
        obs_delay_max_steps = 0           # 观测延迟步 关闭0 建议0~2
        action_delay_max_steps = 0        # 动作延迟步 关闭0 建议0~2

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.28]  # 默认姿态的轮地接触高度（几何值约 0.284，取 0.28 留余量）
        default_joint_angles = {
            'left_hip_joint': 1.0,      # 初始弯曲姿态（能训练成功的配置）
            'left_knee_joint': -2.1,
            'left_wheel_joint': 0.0,
            'right_hip_joint': -1.0,     # 镜像对称的初始姿态
            'right_knee_joint': 2.1,
            'right_wheel_joint': 0.0
        }

    class control(LeggedRobotCfg.control):
        # 关键设置：为腿部提供足够的刚度，为轮子提供速度控制感
        stiffness = {
            'hip_joint': 20.0,
            'knee_joint': 40.0,
            'wheel_joint': 0.4       # 轮子是V，其他两个是P
        }
        damping = {
            'hip_joint': 1.20,
            'knee_joint': 1.20,
            'wheel_joint': 0.005001
        }
        leg_action_scale = 0.5          # rad
        wheel_action_scale = 3.0        # rad/s，为 ±1m/s 线速度和同时转向保留轮速余量
        wheel_target_accel_limit = 0.0  # 关闭额外限速状态，直接执行策略给出的轮速目标
        wheel_acceleration_filter_alpha = 0.8  # D 项加速度低通，越大越平滑
        decimation = 2                  # 提高控制频率到10ms（从20ms提升），平衡控制需要快速响应

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/wbr/urdf/wbr_v1.urdf"
        name = "wbr"
        foot_name = "wheel"                                 # URDF中的末端link关键词
        # 触地惩罚，防止"跪地"
        penalize_contacts_on = ["base_link", "thigh", "shank"]
        terminate_after_contacts_on = ["base_link", "thigh", "shank"]
        flip_visual_attachments = False
        self_collisions = 0

    class rewards(LeggedRobotCfg.rewards):
        class flags:
            enable_turning_tilt = True  # 转向压弯：侧倾目标＝v·ω/g 前馈（启用 `turning_roll_coeff`）
            enable_jump = True        # 跳跃整条通道的总开关
            enable_leg_adjust = True  # 膝角跟踪奖励参与总奖励

        class scales(LeggedRobotCfg.rewards.scales):
            # ── 终止与安全 ────────────────────────────────────────────
            termination = -20.0         # 非超时终止（摔倒）
            collision = -10.0           # base_link / thigh / shank 碰撞
            no_fly = -3.0               # 双轮必须贴地
            # ── 指令跟踪（正奖励，同一个币种：满分 = 权重 × dt）────────
            tracking_lin_vel = 8.0      # vx 跟踪（hinge，带宽见 velocity_tracking_error_limit）
            tracking_ang_vel = 6.0      # 偏航跟踪（hinge）
            tracking_wheel_lin_vel = 4.0  # 轮速等效线速度直接跟 vx 指令
            tracking_base_height = 4.0  # 体高跟踪
            # ── 姿态与平顺 ───────────────────────────────────────────
            orientation = -6.0          # 机身非水平（重力 xy 分量）
            lin_vel_z = -4.0            # 机身 z 向跳动
            ang_vel_xy = -0.80          # 高频俯仰/滚转抖动
            wrong_way = -2.0            # 实速与 vx 指令反向
            no_moonwalk = -20.0         # 两轮前后错位惩罚（机体系 x，容差 25 mm）
            leg_action_rate = -0.20     # 腿部动作变化率
            wheel_action_rate = -0.08   # 轮速目标变化率（已乘 `wheel_action_scale`，单位 rad/s）
            # ── 关节与执行器 ──
            z_adjust_leg = -10.0        # 跟踪解算器给出的膝角目标（量级由 reward_term_probe 定）
            hip_reach = -10.0           # 髋可达单边惩罚：只罚"髋前伸到撑不起当前高度目标"
            dof_pos_limits = -1.0       # 接近关节位置限位
            dof_vel_limits = -0.2       # 接近关节速度限位
            torque_limits = -0.001      # 接近力矩限位
            torques = -0.0001           # 力矩绝对量
            dof_acc = -1e-7             # 关节加速度
            # ── 已停用（保留键名与原因；权重 0 时上游整项丢弃）────────
            base_height = 0.0           # 【已停用：世界系 body 高度平方惩罚】现改用 `tracking_base_height` 的腿伸量 hinge
            dof_vel = -0.0              # 与 dof_acc 重叠
            action_rate = 0.0           # 已拆分为 leg_/wheel_action_rate
            feet_air_time = -0.0        # 【已停用：轮足无迈步】跳跃阶段暂不启用
            feet_stumble = -0.0         # 【已停用：轮子撞垂直面】地形阶段再评估
            stand_still = -0.0          # 零指令静止惩罚与"敢动"冲突
            feet_contact_forces = -0.0  # 接触力过大，实机才需要
            jump_lift = 500.0           # 起飞脉冲奖（单边：够到目标即满分、超了不罚）
            jump_landing = 1200.0       # 落地结算权重：触地那一拍一次结清，满值 = 权重×Δt 再乘本拍乘子
            jump_top_height = 1250.0    # 最高位置奖：顶点一次性结算，量 = 轮底离地峰值，目标 = h_target，σ = 0.04
            jump_shape_track = 12.0     # 空中 / 够地 / 缓冲三段的腿伸量跟踪（到位就给分，不约束形状）
            jump_push_lead = 40.0       # 爆发段引导项权重。⚠不要一次抬到 145/314（追平预蹲需 314），会冲掉相位2已豁免项的平衡
            jump_crouch_track = 15.0    # 预蹲跟踪权重：相位 1 内**逐拍发放**；每拍 = 权重×Δt×
            jump_leg_apex = 500.0       # 顶点腿长奖：量 = 顶点前 leg_apex_win 拍空中腿长（两腿取 max），目标 = 模板 ref_L
        tracking_sigma = 0.1                 # 横向速度分量仍用 exp
        velocity_tracking_error_limit = 0.85  # 线速度 hinge 带宽：|指令 − 实速| 的线性衰减宽度（大 = 梯度平）
        yaw_tracking_error_limit = 0.8       # 偏航 hinge 带宽；必须 ≥ ang_vel_yaw 上限
        height_tracking_error_limit = 0.08   # m 体高 hinge 带宽；带宽过宽会让策略向中间塌
        roll_tracking_gain = 60.0            # 侧倾指令跟踪增益；±0.2 rad 是物理上限
        wrong_way_command_threshold = 0.1    # 指令小于此值不算有运动意图
        wrong_way_velocity_threshold = 0.03  # 实速小于此值不算真的在动

        # ========== 解算器几何参数（`wbr.py::_ik_knee_for_leg` 直用）==========
        thigh_length = 0.2                   # 大腿长 (m)：hip → knee，沿 thigh 局部 x
        shank_length = 0.18                  # 小腿长 (m)：knee → wheel，沿 shank 局部 x
        wheel_radius = 0.08                  # 轮半径 (m)
        roll_lever_eff = 0.407               # m 侧倾有效杠杆（≈2.17 × 轨距）
        roll_gain_denom_floor = 0.12         # m/rad，Σ|∂H/∂knee| 下限 ⇒ 增益上界 0.407/0.12 ≈ 3.4（腿近伸直/折叠时防尖峰）
        K_roll_fb = 0.0                      # roll 误差 → 膝角差动的反馈增益（0 = 只用前馈）
        # no_moonwalk 的口径：罚"两轮前后错位"（机体系 x），下面三个键是它的参数。
        moonwalk_e_tol = 0.025               # m，两轮前后错位容差：平地 p99 6.8mm、台阶 p95 27mm ⇒ 25mm 不误伤爬坎
        moonwalk_ed_tol = 0.30               # m/s，前后错位变化率容差（防快速蹭步）
        moonwalk_rate_weight = 0.05          # 变化率项权重（无量纲；主项权重仍是 no_moonwalk=-20）
        hip_reach_margin = 0.02              # `hip_reach` 的容差带（rad）
        hip_reach_speed_gate = 0.0           # m/s，0 = 全程罚髋；>0 只在该速度以下罚

        class jump:
            """P6 跳跃相位模板的时长与幅度（全部由 `wbr.py::_update_jump_phase` 读）。
            幅度一律用"沿机体轴的腿伸量变化 ΔL (m)"表述；时长是**保险丝不是节拍**，每个
            相位的出口先挂事件（离地 / vz≤0 / 触地），事件不来才按时长强退。"""

            t_crouch = 0.75                             # s 预蹲保险丝时长（出口由 crouch_settle_steps / crouch_vz_tol 决定）
            t_push_max = 0.2                            # s  爆发伸腿最长时长；真离地确认后会提前转腾空
            crouch_reach_tol = 0.006                    # m 预蹲到位的双向容差（机身高度 vs 绝对目标 ± 此值）。0.005→0.006：实测静差补掉后残差 ±4mm 内
            takeoff_vz_min = 0.50                       # m/s 确认起飞所需的最低上升速度
            confirm_steps = 2                           # 离地/落地各需连续几个控制步
            vz_landing_soft = 0.90                      # m/s 落地冲击速度：|vz| ≤ 此值给满分，线性衰减到 0
            off_rate_max = 0.25                         # rad/控制步 模板偏移的最大变化率（爆发相位除外）
            burst_template = True                       # 爆发段用模板腿长目标（关掉＝静默让策略自长）
            hip_rate_max = 0.25                         # rad/控制拍 髋目标偏移限速（与膝同档；空中设定点切换要爬坡）
            chain_leak = 0.05                           # 无量纲 前序未达标时，后续阶段**正分**保留的上限比例（负分一律不抹）
            shape_tol = 0.02                            # m  形状/收腿/够地的达标带（腿竖直伸量误差 ≤ 此值算达标）
            # jump_shape_track 的全局权重只作用于相位 3；相位 4/5 用下面这个独立权重。
            jump_shape_track_ph45 = 0.6                 # 相位 4/5 腿伸跟踪的绝对权重（门为 g1·g2，不含 g3）
            leg_apex_sigma = 0.02                       # m 顶点腿长高斯的 σ（偏差 σ 处得 0.607）；0 = 关闭本项
            top_height_sigma = 0.04                     # m 最高位置奖的 σ（轮底离地口径）；0 = 关闭本项
            top_height_goal_by_h = (0.05, 0.15, 0.070, 0.130)  # (h_lo,h_hi,v_lo,v_hi) 最高位置奖的轮底目标随指令插值；None = 用 h_target
            leg_apex_win = 3                            # 拍 顶点腿长滑动窗长度（3 拍 = 30 ms）；只记相位 3 的帧，整窗都没过长才满分
            land_leg_tol = 0.060                        # m 落地结算的腿伸比值容差：触地那拍误差 ≤ 此值给满，线性衰减到 0
            pulse_sigma_frac = 0.0                      # 离地脉冲的高斯 σ：max(frac×指令, abs)；frac=0 ⇒ σ 恒为固定 65 mm
            pulse_sigma_abs = 0.065                     # m 高斯 σ 的下限（`frac=0` ⇒ σ 恒为此值）
            launch_ok_frac = 0.30                       # 无量纲 起跳达标 = 离地后上升预测 ≥ 此比例 × 指令（g2 门）
            t_flight_min = 0.03                         # s 腾空停留下限（防刚离地就误判顶点）
            t_flight_max = 0.35                         # s 腾空上限（没离地的失败跳也从这里回待机）
            t_land_max = 0.25                           # s 够地段的兜底保险丝（走保险丝不发落地结算的钱）
            t_absorb = 0.20                             # s 缓冲时长，偏移在此期间线性回零
            stroke_gain = 1.2                           # m 附加推程 / m 指令（封顶到超程上限）
            push_to_limit = True                        # 爆发段目标是否钉到该髋角的最大伸量（不钉＝只按行程到位）
            push_overshoot = 0.067                      # m 爆发段超程量：目标腿长留屈膝缓冲、不伸到全直
            burst_L_max = 0.30                          # m 爆发段目标腿长上限；0/None 不限制
            burst_L_by_h = (0.05, 0.15, 0.29, 0.38)     # (h_lo,h_hi,v_lo,v_hi) 爆发目标腿长随起跳指令插值；None = 用 burst_L_max
            burst_retract_L = 0.25                      # m 爆发段内收腿触发腿长：实际腿长 ≥ 此值后目标换成 burst_retract_to（不切相位）
            burst_retract_to = 0.18                     # m 相位2 内收腿到此值后目标腿长换成它（不切相位）。0/None＝关闭
            air_hold_joint = 0                          # 1 = 相位 3 把腿长钉在起飞瞬间的腿长（零刹车）；0 = 用 air_hold_L 的绝对口径
            air_hold_follow = 0.0                       # m/m 相位 3 腿长 = 起飞冻结腿长 + 本值 × 机身升高；0 = 退回 air_hold_L
            air_hold_L = 0.20                           # m 相位 3（腾空）绝对腿长目标，同时是 jump_leg_apex 的奖励目标
            land_hold_L = 0.20                          # m 相位 4（够地）绝对腿长目标，与相位 3 同取以做到零行程
            tuck_depth = 0.05                           # m 相位 3 空中收腿＝腿竖直缩短此量（常数设定点，无曲线）
            air_hold_leg = 0.0                          # 相位 3（腾空）维持离地那一拍的实测腿型；0 = 退回旧的收腿口径
            land_hold_leg = 0.0                         # 相位 4（够地）沿用相位 3 的设定点（腾空→够地零行程）；0 = 退回 `land_extend`
            absorb_abs_height = 0.22                    # 相位 5（缓冲）的腿伸目标绝对机身高度（与预蹲同高）；0 = 回到触发拍的站立基准
            absorb_ease_p = 2.0                         # 相位 5 腿长 ease-out 指数（2 = 起步快、末端速度为 0）
            land_extend = 0.06                          # m 相位 4 够地＝伸到"站立基准+此量"（保留微屈膝吸能）；`land_hold_leg>0` 时不用它
            off_limit = 1.2                             # rad 膝偏移硬上限；几何反解到腿链极限时自己会饱和，这是二重保险
            upright_limit = 0.35                        # rad 触发门的姿态上限（不许在歪着的状态下起跳）
            upright_limit_reward = 0.6                  # 奖励侧姿态门（比触发门宽；摔倒另有 termination 兜底）
            goal_floor = 0.005                          # m 奖励分母下限（指令接近 0 时不放大成无穷奖励）
            knee_damp_extra = [1.2, 0.0, 1.2, 1.2, 1.2]  # 5 维数组，index0..4 = 相位 1..5 的膝阻尼增量（叠加基态 1.2）
            knee_kp_extra = [0.0, 12.0, 0.0, 0.0, 0.0]  # 5 维数组，index0..4 = 相位 1..5 的膝 P 增益增量（叠加基态 40）
            crouch_ramp_end = 0.60                      # 预蹲爬升在 t_crouch×此比例处到位（0.60×0.75＝0.45s ramp）；余下拍数用来等停稳
            crouch_sag_ff = 0.0                         # m 【P6R78】PD 静差**前馈**（可选，默认 0）：指令目标抬此值、机身落点才等于期望。
                                 #   当前用 crouch_z_kp 闭环，故置 0；两者不可同开——ff 会让 z_cmd 高于 crouch_abs_by_h ⇒ 到位判据永远差 ff。
                                 #   实测 ff=0.014 只买到 +4mm（腿伸直后同样的力矩误差对应更大高度差）⇒ 开环修不了静差
            crouch_z_kp = 6.0                           # 【P6R78】机身高度伺服增益（腿长/机身高度，无量纲）：dh += k·(z_cmd−z)。
                                #   为什么必须闭环：dh 是**腿长**变化，机身高度还受**俯仰**影响——实测蹲到 0.45s 机体已后仰 22°、
                                #   base 比腿长等比低约 25mm ⇒ 纯开环一路沉到 0.176，左大腿触地（70 N）终止
            crouch_settle_steps = 5                     # 到位且停稳需连续 N 拍才交棒相位 2（N=5 即 50 ms）
            crouch_vz_tol = 0.02                        # m/s 停稳判据：机身竖直速度 |vz| ≤ 此值算停住
            crouch_hold_time = 0.10                     # s 【P6R78】预蹲命令（ramp 结束＝t_crouch×ramp_end）后至少再保持这么久，才允许交棒
            crouch_deep_tol = 0.010                     # m 【P6R78】过深救助：已比目标低超过此值 ⇒ 不再等停稳、立刻蹬（继续等只会更深）
            crouch_abs_height = 0.22                    # 预蹲的**绝对目标机身高度**（m）。无"=0 回退旧相对口径"的分支——旧键 `crouch_depth` 已删
            crouch_abs_by_h = (0.05, 0.15, 0.22, 0.19)  # (h_lo,h_hi,v_lo,v_hi) 预蹲机身目标随起跳指令插值；None = 用 crouch_abs_height
            crouch_hold_tol = 0.0                       # m 软停止带半宽：>0 时带 = [目标 − tol, 目标 + tol]；0 = 用固定 min / max
            crouch_hold_min = 0.0                       # m 预蹲软停止带下沿；与 crouch_hold_max 一并置 0 即关闭
            crouch_hold_max = 0.225                     # m 预蹲软停止带上沿；0 或 min≥max ⇒ 关闭软停止
            push_sigma_frac = 0.0                       # 离地脉冲高斯 σ 的指令比例项；0 = 用单调锚 clamp(vz/v_req, 0, 1)
            crouch_depth_cap = 0.14                     # m 绝对蹲高模式下的最大下降量帽子
            crouch_sigma_abs = 0.008                    # m 预蹲跟踪给分的高斯 σ（固定值，与误差同量纲）
            crouch_sigma_frac = 0.0                     # 预蹲跟踪高斯 σ 的指令比例项：σ = max(此值×要求, abs)；0 ⇒ 恒取 abs
            burst_rate_max = 0.3                        # rad/拍 爆发段模板偏移限速（0.3 = 30 rad/s，仍低于 URDF 的 30）
            anchor_x_com = None                         # 跳跃期轮子相对髋的水平锚点（m）；None = 用触发帧冻结值
            anchor_curve = [-4.083, 2.560, -0.564, 0.070]  # 平衡曲线 x_ref = f(L_star) 的三次系数；None = 用 anchor_x_com
            anchor_tilt_deg = 5.0                       # 度 锚点整体绕轮心前转角度：把锚向量(x_ref,L*)旋转 θ，髋相对轮前移≈L*·sinθ，用来抵消蹬地后仰。0＝不转
            hip_pitch_gain = 0.0                        # 跳跃期的俯仰 → 髋目标反馈（rad/rad）；0 = 关
            hip_comp_gain = 0.0                         # 已停用（髋目标已由 _jump_2r_solve 一并给出）
            action_authority = 1.0                      # 模板对物理膝目标的授权 0~1：0 = 只进奖励目标、全靠策略；1 = 模板驱动腿
            authority_mode = "override"                 # 预蹲相位改**物理强制**（膝＋髋一起覆盖）
                                                        #   ⇒ 加性授权下"把模板抵回去"仍是理性解，而抵回去 = g1=0 = 全链条锁死。
            override_phases = (1,)                      # 被物理强制的相位号：只强制相位1预蹲（蹲姿必须精确）；相位2走 additive（策略动作+模板偏移叠加）
            air_override_hip = (3,)                     # 只强制髋（膝仍 additive）的相位号，用于钉住腾空髋、消后仰
            rearm_max = 0                               # 0＝关闭连跳（下方 if rearm_max > 0 整块跳过）。连跳样本与"站定后起跳"不同分布，且会绕过装填后的稳定期
            rearm_gap = 0.5
            arm_delay = 2.0                             # s 装填后经过此时长才允许触发（见 trig 里的 arm_at）

            settle_gate = False                         # 每条指令的静定门：等静定再触发；False = 触发门与加门前逐位等价
            cmd_settle_gate = True                      # 指令侧站定门：没站住就不**发**跳跃指令（与触发侧 `settle_gate` 独立）
            settle_steps = 30                           # 静定门需连续成立的拍数（0.3 s）
            settle_vz_max = 0.10                        # m/s 静定门要求的 |v_z| 上限
            settle_vxy_max = 0.10                       # m/s 静定要求的实测水平速度上限（世界系 x/y）
            settle_wxy_max = 0.50                       # rad/s 静定要求的实测姿态角速度上限（世界系 roll/pitch 两轴）
            settle_dz_max = 0.02                        # m 静定门要求的离站立参考高度上限
            settle_ref_tau = 0.03                       # 站立参考高的滑窗系数/拍（≈33 拍时间常数；贴着机身跟 ⇒ 位置带失效）

            class mask:
                """项名 → 该项被关闭的相位号列表。

                相位号：0 待机 1 预蹲 2 爆发 3 腾空 4 够地 5 缓冲（与 ph 一致）。
                不进表的项＝全程生效；爆发段放宽动作率与力矩类，预蹲 / 缓冲段仍照罚 no_fly。
                """
                tracking_base_height = [1, 2, 3, 4, 5]      # 掩蔽"蹲/推/腾/够地"这四段——模板故意让体高目标跑在实测前面
                hip_reach = [1, 2, 3, 4, 5]                 # 掩蔽"腾空/够地"这两拍——策略压不住机身角速度；缓冲相照罚
                lin_vel_z = [1, 2, 3, 4, 5]                 # 竖直速度是本任务的执行量，不是误差
                no_fly = [2, 3, 4]                          # 只有离地三段允许双轮不贴地
                leg_action_rate = [2]                       # 允许爆发推蹬
                torques = [2]
                dof_acc = [2]
                ang_vel_xy = [3, 4]
                no_moonwalk = [2, 3, 4]                     # 爆发 / 腾空 / 够地段豁免（蹬地时左右不对称是必然的）
                wheel_action_rate = [3, 4]
                tracking_lin_vel = [2, 3, 4]                # 掩蔽腾空/够地；缓冲相恢复生效（落地要真的把水平速度带回来）
                tracking_ang_vel = [2, 3, 4]                # 掩蔽"蹲/蹬把机体前后甩"的那几拍
                tracking_wheel_lin_vel = [1, 2, 3, 4, 5]    # 全掩蔽（掩蔽相位 1 会让预蹲段被要求跟踪轮速 ⇒ 腿必须动）
                z_adjust_leg = [1, 2, 3, 4, 5]
        base_height_speed_gain = 0.0         # 已停用（按 |vx| 给高度惩罚加权）

    class commands(LeggedRobotCfg.commands):
        curriculum = False              # 初步站立先不开启课程学习
        max_curriculum = 2.0
        num_commands = 7                # vx / vy / yaw / heading / jump 高度 / 体高 / roll
        resampling_time = 10
        heading_command = False
        threshold = 0.5                 # 控制技能学习采样频率
        smooth_commands = True          # 训练中使用平滑后的 commands，避免每次重采样产生指令阶跃
        turning_command_probability = 0.5  # 抽多少比例的环境练转向（其余练直行）
        yaw_command_deadzone = 0.1      # 偏航指令死区：|ω| 小于此值视作零
        roll_command_probability = 0.5  # 抽多少比例的环境练侧倾（其余练纯直行）
        roll_command_deadzone = 0.02    # roll 指令死区：|roll| 小于此值视作零 roll

        jump_command_probability = 0.0  # 训练分布里的跳跃指令比例（0 = 纯行走训练）
        jump_stand_ratio = 0.0          # 跳跃回合占比（0 = 全是行走回合，走 base_height 量程）
        jump_stand_height = [0.29, 0.29]  # 跳跃回合的体高指令：固定 0.29 是为补偿热启动模型系统性欠高约 40 mm
        jump_vx_gate = 0.10             # m/s |v 指令| ≤ 此值才允许发跳跃指令
        jump_yaw_gate = 0.10            # rad/s |ω 指令| ≤ 此值才允许发
        jump_roll_gate = 0.10           # rad |roll 指令| ≤ 此值才允许发；三条 gate＝速度/角速度/roll 绝对值都 0.10
        jump_cmd_h_min = 0.25           # m 体高指令带下沿：低于此值不发跳跃指令（冗余保险，jump_stand_height 已在带内）
        jump_cmd_h_max = 0.30           # m 体高指令带上沿；与上面三条 gate 合成指令侧站定门
        turning_roll_coeff = -0.102     # 压弯前馈系数（tanφ = v·ω/g ⇒ −1/g）；仅 `enable_turning_tilt` 时生效

        class envelope:
            """目标可达包络：施加在"指令 → 目标"这一步，见 wbr.py::_apply_command_envelope。"""
            yaw_taper_b = 2.14                     # 偏航包络 b：|ω 目标| ≤ b − k·|v|
            yaw_taper_k = 1.34                     # 偏航包络 k：(rad/s)/(m/s)
            expand = 1.0
            height_target_use_reach_clamp = False  # 把高度目标夹到当前髋角的几何可达上限；True = 收缩需求
        class rate_limits:
            lin_vel_x = 0.8     # vx 指令量程（m/s）；上限必须 ≤ `velocity_tracking_error_limit`
            lin_vel_y = 0.8     # vy 指令量程（m/s）
            ang_vel_yaw = 1.5   # rad/s^2
            heading = 100.0     # 航向指令量程（rad）
            jump_height = 100.0  # m/s，跳跃指令保持近似瞬时
            base_height = 0.15  # m/s
            ang_roll = 0.25     # roll 指令量程（rad）；跳跃回合不发侧倾

        class ranges:
            # =====================================================================
            lin_vel_x = [-1.1, 1.1]     # vx 指令量程（m/s）；上限必须 ≤ `velocity_tracking_error_limit`
            lin_vel_y = [0.0, 0.0]      # vy 指令量程（m/s）
            ang_vel_yaw = [-0.8, 0.8]   # 偏航指令量程（与 `yaw_tracking_error_limit` 配套）
            heading = [0, 0]            # 航向指令量程（rad）
            jump_height = [0.05, 0.15]  # 跳跃指令量程（m）= 离地后质心上升；验收看轮底离地
            base_height = [0.20, 0.35]  # 体高指令量程（m）；跳跃回合的高度另由 `jump_stand_height` 给
            ang_roll = [-0.20, 0.20]    # roll 指令量程（rad）；跳跃回合不发侧倾

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            yaw_command = 1.0          # 仅放大偏航指令，不改变机体角速度反馈与平衡观测
            roll_command = 4.0         # 将 ±0.10rad 指令映射到可辨识的 ±0.40 观测范围
            dof_pos = 1.0
            dof_vel = 0.05
            joint_state_history = 0.25
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 5.

class WBRFlatCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 0.25                  # 初始动作噪声标准差
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"                     # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'gru'
        # rnn_hidden_size = 128
        # rnn_num_layers = 1
        """
        在eth20年的sr中提及用本体感知历史信息可以重建特权信息，提升运动表现
        但是本设计并没有采用特权学习，原因是：
        对于两轮足并不像点足机器人需要通过隐式的方式获取局部地形以进行立足点的规划
        实际测试发现相同设置下添加GRU不如直接MLP
        """

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0       # 熵奖励系数（0＝不加熵 bonus）
        num_learning_epochs = 5
        num_mini_batches = 4     # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 3.0e-4   # 固定学习率（`schedule="fixed"`，不随迭代衰减）
        schedule = "fixed"       # 固定学习率（不用自适应调度，避免干扰探索噪声）
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.005       # KL 目标：限制单次更新的步长
        max_grad_norm = 0.5      # 梯度裁剪阈值

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 1200

        save_interval = 50
        experiment_name = 'wbr_flat'
        run_name = 'wbr_standing'

        resume = False

        # 自动选择最新运行及其最后一个 checkpoint：需要时取消下面两行注释，
        # 并注释掉后面的手动选择。注意 get_load_path 当前按目录名字典序选择，
        # 跨月份时应检查终端打印的 Loading model from 路径是否正确。
        # load_run = -1
        # checkpoint = -1

        # 手动选择模型：测试指定运行时保留下面两行有效。
        # 新训练仍由 resume=False 从头开始，不会加载这里指定的模型。
        load_run = 'Jul22_12-49-48_wbr_standing'
        checkpoint = 1200
        resume_path = None
