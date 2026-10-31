# P7F3e — 四轮模式：转向 ±1.0 + 15 项域随机化（交付基准）

> 对应版本 **V0.22**（P7 四轮线首个交付）。task = `wbr_flat4`，模型 `model_11800.pt`
> sha256 `788ff8bc1d0caebe1dff2f7028a2082518ce60153adc363582a41598ce189a1f`
> 训练 run：`logs/wbr_flat4/Oct25_11-35-01_P7F3e/`（warm-start 自 P7F3d@11300，+500 迭代 → 11800）

## 能力
- 四轮静稳构型（躯干水平 pitch=0，四轮着地）
- **vx ∈ [−1, 1] m/s**（机体系前向速度跟踪）
- **yaw ∈ [−1, 1] rad/s**（转向，本轮从 ±0.5 扩到 ±1.0）
- **base_height ∈ [0.175, 0.245] m**（动态可达 0.171~0.246；0.27 端几何不可达）

## 域随机化（15 项，F3e 交付态）
### 模拟器侧（10）
| 项 | 范围 | 实现 |
|---|---|---|
| 驱动轮地摩擦 `friction_range` | μ [0.8, 1.2] | 基类 |
| 基座质量 `added_mass_range` | ±0.1 kg | 基类 |
| 水平推扰 `max_push_vel_xy` | 0.2 m/s @ 每 10 s | 基类 |
| 从动轮质量 `caster_mass_range` | [0.04, 0.07] kg | wbr4 |
| 从动轮侧向摩擦 `caster_friction_range` | μ [0.02, 0.04] | wbr4 |
| 腿部 PD 增益乘子 `pd_gain_range` | [0.9, 1.1] | wbr4（新实现）|
| 电机力矩乘子 `motor_strength_range` | [0.9, 1.1] | wbr4（新实现）|
| 基座质心偏移 `com_offset_range` | ±0.01 m（3 轴） | wbr4（新实现）|
| 从动轮关节阻尼 `caster_joint_damp_range` | [0.015, 0.03] | wbr4（新实现）|
| 推扰俯仰脉冲 `push_pitch_vel_max` | ±0.2 rad/s | wbr4（新实现）|

### 传感器侧（5）
| 项 | 范围 | 实现 |
|---|---|---|
| 陀螺零偏 `imu_gyro_bias_range` | ±0.03 rad/s | wbr4（新实现）|
| 加计零偏 `imu_acc_bias_range` | ±0.1 m/s² | wbr4（新实现）|
| 重力方向偏移 `gravity_offset_range` | ±0.02 rad | wbr4（新实现）|
| 观测延迟 `obs_delay_max_steps` | 0\~2 步 | wbr4（新实现）|
| 动作延迟 `action_delay_max_steps` | 0\~2 步 | wbr4（新实现）|

> 观测白噪声常开：`add_noise=True, noise_level=1.0`（dof_pos 0.01 / dof_vel 1.5 / lin_vel 0.1 / ang_vel 0.2 / gravity 0.05）。
> 全部 15 项均在 `wbr_flat4.py`/基类实现，**未改 `wbr.py`**（合并兼容）。

## 实测指标（body 系；yaw=base_ang_vel_z）
| 工况 | 名义 | 训练DR开 |
|---|---|---|
| vx +1 / −1 | 0.95 / −0.98 | 0.96 / −0.98 |
| vx +0.5 / −0.5 | 0.50 / −0.50 | 0.51 / −0.50 |
| yaw +1 / −1 | 0.93 / 0.91 | 0.95 / 0.91 |
| (0.5, 0.75) | 0.50 / −0.72 | 0.49 / −0.73 |
| 高度 0.175 / 0.245 | +0.5 / −2.7 mm | +0.4 / −2.6 mm |
| 推扰+pitch 恢复 | — | 3/3，0 摔 |

满 DR 压测（最严，抽最坏组合）：vx+1→0.93(摔1)、vx−1→−0.98、yaw±1→0.74/0.81、高度 ±2.5mm、0 名义退化。四轮全程着地。

## play
```bash
python legged_gym/scripts/play02.py --task wbr_flat4 --load_run Oct25_11-35-01_P7F3e --checkpoint 11800
```

## 对照（未纳入交付，留档）
- **P7F3f**：DR 放开到 wbr 满建议值（pd/motor[0.8,1.2]/com±0.02/陀螺±0.05…）→ 满 DR 下极值掉（vx+1→0.79），最保守。
- **P7F3g**：DR 折中值 → 满 DR 压测被 F3e 支配（vx+1 0.87 且摔8）。二者保留在 `logs/` 供对照。
