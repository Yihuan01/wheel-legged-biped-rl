# ==============================================================================
# Date      : 2026-10-29
# Author    : SYH
# File      : wbr_terrain_config.py
# Purpose   : 同一策略的四个地形任务配置，只替换 terrain 段，观测仍 161 维（不含高程）：
# wbr_terrain 随机起伏 / wbr_demo 演示走廊 / wbr_combo 组合地形 / wbr_elements 单元素验收。
# ==============================================================================

import os

from legged_gym import LEGGED_GYM_ENVS_DIR
from legged_gym.envs.wbr.flat.wbr_flat_config import WBRFlatCfg, WBRFlatCfgPPO

# 地形文件默认目录：legged_gym/envs/wbr/terrain/release/
TERRAIN_DIR = os.path.join(LEGGED_GYM_ENVS_DIR, "wbr", "terrain", "release")
TERRAIN_FILE = os.path.join(TERRAIN_DIR, "wbr_rand_curve.npz")


class WBRTerrainCfg(WBRFlatCfg):
    class env(WBRFlatCfg.env):
        num_observations = 161  # 无高程观测 ⇒ 与 wbr_flat 同维

    class terrain(WBRFlatCfg.terrain):
        """P8W 地形参数。全部在此定义，不污染 WBRFlatCfg。"""
        mesh_type = "trimesh"
        measure_heights = False               # 【关键】不加高程观测（实机无视觉/激光）
        track_test = False
        curriculum = True                     # 走得远 ⇒ 升档
        max_init_terrain_level = 0            # 从最易档起步
        # 放开课程上限：W3 零训练就能走完最难行，封顶没有意义。
        curriculum_max_level = None
        # 几何：必须与 wbr_rand_gen.py 的 ENV_LEN/ENV_WID/NUM_ROWS/NUM_COLS 逐字一致
        terrain_length = 5.0                  # 【P8W1】8 → 5 m：让机器人在一段内走得到起伏区
        terrain_width = 5.0
        horizontal_scale = 0.1
        vertical_scale = 0.002                # 2 mm：5 mm 会让起伏呈阶梯状
        border_size = 25.0
        num_rows = 4                          # 行 0 平地 + 3 档起伏
        num_cols = 6
        terrain_proportions = [1.0 / 6] * 6
        # 读图与坡面
        height_field_file = TERRAIN_FILE
        slope_treshold = None                 # 不做坡面修正（会平移顶点、让物理面错开）
        origin_z_use_max = False              # 出生点 z 取中心均值
        add_perlin_noise = False
        # viewer 里的"贴地高度着色网格"（纯可视化，控制台看图用）
        # 可读性外观全套值（网格细分＋加密＋提亮色标）
        class grid(WBRFlatCfg.terrain.grid):
            enable = True
            spacing = 0.25
            z_offset = 0.015
            radius = 16.0
            redraw_every = 4
            color_low = [0.32, 0.44, 0.68]
            color_mid = [0.48, 0.70, 0.50]

    class viewer(WBRFlatCfg.viewer):
        # 三灯平衡（明暗差塑形＋低角度补光）；headless 不进此分支，训练零影响
        light_intensity = [0.90, 0.90, 0.90]
        light_ambient = [0.85, 0.85, 0.85]
        light_direction = [-0.42, 0.42, -0.78]
        light2_intensity = [0.25, 0.25, 0.25]
        light2_direction = [0.5, -0.6, -0.35]

    class commands(WBRFlatCfg.commands):
        """**与 W3 逐字一致**，地形是本轮唯一的变量。

        W3（`releases/W3_vlim085`）已零训练通过 P8 地形判据（行 0~3 全过、
        留出图 seed1 复现）⇒ 本轮的用途只是"让它在地形上多看一眼、抬余量"。
        热启动**唯一该动的东西是地形**：指令量程、奖励带宽只要与 W3 的
        训练配置差一点，就是在换奖励地形而不是在适应地形（P8W1b/c/d 的教训）。
        W3 实配：lin_vel_x ±1.1、ang_vel_yaw ±0.8、velocity 带宽 0.85、
        height 带宽 0.08（见 `releases/W3_vlim085/config.json`）。
        """
        class ranges(WBRFlatCfg.commands.ranges):
            lin_vel_x = [-1.1, 1.1]
            ang_vel_yaw = [-0.8, 0.8]

    class rewards(WBRFlatCfg.rewards):
        # 地形任务的竖直速度口径可切；平地下用原口径即可，这里保持默认 False
        lin_vel_z_height_from_legs = False
        # 侧倾反馈增益：起伏 / 横向坡打破"左右等高"前提，纯前馈会留系统偏差，故从 0 起抬。
        K_roll_fb = 0.5
        # 地形落差被动腾空豁免（独立三态锁存 GROUND/FLIGHT/LANDING）。
        # 只在地形任务生效；平地/跳跃任务继承 WBRFlatCfg 时为 False，零影响。
        # 进 FLIGHT 的三条件：双轮同时离地 + vz<-0.15 + 非跳跃回合（见 wbr_flat.py）。
        terrain_flight_waiver = True
        terrain_flight_vz_min = -0.15
        terrain_flight_max_s = 0.5          # 单次腾空上限；超过判为主动跳，恢复惩罚
        terrain_waiver_budget_s = 3.0       # 每回合累计豁免上限，防"跳着走"刷分
        terrain_land_hold_s = 0.15          # 落地窗口：不豁免，姿态/冲击照罚
        terrain_flight_waiver_terms = ("no_fly", "lin_vel_z", "no_moonwalk", "hip_reach")
        # 竖直墙面的"伸腿探地 + 落地吸收"奖励（见 wbr_flat.py）。
        # 权重只作探索引导：开环实测低速需要伸腿（0.6/15 cm 0→24/32），
        # 但高速过度伸腿会弹跳（0.9 档 k=1.2 时 32→19/32），故不设成主导项。
        drop_extend_ref = 0.20

        class scales(WBRFlatCfg.rewards.scales):
            drop_extend = 500.0
            drop_absorb = 20.0
        # 不覆写带宽与权重 ⇒ 全部继承 WBRFlatCfg（velocity 0.85 / height 0.08 / tracking_lin_vel 8.0）。

    class noise(WBRFlatCfg.noise):
        # 【P8W6a/P8W7b】观测噪声关：P8W7a 实测传感器侧 DR（半档量级）无包络增益、只有名义变慢，
        # 传感器侧收线（包络存活基座已达标 30~32/32）；物理侧探针枪也不混噪声（单变量纪律）。
        add_noise = False

    class domain_rand(WBRFlatCfg.domain_rand):
        # 【P8W7e】末枪＝组合通道累积：push 0.1（P8W7c 已收线）+ 摩擦 ±5%（P8W7d 探针无崩签）
        # 同时开，从 2750 热启动——交付分布=推扰+摩擦双随机。质量/传感器键维持退化
        # （各自包络下基座 zero-shot 已达标，训练无实测增益，见 P8W7a/d 判定）。
        # com_offset 维持退化：本 Isaac 版本忽略 com 写回 ⇒ 键无效。
        push_robots = True
        max_push_vel_xy = 0.10
        push_interval_s = 10.0
        randomize_friction = True
        friction_range = [0.95, 1.05]
        nominal_env_fraction = 0.8     # 20% env 吃随机（推+摩擦），80% 名义稳定梯度源


class WBRTerrainCfgPPO(WBRFlatCfgPPO):
    class algorithm(WBRFlatCfgPPO.algorithm):
        # 【P8W6a】热启动统一降 lr（3e-4→1e-4）：不降则热启动必现行为漂移（速度增益塌缩、更快更矮）。
        learning_rate = 1.0e-4

    class runner(WBRFlatCfgPPO.runner):
        experiment_name = "wbr_terrain"
        run_name = "P8W7e"
        # 热启动写相对轮数：rsl_rl 的 tot_iter = current_learning_iteration + num_learning_iterations，
        # 这里写 300 就是"从 2200 再训 300 轮到 2500"（命令行 --max_iterations 同语义）。
        max_iterations = 300
        save_interval = 25


# ==============================================================================
# 组合地形 / 演示走廊 / 元素验收图三套地形任务
ELEMENTS_FILE = os.path.join(TERRAIN_DIR, "wbr_p8_L3.npz")      # 7 类元素 × 3 档
DROP_FILE = os.path.join(TERRAIN_DIR, "wbr_p8w9_drop.npz")      # P8W9 下台阶 6 行课程图
DROP15_FILE = os.path.join(TERRAIN_DIR, "wbr_p8w9_drop15.npz")  # P8W9c 15 cm 专项图（4 行）
DROP15V_FILE = os.path.join(TERRAIN_DIR, "wbr_p8w9_drop15v.npz")  # 单行 15 cm 回放图
COMBO_FILE = os.path.join(TERRAIN_DIR, "wbr_p8_combo_L3.npz")   # 3 档 × 3 种元素排列
DEMO_FILE = os.path.join(TERRAIN_DIR, "wbr_p8_demo.npz")        # 13.5×7 m 四象限演示场


class WBRDemoCfg(WBRTerrainCfg):
    """演示走廊（1 行 × 1 列，34 m × 8 m）：元素沿 +x 依次排布，控制台回放用。

    段序：平地 / 纵坡 / 单边横坡 / 左右高低不平 / 波浪 / 上台阶 / 下台阶 / 离散障碍 /
          跳石 / 左右高差台（bumps）/ 错位波浪（twave）。长度以 `DEMO_LEN`（34 m）为准。
    """

    class terrain(WBRFlatCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = False
        track_test = False
        curriculum = False
        height_field_file = DEMO_FILE
        num_rows = 1
        num_cols = 1
        # 四象限布局：前区两段各再翻倍 ⇒ 29.5 m (x) × 7.0 m (y)
        terrain_length = 29.5
        terrain_width = 7.0
        horizontal_scale = 0.1
        vertical_scale = 0.002
        border_size = 25.0
        slope_treshold = None
        add_perlin_noise = False
        # grid 与 WBRTerrainCfg 同套外观值（⚠ 本类 terrain 继承自 WBRFlatCfg.terrain，
        # **不会**继承 WBRTerrainCfg.terrain.grid，必须显式；viewer 走 WBRTerrainCfg.viewer 继承 ✓）
        class grid(WBRFlatCfg.terrain.grid):
            enable = True
            spacing = 0.25
            z_offset = 0.015
            radius = 16.0
            redraw_every = 4
            color_low = [0.32, 0.44, 0.68]
            color_mid = [0.48, 0.70, 0.50]
        curriculum_max_level = None

    class env(WBRTerrainCfg.env):
        num_observations = 161


class WBRComboCfg(WBRTerrainCfg):
    """组合地形（3 行 = L1/L2/L3 课程 × 3 列 = 三种元素排列，每格 20 m × 8 m）。"""

    class terrain(WBRFlatCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = False
        track_test = False
        curriculum = True
        max_init_terrain_level = 0
        curriculum_max_level = None
        height_field_file = COMBO_FILE
        num_rows = 3
        num_cols = 3
        terrain_length = 20.0
        terrain_width = 8.0
        horizontal_scale = 0.1
        vertical_scale = 0.002
        border_size = 25.0
        slope_treshold = None
        add_perlin_noise = False
        # 与 WBRTerrainCfg 同套外观值（本类 terrain 继承自 WBRFlatCfg.terrain，需显式）
        class grid(WBRFlatCfg.terrain.grid):
            enable = True
            spacing = 0.25
            z_offset = 0.015
            radius = 16.0
            redraw_every = 4
            color_low = [0.32, 0.44, 0.68]
            color_mid = [0.48, 0.70, 0.50]
        terrain_proportions = [1.0 / 3] * 3

    class env(WBRTerrainCfg.env):
        num_observations = 161


class WBRElementsCfg(WBRTerrainCfg):
    """单元素验收图（4 行：行 0 平地 / 行 1~3 = L1~L3；8 列 = 8 类地形）。"""

    class terrain(WBRFlatCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = False
        track_test = False
        curriculum = False
        max_init_terrain_level = 0
        height_field_file = ELEMENTS_FILE
        num_rows = 4
        num_cols = 7
        terrain_length = 8.0
        terrain_width = 8.0
        horizontal_scale = 0.1
        vertical_scale = 0.002
        border_size = 25.0
        slope_treshold = None
        add_perlin_noise = False
        # 与 WBRTerrainCfg 同套外观值（本类 terrain 继承自 WBRFlatCfg.terrain，需显式）
        class grid(WBRFlatCfg.terrain.grid):
            enable = True
            spacing = 0.25
            z_offset = 0.015
            radius = 16.0
            redraw_every = 4
            color_low = [0.32, 0.44, 0.68]
            color_mid = [0.48, 0.70, 0.50]
        curriculum_max_level = None
        terrain_proportions = [1.0 / 7] * 7

    class env(WBRTerrainCfg.env):
        num_observations = 161


class WBRDemoCfgPPO(WBRTerrainCfgPPO):
    class runner(WBRTerrainCfgPPO.runner):
        experiment_name = "wbr_terrain"
        run_name = "P8_demo"


class WBRComboCfgPPO(WBRTerrainCfgPPO):
    class runner(WBRTerrainCfgPPO.runner):
        experiment_name = "wbr_terrain"
        run_name = "P8_combo"


class WBRElementsCfgPPO(WBRTerrainCfgPPO):
    class runner(WBRTerrainCfgPPO.runner):
        experiment_name = "wbr_terrain"
        run_name = "P8_elements"


class WBRDropCfg(WBRTerrainCfg):
    """P8W9 下台阶训练图：6 行 = 每级落差 0/10/11/12/13/15 cm，4 列 = 落差位置 3.0~4.5 m。

    与 `wbr_elements` 的区别只有地形图与课程。图里 `spawn_center_m=[2.0, 4.0]`，
    出生点落在高台面上（不是格中心），所以训练时不需要 `--spawn-dx`：
    每一步都是"从平台走出去 → 两级落差 → 平地"。
    地形本身由 `wbr_terrain_gen.py --drop` 生成，改档位/距离要重跑生成器。
    """

    class rewards(WBRTerrainCfg.rewards):
        """P8W9 竖直墙面的落地规则（控制层，见 `wbr_flat.py::_apply_drop_rule`）。

        实测（15 cm 竖直面，6 s 窗，V0.27）：不加规则 0.6/0.7/0.75/0.8/0.9 =
        0/10/28/32/29；加规则后 23/23/30/32/31。
        """

        drop_rule_hold_wheel = True      # FLIGHT 期间锁轮速，防空中刹车
        drop_rule_extend_knee = 1.5      # 低速段空中伸腿（膝动作 ±k）。
                                         # below 0.72 × knee {1.4/1.5/1.6} @15cm 竖直面：
                                         # 0.6/0.7 档 = (27,32)/(28,29)/(29,22) —— 1.5 给全场
                                         # 最高地板（全速度档 ≥28/32），修掉 0.7 档 22/32 低谷；
                                         # 取值依据＝eval_runs/2026-11-06_P8W9规则扫描
        drop_rule_extend_below = 0.72    # 只在该 vx 以下伸腿；高速伸腿会弹跳（0.72：门控盖住 0.7 档
                                         # 且 0.75+ 不触发）

    class terrain(WBRTerrainCfg.terrain):
        height_field_file = DROP_FILE
        # 台阶**真正竖直**：heightfield 逐像素高差在 trimesh 里
        # 默认会连成斜面（0.15 m 落差 / 0.1 m 水平 ⇒ 56°）；slope_treshold 让 isaacgym
        # 把超过阈值的边坡顶点水平挪动、生成垂直墙。0.75 = 水平/垂直比，1 格落差即触发。
        # ⚠ 代价：顶点被平移，物理面与高度图采样位置有偏（评测/可视化都以物理面为准）。
        slope_treshold = 0.75
        num_rows = 6
        num_cols = 4
        terrain_length = 8.0
        terrain_width = 8.0
        curriculum = True
        max_init_terrain_level = 5      # 初始随机 0~3 行（含平地行），课程再按行进距离升/降
        terrain_proportions = [1.0 / 6] * 6

    class env(WBRTerrainCfg.env):
        num_observations = 161

    class commands(WBRTerrainCfg.commands):
        """vx 采样只留**中高速段** 0.45~1.1。

        根因：R1（全速段 ±1.1）把 0.6 档 15 cm 从 21/32 拉到 5/32 —— 低速段
        （0.3 下 12 cm 以上物理过不去）产生大量全灭负样本，把策略拉向保守减速。
        0.6~0.65 是 15 cm 的动量临界（0.6→21/32、0.65→42/48、0.7→47/48），
        所以训练分布从 0.45 起，覆盖临界区两侧。

        ⚠ R2 实测：收窄到 0.45~1.1 **仍无效**——15 cm@0.6 21→5、
        @0.75 31→23、@0.9 32→30。结论是不需要训练：V0.27 零训练在 vx 0.7 已 47/48。
        本段保留只为复现 R2；**默认训练建议用全速段 [-1.1, 1.1]**（即注释掉下面两行）。
        """

        class ranges(WBRTerrainCfg.commands.ranges):
            lin_vel_x = [0.45, 1.1]


class WBRDropCfgPPO(WBRTerrainCfgPPO):
    class runner(WBRTerrainCfgPPO.runner):
        experiment_name = "wbr_terrain"
        run_name = "P8W9a"


class WBRDrop15Cfg(WBRDropCfg):
    """P8W9c **15 cm 专项图**：4 行 = 15/15/15/10 cm（每级），75% 的 env 落在 15 cm。

    为什么不用 `wbr_drop` 的 6 行课程：curriculum 按行进距离升档，但策略在 12 cm 以上
    走不动 ⇒ 永远升不到 15 cm 档，15 cm 样本占比接近零（R2 的根因）。这里直接
    `curriculum=False` + 行随机，让 15 cm 成为**主要样本**；保留 1 行 10 cm 防遗忘。
    速度采样继承 WBRDropCfg（0.45~1.1）。
    """

    class terrain(WBRDropCfg.terrain):
        height_field_file = DROP15_FILE
        num_rows = 4
        num_cols = 4
        curriculum = False
        max_init_terrain_level = 3      # 行随机 0~3：3 行 15 cm + 1 行 10 cm
        terrain_proportions = [0.25] * 4

    class commands(WBRDropCfg.commands):
        """R3b 起把速度下沿拉回 0.3（R3a 用 0.45 起，15 cm@0.6 31/32 但 0.3 档 vx 掉到 0.187）。"""

        class ranges(WBRDropCfg.commands.ranges):
            # R1 用 0.5~1.0、R2 用 0.55~0.78（两连负）。默认回到 0.5~1.0 覆盖高速档；
            # 15 cm 竖直面在 vx≥0.75 零训练即可（28~32/32），本任务只在需要重跑时使用。
            lin_vel_x = [0.50, 1.0]


class WBRDrop15CfgPPO(WBRTerrainCfgPPO):
    class runner(WBRTerrainCfgPPO.runner):
        experiment_name = "wbr_terrain"
        run_name = "P8W9c"


class WBRDrop15ViewCfg(WBRDrop15Cfg):
    """单行 15 cm 回放图：`play02 --task wbr_drop15view` 专用，保证每次都在 15 cm 档。"""

    class terrain(WBRDrop15Cfg.terrain):
        height_field_file = DROP15V_FILE
        num_rows = 1
        num_cols = 4
        curriculum = False
        max_init_terrain_level = 0
        terrain_proportions = [0.25] * 4


class WBRDrop15ViewCfgPPO(WBRTerrainCfgPPO):
    class runner(WBRTerrainCfgPPO.runner):
        experiment_name = "wbr_terrain"
        run_name = "P8W9view"
