# ==============================================================================
# Date      : 2026-10-29
# Author    : SYH
# File      : wbr_terrain_config.py
# Purpose   : 同一策略的四个地形任务配置，只替换 terrain 段，观测仍 161 维（不含高程）：
#             wbr_terrain 随机起伏 / wbr_demo 演示走廊 / wbr_combo 组合地形 / wbr_elements 单元素验收。
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
        class grid(WBRFlatCfg.terrain.grid):
            enable = True
            radius = 6.0

    class commands(WBRFlatCfg.commands):
        """【P8W1f 2026-10-29】**与 W3 逐字一致**，地形是本轮唯一的变量。

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
        # 不覆写带宽与权重 ⇒ 全部继承 WBRFlatCfg（velocity 0.85 / height 0.08 / tracking_lin_vel 8.0）。


class WBRTerrainCfgPPO(WBRFlatCfgPPO):
    class runner(WBRFlatCfgPPO.runner):
        experiment_name = "wbr_terrain"
        run_name = "P8W1f"
        # 热启动写相对轮数：rsl_rl 的 tot_iter = current_learning_iteration + num_learning_iterations，
        # 这里写 300 就是"从 2200 再训 300 轮到 2500"（命令行 --max_iterations 同语义）。
        max_iterations = 300
        save_interval = 25


# ==============================================================================
# 组合地形 / 演示走廊 / 元素验收图三套地形任务
ELEMENTS_FILE = os.path.join(TERRAIN_DIR, "wbr_p8_L3.npz")      # 8 类元素 × 3 档
COMBO_FILE = os.path.join(TERRAIN_DIR, "wbr_p8_combo_L3.npz")   # 3 档 × 3 种元素排列
DEMO_FILE = os.path.join(TERRAIN_DIR, "wbr_p8_demo.npz")        # 47 m 单格演示走廊


class WBRDemoCfg(WBRTerrainCfg):
    """演示走廊（1 行 × 1 列，47 m × 8 m）：元素沿 +x 依次排布，控制台回放用。"""

    class terrain(WBRFlatCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = False
        track_test = False
        curriculum = False
        height_field_file = DEMO_FILE
        num_rows = 1
        num_cols = 1
        terrain_length = 47.0
        terrain_width = 8.0
        horizontal_scale = 0.1
        vertical_scale = 0.002
        border_size = 25.0
        slope_treshold = None
        add_perlin_noise = False
        class grid(WBRFlatCfg.terrain.grid):
            enable = True
            radius = 6.0
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
        class grid(WBRFlatCfg.terrain.grid):
            enable = True
            radius = 6.0
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
        num_cols = 8
        terrain_length = 8.0
        terrain_width = 8.0
        horizontal_scale = 0.1
        vertical_scale = 0.002
        border_size = 25.0
        slope_treshold = None
        add_perlin_noise = False
        class grid(WBRFlatCfg.terrain.grid):
            enable = True
            radius = 6.0
        curriculum_max_level = None
        terrain_proportions = [1.0 / 8] * 8

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
