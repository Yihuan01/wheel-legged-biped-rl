# legged_gym/envs/base —— 上游基类

| 文件 | 用途 |
|---|---|
| `base_task.py` | 任务基类：创建仿真、地形、环境句柄，管理 `gym` / `sim` 生命周期 |
| `base_config.py` | 配置基类：按 `class` 嵌套初始化各配置段（sim / control / asset …） |
| `legged_robot.py` | 足式机器人环境基类：观测、奖励、域随机化、地形与相机；WBR 环境的父类 |
| `legged_robot_config.py` | 足式机器人默认配置，WBR 各任务配置逐段覆盖它 |

本目录为上游代码（NVIDIA / ETH Zurich，BSD-3），只做了空白规范化，未改逻辑。
