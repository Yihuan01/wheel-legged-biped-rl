# legged_gym/envs —— 环境与任务注册

## 目录

| 路径 | 用途 |
|---|---|
| `__init__.py` | 唯一的任务注册表：import 各环境并 `task_registry.register(...)`，注册 8 个 WBR 任务 |
| `base/` | 上游基类 `LeggedRobot` 与配置默认值，所有 WBR 环境都从它派生 |
| `wbr/` | WBR 自己的环境，见 `wbr/README.md` |

## 注意

`base/legged_robot.py` 必须排在 `utils` 之前 import：它内部会先 import isaacgym，
若 torch 先被导入，isaacgym 会直接抛 `PyTorch was imported before isaacgym modules`。
