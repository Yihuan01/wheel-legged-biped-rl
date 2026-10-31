# legged_gym —— Isaac Gym 训练框架（WBR 改造版）

代码是 Isaac Gym 的 `legged_gym` 框架，已裁剪为只服务 WBR（轮足机器人）。上游继承的示例机器人、网格与环境代码已全部删除，任务表里只剩 8 个 WBR 任务。

## 目录

| 路径 | 用途 |
|---|---|
| `__init__.py` | 项目根路径常量（`LEGGED_GYM_ROOT_DIR` / `LEGGED_GYM_ENVS_DIR`） |
| `envs/` | 环境与任务注册，见 `envs/README.md` |
| `scripts/` | 训练、回放、遥测与网页控制台入口，见 `scripts/README.md` |
| `utils/` | 任务注册表、地形、日志、数学与命令行工具，见 `utils/README.md` |
| `tests/` | 上游自带的冒烟测试 |

## 快速开始

```bash
pip install -e .                                   # 在本仓库根目录
python legged_gym/scripts/train.py --task wbr_flat --headless
```

8 个已注册任务：`wbr_flat`、`wbr_terrain`、`wbr_demo`、`wbr_combo`、`wbr_elements`、`wbr_flat4`、`wbr_switch`、`wbr_uni`。
