# legged_gym/utils —— 工具模块

| 文件 | 用途 |
|---|---|
| `task_registry.py` | 任务注册表：`register` / `make` / `make_cfg`，train 与 play 都通过它取环境类 |
| `helpers.py` | 命令行参数（`get_args`）与配置导出、路径解析等小工具 |
| `terrain.py` | 地形构建：平地 / 三角网格 / 高度场，含随机起伏与课程 |
| `math.py` | 数学工具：向量的环绕、四元数、随机采样等 |
| `logger.py` | 训练日志与 TensorBoard 记录 |

`__init__.py` 统一导出上述模块。
