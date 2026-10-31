# WBR 轮腿双足机器人 · 强化学习控制

![WBR 轮腿双足机器人](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/robot-stand.jpg)

**WBR 是一台双模式轮腿双足机器人**：两个驱动轮 + 两侧各一条两连杆腿（髋 + 膝），外加被动万向轮。
四种轮腿运动能力——两轮行走、跳跃、四轮行驶、模式切换——由强化学习一体训练，全部在 Isaac Gym 中完成。

[English](README.md)

> **本仓库为个人学习记录使用**：记录我个人的训练过程与产出的权重，不是一个完善的框架，也不提供支持。
> 仓库只保留 WBR 自身：从上游 `legged_gym` 继承的示例机器人（ANYmal、A1、Cassie 等）已全部移除，
> 任务表里只有 8 个 WBR 任务。

## 概述

一个**轮腿双足机器人**的强化学习控制器：躯干由两个轮子支撑，每侧一条两连杆腿（髋 + 膝，单侧 3 自由度），
外加一个被动万向轮。

| 能力 | 任务 | 做什么 |
|---|---|---|
| 两轮行走 | `wbr_flat` | 平地上按速度 / 偏航 / 高度指令行走并保持体高 |
| 两轮跳跃 | `wbr_flat` | 预蹲 → 蹬地 → 空中收腿 → 落地缓冲 |
| 起伏地形 | `wbr_terrain` | 同一套策略跑随机起伏；`wbr_demo` / `wbr_combo` / `wbr_elements` 是脚本化的验收场景 |
| 四轮行驶 | `wbr_flat4` | 四轮静稳构型（两个万向轮着地），行驶与转向 |
| 模式切换 | `wbr_switch`、`wbr_uni` | 由四轮起立到两轮、再趴下，按相位逐步切换 |

三个地形验收任务复用 `wbr_flat` 的策略 —— 它们是场景，不是新的训练问题。

## 功能演示

每个模式一段动图，用已发布的演示权重在 Isaac Gym 中实录。GIF 挂在
[demo-media-v1 release](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/tag/demo-media-v1)，
克隆仓库不会带上它们：

| | |
|---|---|
| **四轮行驶**（`P7F3e`） | **两轮行走**（`P6R81`） |
| ![四轮行驶](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/fourw_drive.gif) | ![两轮行走](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/twow_walk.gif) |
| **跳跃**（`P6R81`） | **起立 / 趴下**（`P7T` 脚本 + `P7L`） |
| ![跳跃](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/jump.gif) | ![起立趴下](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/standup.gif) |

## 网页控制台

`wbr_console` 是仿真之上的浏览器控制台，用 `./run_bench.sh` 启动（→ `http://127.0.0.1:8770`）。
四个页签：

- **仿真运行** —— 启动/停止仿真，实时指令滑条（Vx / Yaw / Height / Roll / Jump），
  一键起立 / 趴下 / 复位 / 装填跳，GPU 状态，以及 Isaac Gym 原生窗口的停靠区。
- **模型配置** —— 四槽衔接链（行驶 → 起立 → 行走 → 趴下），每槽可换任意维度匹配的
  checkpoint；起立可选学习策略（P7S）或 P7T 脚本链；各相位时长参数可调。
- **数据绘图** —— 遥测曲线：速度、轮/腿状态、力矩与相位，可读运行中的仿真或已存的 TSV 日志。
- **使用说明** —— 参数词典、快捷键与存储路径。

| 仿真运行 | 模型配置 |
|---|---|
| ![仿真运行](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/console-run.png) | ![模型配置](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/console-model.png) |
| **数据绘图** | **使用说明** |
| ![数据绘图](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/console-plot.png) | ![使用说明](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/console-help.png) |

控制台代码随仓库分发；上方截图挂在
[demo-media-v1 release](https://github.com/Yihuan01/wheel-legged-biped-rl/releases/tag/demo-media-v1)，
不进仓库，保持克隆轻量。

## 环境要求

| 项 | 版本 / 说明 |
|---|---|
| 显卡 | NVIDIA，支持 CUDA |
| 系统 | Ubuntu 20.04（实测） |
| Python | 3.8 —— Isaac Gym Preview 3 只提供 3.8 的二进制包 |
| 仿真 | **Isaac Gym Preview 3**，需从 NVIDIA 开发者包下载（PyPI 上没有） |
| PyTorch | 用 Isaac Gym Preview 3 环境自带的版本 |
| PPO | `rsl_rl` **1.0.2，已随仓库自带**在 `rsl_rl-1.0.2/`，**不要** `pip install rsl-rl` |

## 安装

```bash
# 1. 安装 Isaac Gym（先从 NVIDIA 开发者网站下载压缩包）
cd isaacgym/python && pip install -e . && cd ../..

# 2. 安装本仓库
pip install -e .

# 3. 绘图与地形生成所需依赖
pip install matplotlib numpy scipy
```

如果 Python 找不到自己的动态库：

```
ImportError: libpython3.8.so.1.0: cannot open shared object file
```

运行前先导出环境的库路径：

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

自检安装是否成功 —— 应输出 `8`：

```bash
python -c "
import isaacgym, torch
import legged_gym.envs
from legged_gym.utils import task_registry
print(len(task_registry.task_classes))
"
```

> 先导入 `legged_gym.envs` 再导入 `legged_gym.utils`：任务表由前者填充；
> 另外 `isaacgym` 必须先于 `torch` 导入。

## 快速开始

```bash
# 训练（无界面；输出写到 logs/<experiment_name>/<run_name>/）
python legged_gym/scripts/train.py --task wbr_flat --headless

# 续训 300 轮
python legged_gym/scripts/train.py --task wbr_terrain --resume --max_iterations 300

# 回放已发布的权重
python legged_gym/scripts/play02.py --task wbr_flat \
    --task wbr_uni --demo_chain

# 网页控制台（四种模式、相位脚本、绘图）→ http://127.0.0.1:8770
cd legged_gym/scripts/wbr_console && ./run_bench.sh
```

`play02.py` 的 `--load_run` 只接受迭代号后缀，所以要把发布包的 `model_*.pt` 与 `config.json`
复制到你自己的 `logs/<task>/` 目录，再把该目录名传给 `--load_run`。

遥测与绘图：`play02.py` 把 TSV 数据段写到 `legged_gym/scripts/wbr_play_data/`，
`plot02.py` 再把它生成自包含 HTML 页面。

## 任务一览

| 任务 | 形态 | 观测 | 动作 | 用途 |
|---|---|---|---|---|
| `wbr_flat` | 两轮（`wbr_v1.urdf`） | 161 | 6 | 平地行走 / 保持体高 / 跳跃 |
| `wbr_terrain` | 两轮 | 161 | 6 | 随机起伏地形行走（观测不含高程） |
| `wbr_demo` | 两轮 | 161 | 6 | 47 m 演示走廊，含 9 个地形元素 |
| `wbr_combo` | 两轮 | 161 | 6 | 三档难度 × 三种元素排列 |
| `wbr_elements` | 两轮 | 161 | 6 | 单元素验收图（8 类 × 3 档） |
| `wbr_flat4` | 四轮（`wbr_v2.urdf`） | 164 | 6 | 四轮静稳构型，行驶与转向 |
| `wbr_switch` | 四轮 | 173 | 6 | 按相位脚本起立 / 趴下 |
| `wbr_uni` | 四轮 | 173 | 6 | 统一相位机，单个环境承载全部模式（只回放） |

三个地形验收任务复用 `wbr_flat` 的策略 —— 它们是场景，不是新的训练问题。

## 目录结构

```
legged_gym/
├── envs/
│   ├── base/          LeggedRobot 基类与配置默认值
│   └── wbr/           ★ 本项目
│       ├── flat/      wbr.py、wbr_flat_config.py —— 两轮行走 + 跳跃
│       ├── flat4/     wbr_flat4.py、wbr_flat4_config.py —— 四轮行驶
│       ├── switch/    起立 / 趴下的相位机
│       ├── uni/       统一多模式环境、状态机、观测拼装
│       └── terrain/   离线高度图生成器与探针
├── scripts/           训练 / 回放 / 遥测，以及 wbr_console 网页控制台
└── utils/             任务注册表、工具函数、地形、日志
rsl_rl-1.0.2/          自带的 PPO 实现
resources/robots/wbr/  URDF 与网格
releases/              仅演示用最小权重集（F3e 四轮、R81 两轮/跳跃、P7L 趴下）；完整训练用 train.py
```

每个目录都有自己的 `README.md`，写明该目录里有什么。

## 版本

[VERSIONS.md](VERSIONS.md) 只记录上传到本公仓的版本——每次公仓上传一条，附该次携带的权重。
随仓库分发的演示权重见各 releases/<包>/README.md。

## 待解决问题

放在前面，因为它们比成绩更能说明这个项目：

- **跳跃高度不随指令单调上升。** 最高档指令得到的跳跃反而比中间档低：当前指令主要影响预蹲深度，而不是离场速度。
- **链上不如单机。** 四槽串联下的站立高度比单机低约 1.5 cm，跳跃短 1–2 cm。
- **障碍地形未解决。** 几厘米以上的离散台阶与断崖仍会导致摔倒。
- **仅有仿真。** 没有实机数据，sim-to-real 安全裕度尚未标定。
- **训练不稳定。** PPO 在长训练末端会崩，权重是按版本逐档挑选的，不是直接取最后一次迭代。

## 致谢

- [**legged_gym**](https://github.com/leggedrobotics/legged_gym)（ETH Zurich，Nikita Rudin）——
  本仓库所基于的训练框架。
- [**rsl_rl**](https://github.com/leggedrobotics/rsl_rl) v1.0.2 —— 本项目使用的 PPO 实现。
- [**NVIDIA Isaac Gym**](https://developer.nvidia.com/isaac-gym) —— 物理仿真后端。
- [**fgmn/Wheel_legged_gym**](https://github.com/fgmn/Wheel_legged_gym) —— 本工作扩展的轮腿机器人基础。
- [**OpenCode**](https://opencode.ai) —— 辅助开发、重构与文档撰写；代码经本人审核并负责。

所有再分发的代码均保留原始许可证；见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 与 `licenses/`。

## 许可证

BSD 3-Clause —— 见 [LICENSE](LICENSE)。这是研究代码，使用风险自担。

---

**作者**：SYH ([@Yihuan01](https://github.com/Yihuan01))
