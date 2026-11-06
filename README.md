# WBR 轮腿双足机器人 · 强化学习控制

<p align="center"><img src="https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/hero-robot.jpg" alt="WBR 轮腿双足机器人" width="420"></p>

**WBR 是一台双模式轮腿双足机器人**：两个驱动轮 + 两侧各一条两连杆腿（髋 + 膝），外加被动万向轮。
四种轮腿运动能力——两轮行走、跳跃、四轮行驶、模式切换——由强化学习一体训练，全部在 Isaac Gym 中完成。

> **本仓库为个人学习记录使用**：记录我个人的训练过程与产出的权重，不是一个完善的框架，也不提供支持。
> 仓库只保留 WBR 自身：从上游 `legged_gym` 继承的示例机器人（ANYmal、A1、Cassie 等）已全部移除，
> 任务表里只有 8 个 WBR 任务。[English](README.en.md) | 中文

## 概述

一个**轮腿双足机器人**的强化学习控制器：躯干由两个轮子支撑，每侧一条两连杆腿（髋 + 膝，单侧 3 自由度），
外加一个被动万向轮。

| 能力 | 任务 | 做什么 |
|---|---|---|
| 两轮行走 | `wbr_flat` | 平地上按速度 / 偏航 / 高度指令行走并保持体高（域随机化交付：推扰 0.1＋摩擦 ±5%） |
| 两轮跳跃 | `wbr_jump` | 预蹲 → 蹬地 → 空中收腿 → 落地缓冲；站跳三档 54.9 / 98.9 / 92.2 mm |
| 复杂地形 | `wbr_terrain` | 同一套策略跑随机起伏；`wbr_p8_L1~L3`（7 类元素 × 3 档）、`wbr_p8_combo_L1~L3`、`wbr_p8_demo`（演示走廊）是脚本化验收场景 |
| 下台阶 | `wbr_drop` / `wbr_drop15` | 被动落差（不靠高程观测）：15 cm 竖直落差全速度档 ≥28/32（落地规则 v2：空中锁轮＋低速伸腿） |
| 四轮行驶 | `wbr_flat4` | 四轮静稳构型（两个万向轮着地），行驶与转向（vx±1 / yaw±1，15 项域随机化） |
| 模式切换 | `wbr_switch`、`wbr_uni` | 由四轮起立到两轮、再趴下；五槽相位机统一编排，起立为平台式脚本链（P7T2：甩+接+驻留交棒） |

地形验收任务复用 `wbr_flat` 的策略 —— 它们是场景，不是新的训练问题。

## 功能演示

<table>
<tr>
<th>四轮行驶（<code>P7F3e</code>）</th>
<th>两轮行走（<code>P8W7</code>）</th>
</tr>
<tr>
<td><img src="https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/fourw_drive.gif" alt="四轮行驶"></td>
<td><img src="https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/twow_walk.gif" alt="两轮行走"></td>
</tr>
<tr>
<th>跳跃（<code>P6R81</code>）</th>
<th>起立 / 趴下（<code>P7T</code> 脚本 + <code>P7L</code>）</th>
</tr>
<tr>
<td><img src="https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/jump.gif" alt="跳跃"></td>
<td><img src="https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/standup.gif" alt="起立趴下"></td>
</tr>
</table>

## 网页控制台

`wbr_console` 是仿真之上的浏览器控制台，用 `./run_bench.sh` 启动（→ `http://127.0.0.1:8770`）。
**双引擎**：Isaac Gym（训练同源）与 MuJoCo（`sim2sim_mujoco` 部署链，免 GPU 初始化、秒级停止）。
五个页签：

- **仿真运行** —— 启动/停止仿真，实时指令滑条（Vx / Yaw / Height / Roll / Jump），
  链模式下一键起立 / 趴下 / 装填跳 / 回两轮 / 回四轮 / 复位 / 切槽（带指令回执条与站定门：
  |vx|≤0.08 ∧ |pitch|≤5.7° 连续 8 拍才切网），GPU 状态，Isaac 原生窗口停靠区。
- **Isaac 配置 / MuJoCo 配置** —— 槽位化：四轮 164 / 两轮行走 161 / 两轮跳跃 161 / 趴下 173 /
  五槽链（两轮或四轮起步）各一套契约＋资产＋权重；MuJoCo 链支持复杂地形（四象限 demo 图）；
  起立引擎可选 P7T2 脚本链（默认）或 P7S 学习策略；各相位时长参数可调。
- **数据绘图** —— 遥测曲线：指令三线、竖直速度、跳跃、相位底色、关节力矩；
  Isaac 与 MuJoCo 遥测同列（MuJoCo 由 serve 增量落盘，运行中可跟随）。
- **使用说明** —— 参数词典、快捷键与存储路径。

<table>
<tr>
<th>仿真运行</th>
<th>模型配置</th>
</tr>
<tr>
<td><img src="https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/console-run.png" alt="仿真运行"></td>
<td><img src="https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/console-model.png" alt="模型配置"></td>
</tr>
<tr>
<th>数据绘图</th>
<th>使用说明</th>
</tr>
<tr>
<td><img src="https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/console-plot.png" alt="数据绘图"></td>
<td><img src="https://github.com/Yihuan01/wheel-legged-biped-rl/releases/download/demo-media-v1/console-help.png" alt="使用说明"></td>
</tr>
</table>

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

# MuJoCo 部署链（免 GPU 初始化；两轮起步的四槽整链 9.6 s 零终止）
python -m legged_gym.scripts.sim2sim_mujoco.deploy.uni_chain --start 2w

# 网页控制台（双引擎、五槽、P7T2 平台式起立、绘图页）→ http://127.0.0.1:8770
cd legged_gym/scripts/wbr_console && ./run_bench.sh
```

`play02.py` 的 `--load_run` 只接受迭代号后缀，所以要把发布包的 `model_*.pt` 与 `config.json`
复制到你自己的 `logs/<task>/` 目录，再把该目录名传给 `--load_run`。

遥测与绘图：`play02.py` 把 TSV 数据段写到 `legged_gym/scripts/wbr_play_data/`，
`plot02.py` 再把它生成自包含 HTML 页面。

## 任务一览

| 任务 | 形态 | 观测 | 动作 | 用途 |
|---|---|---|---|---|
| `wbr_flat` | 两轮（`wbr_v1.urdf`） | 161 | 6 | 平地行走 / 保持体高（域随机化交付基座） |
| `wbr_jump` | 两轮 | 161 | 6 | 含跳跃相位机的两轮任务（交付权重 P8J3d） |
| `wbr_terrain` | 两轮 | 161 | 6 | 随机起伏地形行走（观测不含高程） |
| `wbr_demo` / `wbr_combo` / `wbr_elements` | 两轮 | 161 | 6 | P8 三套验收场景：演示走廊 / 组合地形 / 单元素图 |
| `wbr_drop` / `wbr_drop15` / `wbr_drop15view` | 两轮 | 161 | 6 | 下台阶（被动落差）训练与验收图，含落地规则 |
| `wbr_flat4` | 四轮（`wbr_v2.urdf`） | 164 | 6 | 四轮静稳构型，行驶与转向 |
| `wbr_switch` | 四轮 | 173 | 6 | 按相位脚本起立 / 趴下 |
| `wbr_uni` | 多槽 | 161/164/173 | 6 | 统一相位机（统一 0–11 相位），单个环境承载全部模式 |

地形验收任务复用 `wbr_flat` 的策略 —— 它们是场景，不是新的训练问题。

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
│       └── terrain/   离线高度图生成器与 release/ 验收图（npz）
├── scripts/           训练 / 回放 / 遥测；wbr_console 网页控制台（双引擎）；
│                      sim2sim_mujoco/（MuJoCo 部署链：契约/资产/六槽链/验收脚本/部署清单）
└── utils/             任务注册表、工具函数、地形、日志
rsl_rl-1.0.2/          自带的 PPO 实现
resources/robots/wbr/  URDF 与网格
releases/              演示用最小权重集（P7F3e 四轮、P8W7 行走、P8J3d 跳跃、P6R81 跳跃对照、P7L 趴下）；完整训练用 train.py
```

## 待解决问题

放在前面，因为它们比成绩更能说明这个项目：

- **跳跃高度对指令的跟随不单调。** 站跳三档可达（54.9 / 98.9 / 92.2 mm），但最高档仍低于中间档——策略能力边界，指令到顶点的映射不线性。
- **两引擎存在响应差。** 同一权重在 Isaac 与 MuJoCo 的跳跃顶点差 ±25~42 mm（对 history 敏感的爆发段瞬态）；行走 parity ≤1.3%，部署以 Isaac 口径为基准。
- **链上与单机仍有差项。** 五槽链验收全过（C1×3/C2/C3），链上跳跃与单机的差项尚未逐项归因。
- **离散障碍不做。** 下台阶覆盖被动落差（15 cm 竖直面 ≥28/32）；跳石、断开平台、上台阶不在范围内。
- **仅有仿真。** 没有实机数据；sim-to-real 安全裕度尚未标定（Sim2Sim MuJoCo 链已建立为中间验证层，敏感度成册：两轮策略对 base_lin_vel 观测强依赖）。
- **训练不稳定。** PPO 在长训练末端会崩，权重按版本逐档挑选，不是直接取最后一次迭代。

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
