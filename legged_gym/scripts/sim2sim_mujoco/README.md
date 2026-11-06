# sim2sim_mujoco —— 四轮模式 MuJoCo 部署链（目录索引）

属主：[Ubuntu/OpenCode]｜2026-11-05 建

## 这是什么

把已交付的四轮策略（`releases/P7F3e_yaw1_dr15/model_11800.pt`）从 Isaac Gym（PhysX）搬到 MuJoCo 执行。
**只替换物理层**：观测拼装、动作处理、腿 PD / 轮速度环、力矩节奏全部沿用训练侧语义，逐位对齐有 golden 对账把关。

## 目录

| 路径 | 作用 |
|---|---|
| `assets/mjcf/wbr_v2_demo.xml` | **demo 四象限复杂地形版**（floor 浅色 `rgba 0.74/0.78/0.82`；出生点＝demo 中央十字走廊交点 (4.0,3.5)，与 Isaac `env_origins` 一致）（`tools/make_terrain_asset.py` 由 `wbr_p8_demo.npz` 生成，8-bit PNG 高程场；`terrain/wbr_v2_demo.json` 记 sha／z_offset／出生点） |
| `assets/mjcf/wbr_v2.xml` | 由 URDF 生成的部署模型（freejoint＋地面＋option＋从动轮摩擦）；**视觉＝URDF 的 STL 网格（group 2，纯显示），碰撞＝primitive（group 3）**；生成物，别手改 |
| `assets/deploy_contract.json` | 部署契约（增益/命令范围/观测常量/来源 sha），由 release config ＋代码常量冻结 |
| `assets/joint_map` 语义 | 关节按**名字**映射（Isaac 与 MuJoCo 的 dof 顺序不同），见 `deploy/*.py` 的 `names.index` |
| `deploy/obs_builder.py` | 164 维观测唯一实现（6 维关节块按动作序，排除从动轮；历史滞后一拍；末尾四轮块） |
| `deploy/control.py` | 动作裁剪 → 腿位置 PD / 轮速度环（实测加速度 D 项 α=0.8）→ 逐 dof 限幅 |
| `deploy/policy.py` | `.pt` 直载 actor（512-256-128 ELU），纯 torch，无训练框架 |
| `deploy/com_model.py` | 倾翻裕度要用的质心 FK（`wbr_flat4.py` 的逐字副本，因原文件 import isaacgym） |
| `sim_loop.py` | **运行核（一核两壳）**：MjWbr 物理状态访问 + SimLoop 控制循环（指令限速 vx 0.8 m/s²·h 0.15 m/s → obs → actor → 200 Hz 力矩 → 终止判定 → 遥测；viewer 内建含 `os._exit` 硬退封装） |
| `run_policy.py` | 批量 CLI 壳：固定指令跑 SimLoop（slew 关闭，与旧版逐位一致），出摘要/遥测 |
| `serve.py` | 控制台壳（wbr_bench 按 engine=mujoco 拉起）：spec → SimLoop 实时跑 + mj_bridge + viewer；录制落 `eval/runs/` |
| `mj_bridge.py` | MuJoCo 进程内控制桥 :8771，协议与 wbr_console/bench_bridge 逐字对齐（共享 bridge_common 骨架），engine 报 "mujoco" |
| `tools/build_mjcf.py` | URDF→MJCF（补 freejoint、地面、`timestep 0.005`、`implicitfast`、从动轮 μ=0.02、视觉网格）；`--no-mesh` 生成纯碰撞版 |
| `tools/make_contract.py` | 冻结契约；交付包与代码常量对不上就报错 |
| `tools/dump_isaac_static.py` | Isaac 侧静态导出（**CPU 管线，不占 GPU**） |
| `tools/dump_isaac_golden.py` | Isaac 侧链路 golden（状态/观测/动作/力矩，含两个物理子步） |
| `tools/check_static.py` | MJCF ↔ Isaac 静态对账（89 项） |
| `tools/run_parity.py` | 回放 golden 做三对齐（obs / act / τ），不跑物理 |
| `deploy/p7t2_platform_standup.py` | **P7T2 平台式起立（✅ 成功；2026-11-06 由代号 P9 更名，作者定）**：论文 §8 移植＋甩/接/驻留交棒（重力补偿随 pitch 衰减＋过零纯阻尼＋roll 差动＋交棒渐入）。六槽自动链两次 RISE 全部存活；早前"物理不可行"定案因符号/重力矩误算被推翻。详见 `eval_runs/2026-11-06_S3c_平台式起立/RESULT.md` |
| `tools/make_terrain_asset.py` | **地形资产生成**：npz→8-bit L PNG＋带 `<hfield>` 的 MJCF（MuJoCo 3.17 只认该 PNG 格式；base>0、删 plane size 残留、data 转置、size 匹配真实半宽） |
| `deploy/uni_chain.py` | **S3 五槽链**（`terrain=True` 走 demo 地形；跳完原地交还 P8W7 网；`h_2w` 记面板两轮高度（默认 0.25）、`h_jump` 记下一跳高度）：`interactive=False` 按固定时长自动放完（CLI 评估）；`interactive=True` 事件驱动（控制台等面板指令，含守卫表与站定门） |
| `eval/compare_isaac.py` | Isaac 与 MuJoCo 同协议指标对照 |

## 环境与运行

```bash
# MuJoCo 侧（CPU，零 GPU）
conda activate mujoco
cd /home/syh/code/Wheel_legged_gym
python -m legged_gym.scripts.sim2sim_mujoco.tools.build_mjcf
python -m legged_gym.scripts.sim2sim_mujoco.tools.make_contract
python -m legged_gym.scripts.sim2sim_mujoco.tools.check_static
python -m legged_gym.scripts.sim2sim_mujoco.tools.run_parity
python -m legged_gym.scripts.sim2sim_mujoco.run_policy --vx 0.3 --h 0.21 --dur 10    # 只出数据
python -m legged_gym.scripts.sim2sim_mujoco.run_policy --vx 0.3 --h 0.21 --dur 20 --viewer          # 开窗口实时看
python -m legged_gym.scripts.sim2sim_mujoco.run_policy --vx 0.3 --h 0.21 --dur 20 --viewer --speed 0.3  # 慢放

# 控制台驱动（wbr_bench 按 engine=mujoco 自动拉起 serve.py；网页 → http://127.0.0.1:8770）
#   运行页分段控件选 MuJoCo / 配置页 /mujoco.html；详见 wbr_console/README.md 的「MuJoCo 引擎」节
python -m legged_gym.scripts.sim2sim_mujoco.eval.plot_run --run <run目录> --scenario walk03         # 出四联图

# Isaac 侧采集（CPU 管线；必须带 LD_LIBRARY_PATH）
conda activate isaacgym01
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \
  python -m legged_gym.scripts.sim2sim_mujoco.tools.dump_isaac_static
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \
  python -m legged_gym.scripts.sim2sim_mujoco.tools.dump_isaac_golden
```

器材版本：mujoco 3.17.0 / torch 2.14.1（推理）、torch 2.4.1（训练/采集）。

> 窗口模式的小坑：GLFW/NVIDIA 在解释器**退出阶段**会段错误（窗口与模型已正常关过），
> `run_policy.py` 因此在窗口模式下用 `os._exit` 硬退；命令行窗口建议 `--dur ≤ 60`，关窗/`Ctrl+C` 退出。物理 `dt=0.005`、策略 100 Hz、**力矩 200 Hz 重算**。

## 与训练侧的四个已核实契约（改任何一条都要重跑 golden）

1. 观测 164 ＝ 33 基础 ＋ 120 历史 ＋ 8 跳跃块 ＋ 3 四轮块；6 维关节块顺序＝动作序 `[hip_l,knee_l,hip_r,knee_r,wheel_l,wheel_r]`。
2. 高度指令归一化用 **`h_half = max(半宽, 0.05)`**（F3 区间半宽 0.035 ⇒ 实际 0.05）。
3. 包络从 `command_raw_yaw` 读原始偏航；只写 `command_targets` 不写它，yaw 会被旧值覆盖。
4. `reset_idx` 里 `history[env_ids].zero_()` 因高级索引取副本而**不生效**（既有 bug，未改环境码）；部署冷启动用零历史，golden 回放按落盘初值。

## 视觉网格

URDF 的 7 个 STL（`resources/robots/wbr/meshes_v1/`，共 24 MB）作为**纯视觉几何**挂进来：
`group=2`、`contype=0/conaffinity=0`，碰撞仍是 box/cylinder/sphere（`group=3`，默认不显示）。物理量一条不变：
静态对账仍 89/89、golden 三对齐仍 PASS、同一场景 z/vx 与无网格版逐位相同。

两个 MjSpec 的坑（已在脚本里绕过）：
1. **必须先 `spec.compile()` 再 `spec.to_xml()`**，否则 `<asset>` 与 mesh geom 整段被静默丢弃；
2. 在 MjSpec 里同时把 `contype` 与 `conaffinity` 设成 0，`compile()` 会把该 geom 丢掉；
   改成在 XML 里写这两个属性（`*_vis` 几何）才保留。

## 已知差异（登记，不修）

- 轮碰撞：Isaac `replace_cylinder_with_capsule=True`（capsule）vs MuJoCo URDF 原生 cylinder。
- 接触模型：PhysX TGS vs MuJoCo soft-constraint；静态沉降已对上（0.7 mm），不做接触调参。
- 从动轮关节阻尼叠了两层：URDF `0.01`（物理）＋控制器 `-0.02·q̇`（力矩，clip ±1.0）。
