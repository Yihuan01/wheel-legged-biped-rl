# WBR P8 地形文件

本目录的 `.npz` 全部由**纯 numpy 的离线生成器**产出，训练/回放侧只读图（不需要 GPU）。

| 生成器 | 产物 | 用途 |
|---|---|---|
| `wbr_rand_gen.py` | `release/wbr_rand_curve.npz`（seed0）、`release/s1/…`（seed1 留出） | **训练分布**：无向随机起伏，4 行 × 6 列 |
| `wbr_terrain_gen.py` | `release/wbr_p8_L{1,2,3}.npz` | **单元素验收图**：8 类元素 × 3 档（行 0 平地） |
| `wbr_composite_gen.py` | `release/wbr_p8_demo.npz`、`release/wbr_p8_combo_L{1,2,3}.npz` | **组合地形**：元素首尾相接的走廊 + 3 档 × 3 种排列 |
| `check_terrain_rand.py` / `check_terrain_elements.py` / `check_terrain_composite.py` | — | 各自的探针（必须通过才拿来用） |

## 地形任务（`legged_gym/envs/__init__.py`）

| task | 高度图 | 几何 | 说明 |
|---|---|---|---|
| `wbr_terrain` | `wbr_rand_curve.npz` | 4×6，5 m 格 | P8W 训练用（**未改动**） |
| `wbr_demo` | `wbr_p8_demo.npz` | 1×1，47 m × 8 m | **控制台演示走廊**：9 个元素沿 +x 依次排布 |
| `wbr_combo` | `wbr_p8_combo_L3.npz` | 3×3，20 m 格 | **组合地形**：行 = L1/L2/L3（课程升档），列 = 3 种元素排列 |
| `wbr_elements` | `wbr_p8_L3.npz` | 4×8，8 m 格 | **单元素验收**：行 0 平地 / 行 1~3 = L1~L3；列 = 8 类 |

- 三者观测同为 **161 维**（不放高程），**W3 权重可直接回放**。
- 控制台：模型配置页「单策略」→ task 选 `wbr_demo` / `wbr_combo` / `wbr_elements`，
  checkpoint 选 `releases/W3_vlim085/model_2200.pt`；`--from_run_config` 要指向 **W3 那个 run 目录**
  （`logs/wbr_flat/Oct22_00-08-53_W3`），不要指向地形 run（会把它自己的地形参数回填进来）。
- 评测：`terrain_eval.py --terrain-task wbr_demo`（缺省时长按 `terrain_length/|vx|×1.05` 换算，
  20 s 只适合 5 m 格）。

## 八类元素与三档（2026-10-29 渐进化）

**L3 = 各元素的"最高档"**（T1/T3/T4 统一打到侧向/峰高上限 0.12 m），L1/L2 在这条几何区间里过渡，
**不再出现"两档完全相同"**（旧版 T3 三档都是 0.12 m、T4 的 L2 = L3）。

| 元素 | L1 | L2 | L3 | 说明 |
|---|---|---|---|---|
| flat 平地 | 0 | 0 | 0 | 行 0 基线 |
| t1 左右高低不平 | 2 cm | 3 cm | 9\~12 cm | 抬到 "半宽 × tan10°" 的上限 0.353 m，再受 **10° 横向坡度** 限制 |
| t2 波浪 | ±2 cm | ±3 cm | ±4 cm | 波长 0.5/1.0/1.5 m |
| t3 单边横坡 | 2 cm / 3° | 4 cm / 5° | 12 cm / 8° | 三档都是"格边达到最大横滚需求"，只有坡度不同 |
| t4 纵向坡 | 4 cm | 8 cm | 12 cm | 峰高为主标称，坡度由坡长反算 |
| t5 台阶跳上 | 5 cm | 10 cm | 15 cm | 缓坡上台 → 一格主台沿 → 台面 → 缓降 |
| t6 台阶跳下 | 5 cm | 10 cm | 15 cm | 缓坡上台 → 台面 → 一格主台沿 → 缓降 |
| t7 跳石(浅坑) | 8 cm | 8 cm | 8 cm | 石墩 + 石间 −6 cm 浅坑（不是基类那种 −10 m 深渊） |

### 三条生成铁律（都是实测踩出来的）

1. **四边 1 m 收进地面**：格子边界带高度会让相邻格在格界处撞墙（实测 50\~150 mm）。
   现在所有元素都在格内四边做"每级 ≤1 格竖向"的收边，探针的"格间缝"判据是 **≤2 mm**。
2. **单个采样格（0.1 m）的竖向差 ≤ 12 mm**：0.1 m 采样的 heightfield 会把"1\~2 cm 的坡"
   退化成"1 格竖墙"。起伏类统一做了 `_avert_steep_step` 限幅。
   ⚠ **组合地形在控制台里仍只放"缓"的那一档**：demo 用 0.086 的台阶，120 mm 以上的竖向截断
   在 PhysX 里会开出假坡度（实测 demo 跑 30 s 存活 4/8、60 s 3/8）。
3. **别做单格竖墙**：80\~140 mm/格 的竖面在 PhysX 上不可靠。要"断崖"用 5\~10 格、每格 ≤17.6 mm
   的陡升（合计可达 0.1\~0.18 m）。

## 探针

```bash
python3 legged_gym/envs/wbr/terrain/check_terrain_rand.py            # 训练图（7 项）
python3 legged_gym/envs/wbr/terrain/check_terrain_elements.py         # 单元素图（8 项，含渐进化后的标称对照）
python3 legged_gym/envs/wbr/terrain/check_terrain_composite.py       # 组合地形 / 演示走廊（逐段）
```

三个探针当前全部通过（2026-10-29）。

## 关键约定（沿用）

- **观测不加高程块**：`measure_heights=False` ⇒ 观测仍 161 维，W 线权重可直接热启动；
  实机没有视觉/激光，加高程等于给策略一个不存在的传感器。
- **出生点 z 取中心均值**：在起伏地形上取 `max` 会让机器人悬空出生。
- **格与格连续**：随机起伏整图一次性生成后切格，不做逐格独立随机。
- **垂直分辨率 2 mm**：5 mm 下 2 cm 档只剩 8 个高度级，剖面呈阶梯状。
