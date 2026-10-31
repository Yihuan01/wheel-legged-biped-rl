# legged_gym/envs/wbr/flat —— 两轮：行走与跳跃

| 文件 | 用途 |
|---|---|
| `wbr.py` | 两轮环境 `WBRFlatMob`：观测拼装、奖励、指令目标层（包络 / 压弯 / 限速）、跳跃相位机、遥测快照 |
| `wbr_flat_config.py` | 任务 `wbr_flat` 的配置（161 维观测 / 6 维动作 / 平地），其余 WBR 配置都从它继承 |
| `wbr_terrain_config.py` | 同一策略的四个地形任务配置：`wbr_terrain` / `wbr_demo` / `wbr_combo` / `wbr_elements` |

地形任务共用 `wbr.py` 的环境类，只替换 terrain 段，观测仍是 161 维（不含高程）。
