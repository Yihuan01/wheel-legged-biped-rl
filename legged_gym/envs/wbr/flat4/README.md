# legged_gym/envs/wbr/flat4 —— 四轮模式

| 文件 | 用途 |
|---|---|
| `wbr_flat4.py` | 四轮环境 `WBRFlat4Mob`：在两轮之上加从动轮接触、纵向倾翻裕度与四轮域随机化 |
| `wbr_flat4_config.py` | 任务 `wbr_flat4` 的配置：164 维观测 / 6 维动作（2 轮 + 4 腿），复用 `wbr_v2.urdf` |

跳跃机构保留在本类中，但配置里 `enable_jump=False` 关死；`switch/` 与 `uni/` 都从它派生。
