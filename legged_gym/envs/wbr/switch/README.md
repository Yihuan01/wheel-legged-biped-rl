# legged_gym/envs/wbr/switch —— 起立 / 趴下模式切换

| 文件 | 用途 |
|---|---|
| `wbr_switch.py` | 切换环境 `WBRSwitchMob`：单个 env 承载四轮与两轮两套策略，按相位推进脚本过渡 |
| `wbr_switch_config.py` | 任务 `wbr_switch` 的配置，并定义 7 个相位号与观测维度（164 + 9 = 173） |

相位顺序：`SP_HOLD4W → SP_RISE → SP_HANDOVER → SP_HOLD2W → SP_LOWER → SP_TILT → SP_ABSORB`。
回放时由 `play02.py` 每步写入相位（`playback_hold`），训练时相位机自动推进。
