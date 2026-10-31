# legged_gym/envs/wbr/uni —— 统一多模式环境

只做回放，不训练。单个 `wbr_uni` env 用一套相位编号完成四轮 / 起立 / 两轮 / 跳跃 / 趴下的切换。

| 文件 | 用途 |
|---|---|
| `phase.py` | 统一相位枚举（0-11）、相位 → 控制模式映射与相位名 |
| `fsm.py` | 相位状态机：由外部请求码与事件推进相位，含超时保护 |
| `wbr_uni.py` | 统一环境 `WBRUniMob`：多策略观测拼接与模式路由 |
| `wbr_uni_config.py` | 任务 `wbr_uni` 的配置（173 维观测） |
| `obs_builder.py` | wbr_v1（161 维）与 wbr_v2（164 维）的观测 / 动作重排及分段常量 |
| `rise_script.py` | 起立脚本：RISE 内分 HOLD → CROUCH → CROUCH_HOLD → EXTEND → LIFT → DONE |
| `demo_cycle.py` | 四策略连跑自检：四轮 → 起立 → 两轮 → 趴下 → 四轮，退出码 0 表示全通过 |
| `check_uni.py` | 统一环境冒烟：建 env 跑几步，确认不报错 |
| `test_*.py` | 按功能的单点测试（FSM / 起立脚本 / 跳跃 / 链式脚本 / 扰动） |
