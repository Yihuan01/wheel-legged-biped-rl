# legged_gym/envs/wbr —— WBR 环境

四种运行形态都从 `flat/` 派生，观测维度依次加宽：

| 目录 | 任务 | 观测 | 说明 |
|---|---|---|---|
| `flat/` | `wbr_flat`、`wbr_terrain`、`wbr_demo`、`wbr_combo`、`wbr_elements` | 161 | 两轮行走 + 跳跃；地形任务只换 terrain 段 |
| `flat4/` | `wbr_flat4` | 164 | 四轮静稳构型行驶，从动轮被动 |
| `switch/` | `wbr_switch` | 173 | 四轮 ⇄ 起立 ⇄ 两轮站立，7 个相位脚本过渡 |
| `uni/` | `wbr_uni` | 173 | 统一相位枚举 + FSM 的多模式环境，只回放不训练 |
| `terrain/` | — | — | 地形高度图生成器与探针，见 `terrain/README.md` |

## 文件（直接位于本层）

无。本层只有子目录，每个子目录在自己的 `README.md` 里列文件。

## 代码约定

- 文件头固定为 `Date / Author / File / Purpose` 五行块；`Purpose` 描述**当前**功能，改代码时同步改它。
- 注释只写"这段代码做什么"和不写就会踩的坑；改动原因与轮次记录交给 git。
