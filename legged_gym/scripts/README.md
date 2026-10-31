# legged_gym/scripts —— 训练、回放与控制台入口

| 文件 | 用途 |
|---|---|
| `train.py` | 训练入口：按 `--task` 建环境与 PPO runner，落盘到 `logs/<实验名>/` |
| `play.py` | 上游回放入口（无 GUI 面板），加载 checkpoint 后跑策略 |
| `play02.py` | 遥测采集端：面板下发指令 → 每控制步采样 → 每 10 s 落一个 TSV；`--play_mode switch` 走模式切换 |
| `plot02.py` | 遥测出图端：读 play02 落的 TSV，生成自包含 HTML |
| `check_geometry_4w.py` | 几何预检：URDF 结构 + 运动学五项检查 + 起立扫掠（纯 numpy） |
| `check_geometry_figs.py` | 几何预检配图：3D 骨架、起立扫掠与交互 HTML |
| `check_flat4_eval.py` | 四轮模式批量评测 |
| `check_switch_corridor.py` | 切换走廊标定：扫起立 / 趴下的时间窗 |
| `check_flat4.py` | 四轮任务冒烟：建 env 跑几步 |
| `wbr_console/` | 本地网页控制台（独立部署），见 `wbr_console/README.md` |
| `wbr_play_data/` | play02 的遥测输出目录（本地数据，不入库） |

## 常用命令

```bash
python legged_gym/scripts/train.py --task wbr_flat --headless
python legged_gym/scripts/play02.py --task wbr_flat --from_run_config releases/W3_vlim085
python legged_gym/scripts/plot02.py            # 打开生成的 HTML 图表
```

注意：play02 / plot02 与控制台里的 `deps/*_core.py` 是同一份逻辑的两份副本，
改其中一份时另一份需同步，否则控制台与命令行会给出不同结果。
