# P7L_liedown — 两轮→四轮趴下（**混合部署的趴下槽**）

> task = `wbr_switch`，模型 `model_300.pt`
> sha256 `a6ca192cb1287b75781af0d89d5a8f04121a1f7ac5396bacb28ceb9d84dfac4e`
> 训练 run：`logs/wbr_switch/Oct25_17-46-44_P7L/`（本包含训练时 `config.json`，可 `--from_run_config` 复现）
> obs 173 / act 6；相位 `HOLD2W → LOWER → TILT → ABSORB → HOLD4W`

## 能力

| 项 | 实测 |
|---|---|
| 两轮站立 → 四轮 | 单机 **2.6~2.9 s**（链上四槽 2.34~3.0 s） |
| 末态 | 四轮四点触地、z **0.189**、pitch **+3.0°**、从动轮离地 |
| 稳定性 | 2026-10-28 C1 循环 ×5（共 6 轮）连续趴下**全通**，0 终止；每轮末态完全一致 |

## 前提（实测得出，必须遵守）

**必须先停稳再发趴下**：P7T 恒力矩抬升让交两轮时带 0.4~0.6 m/s 后滚，R81 需 1~2 s 刹住；
在还带着后滚时切 LOWER 会**连人带车翻过去**（2026-10-28 首测第 4/5 轮 z 0.09、pitch +86°）。
控制台/驱动统一用"停稳门"（`|vx|≤0.08` 连续 3 拍），与 C4 裁定"停止状态切、姿态回正再切"一致。

## 复现

```bash
python play02.py --task=wbr_switch --load_run=releases/P7L_liedown --checkpoint=300 \
  --from_run_config=releases/P7L_liedown
# 链上（四槽）：场景 P7T_脚本起立R81 + wbr_console/wbr_chain_check.py --cycles 5
```

## 备注

- 本包只归档**模型 + 训练 config**；`model_300` 是交付档（同 run 后续档未采用）。
- 四槽另三件：四轮 `releases/P7F3e_yaw1_dr15/`（V0.22）、两轮/跳跃 `releases/P6R81_goalv2/`（V0.21）、起立 `releases/P7S1b_standup/` 或相位脚本 `uni/rise_script.py`（P7T，无需权重）。
