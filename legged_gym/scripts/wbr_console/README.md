# WBR Console（WBR 控制台 · 独立部署版）

WBR 综合试验台的本地网页控制台。**P2/P3/P4 已完成**（启动与控制 / 原生窗口停靠 / 绘图页），
**P5 独立化完成（2026-10-26）**：整个控制台连同它需要的 play/plot 核心打包在**本目录内**，
不再依赖 `scripts/play02.py` / `scripts/plot02.py`（两份原文件保持一行不改；2026-10-26 已裁「保留不删」）。

> **独立化方式**：`deps/play02_core.py` 与 `deps/plot02_core.py` 是 `scripts/play02.py`、`scripts/plot02.py`
> 的逐字副本（函数体未动），差异只有 `DATA_DIR` / `REPO_ROOT` 改为按 `__file__` 推导、顶部加了出处说明。
> 改命令行版时需同步复制到本目录，否则两边结果会不一致。

## 目录

```
wbr_console/
├── app.html / app.css / app.js   前端（3 页签 + 3 弹窗；后端不可达时自动退回 mock 离线预览）
├── wbr_bench.py        后端：静态站点 + 试验台 API + 控制桥代理（纯标准库）
├── wbr_bench_run.py    仿真进程入口：from deps import play02_core as play02，
│                       替换 start_gui_thread / _build_switch 后跑 play02.play02(args)
├── bench_bridge.py     仿真进程内 HTTP 控制口（读写 gui_cmds/mode_shared/jump_flags/readout）
├── bench_store.py      存储层：settings / scenarios / saved / runs（STORE_DIR=本目录内）
├── bench_models.py     checkpoint 扫描 + obs/act 探测（批量子进程 + mtime 缓存）
├── bench_plot.py       绘图数据层：from deps import plot02_core，补 switch 面板
├── deps/
│   ├── play02_core.py  play02.py 逐字副本（差异仅 DATA_DIR/REPO_ROOT 绝对路径 + 出处头）
│   └── plot02_core.py  plot02.py 逐字副本（差异仅 DATA_DIR 绝对路径 + 出处头）
├── samples/            示例遥测（离线预览用）
├── xwin.py             原生窗口停靠：ctypes 直调 libX11
├── run_bench.sh        一键启动（自动选 conda python）
├── wbr_bench_store/    数据目录（settings.json / scenarios/ / saved/ / runs/）
└── shots/              实机截图
```

## 启动

```bash
cd legged_gym/scripts/wbr_console
./run_bench.sh                # → http://127.0.0.1:8770
# 或
python3 wbr_bench.py --port 8770 --bridge-port 8771
```

后端本身是纯标准库，可用系统 python3；探测 checkpoint 维度会自动调用
一个带 torch 的解释器（如本机 conda 环境里的 python）。用 `WBR_PYTHON` 指定，缺省用当前 `python3`。

**仿真必须独占显卡**：检测到 `legged_gym.scripts.train` 进程时会拒绝启动，界面上可二次确认强制启动。

## 架构

```
浏览器 :8770
  ├─ 静态页 /api/status /api/models /api/scenarios /api/saved /api/runs /api/tellogs
  └─ /api/start /api/stop /api/preflight /api/cmd /api/action
        │  （/api/cmd、/api/action 代理到 ↓）
        ▼
wbr_bench.py（试验台服务，纯标准库）
        │  subprocess
        ▼
wbr_bench_run.py ── from deps import play02_core as play02（原 scripts/play02.py 不参与）
                    └─ play02.start_gui_thread = bench_bridge.install（桥与副本共享同一模块对象）
                    └─ play02._build_switch 运行时包装（注入衔接参数 + 腿 P 增益倍率 + 传统起立参数）
        ▼
仿真进程 :8771（bench_bridge）
   读 readout → /api/state；写 gui_cmds / mode_shared / jump_flags
```

**为什么不用改 play02**：它把面板与主循环解耦成四个模块级字典
（`gui_cmds` 指令、`mode_shared` 模式切换/复位、`jump_flags` 跳跃、`readout` 读数），
`start_gui_thread` 是模块全局函数 —— 运行时替换即可，主循环一行不动。
独立化后 `bench_bridge` 也改为 `from deps import play02_core`，保证桥读写的是**同一个**模块对象
（若仍 `import play02` 会绑到 scripts/ 原文件，共享字典断线——独立化时排掉的雷）。

## 起立方案：P7T 传统起立（脚本链）与 P7S 快速起立（学习策略）

`MODE_TRAD_RISE` 已落进 `deps/play02_core.py`（副本，原 play02.py 不动），配方逐条对齐
2026-10-25 harness 验证（`/tmp/chain.py`，num_envs=1/3/6 全过）：

**链路**：4W 静稳 0.6s → smoothstep 折蹲 0.5s（hip ±1.5391 / knee ∓2.2722）→ 蹲稳 0.4s
→ smoothstep 伸腿到 2W 0.5s（腿 P ×`trad_boost`，轮零矩）→ 恒力矩 C=`trad_torque` 抬升
（门限 p_g>−0.26 rad 且 z>0.20，实测 ~0.2s）→ 交 W3（hcmd=`trad_hcmd` 0.22 保持 `trad_hcmd_hold` 3s 后回常规 0.28）。
harness 三处修复一并落地：交 W3 前恢复原 `_compute_torques`、`env.commands=command_targets` + 刷观测、
进入时 `_dr_act_delay/_dr_obs_delay` 清零。**复位键随时可中断**（自动卸力矩补丁、恢复名义腿增益）。

**场景接入（纯参数，不改代码）**：场景 `params` 里 `standup_mode: "trad"` + 上述 `t_trad_*/trad_*`
（缺省值＝harness 验证值，全部可省略）；**没有 `standup_mode` 时走 P7S 学习策略，行为不变**。

| 场景 | params | 实测（2026-10-26） |
|---|---|---|
| `P7T_传统起立` | `standup_mode=trad` + trad 参数 | 4W→TWOW **3.57s**（含静稳/折蹲全程；交 W3 时 p_g −14.8°、z 0.260，与 harness 一致）；趴下 2.64s；连续两轮循环 ✅ |
| `P7S_快速起立` | `t_rise=0.2`、`t_handover=0.05`、`leg_p_gains=2.0` | 4W→TWOW **1.06~1.08s**（默认参数基线 2.54s）；趴下 2.66~2.85s |

> `leg_p_gains`（P7S 代理用）＝`env.p_gains[腿]` 在 SP_RISE/SP_HANDOVER 相位 ×倍率，其余相位恢复名义值
> （`wbr_bench_run.py` 包装 `env.playback_set_mode` 实现，只动运行时张量）。
> 已知未测边界：交 W3 满 3s 时 hcmd 0.22→0.28 的切换未单独验证（实测两次从 0.22 窗口出来后
> z 0.24→0.30 平滑站高，无异常）；人工复验起立过程时留意此处。

**前端参数卡（2026-10-26 补全，两处 UI 缺陷修复）**：模型配置页「衔接参数」卡含**起立方案下拉**
（P7S 学习策略 / P7T 传统脚本链）与全部 `trad_*`/`leg_p_gains` 字段（trad 组在 P7S 下灰显）。
此前 `buildSpec` 只认老 9 键，UI 启动会把 `standup_mode`/trad 参数/`leg_p_gains` 全丢掉——**两个场景
都跑成默认 P7S**（「两个场景一样」的根因之一）；另外刷新页面时活动场景只换 pill 文字、参数回落
settings 旧值（根因之二），均已修复并走全 UI 链路复验（见实测记录）。

## 页面

| 页签 | 内容 |
|---|---|
| **仿真运行** | 上方仿真视口（填充剩余高度，P3 停靠原生窗口；工具条含**相机跟随开关**，默认关）；下方控制台（5 路滑块 **Vx / Yaw / Height / Roll / Jump**（首字母大写）+ in/tgt/act 三联读数、**每路滑块两侧显示当前加载模型的实际指令量程（黑色）**，四轮 h[0.175,0.245]/roll 固定、两轮 h[0.2,0.35]/roll±0.2，量程退化的通道**整卡置灰**；起立/趴下/复位/装填跳、连跳、收藏当前、状态条）；底部抽屉（运行记录 / 控制台输出 / 复现命令）。**键盘**：W/S 前后、A/D 偏航、**I/K 体高 +/−**、J/L 侧倾、**H 装填跳、空格 指令重置到当前模式默认位**、M 切模式、R 复位、1/2/3/4 切页签（输入框内不触发，数字键按导航顺序） |
| **数据绘图** | 真实遥测看图（复用 plot02_core）：时间窗 + body_z 缩略轴（可拖）+ 子图网格；单策略 18 图、双模式 9 图（模式色带，传统起立与起立中同色）；顶部「模型配置」胶囊显示这份录像用的权重（双模式列出四轮/起立/两轮/趴下四个模型，传统起立标「P7T 传统脚本链」）；支持整页/分图导出、自动跟随 |
| **模型配置** | 左右分区：**左侧**「启动预检」+「场景管理 / 检查点库」；**右侧**「模型配置」——双模式、单策略上下两张卡，双模式卡内含**可编辑的衔接参数**（起立方案、相位时长、P7S/trad 参数）。场景/固化存档条目**双击即加载**（场景→套用；存档→加载到单策略）；每条最右两个图标：**笔**（悬停「覆写到本条」）弹保存窗口、命名确认后用当前右侧内容覆写该条，**垃圾桶**（悬停「删除本条」）弹确认后删除。单策略区有「固化检查点」按钮（命名后把当前 checkpoint 存入检查点库） |
| **使用说明** | 参数说明 / 功能说明 / 快捷键说明 / 存储路径说明（四个区域标题行浅蓝底，无编号；取代原「?」帮助弹窗） |

> **配色与导航位置**：导航默认在**左侧**——左栏自上而下：图标+WBR Console、四个页签（字号放大、未选中黑字、**选中蓝底**）；仿真状态、导航位置、三个配色**贴到左栏最底部**，其中导航位置/配色**选中为灰底**。切「顶部导航」回到顶栏、切「右侧导航」镜像到右栏。均存 `localStorage`。

> **衔接参数（可编辑）**：`t_align/t_rise/t_handover/t_lower/t_tilt/t_absorb`（对齐/起立/交接/趴下/倾覆/缓冲，秒）是
> **真实相位时长**——启动时由 `wbr_bench_run.py` 运行时包装 `play02_core._build_switch`，把值写进
> `env.cfg.rewards.switch.*`，`run_switch` 的相位机按新值推进，**重启仿真生效**；
> 启动日志会打印「⚙️ 衔接相位参数已应用」（含 `leg_p_gains` 时一并打印）。
>
> 【2026-10-26】原 `t_hold4w / t_hold2w / blend` 与「循环 / 自动演示」开关**后端并不消费**，已从参数卡移除
> （真正连跳走 `jump_flags`）；现参数卡＝起立方案 + 6 个相位时长 + `leg_p_gains`(P7S) + 9 个 trad 参数。

## 原生窗口停靠（P3）

运行页的仿真视口不抓流：把 Isaac Gym 的**原生窗口**去边框、置顶、按视口矩形摆好，帧率保持原生。
实现全在 `xwin.py`（ctypes 直调 libX11）：找窗口（标题含 `Isaac Gym` 且面积最大）→
`_MOTIF_WM_HINTS` 去边框 → `_NET_WM_STATE` 置顶 → `XConfigureWindow` 定位 → 切页签 `XUnmapWindow` 降下。
行为：窗口一出现自动停靠；视口变化 250ms 防抖重贴；拖动每 2s 比对矩形自动跟上；切页签自动降下。

> **什么时候能用**：页面必须跑在**真实浏览器窗口**里（`--app` 窗口或普通标签页）。
> OpenChamber 浏览器面板是嵌入式 webview，实测 `screenX/outerWidth/outerHeight` 全为 0，页面据
> `outerWidth/outerHeight<200` 判定并提示「拿不到屏幕坐标」；这种视图下点「独立窗口」，后端起一个
> `microsoft-edge --app=...` 真窗口，在那儿自动停靠。
>
> 【2026-10-26 修】旧判据用 `outer-inner` 差值（chromeX>60||chromeY>160）判定嵌入面板，会**误杀真浏览器
> 标签页**（Edge 垂直标签页侧栏让 chromeX 轻松 >300）⇒ 表现为「标签页里不能停、独立窗口能停」。现改为
> 看 `outerWidth/outerHeight` 是否 ≥200（嵌入 webview 恒 0，决定性特征），矩形下方是否真有窗口由后端
> `rect_inside_window`（X11 实测）兜底。后端 dock→undock 已用真实 X11 坐标实测通过。

**相机跟随（30Hz，可手动开关，默认关；2026-10-26 定档）**：play02 只在第 1 步设一次相机。`bench_bridge.install` 运行时包装
`env.step` 与 `env.playback_reset`，**仅当开关打开时**按**时间间隔 1/30s** 调 `env.set_camera` 跟住机器人（时间基准，与仿真步频无关）。
开关在后端 `_STATE['camera']`（默认 False），运行页视口工具条「相机跟随：开/关」按钮经 `/api/action {action:'camera',value}` 切换。

## 绘图页说明（P4）

- 数据层 `bench_plot.py` 用 `deps/plot02_core.py`：复用 `Index`（行索引/增量扫描）、
  `read_overview`（缩略轴包络）、`read_window`（单策略时间窗切片）、`KIND`（曲线分类配色）。
- 四槽日志（`wbr_switch_tel_*.tsv`）没有关节列 ⇒ 另给一套 9 图面板（3 组 × 3 图）：
  线速度/偏航/高度跟随、机身高度/俯仰/模式色带+相位、触地轮数/从动轮离地/起立锁存，共 18 条曲线。
- 遥测文件仍落在 `scripts/wbr_play_data/`（`play02_core.DATA_DIR` 绝对路径指定）。

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/status` | 仿真进程状态、GPU、训练进程检测 |
| GET | `/api/state` | 代理控制桥：模式/相位/指令 in-tgt-act/遥测/params 回显 |
| GET | `/api/models` | 全量 checkpoint（含 obs/act，缓存） |
| GET | `/api/scenarios` `/api/saved` `/api/runs` `/api/tellogs` | 场景 / 固化 / 运行记录 / 遥测文件 |
| GET | `/api/plot/list` `/api/plot/meta` `/api/plot/overview` `/api/plot/window` `/api/plot/kinds` | 绘图数据 |
| GET | `/api/window` | 原生窗口是否找到 + 位置尺寸 |
| POST | `/api/window/dock` `{x,y,w,h}` | 去边框+置顶+移到该矩形（**设备像素**） |
| POST | `/api/window/undock` `/api/window/reset` | 降下 / 恢复 |
| POST | `/api/app/open` `{url}` | 用 `--app` 开一个真浏览器窗口（嵌入式面板里停靠的出路） |
| POST | `/api/plot/save` | 分图 PNG 存到 ~/Downloads |
| GET | `/api/runlog?id=` | 某次运行的 console.log 尾部 |
| POST | `/api/start` `{spec,force}` | 起仿真（先预检；训练占用默认拦截） |
| POST | `/api/stop` | 停仿真（SIGTERM→SIGKILL，先收仿真防孤儿） |
| POST | `/api/preflight` `{spec}` | 预检：权重存在/维度匹配/显卡/训练进程 |
| POST | `/api/cmd` `{key,value}` | 下发指令（代理到桥） |
| POST | `/api/action` `{action,value}` | toggle/rise/lower/reset/armjump/auto |
| POST/DELETE | `/api/scenarios` `/api/saved` | 保存/删除场景、固化/删除存档 |

## 实测记录（2026-10-25，旧目录 wbr_bench）

- 1182 个 checkpoint 首次探测 6.2s（批量子进程），二次 0.01s（缓存）。
- switch 档全链路：四轮行驶 → 起立 → 两轮站立（pitch −5.0°、z 0.30、ev_upright ✓）→ 趴下 → 回四轮，全程网页按钮驱动。
- 衔接参数 A/B（重启生效）：默认（t_rise 0.6/t_lower 0.5）起立 2.54s/趴下 2.65s；改 1.6/1.5 → 4.19s/4.19s。
- 窗口停靠真机验证：`--app` 窗自动停靠、几何一致、切页签 IsUnMapped、拖动 1s 内跟随、重启自动重停。
- 停止后进程与显存正常释放（5.3G → 2.4G 基线）。

## 实测记录（2026-10-26，独立部署版 wbr_console）

- **独立化打包**：`deps/play02_core.py`（61KB）与 `deps/plot02_core.py`（47KB）为原文件逐字副本；
  imports 全部指向 deps；存储收进 `wbr_console/wbr_bench_store/`。排雷 3 处搬迁陷阱：
  `plot02_core.DATA_DIR`、`play02_core.REPO_ROOT`（`__file__` 上溯层级错位 → 绝对路径）、
  `bench_bridge` 的 `import play02`（会绑到原文件，共享字典断线 → 改 deps 别名）。
- **P7T 场景端到端**：`/api/start`（slots+params）→ 日志打印「⚙️ 衔接相位参数已应用：t_rise=0.2,
  t_handover=0.05, leg_p_gains=2.0（起立/交接相位腿 PD ×2.0）」+「🦵 腿 P 增益倍率已挂接」；
  起立 **1.06s**（基线 2.54s）、趴下 2.85s；`/api/state` params 回显正确。
- **绘图层**：`/api/plot/list|meta|overview|window` 全通（9 图 3 组 18 曲线、1250 桶缩略轴、meta 含四槽配置）。
- **single 档冒烟**：`wbr_flat4` + `logs/wbr_flat4/Oct25_11-35-01_P7F3e`（含 load_run）6s 建好 env，
  桥与遥测正常，干净停止。注意：single 预设必须给 `load_run`（否则 train_cfg 默认 7 月旧 run 找不到权重），
  且 `from_run_config` 指向**带 config.json 的训练 run 目录**（releases/ 精简目录没有 config.json）。
- **截图**：`shots/console-p7t-running.jpg`（运行态：P7T 场景、遥测 5998 行、GPU 5.2/8G）。
- **传统起立 `MODE_TRAD_RISE` 落地**（2026-10-26 下午，作者裁「①做」）：场景 `P7T_传统起立` 端到端——
  交 W3 时 p_g −14.8°、z 0.260、抬升仅 0.17s（与 harness 完全一致），4W→TWOW 全程 3.57s；
  趴下 2.64s；连续两轮循环 ✅；hcmd 0.22 保持窗后 z 0.24→0.30 平滑站高。原代理参数另存为场景
  `P7S_快速起立`；P7S 回归测试（无 `standup_mode`）1.08s 走 RISING 分支不受影响。
- **相机 30Hz**（作者定档）：`bench_bridge` 相机更新改时间基准 1/30s（原 20 步 ≈3Hz）。
- **single 预设修正**（作者裁「④做」）：`settings.json` 的 single 从 `releases/P7L_liedown`（无 config.json，起不来）
  改指 `logs/wbr_flat4/Oct25_11-35-01_P7F3e`（wbr_flat4 / F3e 164 维，含 config.json；UI 的 run 字段即 load_run）。
- **原文件处置**（作者裁「③不删」）：`scripts/play02.py`/`plot02.py` 保留不删。
- **两轮模式 height/roll 可调 + 滑块范围显示 + 停靠误判修复**（2026-10-26）：
  - 根因：`run_switch` 两轮分支把 `in_h` 硬编码 0.28、`base_roll_targets` 恒 0；且 flat4/switch 的
    `ang_roll=[0,0]` 使 `_update_turning_roll_targets` 每步把 roll 目标钳成 0。修法：两轮分支改吃
    `gui_cmds['base_height']`/`gui_cmds['ang_roll']`（各夹 W3 训练量程 [0.2,0.35]/±0.2）；`_apply_mode`
    两轮时把 `ang_roll` 量程临时开 [-0.2,0.2]、回四轮/趴下收回 [0,0]；切模式时 `_snap_cmds_for` 把面板
    体高拽回该模式合理初值（进两轮 <0.2 抬到 0.28、回四轮 >0.245 收回 0.19）。`bench_bridge` 把真实
    `cmdRanges` 透给前端，滑块上下限与「范围标签」随之更新（本轮先按并集 vx/yaw ±1、height 0.175~0.350、
    roll ±0.2；**2026-10-26 傍晚 UI 大改后改为按当前模式量程**，见下方条目）。
  - 实测（GPU）：两轮站定后 h 设 0.30→act 0.315、0.22→0.230；roll 设 +0.15→act +0.130、−0.15→−0.137、
    回 0→0；readout roll 三联（in/tgt/act）随滑块实时更新 ✅。
  - 停靠：见「原生窗口停靠」章——旧 `outer-inner` 判据误杀真浏览器标签页已修，后端 dock→undock 用真实
    X11 坐标实测通过（Isaac 移到指定矩形、undock 隐藏、荒谬矩形 409）。
- **UI 链路验证与前端缺陷修复**（2026-10-26 下午，作者报「两个预置场景好像一样」后排查）：
  根因① `app.js buildSpec/persistParams` 只认老 9 键，UI 启动丢 `standup_mode`/trad 参数/`leg_p_gains`
  （API 直发不经过这条路径，所以此前 API 验证没暴露）；根因② `boot()` 刷新时活动场景只换 pill 文字、
  参数不套用。修复＝PARAM_DEFS 扩充（起立方案下拉 + trad 组 + leg_p_gains，trad 组 P7S 下灰显）+
  boot 套用活动场景 slots+params。复验（浏览器实测）：P7S 场景 UI 启动 → 日志 `leg_p_gains=2.0` 生效、
  走 RISING 1.1s ✅；P7T 场景 UI 加载（参数卡 trad 化）→ UI 启动 → 日志「🦿 传统起立参数已应用
  standup_mode=trad」→ UI 点起立 → **TRAD_RISE 3.2s → TWOW**，交 W3 p_g −14.8°/z 0.260 与 harness
  一致，hcmd 保持窗后 z 0.24→0.30 平滑 ✅。
- **UI 大改一轮**（2026-10-26 傍晚，作者逐条定稿）：
  - **页签**：运行→**仿真运行**、绘图→**数据绘图**、模型配置不变、新增**使用说明**页（参数/功能/快捷键/存储路径），
    删掉原「?」帮助弹窗；数字键 **1/2/3/4** 切页签。
  - **快捷键**：跳跃改 **H**（原空格）；**空格＝指令重置到当前模式默认位**（滑块与下发值一起回：四轮 h=0.19、
    两轮 h=0.28，vx/yaw/roll 归 0，机器人随指令回默认站姿）；新增 **J/L 调 roll**。
  - **范围显示**：每路滑块**两侧贴当前量程**（左低右高）；量程由后端 `_cmd_ranges()` 按 env 当前 `command_ranges`
    给（vx/yaw/jump 用播放量程，height/roll 随模式动态），**量程退化（min==max）的通道自动置灰**
    （四轮 roll=[0,0] 灰、两轮 ±0.2 可用、jump 四槽灰）。四轮 h 上限 0.245、两轮 0.35，非并集。
  - **死参数清理**：删 `t_hold4w/t_hold2w/blend` 与「循环/自动演示」开关（后端不消费）。
  - **圆角统一**：所有标签/按钮/胶囊由跑道形 999px 改为 **8px 圆角矩形**（开关轨道除外）。
  - **右侧导航**：切「右侧导航」时**整条顶栏隐藏**，页签 + 仿真状态 + 导航位置 + 三个配色**竖排收进右侧状态栏**，
    主内容上移到顶、不再占上方空间（`app[data-nav=right]`）。
  - **绘图**：`MODE_BANDS` 补 mode=4（传统起立）同「起立中」色。
  - **停靠诊断**：新增 `/api/window/list`（列本机可见顶层窗口）+ 前端 **Ctrl+Alt+Shift+D** 打印页面几何与窗口列表；
    停靠失败时角标显示算出的视口坐标便于现场核对。
  - 实测（GPU）：四轮 `cmdRanges` h[0.175,0.245]/roll[0,0]（roll 滑块灰）、起立 0.90s→两轮 h[0.2,0.35]/roll±0.2
    （roll 解禁，in/tgt 自动 0.28）；J/L 设 roll 0.12→act 0.103；h 0.30→act 0.322；回四轮量程复原；右栏竖排布局桌面核对 ✅。
- **导航左置 + 卡片/使用说明微调**（2026-10-26 傍晚，作者三条定稿）：
  - **导航改左侧**：默认左侧栏，自上而下＝图标+WBR Console、四个页签（字号 0.92rem、未选中黑字、**选中蓝底**）；仿真状态、导航位置（顶部/左侧/右侧三选）、三个配色**贴到左栏最底部**，导航位置/配色**选中灰底**。顶栏在左/右侧模式下整体隐藏；`data-nav` 支持 left/right/top，控制块随位置搬进/搬出侧栏（`margin-top:auto` 贴底）。
  - **指令卡**：标签首字母大写（Vx/Yaw/Height/Roll/Jump）；范围数字改黑色；去掉 Jump 的「四槽未用」小标签；不可用通道**整卡置灰**（`.chan.off`）。修复两个隐患：①`off` 由 `undefined` 改为严格布尔，避免 `classList.toggle('off', undefined)` 变成每帧翻转；②去掉 `.chan` 的 `opacity .15s` 过渡（后台标签页里过渡不推进，导致置灰算成 1 不生效）。可用性每帧重算（不再用签名门控）。
  - **使用说明页**：四个区域去掉①~④编号；每个标题行加**浅蓝底**（`color-mix(primary 10%, card)`，与模型配置区域标题行同款）；存储路径「权重 / 控制台设置」两行的说明改为换行、不再与 `<b>` 标签混排，修好对不齐。
  - 实测（GPU）：四轮起立 1.10s→两轮；四轮 roll 卡 opacity .45（禁用）、两轮 opacity 1 且量程 ±0.20（可用）、jump 恒灰；左/顶/右三种导航位置切换正常；使用说明标题行 `color(srgb .915 .939 .992)` 与模型配置标题行一致 ✅。
- **Height 默认对齐 + 体高键 I/K + 页签顺序 + 相机开关 + 停靠纠偏**（2026-10-26）：
  - **Height 起立后没跟默认**：根因＝`_snap_cmds_for(MODE_TWOW)` 是条件式（`base_height<0.2` 才抬到 0.28），若进两轮前手调过四轮体高（如 0.22）就不满足条件、面板 Height 停在旧值、与机器人实际站高（0.28）不符。改为**无条件对齐**：进两轮一律 0.28、回四轮/其它一律 0.19。实测：四轮手动设 0.22 → 起立 0.88s → 两轮 `cmd.h in=0.280 / tgt=0.280 / act=0.293`，面板同步 0.280 ✅。
  - **体高键改 I/K**：`i` 升高（+0.005）、`k` 降低（−0.005）；去掉旧 Q/E 绑定（WASD 移动/偏航保留，J/L 侧倾不变）。
  - **页签顺序**：模型配置第二、数据绘图第三（仿真运行 / 模型配置 / 数据绘图 / 使用说明）；数字键 1–4 按导航顺序切换（改为从 TABS 顺序派生，改顺序无需再改键码）。
  - **相机跟随可手动开关、默认关闭**：后端 `_STATE['camera']=False`，`_step`/`reset` 仅在开时调 `set_camera`；运行页视口工具条「相机跟随：开/关」按钮 + `/api/action {action:'camera'}`。实测 UI 点击 → `camera: True/False`、按钮文案与高亮同步 ✅。
  - **仿真窗口偶尔靠左偏**：`vpRectDevice` 原来一律按 `screenX + chromeX/2` 估窗口内容区原点，**最大化窗口**时左右边框估算会偏；改为检测最大化/贴边（`outerWidth≈screen.availWidth` 或 `screenX/screenY≈0`）时直接用 `screenX`。另加**有界自校正**：`dockWindow` 拿到 X11 回报的实际左上角与目标比对，偏差 >6px 就累计校准偏移重贴（≤3 次）。真浏览器实测待现场（嵌入面板拿不到屏幕坐标，本环境无法端到端验证停靠像素）。

## 实测记录（2026-10-27，双模式更名 + 控制台清理 + 场景/检查点库条目交互）

- **命名统一「双模式」**（`fed69cb`）：原「多模式」全站改为「双模式」——运行页模式胶囊、模型配置「当前模式」与卡片标题、场景列表标签、绘图日志标签与类型徽章、使用说明、README，各处一致。
- **控制台清理**（`fdb4496`）：删除 10-25 旧重复版 `scripts/wbr_bench/`（32 文件，无引用、已被 `wbr_console` 完全取代）与旧遗留 store `scripts/wbr_bench_store/`（11 条 10-25/10-26 旧运行）；其 `mock.js/mock.json` 并入 `wbr_console`（`app.html` 本就引用 `mock.js`，原先缺失 404，现修复离线预览）。运行记录 `wbr_console/wbr_bench_store/runs/` 由 54 条清到**最近 5 条**；`bench_plot.STORE_RUNS` 收敛为控制台自身目录（去掉旧 store 兜底）。控制台体积 8.9M→7.4M。
- **场景 / 检查点库条目交互**（`75b8765`）：
  - **双击即加载**：场景→套用该场景（四槽/单策略）；固化存档→加载到单策略（用记录源路径 + 检查点号，不额外复制显存）。
  - 每条右侧两枚**图标按钮**：**笔**（悬停「覆写到本条」）弹「保存场景」窗口、名称预填本条，命名确认后用当前右侧编辑区（场景）或当前单策略 checkpoint（存档）覆写该条；**垃圾桶**（悬停「删除本条」）弹确认后删除。
  - 单策略区新增**「固化检查点」**按钮：命名后把当前 checkpoint 固化进检查点库。
  - 后端 `bench_store.solidify` 增 `folder` 参数（覆写沿用旧目录名）、`/api/saved` 支持 `name`；新增通用「保存/覆写」与「确认删除」两个 dialog；使用说明/README 同步。
  - 实测：四类弹窗标题/预填/确认文案正确（覆写场景：LQR起立R81跳跃 / 确定删除场景 P7S_快速起立 吗？/ 覆写检查点：… / 固化检查点 预填 `Oct25_11-35-01_P7F3e__model_11800`）；后端覆写单测通过（同目录覆写、清理无误）；预检 4/4、5 场景 + 1 固化存档齐全、无 JS 报错。

## 后续阶段

| 阶段 | 内容 |
|---|---|
| ~~P3~~ | ~~原生窗口停靠~~ 已完成 |
| ~~P5~~ | ~~独立化打包 + P7T 参数代理 + 测试 + 文档~~ 已完成（2026-10-26） |
| ~~P7~~ | ~~真·传统起立 `MODE_TRAD_RISE`~~ 已落进 deps 副本（2026-10-26，作者裁「①做」） |
| P6 | 可选：MJPEG 抓流（远程/兜底）——作者暂不需要（2026-10-26 裁） |
| 尾巴 | 真窗口人工复验 P7T 传统起立过程（下次作者来时提醒，重点看 hcmd 0.22→0.28 切换）；原 play02.py/plot02.py 保留不删（已裁） |
