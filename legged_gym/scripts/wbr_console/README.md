# WBR Console（WBR 控制台 · 独立部署版）

WBR 综合试验台的本地网页控制台。**P2/P3/P4 已完成**（启动与控制 / 原生窗口停靠 / 绘图页），
**P5 独立化完成（2026-10-26）**：整个控制台连同它需要的 play/plot 核心打包在**本目录内**，
不再依赖 `scripts/play02.py` / `scripts/plot02.py`（两份原文件保持一行不改；2026-10-26 已裁「保留不删」）。

**MuJoCo 引擎（Sim2Sim）已接入（2026-11-05，P1~P3）**：运行页左上角分段控件 `[Isaac Gym │ MuJoCo]`
选引擎，同一运行页驱动；配置 MJ 是主导航里的**「配置MJ」页签**（与快启卡共用同一份设置）。详见下文「MuJoCo 引擎」一节。

> **独立化方式**：`deps/play02_core.py` 与 `deps/plot02_core.py` 是 `scripts/play02.py`、`scripts/plot02.py`
> 的逐字副本（函数体未动），差异只有 `DATA_DIR` / `REPO_ROOT` 改为按 `__file__` 推导、顶部加了出处说明。
> 改命令行版时需同步复制到本目录，否则两边结果会不一致。

## 目录

```
wbr_console/
├── app.html / app.css / app.js   前端（5 页签：运行/配置Isaac/配置MJ/绘图/说明；后端不可达时自动退回 mock 离线预览）
├── wbr_bench.py        后端：静态站点 + 试验台 API + 控制桥代理（纯标准库；ENGINES 表分引擎）
├── wbr_bench_run.py    Isaac 仿真进程入口：from deps import play02_core as play02，
│                       替换 start_gui_thread / _build_switch 后跑 play02.play02(args)
├── bench_bridge.py     Isaac 仿真进程内 HTTP 控制口（HTTP 骨架来自 bridge_common）
├── bridge_common.py    控制桥共享 HTTP 骨架：路由/端口扫描/SIGINT 自杀（Isaac 与 MuJoCo 桥同一份协议实现）
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
wbr_bench.py（试验台服务，纯标准库；ENGINES 表按 spec.engine 分支）
        │  subprocess（按引擎选解释器：isaacgym01 / mujoco env）
        ├─ engine=isaac ─► wbr_bench_run.py ── from deps import play02_core as play02
        │                   └─ play02.start_gui_thread = bench_bridge.install（桥与副本共享同一模块对象）
        │                   └─ play02._build_switch 运行时包装（衔接参数 + 腿 P 增益倍率 + 传统起立参数）
        └─ engine=mujoco ─► sim2sim_mujoco/serve.py（一核两壳，见其目录 README）
                            └─ sim_loop.SimLoop（限速→obs→actor→200Hz 力矩）+ mj_bridge 控制桥
        ▼
仿真进程 :8771（bench_bridge / mj_bridge，同一份协议）
   读 readout → /api/state；写 gui_cmds / mode_shared / jump_flags
```

**为什么不用改 play02**：它把面板与主循环解耦成四个模块级字典
（`gui_cmds` 指令、`mode_shared` 模式切换/复位、`jump_flags` 跳跃、`readout` 读数），
`start_gui_thread` 是模块全局函数 —— 运行时替换即可，主循环一行不动。
独立化后 `bench_bridge` 也改为 `from deps import play02_core`，保证桥读写的是**同一个**模块对象
（若仍 `import play02` 会绑到 scripts/ 原文件，共享字典断线——独立化时排掉的雷）。

## 起立方案：P7T2 相位脚本（默认）/ P7T 恒力矩（回退）/ P7S 学习策略

`MODE_TRAD_RISE` 已落进 `deps/play02_core.py`（副本，原 play02.py 不动）。

**P7T2 链路（2026-11-07 定版，`standup_mode=script` 或 MuJoCo 五槽链）**：
4W 静稳 `t_trad_hold` → smoothstep 折蹲 `t_trad_crouch`（hip ±1.5391 / knee ∓2.2722）→
蹲稳 `t_trad_crouch_hold` → smoothstep 伸腿 `t_trad_extend` 到 **W3 稳态腿姿 (1.415, −2.113)** →
抬升。**轮矩律从折蹲第一拍就生效**：`u = −(kp·pitch + kd·rate) + kv·vx`
（pitch 正=后仰、u<0=机体前翻；缺省 kp 12 / kd 3 / kv 14，MuJoCo 侧钳位 kp∈[8,16] kd∈[4,6] kv∈[4,10]
并另加 `u_bias` 折蹲前馈）。**交棒＝三闸驻留**：|pitch| ≤5.7° ∧ |rate| ≤0.5 ∧ |vx| ≤`trad_vx_tol` 连续 0.1 s，
保险丝 `t_trad_lift_max` 兜底。
（旧版 `grav_k·|sin(pitch)|+grav_c` 恒正＝前驱，实测只前移不翻身；`const` 剖面＝旧恒力矩 C 保留为回退。）

**场景接入**：`standup_mode` 三选一 —— `p7s`（学习策略）/ `trad`（P7T 恒力矩）/ `script`（P7T2，默认场景用它）；
`P7T2_五槽链_P8` 即 script 档。参数见模型配置页「② 起立 · 相位脚本 (P7T2)」组。

| 场景 | params | 实测 |
|---|---|---|
| `P7T2_五槽链_P8` | `standup_mode=script` + 上表参数 | Isaac：起立 2.8~3.2 s、净位移 −0.24 m、交棒后两轮 z 0.28 稳站；MuJoCo：起立 2.85 s（C1×3 全过） |
| `P7T_脚本起立R81` | `standup_mode=script`（R81 作两轮槽） | 对照验收用（两轮为 R81，不是 P8W7） |
| P7S 档 | `t_rise=0.2`、`t_handover=0.05`、`leg_p_gains=2.0` | 4W→TWOW 1.06~1.08 s（默认参数基线 2.54 s） |

> `leg_p_gains`（P7S 代理用）＝`env.p_gains[腿]` 在 SP_RISE/SP_HANDOVER 相位 ×倍率，其余相位恢复名义值。
> **起立必然有位移**：滚轮要把从动轮抬过支撑，Isaac 实测 −0.24 m、MuJoCo −0.34 m（含折蹲前馈）；
> 这是物理代价，不是缺陷。

**前端参数卡（2026-10-26 补全，两处 UI 缺陷修复）**：模型配置页「衔接参数」卡含**起立方案下拉**
（P7S 学习策略 / P7T 恒力矩 / P7T2 相位脚本）与全部 `trad_*`/`leg_p_gains` 字段（trad 组在 P7S 下灰显，script 与 trad 共用该组）。
此前 `buildSpec` 只认老 9 键，UI 启动会把 `standup_mode`/trad 参数/`leg_p_gains` 全丢掉——**两个场景
都跑成默认 P7S**（「两个场景一样」的根因之一）；另外刷新页面时活动场景只换 pill 文字、参数回落
settings 旧值（根因之二），均已修复并走全 UI 链路复验（见实测记录）。

## 页面

| 页签 | 内容 |
|---|---|
| **仿真运行** | 上方仿真视口（填充剩余高度，P3 停靠原生窗口；工具条含**相机跟随开关**，默认关）；下方控制台（5 路滑块 **Vx / Yaw / Height / Roll / Jump**（首字母大写）+ in/tgt/act 三联读数、**每路滑块两侧显示当前加载模型的实际指令量程（黑色）**，四轮 h[0.175,0.245]/roll 固定、两轮 h[0.2,0.35]/roll±0.2，量程退化的通道**整卡置灰**；起立/趴下/复位/装填跳、连跳、收藏当前、状态条）；底部抽屉（运行记录 / 控制台输出 / 复现命令）。**键盘**：W/S 前后、A/D 偏航、**I/K 体高 +/−**、J/L 侧倾、**H 装填跳、空格 指令重置到当前模式默认位**、M 切模式、R 复位、1/2/3/4 切页签（输入框内不触发，数字键按导航顺序） |
| **数据绘图** | 真实遥测看图（复用 plot02_core）：时间窗 + body_z 缩略轴（可拖）+ 子图网格；单策略 18 图、双模式 9 图（模式色带，传统起立与起立中同色）；顶部「模型配置」胶囊显示这份录像用的权重（双模式列出四轮/起立/两轮/趴下四个模型，相位脚本起立标「P7T2 相位脚本（无权重）」）；支持整页/分图导出、自动跟随 |
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

## 日志命名规则（2026-11-07 作者定）

| 前缀 | 引擎/模式 | 产出 |
|---|---|---|
| `Isaac_1_<时间戳>.tsv` | Isaac 单策略 | play02 single 档 |
| `Isaac_2_<时间戳>.tsv` | Isaac 双模式 | play02 uni/switch 档 |
| `MJ_<时间戳>.tsv` | MuJoCo | serve.MjTelWriter |

旧前缀 `wbr_tel_*` / `wbr_switch_tel_*` 在绘图页**兼容显示**（PATTERNS 新旧并存），
仅新日志按此命名。MuJoCo 日志含**全部 8 dof × 6 关节列**（pos/vel/tau × tgt/act，
含 caster —— 不能只写 action_dof_names 的 6 个：`left_knee_caster_joint` 名含 "knee"，
plot02 会为它建"左膝 速度/扭矩"面板，缺列就显示"无该列数据"）与
`#joints/#leg_idx/#torque_limits` 元数据 ⇒ 两引擎绘图页完全同构。

## 前端设计规则（2026-11-07 作者定，硬规则）

以下为控制台前端的**统一契约**，改 `app.css` / `app.js` 时必须同时满足；实现集中在 `app.css` 末尾
「卡片统一」块与 `app.js` 的 `renderPreflight()`，不要在各页单独写覆盖样式。

1. **三个内容页（配置Isaac / 配置MJ / 使用说明）的卡片样式完全一致**：
   - 标题栏高度 **2rem（32px @16px 根字号）**、字号 **.82rem**、字体 `var(--font-sans)`、字重 600、
     primary 10% 底 + 1px 下边框、内边距 `.3rem .6rem`；
   - 同一条选择器覆盖所有标题栏：`.pf-wide > h4`、`.mj-box > h4`、`.model-side .box > h4`、
     `.usegrid .box > h4`、`.cfg-head`（双模式/单策略卡的标题栏）；
   - `.cfg-head` 里的图标/胶囊/按钮不得撑高标题栏（固定 `height:2rem` + `overflow:hidden`）；
   - 「当前使用」态只把底色加深到 primary 16%，不改高度/字号。
   - 校验：三页任一 `> h4` 与 `.cfg-head` 的 computed `height / fontSize / fontFamily / fontWeight` 相同。

2. **两个配置页「启动预检」卡片位置、大小、格式一致**：
   - 卡片全宽、位于页首（两页 `pagehead` 都限制 `min-height:2rem` + 长描述单行省略，保证 y 对齐）；
   - 正文区固定 **3 列 × 4 行 = 12 格**：`grid-template-columns: repeat(3,1fr)`、
     `grid-template-rows: repeat(4,1.5rem)`（+ `grid-auto-rows` 同高兜底）、字号 `.7rem`；
   - 两页共用 `renderPreflight(items, boxId)`，恒输出至少 12 个 `<span>`：不足补透明空格
     （`.pf-blank`）；**不截断、不隐藏任何检查项**（MuJoCo 最后一条也必须显示），
     超过 12 项时网格自动加行、两页规则仍一致；
   - 正文区 `.bd` 一律 `display:block`（MuJoCo 的 `.mj-box > .bd` 是 flex，需显式覆盖）。
   - 校验（1920×1080 CDP 实测）：两页预检卡 `x / y / w / h` 与检查网格 `cols / rows / fontSize / 格数` 全部相同。

3. **布局补充**：双模式/单策略卡片按 3:1 分配右侧剩余高度、内部放大字号；常见分辨率（≥1600 宽）
   下双模式内容完整显示且**无滚动条**（卡片内允许略微留空），更小视口才由 `overflow-y:auto` 兜底；
   侧栏「场景与运行 / 检查点库」撑满剩余高度、条数增多时列表内滚动（卡片高度不变）。

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
- 【2026-11-07 更新】旧版四槽日志（`wbr_switch_tel_*.tsv`）没有关节列 ⇒ 曾另给一套
  9 图面板；**新日志（`Isaac_2_*` / `MJ_*`）都带全部关节列**，单策略/双模式/MuJoCo
  三类统一走 `plot02.build_panel_cols` 的 19 图面板（两引擎同构）；旧四槽日志缺列时
  `read_window` 只画有数据的系列，缺的面板留空（向后兼容）。
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
| POST | `/api/start` `{spec,force}` | 起仿真（先预检；Isaac 训练占用默认拦截，MuJoCo 纯 CPU 放行） |
| POST | `/api/stop` | 停仿真（SIGTERM→SIGKILL，先收仿真防孤儿） |
| POST | `/api/preflight` `{spec}` | 预检（按引擎分发：Isaac 权重/显卡/训练；MuJoCo 解释器/MJCF/契约/端口） |
| POST | `/api/cmd` `{key,value}` | 下发指令（代理到桥） |
| POST | `/api/action` `{action,value}` | toggle/rise/lower/reset/armjump/auto |
| POST/DELETE | `/api/scenarios` `/api/saved` | 保存/删除场景、固化/删除存档 |
| GET | `/api/mujoco/manifest` | 配置MJ 页清单：MJCF+sha / S0 parity 读数 / 契约要点 |
| POST | `/api/mujoco/export` | 用 mujoco env 跑 `eval/plot_run.py --latest` 出三联图 PNG |

## MuJoCo 引擎（Sim2Sim，P1~P3 已实施 2026-11-05）

设计稿：`Shared/WBR/docs/02-规划与技术路线/WBR_Sim2Sim_MuJoCo控制台_设计与实施.md`。运行核在
`../sim2sim_mujoco/`（一核两壳：`sim_loop.py` 运行核，`run_policy.py` 批量 CLI 壳，`serve.py` 控制台壳）。

- **引擎选择**：运行页左上角分段控件 `[Isaac Gym │ MuJoCo]`，只切启动面板；运行中按钮锁定；
  停止后解锁。引擎徽标 `● ENGINE: …` 取自桥 `/api/state.engine`（Isaac 橙 / MuJoCo 青，未启动灰）。
- **配置入口**：主导航**「配置MJ」页签**（**四轮/两轮模式选择** / 模型按维度过滤 / MJCF+sha / 场景预设 / 时长·倍速·录制·窗口 / 预检 / S0 parity 对账 / 导出图；持久化 localStorage `wbr.mj.cfg`）。原运行页快启卡已按作者要求移除，配置页签是唯一入口。
- **引擎差异收敛在 `ENGINES` 表**（wbr_bench.py）：解释器（`WBR_PYTHON` / `WBR_PY_MUJOCO`，缺省
  `~/miniconda3/envs/mujoco/bin/python`）、启动 argv（脚本 / `-m serve`）、LD_LIBRARY_PATH、
  训练拦截（Isaac 拦、MuJoCo 放行——纯 CPU 可与训练并行）、preflight 分支。
- **桥协议单一实现**：`bridge_common.py` 承载 HTTP 骨架，`bench_bridge`（Isaac）与
  `mj_bridge`（MuJoCo）各只写自己的状态映射；协议与 2026-11-05 前逐字一致，仅新增
  `/api/state.engine` 字段（前端对缺省值按 isaac 处理，向后兼容）。
- **MuJoCo 运行语义**：单策略，四轮 164 维 / 两轮 161 维按配置页模式；cmd 四通道 `[in,tgt,act]`，vx 限速 0.8 m/s²、
  h 0.15 m/s（滑条渐进）；复位＝零历史冷启动（HTTP 线程只置 flag，主循环 step 安全点执行，
  与 play02 的 mode_shared['reset'] 同构）；无窗口也按墙钟 100 Hz 实时节律；录制 TSV 落
  `sim2sim_mujoco/eval/runs/`，「导出图」取最新一份画三联图 PNG。
- **槽位化（2026-11-05 接手，V0.31）**：配置页改为**槽位下拉**——① 四轮行驶 164 / ② 两轮行走 161 /
  ③ 两轮跳跃 161 / ④ 趴下 173 / ⛓ 五槽链·两轮起步 / ⛓ 五槽链·四轮起步。每槽绑定契约＋MJCF＋默认权重＋预设＋
  可用控制项（跳跃槽解锁「装填跳」，roll 只在两轮槽可用）。链模式 `spec.wmode="uni"`：`serve.py` 建
  `deploy/uni_chain.py` 的 `UniChain`，桥 `/api/state` 增 `chain/slot/phase/marks/canJump`，运行页控制台下方出
  **链时序条**（每槽事件：时间/z/终止原因）；链预检 `_preflight_mujoco_chain()` 逐份核对四槽权重维度与 MJCF sha。
  **两轮/四轮起步链均已验证**（四轮起步 = 4W→RISE 真起立→IDLE2W，2026-11-07 起预检不再给 warn）。
- **复杂地形（demo，2026-11-06）**：配置MJ 页选「⛓ 五槽链」后多出「复杂地形（demo）」勾选，
  `spec.terrain` → `UniChain(terrain=True)` → `wbr_v2_demo.xml`（`wbr_p8_demo.npz` 四象限：波浪/斜坡/
  减速带/台阶/断开平台，中央十字走廊平地，地面浅色）。出生点＝中央十字走廊交点、初速 0、
  roll ±0.2、height 0.20~0.35、跳跃高度 0.05~0.15 均可调（跳跃高度在**行走时**预设，装填跳时才下发；
  通道可用性由后端 `cmdRanges` 表达——MuJoCo 下**不要**再用 `S.mode` 判「仅 TWOW 可跳」，
  桥不返回 `modeKey` 会让滑条永远置灰）。两轮默认站立高度 **0.25**。
  **链启动一律静止（vx=0）**，运行中滑条生效；**默认勾选复杂地形**（可取消）。两轮行走（P8W7）实测 vx 0.295、零终止；
  **跳跃策略 P8J3d 在起伏地形未验证** ⇒ 预检给 warn、自动播放链跳过 JUMP 槽、交互模式装填跳回执提示
  「建议到中央走廊平地再跳」。地形模式跳完**原地交还行走网**（不瞬移回出生点）。

- **面板控制器 A+B（2026-11-06）**：五槽链改**事件驱动**——槽位停住等面板指令，按钮
  `起立/趴下/装填跳/回两轮/回四轮/复位/切槽` 按守卫表动态解禁并真正下发；
  运行页出指令回执条（接受/拒绝＋原因、可下指令清单、事件时序）。过渡请求先过**站定门**
  （|vx|≤0.08 且 |pitch|≤5.7° 连续 8 拍，超时 1.5 s），避免任意时刻立刻切网导致策略吃分布外输入翻倒。
  ［2026-11-07 更新］起立在 MuJoCo 已收口（4W→RISE 真起立 2.85 s，C1×3 全过），不再拒绝；
  只有显式选 `standup_engine="p7t"`（旧恒力矩剖面）时守卫才回绝并回显原因。CLI 自动播放保持原行为、产物可复现。

- **停止/响应优化（2026-11-06）**：MuJoCo 侧 stop 走桥的 `loop.stop_requested` 确定性通道
  （主循环下一拍退出并 flush），不再单靠 SIGINT；实测 **21 s → 0.2 s**。`serve.py` 入口限
  `OMP/MKL/torch` 单线程（小网络推理拉满 9 线程会占 700%+ CPU 并拖慢主线程）；
  `tick()` 节律基准独立于 `reset()`（此前 reset 后目标时刻错乱导致空转 856% CPU）。
  链回执条 `#mjChainBar` 固定高度＋内部滚动、`renderStatus` 内容签名、`onViewportResize` 忽略
  <8 px 变化并防抖到 600 ms ⇒ 按钮说明不再让卡片高度跳变、原生窗口不再被反复重贴。

- **两轮模式（任务③，2026-11-05，已被上面的槽位化取代为子集）**：配置页分段控件 `[四轮 164 │ 两轮 161]`——契约（`deploy_contract{_two}.json`）、
  模型列表（按 obs 过滤）、MJCF（wbr_v2/wbr_v1）、预设（两轮站定 0.275/走 0.3/走 0.6/转向 0.5/高度 0.20/0.35）、
  高度量程（0.175–0.245 → 0.20–0.35）随模式切换；两轮 Roll 指令 ±0.2 解锁（四轮仍置灰）。
  **契约透传**：前端把 `contract` 写进 spec → `_preflight_mujoco` 按 spec 契约对账（obs 维度不符报
  「不是两轮/四轮 N 维策略」）→ `serve.py` 按契约加载（相对路径按仓根解析）；spec 缺 contract 字段回退四轮
  （向后兼容旧 spec）。preflight 端口探测补 SO_REUSEADDR（与真桥 ThreadingHTTPServer 同语义，
  修掉上一场仿真收尾后 TIME_WAIT 误报「端口被占用」）。
- **置灰矩阵（设计稿 §3）**：静态 `data-mj-locked`/`data-mj-why` 标记 + 一个 capture 委托点击
  守卫 + 一个键盘守卫（M/H/U/O/J/L），CSS `body.engine-mujoco` 统一置灰外观；置灰不隐藏。
  滑块量程走既有 cmdRanges 动态机制（Height 自动收窄 0.175–0.245，Roll/Jump 量程退化自动置灰）。
- **已知限制**：① 桥的 400（如 jump_height）经后端代理表现为 `bridgeDown`——前端置灰挡住不会发；
  ② 录制行在内存累积，超长会话（小时级）注意内存；③ ~~MuJoCo 遥测列名与 play02 不同，绘图页不接~~ → **2026-11-06 已接**：serve 落 `MJ_*.tsv`（plot02 口径、含关节列），绘图页两引擎**完全同构**（含关节面板）。

### 实测（2026-11-05，P1~P3 验收）

- 门 1：`run_parity` 7 场景 ALL PASS（obs 1.3e-7 / act 9.5e-7 / τ 1.2e-7）；重构后 run_policy 与旧版
  同参数输出**逐字节一致**（slew 关闭批量模式）。
- 门 2：bench_bridge 换骨架前后 16 条协议采样（含 400/404 语义）diff 零差异，仅 engine 字段新增。
- 门 3：preflight(isaac)×3 spec + build_launch(isaac) argv/env 快照全一致。
- 门 4：五槽链检 `wbr_chain_check.py --scenario WBR统一五槽_P8` **ALL PASS**（起立 4.84s、
  三档跳 60.0/82.7/86.4 mm、行走 act 0.4、趴下 3.16s——与 11-05 基线逐项复现）。
- MuJoCo e2e（API + 浏览器 UI 双通道）：启动→`● ENGINE: MuJoCo`→滑条驱动（vx 0.3 → 稳态 0.262 m/s，
  与 S1 口径一致）→复位零历史重走→急停→落盘 TSV+summary→停止解锁切回 Isaac。
- 停靠回归：MuJoCo 窗口（GLFW 标题 `MuJoCo : wbr_v2`）`/api/window` found、dock/undock/reset 全通
  ——根因是 `xwin.status/_status/undock` 默认匹配写死 `"isaac gym"`，已改为随 `_DEFAULT_MATCH`（引擎）。
- **停靠稳定性加固（作者反馈「不太稳定」后）**：① `xwin.dock()` 改回**实测几何**（窗口 id 在手直接
  `_attrs`，避开 11-04 的 find_window 瞬态误判）→ 前端自校正复活；② 自校正只认 <200px 小偏差为
  换算误差（实测 MuJoCo 新窗会被 mutter 初始放置覆盖，大偏差纠不得，交给看门狗）；③ 前端**停靠保持
  看门狗**：DOCK.on 期间每 3 s 比对实测几何，漂移 >8px 静默重贴（实测挪窗 1 s 内回位）；④ 贴前等
  连续两拍几何一致（避开 viewer 初始化挪窗期）；⑤ 仿真自行退出（如 MuJoCo 终止判定摔倒→窗口消失）
  前端拉尾日志 toast 一行原因，不再无声。
- **已知行为（待作者裁决）**：MuJoCo 下「站定 0.19」预设（vx=0）用 P7F3e@11800 会缓慢倒下触发终止
  判定，进程退出→窗口消失（此前烟测均为行走 vx≥0.3 未踩到）；站不住是策略/工作点问题非控制台问题，
  暂用「走 0.3」等预设验证停靠。
- 烟测中修掉的三个真 bug：spec.mode 语义撞名导致零动作（policy 不加载）、无窗口时缺实时节律
  （13 倍速狂奔）、复位跨线程竞态+节律基准未重置。

### 实测（2026-11-05 晚，两轮模式验收，[Ubuntu/zcode] 任务③）

- manifest 双契约：four 164 / two 161，parity_two 六场景 PASS（stand/walk03/walk06/yaw05/h020/h035）。
- 预检三例：两轮 spec（wbr_v1 sha 75823356 + P8W7@2800 obs 161 ✓）；四轮 spec 无 contract 字段回退默认 ✓；
  两轮契约×四轮权重 → 「obs 164 与契约 161 不符——不是两轮 161 维策略」✓。
- e2e（API 直测）：两轮 vx0.3 走 6 s（mean vx 0.283、z 0.272 vs 指令 0.275、无终止）；两轮 roll 指令 0.1
  （mean roll 0.0967 跟随、无终止）；四轮回归 vx0.3@h0.19（mean vx 0.317、无终止）。
- 连续起仿暴露 preflight 端口探测误报：真桥 ThreadingHTTPServer 自带 SO_REUSEADDR，探测裸 bind 没有
  → TIME_WAIT 残留误报「桥端口被占用」；探测已补同语义 socket option。
- **协作教训**：并行会话（任务②）同刻重启控制台产生 8771 重复实例（web 端口顺延撞上默认桥端口）。
  控制台只允许一个实例（8770）；重启前先 `ss -tlnp | grep 8770` 确认，或直接 POST /api/stop 后复用。

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
