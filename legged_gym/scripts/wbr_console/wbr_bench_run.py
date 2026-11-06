# ==============================================================================
# Date      : 2026-10-26
# Author    : SYH
# File      : wbr_bench_run.py
# Purpose   : 仿真进程入口：替换 play02_core 的 GUI 线程与切换构造后驱动仿真，供控制台启停。
# ==============================================================================

import json
import os
import sys
import signal

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SCRIPTS))

# libpython 只在解释器自己的 lib 下；没设 LD_LIBRARY_PATH 时自我重启一次再 import isaacgym。
_conda_lib = os.path.join(sys.prefix, "lib")
if os.path.isdir(_conda_lib) and _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _conda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
os.chdir(SCRIPTS)   # play02 的相对路径依赖（releases/... / logs/...）都以仓库根解析，但 cwd 在 scripts 更贴近习惯

# conda 环境的动态库（子进程若继承不到 LD_LIBRARY_PATH 时兜底）
_conda_lib = os.path.join(sys.prefix, "lib")
if os.path.isdir(_conda_lib):
    os.environ["LD_LIBRARY_PATH"] = _conda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")


def _abs(p):
    """相对路径按仓库根解析成绝对路径（子进程 cwd 在 scripts，play02 的 from_run_config
    是相对 cwd 判断的，必须给绝对路径才找得到）。"""
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def _build_argv(spec):
    """把 spec 翻译成 play02 认识的命令行（等价于手工敲的那条）。"""
    argv = ["wbr_bench_run.py"]
    mode = spec.get("mode")
    if mode in ("switch", "uni"):
        # switch / uni 同构（四槽）：uni 的 env=wbr_uni、两轮槽=R81（161，与 W3 同规格）
        argv += ["--play_mode", mode]
        argv += ["--fourw_checkpoint", spec["slots"]["fourw"]]
        # 起立槽可为相位脚本（dict）：无权重，传空串，_build_switch 里跳过加载。
        _su = spec["slots"]["standup"]
        argv += ["--standup_checkpoint", ("" if isinstance(_su, dict) else _su)]
        argv += ["--w3_checkpoint", spec["slots"]["w3"]]
        argv += ["--lie_checkpoint", spec["slots"]["lie"]]
        # 【P8 五槽】跳跃槽（事件级）：空=不加载
        argv += ["--jump_checkpoint", (spec["slots"].get("jump") or "")]
        argv += ["--task", "wbr_uni" if mode == "uni" else "wbr_switch"]
    else:
        single = spec.get("single") or {}
        argv += ["--play_mode", "single"]
        argv += ["--task", single.get("task", "wbr_flat")]
        if single.get("load_run"):
            argv += ["--load_run", _abs(single["load_run"])]
        if single.get("checkpoint") is not None:
            argv += ["--checkpoint", str(single["checkpoint"])]
        if single.get("from_run_config"):
            argv += ["--from_run_config", _abs(single["from_run_config"])]
        try:
            _rb = max(1, min(3, int(single.get("robots"))))
        except (TypeError, ValueError):
            _rb = None
        if _rb is not None:
            argv += ["--num_envs", str(_rb)]   # 机器人数 1~3（模型配置页选择）
    return argv


def main():
    # 控制台用 setsid 后台启动时，父进程的 SIG_IGN 会遗传给本进程，play02 的 SIGINT 收尾就失效。
    # 恢复 Python 默认处理（raise KeyboardInterrupt）才能走到 finally flush；不能用 SIG_DFL（立即终止）。
    signal.signal(signal.SIGINT, signal.default_int_handler)
    spec_path = None
    for i, a in enumerate(sys.argv):
        if a == "--spec" and i + 1 < len(sys.argv):
            spec_path = sys.argv[i + 1]
    if not spec_path or not os.path.isfile(spec_path):
        print("❌ 缺少 --spec <spec.json>")
        sys.exit(2)
    with open(spec_path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    os.environ["WBR_BRIDGE_PORT"] = str(spec.get("bridge_port", 8771))

    # 只让 play02 看到它认识的参数（gymutil.parse_arguments 不接受未知项）
    sys.argv = _build_argv(spec)
    print("🧾 [bench] 等效命令：" + " ".join(sys.argv))

    from deps import play02_core as play02
    import bench_bridge


    # 运行时替换面板：签名一致，只起 HTTP 控制口
    play02.start_gui_thread = bench_bridge.install

    # 衔接相位参数：写进 env.cfg.rewards.switch.*（run_switch 的相位机按它推进），
    #   不改 play02.py —— 仅替换其 _build_switch 的返回对象。single 档无此概念，跳过。
    _PHASE_KEYS = ("t_align", "t_rise", "t_handover", "t_lower", "t_tilt", "t_absorb")
    _TRAD_KEYS = ("t_trad_hold", "t_trad_crouch", "t_trad_crouch_hold", "t_trad_extend",
                  "t_trad_lift_max", "trad_torque", "trad_boost", "trad_hcmd", "trad_hcmd_hold")
    params = spec.get("params") or {}
    if spec.get("mode") in ("switch", "uni"):
        _orig_build = play02._build_switch

        def _build_switch_with_params(args, task="wbr_switch"):
            env, p4w, pstand, p_w3, p_lie, p_jump = _orig_build(args, task)
            sw = env.cfg.rewards.switch
            applied = {}
            for k in _PHASE_KEYS:
                v = params.get(k)
                if v is not None:
                    try:
                        setattr(sw, k, float(v))
                        applied[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            # 【P8 五槽】跳跃模板保真：params["jump_cfg"]=P8J config.json ⇒ 整块 rewards.jump
            #   写进 env.cfg（uni 继承的 wbr_flat4 跳跃机由此读到 crouch_abs_by_h_moving/
            #   moving_crouch_vx_gate/1c 三锚点等 P8J 键），并附 commands.ranges 供事件窗切量程；
            #   然后恢复控制台的 arm_delay=0.1（瞬时起跳节流，_build_switch 原语义）。
            _jc = params.get("jump_cfg")
            if _jc:
                import json as _json
                _jp = _abs(_jc)
                with open(_jp, "r", encoding="utf-8") as _fh:
                    _jd = _json.load(_fh)
                _jk = (_jd.get("rewards") or {}).get("jump") or {}
                for _k, _v in _jk.items():
                    setattr(env.cfg.rewards.jump, _k, tuple(_v) if isinstance(_v, list) else _v)
                env.cfg.rewards.jump.arm_delay = 0.1
                _jr = (_jd.get("commands") or {}).get("ranges") or {}
                env._jump_cfg_ranges = {k: tuple(v) for k, v in _jr.items()}
                applied["jump_cfg"] = {"path": _jp, "keys": len(_jk),
                                       "ranges": sorted(env._jump_cfg_ranges.keys())}
            # 传统起立（方案①脚本链）参数：standup_mode + t_trad_*/trad_*（缺省值在 play02_core.run_switch 内）
            sm = params.get("standup_mode")
            if sm is not None and str(sm).strip():
                setattr(sw, "standup_mode", str(sm).strip())
                applied["standup_mode"] = str(sm).strip()
            for k in _TRAD_KEYS:
                v = params.get(k)
                if v is not None:
                    try:
                        setattr(sw, k, float(v))
                        applied[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            # 脚本起立的两个非浮点参数：
            #   trad_advance = 'pitch'|'timer'（字符串）；trad_z_handover_min = 数值 或 null/'none'（None=纯俯仰）
            _adv = params.get("trad_advance")
            if _adv is not None and str(_adv).strip():
                setattr(sw, "trad_advance", str(_adv).strip().lower())
                applied["trad_advance"] = str(_adv).strip().lower()
            _zh = params.get("trad_z_handover_min")
            if _zh is not None:
                try:
                    setattr(sw, "trad_z_handover_min",
                            None if str(_zh).strip().lower() in ("none", "", "null") else float(_zh))
                    applied["trad_z_handover_min"] = getattr(sw, "trad_z_handover_min")
                except (TypeError, ValueError):
                    pass
            # P7T 代理：leg_p_gains —— 起立/交接两相位把腿 P 增益乘倍率，其余相位恢复名义值。
            #   只改 env.p_gains 运行时张量，cfg 与 play02_core 文件不动。
            boost = params.get("leg_p_gains")
            if boost is not None:
                try:
                    boost = float(boost)
                except (TypeError, ValueError):
                    boost = None
            if boost and boost != 1.0:
                _install_leg_boost(env, boost)
                applied["leg_p_gains"] = boost
            # env.sw 与 env.cfg.rewards.switch 是同一对象，run_switch 里 sw.t_* 立刻生效
            bench_bridge.set_applied_params(applied)
            if applied:
                extra = (f", leg_p_gains={applied['leg_p_gains']}"
                         f"（起立/交接相位腿 PD ×{applied['leg_p_gains']}）") if "leg_p_gains" in applied else ""
                print("⚙️ [bench] 衔接相位参数已应用："
                      + ", ".join(f"{k}={applied[k]}" for k in _PHASE_KEYS if k in applied) + extra)
            trad_bits = [k for k in ("standup_mode",) + _TRAD_KEYS if k in applied]
            if trad_bits:
                print("🦿 [bench] 传统起立参数已应用：" + ", ".join(f"{k}={applied[k]}" for k in trad_bits))
            return env, p4w, pstand, p_w3, p_lie, p_jump

        def _install_leg_boost(env, boost):
            """包一层 env.playback_set_mode（run_switch 的 _apply_mode 每步都以当前相位调它）：
            SP_RISE/SP_HANDOVER 期腿 P 增益 = 名义×boost，其余相位恢复名义值。"""
            leg = [int(i) for i in env.leg_dof]   # env.leg_dof 是张量，转成 python int 索引列表
            nominal = env.p_gains[leg].clone()
            orig_set_mode = env.playback_set_mode
            boost_phases = (play02.SP_RISE, play02.SP_HANDOVER)   # 1=起立伸腿 2=交接

            def _with_boost(ids, ph, task=1, ph_t=0.0, **kw):
                env.p_gains[leg] = nominal * boost if ph in boost_phases else nominal
                return orig_set_mode(ids, ph, task=task, ph_t=ph_t, **kw)

            env.playback_set_mode = _with_boost
            print(f"🦵 [bench] 腿 P 增益倍率已挂接：起立/交接相位 ×{boost}（其余恢复名义）")

        play02._build_switch = _build_switch_with_params

    args = play02.get_args()
    try:
        play02.play02(args)
    except SystemExit:
        pass
    finally:
        print("🛑 [bench] 仿真进程退出")


if __name__ == "__main__":
    main()
