#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Date      : 2026-10-28
# Author    : SYH
# File      : wbr_chain_check.py
# Purpose   : 链上验收驱动：按固定序列调用控制台 API 跑完整链路（四轮 → 起立 → 跳跃 → 行走 →
#             趴下），输出各段实测值与判据。
# ==============================================================================

"""P7 混合部署 · 链上验收驱动（不训练，只驱动控制台跑一条固定序列）。

序列（约 30 s）：四轮静稳 → 脚本起立(P7T) → 两轮站定 → 跳跃 0.05/0.10/0.15（每次落地站住）
                → 给 vx 行走 → 减速到停 → 趴下(P7L) → 四轮

它只调控制台的只读/控制 API（`/api/state`、`/api/start`、`/api/cmd`、`/api/action`），
跑完把 TSV 路径与逐段实测值打出来；TSV 的 `ev_takeoff/ev_land/jump_apex/v_impact/rise_sub`
是 2026-10-27 新加的事件列，判据表直接读这些列，不再需要探针复算。

用法：
    python wbr_chain_check.py --scenario P7T_脚本起立R81 --port 8770
    python wbr_chain_check.py --scenario LQR起立R81跳跃 --jumps 0.05,0.10,0.15 --walk 0.4
退出码 0 = 全段通过。
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "wbr_bench_store", "scenarios")


def _req(base, path, body=None, timeout=30):
    url = base + path
    if body is None:
        r = urllib.request.Request(url, method="GET")
    else:
        r = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                   headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return json.loads(f.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"HTTP": e.code, "body": e.read().decode("utf-8")[:600]}


class Driver:
    def __init__(self, base):
        self.base = base
        self.log = []

    def get(self, p):
        return _req(self.base, p)

    def post(self, p, b):
        return _req(self.base, p, b)

    def state(self):
        return self.get("/api/state")

    def wait(self, pred, timeout, what, poll=0.25, grace=0.0):
        """grace 秒内允许桥未起来（仿真冷启动要 ~10 s，期间 /api/state 会报 bridgeDown）。"""
        t0 = time.time()
        last = None
        while time.time() - t0 < timeout:
            st = self.state()
            if st.get("bridgeDown") and (time.time() - t0) >= grace:
                raise RuntimeError(f"[{what}] 桥断了：{st.get('error')}")
            last = st
            if not st.get("bridgeDown") and pred(st):
                return st
            time.sleep(poll)
        raise RuntimeError(f"[{what}] {timeout}s 超时；最后状态={_brief(last)}")

    def wait_settle(self, min_s=0.0, vx_tol=0.08, timeout=12.0, what="停稳"):
        """等两轮真正停住（|act_vx|≤vx_tol 连续 3 拍）再切模式——C4 裁定的"停止状态切"。
        交接后 P7T 的抬升力矩会把机器人推着后滚 0.4~0.6 m/s，R81 要 1~2 s 才刹住；
        不等就趴下会连人带车翻过去（2026-10-28 C1 循环实测）。"""
        t0 = time.time()
        need, hit, last = 3, 0, None
        while time.time() - t0 < timeout:
            st = self.state()
            last = st
            if st.get("modeCN") != "两轮站立":
                raise RuntimeError(f"[{what}] 不在两轮站立（{st.get('modeCN')}）")
            c = (st.get("cmd") or {}).get("vx") or []
            vx = abs(float(c[2])) if len(c) > 2 else 9.9
            hit = hit + 1 if vx <= vx_tol else 0
            if hit >= need and (time.time() - t0) >= min_s:
                return st
            time.sleep(0.25)
        raise RuntimeError(f"[{what}] {timeout}s 内没停稳；最后 {_brief(last)}")

    def cmd(self, **kw):
        for k, v in kw.items():
            r = self.post("/api/cmd", {"key": k, "value": v})
            if not r.get("ok"):
                raise RuntimeError(f"cmd {k}={v} 失败：{r}")
        return True

    def action(self, a, **kw):
        body = {"action": a}
        body.update(kw)
        r = self.post("/api/action", body)
        if not r.get("ok"):
            raise RuntimeError(f"action {a} 失败：{r}")
        return r


def _brief(st):
    if not st:
        return "None"
    return (f"mode={st.get('modeCN')} phUni={st.get('phUni')} armed={st.get('armed')} "
            f"line={st.get('modeLine')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="P7T_脚本起立R81")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--jumps", default="0.05,0.10,0.15", help="逗号分隔的跳跃指令档")
    ap.add_argument("--walk", type=float, default=0.4, help="行走段 vx 指令（m/s），0=跳过")
    ap.add_argument("--walk-s", type=float, default=3.0)
    ap.add_argument("--rise-settle", type=float, default=3.0,
                    help="交两轮后静置秒数（太短会在交接摇摆里起跳，0.05 档会被相位机截断）")
    ap.add_argument("--jump-settle", type=float, default=2.5, help="每次落地后静置秒数")
    ap.add_argument("--land-walk", action="store_true",
                    help="C2-a：落地当拍立刻给 vx（不等静置），看能不能直接走起来")
    ap.add_argument("--land-lower", action="store_true",
                    help="C2-b：落地当拍立刻趴下（不等静置）")
    ap.add_argument("--land-jump", type=float, default=0.10, help="C2 用的跳跃档")
    ap.add_argument("--moving-jump", action="store_true",
                    help="【P8 五槽】C4 行进跳：行走中装填跳跃（事件级切跳跃网，落地交还行走网）")
    ap.add_argument("--mj-walk", type=float, default=0.4, help="行进跳的行走 vx（m/s）")
    ap.add_argument("--cycles", type=int, default=0, help="C1：起立⇄趴下循环次数（0=不做）")
    ap.add_argument("--out", default="", help="结果 JSON 落盘路径")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    jumps = [float(x) for x in args.jumps.split(",") if x.strip()]

    spec_path = os.path.join(STORE, args.scenario + ".json")
    if not os.path.isfile(spec_path):
        print(f"❌ 场景不存在：{spec_path}")
        return 2
    spec = json.load(open(spec_path, encoding="utf-8"))
    spec.pop("name", None)
    spec.pop("when", None)
    spec.pop("bridge_port", None)

    d = Driver(base)
    fails = []
    res = {"scenario": args.scenario, "jumps": jumps, "walk": args.walk}

    # 0) 干净起点
    st = d.get("/api/status")
    if st.get("status") == "running":
        print("· 已有仿真在跑，先停")
        d.post("/api/stop", {})
        time.sleep(2.5)

    pre = d.post("/api/preflight", {"spec": spec})
    bad = [i for i in pre.get("items", []) if i.get("level") == "bad"]
    print("预检：", " / ".join(f"{i['level']}:{i['text']}" for i in pre.get("items", [])))
    if bad:
        print("❌ 预检有硬错误，停")
        return 2

    # 1) 起仿真
    r = d.post("/api/start", {"spec": spec})
    if not r.get("ok"):
        print(f"❌ 启动失败：{r}")
        return 2
    run_id = r.get("run")
    res["run"] = run_id
    print(f"▶ 已启动 run={run_id}")
    st = d.wait(lambda s: s.get("modeCN") == "四轮" and s.get("hasReadout"), 45, "等四轮就绪", grace=25)
    # 桥的 telemetry 字段是 {"file": ..., "sizeMB": ..., "rows": ...}，且只给文件名（不带目录）。
    #   2026-11-05 修：原来读的 "name" 键不存在 ⇒ tsv 恒 None ⇒ 三档站定跳读不到事件列。
    #   兜底：直接 glob 最新一份 wbr_switch_tel_*.tsv（每次仿真启动新建一份）。
    import glob as _glob
    tsv = None
    for _ in range(40):
        _tel = d.state().get("telemetry") or {}
        _nm = _tel.get("file") or _tel.get("name")
        if _nm and _nm not in ("—", ""):
            tsv = _nm if os.path.isabs(_nm) else os.path.join(_data_dir(), os.path.basename(_nm))
        if not tsv or not os.path.isfile(tsv):
            _c = _glob.glob(os.path.join(_data_dir(), "wbr_switch_tel_*.tsv"))
            tsv = max(_c, key=os.path.getmtime) if _c else tsv
        if tsv and os.path.isfile(tsv) and os.path.getsize(tsv) > 0:
            break
        time.sleep(0.25)
    res["tsv"] = tsv
    print(f"  遥测：{tsv}")

    # 2) 四轮静稳 1.5 s（看起点高度/姿态）
    time.sleep(1.5)
    st = d.state()
    print(f"① 四轮：{st.get('modeLine')}")
    res["fourw_line"] = st.get("modeLine")

    # 3) 起立
    t_rise0 = time.time()
    d.action("rise")
    st = d.wait(lambda s: s.get("modeCN") == "两轮站立", 20, "等起立完成")
    res["rise_s"] = round(time.time() - t_rise0, 2)
    res["rise_line"] = st.get("modeLine")
    print(f"② 起立→两轮：{res['rise_s']}s  {st.get('modeLine')}")
    d.wait_settle(min_s=args.rise_settle, what="交两轮后停稳")   # 至少站 args.rise_settle 秒，且真的停住

    # 4) 三档跳
    res["jump"] = []

    def jump_once(h, settle):
        """装填→等起飞→等回 ph0→静置 settle 秒；返回 (state, 墙钟秒, TSV 事件指标)。"""
        d.cmd(jump_height=h)
        time.sleep(0.3)
        d.action("armjump")
        t0 = time.time()
        d.wait(lambda s: (s.get("phUni") or 0) >= 1, 3.0, f"跳 {h} 起飞")
        st_ = d.wait(lambda s: (s.get("phUni") or 0) == 0, 8.0, f"跳 {h} 落地回 0")
        air_ = round(time.time() - t0, 2)
        if settle:
            time.sleep(settle)
        m_ = _last_jump_metrics(os.path.join(_data_dir(), tsv)) if tsv else None
        return st_, air_, m_

    for h in jumps:
        st, air, m = jump_once(h, args.jump_settle)
        ok = st.get("modeCN") == "两轮站立"
        if not ok:
            fails.append(f"跳 {h}: 落地后不在两轮站立（{st.get('modeCN')}）")
        res["jump"].append({"cmd": h, "wall_s": air, "metrics": m, "line_after": st.get("modeLine")})
        print(f"③ 跳 {h:.2f}：{m}  落地后 {st.get('modeCN')}")
        d.cmd(jump_height=0.0)

    # 4b) C2-a：落地→走（落地当拍就给 vx，不留静置窗）
    if args.land_walk:
        st, air, m = jump_once(args.land_jump, 0.0)
        d.cmd(lin_vel_x=args.walk or 0.4)
        t_w = time.time()
        time.sleep(2.5)
        st = d.state()
        vx_now = (st.get("cmd") or {}).get("vx")
        res["land_walk"] = {"cmd": args.land_jump, "metrics": m, "vx": vx_now,
                            "mode": st.get("modeCN")}
        print(f"③b 落地→走（{args.land_jump:.2f} 落地当拍给 vx {args.walk or 0.4}）："
              f"2.5s 后 act={vx_now}  mode={st.get('modeCN')}")
        if st.get("modeCN") != "两轮站立":
            fails.append(f"落地→走：不在两轮站立（{st.get('modeCN')}）")
        if vx_now is None or abs(float(vx_now[2] if isinstance(vx_now, (list, tuple)) else vx_now)) < 0.5 * (args.walk or 0.4):
            fails.append(f"落地→走：vx 没跟上（{vx_now}）")
        d.cmd(lin_vel_x=0.0)
        time.sleep(2.0)
        d.cmd(jump_height=0.0)

    # 4c) C2-b：落地→趴下（落地当拍就发 lower）
    if args.land_lower:
        st, air, m = jump_once(args.land_jump, 0.0)
        t_l = time.time()
        d.action("lower")
        st = d.wait(lambda s: s.get("modeCN") == "四轮", 20, "落地→趴下完成")
        res["land_lower"] = {"cmd": args.land_jump, "metrics": m,
                             "s": round(time.time() - t_l, 2), "line": st.get("modeLine")}
        print(f"③c 落地→趴下：{res['land_lower']['s']}s  {st.get('modeLine')}")
        time.sleep(1.0)
        d.cmd(jump_height=0.0)

    # 4d) 【P8 五槽】C4 行进跳：行走中装填（条件满足直接切跳跃网，事件结束交还行走网）
    if args.moving_jump:
        _mj = args.mj_walk or 0.4
        d.cmd(base_height=0.28)              # 体高设 P8J 带内（否则事件尾从夹取的 0.30 弹回滑块
        d.cmd(lin_vel_x=_mj)                 #   默认 0.19，110mm 姿态阶跃拖慢恢复——操作员现实行为）
        time.sleep(2.0)                      # 到达稳态速度
        d.cmd(jump_height=args.land_jump)
        d.action("armjump")
        # 等「进事件窗（phUni≥1）→ 事件结束（phUni==0）」：恢复判定从事件尾起算
        #   （与 P8J 交付口径一致——从装填起算会把事件时长误算进恢复窗）
        d.wait(lambda s: (s.get("phUni") or 0) >= 1, 6, "行进跳：进入事件窗")
        d.wait(lambda s: s.get("phUni") == 0, 6, "行进跳：事件结束")   # phUni=0 是合法值，不能走 or 短路
        time.sleep(1.0)                      # 交还行走网后重新加速
        st = d.state()                       # 【修复】事件后状态在此刻取（原版用了③段的过期 st）
        # 遥测 TSV 用 glob 最新（桥的 telemetry 字段不带 name；wbr_switch_tel_*.tsv 每次
        #   仿真启动新建一份，行进跳时刻的最新文件必是本 run 的）
        import glob as _glob
        _cands = _glob.glob(os.path.join(_data_dir(), "wbr_switch_tel_*.tsv"))
        _tsv_now = max(_cands, key=os.path.getmtime) if _cands else tsv
        m = _last_jump_metrics(_tsv_now) if _tsv_now else None
        cw = _ctrl_window_stats(_tsv_now) if _tsv_now else {}
        vx1 = (st.get("cmd") or {}).get("vx") or []
        res["moving_jump"] = {"walk": _mj, "metrics": m, "ctrl": cw,
                              "mode": st.get("modeCN"), "vx": vx1}
        print(f"③d 行进跳：ctrl {cw}  metrics {m}  落地后 {st.get('modeCN')}")
        if st.get("modeCN") != "两轮站立":
            fails.append(f"行进跳：落地后不在两轮站立（{st.get('modeCN')}）")
        if not cw.get("covered"):
            fails.append(f"行进跳：事件窗 ctrl 未翻转（{cw}）")
        _vx1 = float(vx1[2]) if len(vx1) > 2 else 0.0
        if _vx1 < 0.5 * _mj:
            fails.append(f"行进跳：落地后 vx 没跟上（{_vx1}）")
        d.cmd(lin_vel_x=0.0)
        time.sleep(1.5)
        d.cmd(jump_height=0.0)

    # 5) 行走 → 减速到停（只在两轮站立时做；C2-b 已趴下就跳过）
    if args.walk and d.state().get("modeCN") == "两轮站立":
        d.cmd(lin_vel_x=args.walk)
        time.sleep(args.walk_s)
        st = d.state()
        vx_run = (st.get("cmd") or {}).get("vx")
        res["walk_vx"] = vx_run
        print(f"④ 行走 vx={args.walk}：act={vx_run}")
        d.cmd(lin_vel_x=0.0)
        time.sleep(2.0)
        st = d.state()
        res["stop_vx"] = (st.get("cmd") or {}).get("vx")
        print(f"   减速到停：act={res['stop_vx']}")
        if st.get("modeCN") != "两轮站立":
            fails.append(f"减速后不在两轮站立（{st.get('modeCN')}）")

    # 6) 趴下（若已在四轮——例如刚跑过"落地→趴下"——则跳过）
    if d.state().get("modeCN") == "四轮":
        res["lower_s"] = None
        print("⑤ 趴下：已在四轮，跳过")
    else:
        t_lie0 = time.time()
        d.action("lower")
        st = d.wait(lambda s: s.get("modeCN") == "四轮", 20, "等趴下完成")
        res["lower_s"] = round(time.time() - t_lie0, 2)
        res["lower_line"] = st.get("modeLine")
        print(f"⑤ 趴下→四轮：{res['lower_s']}s  {st.get('modeLine')}")
        time.sleep(1.0)

    # 6b) C1：起立⇄趴下循环 ×N（每轮都要求回到四轮且无摔倒）
    if args.cycles:
        res["cycles"] = []
        for i in range(args.cycles):
            t0 = time.time()
            d.action("rise")
            st = d.wait(lambda s: s.get("modeCN") == "两轮站立", 20, f"循环{i+1} 起立")
            t_up = round(time.time() - t0, 2)
            # 停稳再切（C4：停止状态切 + 姿态回正再切）；交接后还会后滚 0.4~0.6 m/s
            st = d.wait_settle(min_s=0.8, what=f"循环{i+1} 停稳")
            time.sleep(1.0)
            t1 = time.time()
            d.action("lower")
            st = d.wait(lambda s: s.get("modeCN") == "四轮", 20, f"循环{i+1} 趴下")
            t_dn = round(time.time() - t1, 2)
            time.sleep(1.0)
            res["cycles"].append({"i": i + 1, "rise_s": t_up, "lower_s": t_dn})
            print(f"⑥ 循环 {i+1}/{args.cycles}：起立 {t_up}s / 趴下 {t_dn}s")
            if st.get("modeCN") != "四轮":
                fails.append(f"循环{i+1} 未回到四轮")
        print(f"   C1 ×{args.cycles} 完成："
              + " / ".join(f"{c['rise_s']}+{c['lower_s']}" for c in res["cycles"]))

    # 7) 收工
    d.post("/api/stop", {})
    time.sleep(2.0)
    print(f"\nTSV：{os.path.join(_data_dir(), tsv) if tsv else '?'}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"结果：{args.out}")
    print(f"\n==> {'ALL PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    return 0 if not fails else 1


def _data_dir():
    # 【P8 五槽】不再经 play02_core（它 import isaacgym，链式驱动是纯 stdlib）：
    #   DATA_DIR = <repo>/legged_gym/scripts/wbr_play_data（与 play02_core.DATA_DIR 同源）
    return os.path.normpath(os.path.join(HERE, "..", "wbr_play_data"))


def _read_tsv(path):
    if not path or not os.path.isfile(path):
        return [], []
    rows = []
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    hdr = None
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            continue
        cols = ln.split("\t")
        if hdr is None:
            hdr = cols
            continue
        rows.append(dict(zip(hdr, cols)))
    return hdr or [], rows


def _ctrl_window_stats(path):
    """【P8 五槽】事件级切换核查：ctrl==1 拍数＋最近一次 ev_takeoff 拍 ctrl 是否=1（窗覆盖）。"""
    hdr, rows = _read_tsv(path)
    if not rows or "ctrl" not in (hdr or []):
        return {"present": False, "covered": False}
    n1 = sum(1 for r in rows if r.get("ctrl") == "1")
    takeoff_ok = None
    for r in rows:
        if r.get("ev_takeoff") == "1":
            takeoff_ok = (r.get("ctrl") == "1")
    return {"present": True, "n1": int(n1), "takeoff_ctrl": takeoff_ok,
            "covered": bool(n1 > 0 and takeoff_ok)}


def _last_jump_metrics(path):
    """从 TSV 里取最近一次完成的跳跃（ev_land==1 的最后一条）峰值/腾空/冲击。"""
    hdr, rows = _read_tsv(path)
    if not rows or "ev_land" not in (hdr or []):
        return None
    land_idx = None
    for i, r in enumerate(rows):
        if r.get("ev_land") == "1":
            land_idx = i
    if land_idx is None:
        return None
    r = rows[land_idx]
    return {"apex_mm": round(float(r.get("jump_apex", 0)) * 1000, 1),
            "air_s": round(float(r.get("jump_air_s", 0)), 2),
            "v_impact": round(float(r.get("v_impact", 0)), 2),
            "z": round(float(r.get("body_z", 0)), 3),
            "pitch_deg": round(float(r.get("pitch", 0)) * 57.2958, 1)}


if __name__ == "__main__":
    raise SystemExit(main())
