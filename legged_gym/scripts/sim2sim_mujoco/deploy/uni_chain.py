# ==============================================================================
# Date      : 2026-11-05（2026-11-06 事件驱动改造）
# Author    : SYH
# File      : uni_chain.py
# Purpose   : S3 五槽整链（MuJoCo）——把四个策略槽接成一条链，供 CLI 与控制台共用。
#             槽序与训练侧 wbr_uni 相位编号对齐（phase.py 0-11）：
#               IDLE2W(0) 两轮行走 → JUMP(1-5) 事件级跳跃 → LOWER(9)/TILT(10)/ABSORB(11) 趴下
#               → HOLD4W(6) 四轮 → （可选）RISE(7)/HANDOVER(8) 起立
#             ⚠ 起立槽（P7T）在 MuJoCo 尚未收口：默认请求会被守卫明确拒绝并回显原因，
#               `allow_rise=True` 才真的进 RISE（用于后续调试）。
#
#   两种运行模式（互不影响，评估产物可复现）：
#     interactive=False（CLI 默认）：按固定时长表自动放完 PLAN_2W / PLAN_FULL。
#     interactive=True （控制台）：槽位**停住等指令**，由 /api/action → chain.request()
#       驱动；守卫与训练侧 PhaseFSM 同语义（跳跃/趴下过渡态拒绝插队）。
#       过渡请求先排队，当前槽继续保持，等姿态平稳（站定门）或超时后执行切换——
#       实测任意时刻立刻切网会让策略吃分布外输入而翻倒。
#
# 契约切换：164（四轮）/ 161@v2（两轮，8 dof 载体）/ 173（趴下，switch 尾块）。
# ==============================================================================

import json
import os
from collections import deque

import numpy as np
import torch

from legged_gym.scripts.sim2sim_mujoco.deploy.jump_template import JumpTemplate
from legged_gym.scripts.sim2sim_mujoco.deploy.obs_builder import ObsBuilder
from legged_gym.scripts.sim2sim_mujoco.deploy.policy import Policy
from legged_gym.scripts.sim2sim_mujoco.sim_loop import PKG, SimLoop, load_contract, resolve_model

MJCF = os.path.join(PKG, "assets/mjcf/wbr_v2.xml")
TERRAIN_META = os.path.join(PKG, "assets/terrain/wbr_v2_demo.json")
LEGS4 = ("left_hip_joint", "left_knee_joint", "right_hip_joint", "right_knee_joint")
POSE_2W = [1.0, -2.1, -1.0, 2.1]
Z_2W_START = 0.2747

# 自动播放模式的槽默认时长（秒）——仅 interactive=False 使用
PLAN_2W = [("IDLE2W", 3.0), ("JUMP", 2.0), ("LOWER", 1.6), ("HOLD4W", 3.0)]
PLAN_FULL = [("HOLD4W", 1.0), ("RISE", 3.0), ("IDLE2W", 3.0),
             ("JUMP", 2.0), ("LOWER", 1.6), ("HOLD4W", 1.0)]

# 趴下 switch 相位时序（console 同款：HOLD2W 0.3 → LOWER 0.5 → TILT 0.3 → ABSORB 0.4 → 0）
LIE_T = (0.30, 0.80, 1.10, 1.50)
LIE_END = 1.50

# 事件驱动：过渡请求的站定门（与 P7 链检 wait_settle 同口径）
SETTLE_VX = 0.08
SETTLE_PITCH = 0.10          # rad ≈ 5.7°
SETTLE_STEPS = 8
SETTLE_TIMEOUT = 1.5         # s，超时也执行（不让指令石沉大海）

# 事件驱动模式的守卫表：当前槽 → 允许的请求（与 envs/wbr/uni/fsm.py 的 rejected 语义一致）
ALLOWED = {
    "IDLE2W": ("jump", "lower", "four", "reset", "next"),
    "JUMP": ("reset",),
    "LOWER": ("reset",),
    "HOLD4W": ("rise", "two", "reset", "next"),
    "RISE": ("reset",),
}
# 面板按钮 → 链请求（桥 /api/action 用这套映射）
ACTION_TO_REQ = {
    "armjump": "jump", "jump": "jump",
    "lower": "lower", "rise": "rise",
    "toggle": "next", "next": "next",
    "reset": "reset",
    "two": "two", "four": "four",
}
REQ_LABEL = {
    "jump": "装填跳", "lower": "趴下", "four": "回四轮", "rise": "起立",
    "two": "回两轮", "reset": "复位", "next": "下一步",
}


def _lie_phase(t):
    if t < LIE_T[0]:
        return 3
    if t < LIE_T[1]:
        return 4
    if t < LIE_T[2]:
        return 5
    if t < LIE_T[3]:
        return 6
    return 0


class UniChain:
    """一条链的运行时状态：槽推进、契约切换、事件 marks、终止判定、指令队列。"""

    def __init__(self, start="2w", h_jump=0.10, vx=0.3, p7t_params=None,
                 interactive=False, allow_rise=False, terrain=False):
        self.terrain = bool(terrain)
        self.tinfo = json.load(open(TERRAIN_META, encoding="utf-8")) if self.terrain else None
        self.c4 = load_contract(os.path.join(PKG, "assets/deploy_contract.json"))
        self.c2 = load_contract(os.path.join(PKG, "assets/deploy_contract_two_v2.json"))
        self.cl = load_contract(os.path.join(PKG, "assets/deploy_contract_lie.json"))
        self.ck4 = resolve_model(self.c4)
        self.ck2 = resolve_model(self.c2, "releases/P8W7_push01_fric05", "model_2800.pt")
        self.ckl = resolve_model(self.cl)
        self.ckj = resolve_model(self.c2, "releases/P8J3d_wheelfollow_lite", "model_5650.pt")
        if self.terrain:
            sx, sy = self.tinfo["spawn_xy"]
            for c in (self.c4, self.c2, self.cl):
                c["initial_root_pos"] = [sx, sy, float(c["initial_root_pos"][2])]
        self.h_jump = float(h_jump)
        self.vx = float(vx)
        self.h_2w = 0.250          # 两轮默认站立高度（作者 2026-11-06 定；面板可调，切槽沿用）
        self.p7t_params = dict(p7t_params or {})
        self.interactive = bool(interactive)
        self.allow_rise = bool(allow_rise)
        self.start = "4w" if start == "4w" else "2w"

        self.plan = PLAN_FULL if self.start == "4w" else PLAN_2W
        if self.terrain and not self.interactive:
            # 自动播放（CLI 评估）：地形模式跳过跳跃槽——P8J3d 是平地/行进跳，
            #   在 demo 的起伏/台阶上落地必翻。交互模式下用户仍可自行装填（守卫会提示）。
            self.plan = [p for p in self.plan if p[0] != "JUMP"]
        mjcf = os.path.join(PKG, "assets/mjcf/wbr_v2_demo.xml") if self.terrain else MJCF
        self.loop = SimLoop(mjcf, self.c4, ckpt=self.ck4, mode="policy")
        # 8 dof 载体：链全程不重建（历史/default 连续），只换 obs 布局
        self.loop.obs_b = ObsBuilder(self.c4, dof_names=self.c4["dof_names"],
                                     default_dof_pos=self.c4["default_dof_pos"])
        self.loop.slew = False
        self._inbox = deque()            # HTTP 线程 append / 主循环 popleft（deque 两端操作线程安全）
        self.slot_i = -1
        self.slot = None
        self.t_slot = 0.0
        self.t_total = 0.0
        self.phase = 6 if self.start == "4w" else 0
        self.marks = []
        self.events = []
        self.last_request = None
        self.n_requests = 0
        self.n_rejected = 0
        self.done = False
        # 事件驱动：过渡请求排队 + 站定门
        self._pending = None             # (req, value, 已等待秒数, 连续站定拍数)
        self._jump_seen = False
        self._enter(self.plan[0][0])
        if self.start != "4w":
            self._init_2w_pose()
        self.marks[-1] = (self.slot, 0.0, round(float(self.loop.sim.d.qpos[2]), 4))

    # ------------------------------------------------------------------ 指令
    def request(self, req, value=None):
        """面板/桥调用：排队一条指令，返回「已排队」给 HTTP 线程（结果由 /api/state 回显）。"""
        req = ACTION_TO_REQ.get(str(req), str(req))
        self._inbox.append((req, value))
        return True

    def _drain(self):
        out = []
        while self._inbox:
            out.append(self._inbox.popleft())
        return out

    # ------------------------------------------------------------------ 切换
    def _switch(self, contract):
        self.loop.c = contract
        self.loop.obs_b.set_contract(contract)
        self.loop.ctl.set_contract(contract)
        self.loop.wheel_i = [i for i, n in enumerate(contract["dof_names"]) if "wheel" in n]

    def _set_legs(self, pose):
        for k, idx in enumerate([self.loop.c["dof_names"].index(n) for n in LEGS4]):
            self.loop.ctl.default[idx] = float(pose[k])

    def _init_2w_pose(self):
        """两轮起步：把初始状态摆到两轮站姿（z2w＋5 mm 净空），否则 4W 蹲姿直接倒。"""
        import mujoco
        for n, v in zip(self.c4["dof_names"], self.c4["default_dof_pos"]):
            self.loop.sim.d.qpos[self.loop.sim.qadr[n]] = v
        for n, v in zip(("left_hip_joint", "left_knee_joint",
                         "right_hip_joint", "right_knee_joint"), POSE_2W):
            self.loop.sim.d.qpos[self.loop.sim.qadr[n]] = v
        # 初始高度＝**指令高度**（与滑条默认/空格复位同值：两轮 0.25），实测两者都能稳站
        #   （0.25 起始比站姿几何 0.2747 低 25 mm，PD 一拍内收敛；这样"初始姿态/默认显示/复位"三处一致）
        self.loop.sim.d.qpos[2] = float(self.h_2w)
        self.loop.sim.d.qvel[:] = 0.0
        mujoco.mj_forward(self.loop.sim.m, self.loop.sim.d)
        self.loop.warmup(3)

    def _enter(self, name):
        prev_slot = self.slot                  # 【P7T2】交棒来源槽（IDLE2W 渐入判断要用）
        self.slot = name
        self.t_slot = 0.0
        self.slot_i = next((i for i, (n, _d) in enumerate(self.plan) if n == name), 0)
        loop = self.loop
        loop.p7t = None
        loop.term_pitch_ovr = None
        loop.term_roll_ovr = None
        if name == "HOLD4W":
            self._switch(self.c4)
            loop.pol = Policy(self.ck4, 164, 6)
            loop.jt = None
            loop.ctl.off_knee = 0.0
            loop.ctl.off_hip = 0.0
            loop.ctl.leg_override = None
            loop.ctl.wheel_override = None
            self._set_legs([self.c4["default_dof_pos"][self.c4["dof_names"].index(n)]
                            for n in LEGS4])
            loop.set_cmd("base_height", 0.19)
            loop.set_cmd("lin_vel_x", 0.0)
            loop.snap_cmds()
            self.phase = 6
        elif name == "RISE":
            # 【P7T2】standup_engine: "p7t2"（平台式甩+接，MuJoCo 验证通过；旧值 "p9" 兼容读）|"p7t"（传统甩尾，MJ 未收口）
            engine = str(self.p7t_params.get("standup_engine", "p7t2")) if self.p7t_params else "p7t2"
            if engine in ("p7t2", "p9"):  # "p9"＝2026-11-06 前旧代号，兼容读
                loop.activate_p7t2(dict(kp=0, kd=0, kd2=2.0, T=0.5, k_roll=2.5, throw_c=0.4))
                loop.term_pitch_ovr = 1.3
                loop.term_roll_ovr = 1.3
                loop.caster_air_ok = True
            else:
                loop.activate_p7t(self.p7t_params or None)
                loop.term_pitch_ovr = 1.0
            self.phase = 7
        elif name == "IDLE2W":
            from_ = prev_slot
            self._switch(self.c2)
            loop.pol = Policy(self.ck2, 161, 6)
            loop.jt = None
            loop.ctl.off_knee = -0.1          # w3 训练分布（flat 语义）
            loop.ctl.off_hip = 0.0
            loop.jump["off"] = -0.1
            loop.ctl.leg_override = None
            loop.ctl.wheel_override = None
            loop.ctl.wheel_override_l = None
            loop.ctl.wheel_override_r = None
            self._set_legs(POSE_2W)
            if from_ == "RISE":
                # 【P7T2】交棒（scan13 PASS 全配置）：冷启动（历史/轮速环/last_action 清零）
                #   + 2s 慢渐入 + h=0.19 保持甩接期口径（h_2w 稍后恢复）
                loop.obs_b.reset()
                loop.ctl.reset()
                loop.last_action = np.zeros(loop.c["act_dim"], dtype=np.float32)
                ramp_n = int(2.0 / float(self.c2["control_dt"]))
                loop.action_ramp = (ramp_n, ramp_n)
                loop.term_pitch_ovr = 1.0
                loop.term_roll_ovr = 1.0
                self._grace_until = self.t_total + 2.0
                loop.set_cmd("lin_vel_x", 0.0)
                loop.set_cmd("base_height", 0.19)   # scan13 PASS 口径（勿改 h_2w）
                self._pending_vx = self.vx
                self._pending_h = self.h_2w
            else:
                loop.set_cmd("lin_vel_x", self.vx)
            # 高度：面板在本槽内调过就沿用（链不再硬写死），否则用默认 0.25
            loop.set_cmd("base_height", float(self.h_2w))
            if from_ != "RISE":
                loop.snap_cmds()      # 【P7T2】交棒不 snap：甩接期间 cmds 残留（h=0.19）与新
                                      # 契约混算的瞬变，scan13（无 snap）PASS、有 snap 必翻
            self.phase = 0
        elif name == "JUMP":
            # 进入即切 P8J3d 与相位机；装填由 step() 的站定门在姿态平稳后执行
            loop.pol = Policy(self.ckj, 161, 6)
            loop.jt = JumpTemplate(k_lim=tuple(self.c2["knee_limits"]),
                                   h_lim=tuple(self.c2["hip_limits"]),
                                   dt=float(self.c2["control_dt"]))
            loop.set_cmd("lin_vel_x", 0.0)
            loop.snap_cmds()
            self.phase = 1
            self._jump_seen = False
            self._jump_armed = not self.interactive
            self._quiet_steps = 0
            if not self.interactive:
                loop.jump_request(self.h_jump)
        elif name == "LOWER":
            self._switch(self.cl)
            loop.pol = Policy(self.ckl, 173, 6)
            loop.jt = None
            loop.set_switch_phase(3)
            loop.set_cmd("lin_vel_x", 0.0)
            loop.set_cmd("base_height", 0.19)
            loop.snap_cmds()
            self.phase = 9
        self.marks.append((name, round(self.t_total, 2),
                           round(float(loop.sim.d.qpos[2]), 4)))
        self.events.append({"slot": name, "t": round(self.t_total, 3),
                            "z": round(float(loop.sim.d.qpos[2]), 4), "phase": self.phase})

    # ------------------------------------------------------------------ 守卫
    def _handle(self, req, value=None):
        self.n_requests += 1
        allowed = ALLOWED.get(self.slot, ())
        t = round(self.t_total, 3)
        rec = {"t": t, "slot": self.slot, "event": "request",
               "req": req, "label": REQ_LABEL.get(req, req)}
        reason = None

        if req == "reset":
            self._reset()
            rec["accepted"] = True
            rec["reason"] = "已冷启动复位到起始槽"
        elif req not in allowed:
            self.n_rejected += 1
            rec["accepted"] = False
            reason = f"当前槽 {self.slot} 不接受「{REQ_LABEL.get(req, req)}」"
            if self.slot in ("JUMP", "LOWER", "RISE"):
                reason += "（过渡态禁止插队，可先复位）"
            rec["reason"] = reason
        elif req == "rise" and self.allow_rise is False and self.p7t_params and \
                self.p7t_params.get("standup_engine") == "p7t":
            # 仅当显式选 p7t 引擎时拒绝（P7T 在 MJ 未收口）；P7T2 默认引擎已收口
            self.n_rejected += 1
            rec["accepted"] = False
            reason = "P7T 脚本起立在 MuJoCo 未收口（抬升段弹射）；请用 P7T2 引擎（standup_engine=p7t2，默认）"
            rec["reason"] = reason
        elif self._pending is not None and req != "reset":
            # 已有一条过渡指令在等站定门：不覆盖、不插队，明确回绝（避免「接受 2 条只执行 1 条」）
            self.n_rejected += 1
            rec["accepted"] = False
            reason = "已有指令「" + REQ_LABEL.get(self._pending[0], self._pending[0]) + "」在等待执行，请等它完成或复位"
            rec["reason"] = reason
        else:
            # 过渡请求：排队等站定门，当前槽继续跑（不立刻切网）。
            # 实测任意时刻立刻切 161/173 网会让策略吃分布外输入而翻倒。
            self._pending = (req, value, 0.0, 0)
            rec["accepted"] = True
            rec["reason"] = "已接受，等姿态平稳后执行（" + REQ_LABEL.get(req, req) + "）"
            if req == "jump" and self.terrain:
                rec["reason"] += " ⚠ 跳跃策略在复杂地形未验证，建议到中央十字走廊平地再跳"
        self.last_request = rec
        self.events.append(rec)
        if reason:
            print(f"⛔ [chain] 指令被拒：{reason}", flush=True)
        else:
            print(f"✅ [chain] 指令 {req} 已排队", flush=True)

    def _apply_pending(self, req, value=None):
        """站定门满足/超时后执行真正的槽切换。"""
        if req == "jump":
            self._enter("JUMP")
        elif req == "lower":
            self._enter("LOWER")
        elif req == "four":
            self._enter("HOLD4W")
        elif req == "rise":
            self._enter("RISE")
        elif req == "two":
            self.loop.reset()
            self._init_2w_pose()
            self._enter("IDLE2W")
        elif req == "next":
            self._enter("LOWER" if self.slot == "IDLE2W" else "IDLE2W")
            if self.slot == "IDLE2W":
                self.loop.reset()
                self._init_2w_pose()
        else:
            return
        self.events.append({"t": round(self.t_total, 3), "slot": self.slot,
                            "event": "applied", "req": req,
                            "label": REQ_LABEL.get(req, req) + " 已执行"})
        print(f"✅ [chain] 指令 {req} 已执行 → {self.slot}", flush=True)

    def _reset(self):
        self.loop.reset()
        self.loop.warmup(1)
        self._pending = None
        self.done = False
        self.t_total = 0.0
        self.vx = 0.0                     # 【复位】回到起始状态：速度归零（作者口径：初始速度应为 0）
        self.loop.set_cmd("lin_vel_x", 0.0)
        self.marks = []
        self.events = []
        self._jump_seen = False
        self._enter(self.plan[0][0])
        if self.start != "4w":
            self._init_2w_pose()

    # ------------------------------------------------------------------ 站定门
    def _pose_quiet(self):
        """物理状态是否平稳（不依赖 loop.rows，record=False 也能用）。"""
        loop = self.loop
        R, _q, lin_b, _ang = loop.sim.base_state()
        pg = loop.sim.projected_gravity(R)
        pitch, _roll = loop.sim.body_pitch_roll(pg)
        return abs(float(lin_b[0])) <= SETTLE_VX and abs(float(pitch)) <= SETTLE_PITCH

    # ------------------------------------------------------------------ 一拍
    def step(self, record=True):
        if self.done:
            return False
        for req, val in self._drain():
            self._handle(req, val)
            if self.done:
                return False
        loop = self.loop
        dt = float(loop.c["control_dt"])

        # —— 过渡请求排队：当前槽继续跑，等站定门或超时 ——
        if self._pending is not None and self.interactive:
            req, val, waited, quiet = self._pending
            waited += dt
            quiet = quiet + 1 if self._pose_quiet() else 0
            if quiet >= SETTLE_STEPS or waited >= SETTLE_TIMEOUT:
                self._pending = None
                self._apply_pending(req, val)
            else:
                self._pending = (req, val, waited, quiet)

        # 趴下 slot：按外部时序推进 switch 相位
        if self.slot == "LOWER":
            ph = _lie_phase(self.t_slot)
            loop.set_switch_phase(ph)
            self.phase = ph
        elif self.slot == "JUMP":
            self.phase = int(loop.jump["ph"]) if loop.jump["ph"] else 1
        elif self.slot == "RISE" and loop.p7t_done:
            self.phase = 8

        loop.step(record=record)
        self.t_slot += dt
        self.t_total += dt
        if loop.terminated:
            self.events.append({"slot": self.slot, "t": round(self.t_total, 3),
                                "event": "terminated", "why": loop.terminated})
            self.done = True
            return False

        if not self.interactive:
            # 自动播放：按固定时长表推进（CLI 评估口径，产物可复现）
            dur = self.plan[self.slot_i][1]
            if self.t_slot >= dur:
                if self.slot_i + 1 >= len(self.plan):
                    self.done = True
                    return False
                self._enter(self.plan[self.slot_i + 1][0])
            return True

        # grace 期结束 → 恢复正常姿态阈 + 补发行走速度
        if getattr(self, "_grace_until", None) is not None and self.t_total >= self._grace_until:
            loop.term_pitch_ovr = None
            loop.term_roll_ovr = None
            self._grace_until = None
            if getattr(self, "_pending_vx", None) is not None:
                loop.set_cmd("lin_vel_x", float(self._pending_vx))
                loop.set_cmd("base_height", float(getattr(self, "_pending_h", self.h_2w)))
                self._pending_vx = None
        # 事件驱动：槽位停住，只有物理事件或新指令才切
        if self.slot == "JUMP":
            ph = int(loop.jump["ph"])
            if ph > 0:
                self._jump_seen = True
            elif self._jump_seen:
                # 跳完自动回两轮行走：
                #   地形模式＝原地交还 P8W7 网（保留地形位置/速度/历史，继续在复杂地形上走）；
                #   平地模式＝重建干净站姿（历史验证过的口径，避免落地残余动态带进趴下）。
                if not self.terrain:
                    self.loop.reset()
                    self._init_2w_pose()
                self._enter("IDLE2W")
                print("✅ [chain] 跳跃完成 → " +
                      ("原地交还 P8W7，继续复杂地形行走" if self.terrain else "重建两轮站姿") +
                      "，等下一条指令", flush=True)
            elif not self._jump_armed:
                self._quiet_steps = self._quiet_steps + 1 if self._pose_quiet() else 0
                if self._quiet_steps >= SETTLE_STEPS:
                    loop.set_cmd("lin_vel_x", 0.0)
                    loop.snap_cmds()
                    loop.jump_request(self.h_jump)
                    self._jump_armed = True
                    print(f"✅ [chain] 已站定（{self._quiet_steps} 拍）→ 装填跳 {self.h_jump}", flush=True)
        elif self.slot == "LOWER" and self.t_slot >= LIE_END:
            self._enter("HOLD4W")            # 趴下时序走完停到四轮，等下一条指令
        elif self.slot == "RISE" and loop.p7t_done:
            self._enter("IDLE2W")
        return True

    # ------------------------------------------------------------------ 读数
    def status(self):
        loop = self.loop
        d = loop.sim.d
        R, _q, lin_b, _ang = loop.sim.base_state()
        pg = loop.sim.projected_gravity(R)
        pitch_now, _roll_now = loop.sim.body_pitch_roll(pg)
        return {
            "slot": self.slot,
            "phase": int(self.phase),
            "t": round(self.t_total, 2),
            "slot_t": round(self.t_slot, 2),
            "z": round(float(d.qpos[2]), 4),
            "vx": round(float(lin_b[0]), 3),
            "pitch_deg": round(float(np.degrees(pitch_now)), 1),
            "terminated": loop.terminated,
            "marks": list(self.marks),
            "events": list(self.events),
            "interactive": self.interactive,
            "allowed": list(ALLOWED.get(self.slot, ())),
            "lastRequest": self.last_request,
            "nRequests": self.n_requests,
            "nRejected": self.n_rejected,
            "allowRise": self.allow_rise,
            "pending": (self._pending[0] if self._pending else None),
        }

    def summary(self):
        return {
            "mode": "interactive" if self.interactive else "auto",
            "start": self.start,
            "plan": [(n, d) for n, d in self.plan],
            "marks": [list(m) for m in self.marks],
            "events": self.events,
            "frames": len(self.loop.rows),
            "total_s": round(self.t_total, 2),
            "term": self.loop.terminated,
            "n_requests": self.n_requests,
            "n_rejected": self.n_rejected,
            "z_min": round(min((r["z"] for r in self.loop.rows), default=0.0), 4),
            "z_max": round(max((r["z"] for r in self.loop.rows), default=0.0), 4),
        }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2w", choices=("2w", "4w"))
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--jump", type=float, default=0.10)
    ap.add_argument("--out", default="chain_uni.json")
    ap.add_argument("--interactive", action="store_true", help="事件驱动（不自动放完；CLI 下需自行注入指令）")
    ap.add_argument("--allow-rise", action="store_true")
    ap.add_argument("--terrain", action="store_true", help="两轮槽用 wbr_p8_demo 四象限复杂地形")
    a = ap.parse_args()
    chain = UniChain(start=a.start, h_jump=a.jump, vx=a.vx,
                     interactive=a.interactive, allow_rise=a.allow_rise,
                     terrain=a.terrain)
    while chain.step():
        pass
    s = chain.summary()
    out = os.path.join(PKG, "../../..", "eval_runs/2026-11-05_S3_五槽uni整链")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, a.out), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    print(json.dumps(s, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
