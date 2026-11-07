# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : dump_isaac_golden.py
# Purpose   : 采集链路 golden：Isaac Gym（CPU 管线）1 env，名义口径，按场景跑策略并逐步记录
#             状态/观测/动作/力矩（含两个物理子步各自的 dof 状态与力矩），供 MuJoCo 部署链回放对账。
#             运行环境：conda activate isaacgym01（需 LD_LIBRARY_PATH=$CONDA_PREFIX/lib）
# ==============================================================================

import argparse
import json
import os
import sys

import numpy as np

import isaacgym  # noqa: E402,F401  必须先于 torch
import torch  # noqa: E402
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402

from legged_gym.scripts.sim2sim_mujoco.tools._isaac_common import patch_nominal, assert_nominal

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(PKG, "../../.."))
RELEASE = "releases/P7F3e_yaw1_dr15"


class TorqueRecorder:
    """包住 env._compute_torques，逐步记录每个物理子步的输入状态与输出力矩。"""

    def __init__(self, env):
        self.env = env
        self.calls = []
        self._orig = env._compute_torques
        env._compute_torques = self._fn

    def _fn(self, actions):
        out = self._orig(actions)
        self.calls.append({
            "tau": out[0].detach().cpu().numpy().copy(),
            "dof_pos": self.env.dof_pos[0].detach().cpu().numpy().copy(),
            "dof_vel": self.env.dof_vel[0].detach().cpu().numpy().copy(),
            "actions": actions[0].detach().cpu().numpy().copy(),
        })
        return out

    def take(self):
        out = self.calls
        self.calls = []
        return out


def load_policy(ckpt, obs_dim, act_dim, device="cpu"):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    pol = ActorCritic(obs_dim, obs_dim, act_dim,
                      actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                      activation="elu", init_noise_std=1.0).to(device)
    pol.load_state_dict(ck["model_state_dict"], strict=True)
    pol.eval()
    return pol


def set_commands(env, vx, yaw, h, jump=0.0):
    c = torch.zeros(7, device=env.device, dtype=torch.float)
    c[0], c[2], c[5] = float(vx), float(yaw), float(h)
    c[4] = float(jump)                     # 【S2b 跳跃】跳跃指令通道（平滑后驱动 h_request）
    env.commands[0] = c
    env.command_targets[0] = c
    # 包络 `_apply_command_envelope` 从 command_raw_yaw 读原始偏航并写回 command_targets，
    # 不设它的话 yaw 会被复位时随机采样的旧值覆盖（vx/height 不受影响，只有 yaw 会漂）。
    env.command_raw_yaw[0] = float(yaw)


def run_scenario(env, pol, rec, name, vx, yaw, h, steps, h2=None, switch=100, jump=0.0, pol2=None):
    pol_stance = pol2 or pol                  # 站立段网络（lie 双网接力：HOLD2W 用 w3）
    set_commands(env, vx, yaw, h)
    obs, _ = env.reset()
    env.actions[:] = 0.0          # 每个场景从"上一动作=0"开始（obs 的 last-action 通道）
    set_commands(env, vx, yaw, h)
    # reset 的 obs 用的是随机指令；置好指令后重算一次，保证第 0 帧与后续帧同口径
    env.compute_observations()
    if name == "lie":
        # 【S2c 趴下·A 口径】接受 switch env 的 off=0（obs[159]=0、无 additive）。
        # 运行语义＝console 五槽链：playback_hold＋外部趴下时序脚本，策略（P7L@300）跟随相位。
        env.playback_hold = True
        env._reset_root_states(torch.tensor([0], device=env.device))
        env._reset_dofs(torch.tensor([0], device=env.device))
        env.playback_set_mode(torch.tensor([0], device=env.device), ph=3, task=0, ph_t=0.0)
        env.compute_observations()
        env.actions[0] = 0.0
        print("[lie] A 口径起点：2W 站姿/HOLD2W/task=0/off=0", flush=True)

    if jump > 0.0:
        # 【S2b 跳跃】评测/回放的装填口径：直接写 h_request＋is_armed（h_request 只由指令重采样
        # 设置，写 commands[4] 不进触发链）；站立期 command_targets[:,4] 恒 0，触发拍相位机写入。
        env.h_request[0] = float(jump)
        env.is_armed[0] = True
    obs = torch.clip(env.obs_buf, -env.cfg.normalization.clip_observations,
                     env.cfg.normalization.clip_observations).clone()
    # ⚠ env.reset_idx 里的 history[env_ids].zero_() 因高级索引取到副本而**不生效**（既有 bug，
    #   不改环境码）：历史会跨回合残留 ⇒ 必须把本场景起始历史原样记下来，回放才能同初值。
    hist0_pos = env.dof_pos_error_history[0].detach().cpu().numpy().copy()
    hist0_vel = env.dof_vel_history[0].detach().cpu().numpy().copy()
    # reset() 内含一步 step ⇒ 轮速环的 D 项状态已被推进过；照原样记录，回放同初值。
    ctrl0_last = env.last_wheel_control_vel[0].detach().cpu().numpy().reshape(-1).copy()
    ctrl0_filt = env.filtered_wheel_acceleration[0].detach().cpu().numpy().reshape(-1).copy()
    ctrl0_target = env.wheel_target_vel[0].detach().cpu().numpy().reshape(-1).copy()

    data = {k: [] for k in [
        "root", "dof_pos", "dof_vel", "commands", "last_action", "obs", "act", "tau",
        "call_dof_pos", "call_dof_vel", "call_actions", "rb_state", "caster_force",
        "caster_contacts", "tip_margin", "ph", "off_knee", "off_hip", "h_target", "air_z", "air_latched",
    ]}

    for t in range(steps):
        if h2 is not None and t == switch:            # 【S2 两轮】高度过渡：站稳 1 s 后再切目标高度
            set_commands(env, vx, yaw, h2)
        # ---- 观测时刻的状态（Isaac 的顺序：obs 由上一拍 step 末尾算出） ----
        root = env.root_states[0].detach().cpu().numpy().copy()
        dof_pos = env.dof_pos[0].detach().cpu().numpy().copy()
        dof_vel = env.dof_vel[0].detach().cpu().numpy().copy()
        last_action = env.actions[0].detach().cpu().numpy().copy()
        commands = env.commands[0].detach().cpu().numpy().copy()
        rb_state = env._rb_state[0].detach().cpu().numpy().copy()
        if env.num_dof == 8 and hasattr(env, "caster_body_indices"):
            caster_force = env.contact_forces[0, env.caster_body_indices].detach().cpu().numpy().copy()
            caster_contacts = env._caster_contacts()[0].detach().cpu().numpy().copy()
            tip_margin = float(env._tip_margin()[0])
        else:
            # 【S2 两轮】v1 资产无从动轮：字段占位 0（two-parity 不读，结构保持 npz 一致）
            caster_force = np.zeros(1)
            caster_contacts = np.zeros(2)
            tip_margin = 0.0
        ph = int(env.ph[0])
        off_knee = float(env.off_knee[0])
        off_hip = float(env.off_hip[0]) if hasattr(env, "off_hip") else 0.0
        h_target = float(env.h_target[0])
        air_z = float(env.air_z[0])
        air_latched = float(env.air_latched[0].item()) if hasattr(env.air_latched[0], "item") else float(env.air_latched[0])

        obs_t = obs[0].detach().cpu().numpy().copy()
        with torch.no_grad():
            # 【lie 双网接力】HOLD2W(3) 段用站立网（164 前缀观测）；LOWER(4) 起切趴下网（173 全维）
            _net = pol                                # 【S2c】单网 P7L：HOLD2W 对齐站立＋LOWER 趴下同网（sw_align 训练语义）
            _obs_in = obs
            act = _net.act_inference(_obs_in).detach().cpu().numpy()[0].copy()

        rec.take()
        obs, _, _, reset_buf, _ = env.step(torch.from_numpy(act).unsqueeze(0).to(env.device))
        calls = rec.take()
        assert len(calls) == env.cfg.control.decimation, f"子步数异常 {len(calls)}"
        if bool(reset_buf.any()):
            if getattr(env, "sw_done", None) is not None and bool(env.sw_done[0]):
                print(f"  [done] {name} 第 {t} 步任务完成（sw_done），正常收卷", flush=True)
                break
            raise RuntimeError(f"场景 {name} 第 {t} 步触发 reset（终止条件），golden 不可用")

        if name == "lie":
            # 趴下时序（console 参数）：HOLD2W 0.3s → LOWER 0.5s → TILT 0.3s → ABSORB 0.4s → HOLD4W
            _tt = t * 0.01
            if _tt < 0.30: _p, _pt = 3, _tt
            elif _tt < 0.80: _p, _pt = 4, _tt - 0.30
            elif _tt < 1.10: _p, _pt = 5, _tt - 0.80
            elif _tt < 1.50: _p, _pt = 6, _tt - 1.10
            else: _p, _pt = 0, _tt - 1.50
            env.playback_set_mode(torch.tensor([0], device=env.device), ph=_p, task=0, ph_t=_pt)
        if name == "lie" and t % 10 == 0:
            import math as _m
            _c = env._caster_contacts()[0].detach().cpu().numpy()
            _w = (env.contact_forces[0, env.feet_indices, 2] > 0.1).detach().cpu().numpy()
            print(f"[lie] t={t} ph={int(env.ph[0])} z={float(root[2]):.3f} pitch={float(env._body_pitch()[0]):+.3f} "
                  f"cast={_c.astype(int)} wheel={_w.astype(int)} done={bool(env.sw_done[0])}", flush=True)
        data["root"].append(root)
        data["dof_pos"].append(dof_pos)
        data["dof_vel"].append(dof_vel)
        data["commands"].append(commands)
        data["last_action"].append(last_action)
        data["obs"].append(obs_t)
        data["act"].append(act)
        data["tau"].append(calls[0]["tau"])
        data["call_dof_pos"].append(np.stack([c["dof_pos"] for c in calls]))
        data["call_dof_vel"].append(np.stack([c["dof_vel"] for c in calls]))
        data["call_actions"].append(np.stack([c["actions"] for c in calls]))
        data["rb_state"].append(rb_state)
        data["caster_force"].append(caster_force)
        data["caster_contacts"].append(np.asarray(caster_contacts, dtype=np.float64))
        data["tip_margin"].append(np.asarray([tip_margin]))
        data["ph"].append(np.asarray([ph]))
        data["off_knee"].append(np.asarray([off_knee]))
        data["off_hip"].append(np.asarray([off_hip]))
        data["h_target"].append(np.asarray([h_target]))
        data["air_z"].append(np.asarray([air_z]))
        data["air_latched"].append(np.asarray([air_latched]))

    out = {k: np.stack(v) for k, v in data.items()}
    out["hist0_pos"] = hist0_pos
    out["hist0_vel"] = hist0_vel
    out["ctrl0_last"] = ctrl0_last
    out["ctrl0_filt"] = ctrl0_filt
    out["ctrl0_target"] = ctrl0_target
    out["meta"] = np.asarray([name, vx, yaw, h, steps])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, RELEASE, "model_11800.pt"))
    ap.add_argument("--out", default=os.path.join(PKG, "eval/golden/isaac_golden.npz"))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--task", default="wbr_flat4", choices=["wbr_flat4", "wbr_flat", "wbr_switch"],
                    help="四轮 164（默认）/ 两轮 161（S2）/ 趴下 173（S2c）：各用现役槽交付权重")
    ap.add_argument("--matrix", action="store_true",
                    help="命令矩阵场景集（vx 全幅/负向/大 yaw/组合高度），写独立 npz，不动 S0 冻结 golden")
    ap.add_argument("--ckpt2", default=os.path.join(ROOT, "releases/P8W7_push01_fric05/model_2800.pt"),
                    help="【lie 双网接力】站立段网络（w3 槽 164 维）：HOLD2W 段用它站住，LOWER 边沿切主 ckpt")
    ap.add_argument("--only", default=None,
                    help="只采指定场景（逗号分隔）。跨回合历史残留 bug（reset_idx 不清零）会让部分"
                         "连跑场景互扰——分进程逐场景采集是规避手段")
    a = ap.parse_args()
    if a.task == "wbr_flat":
        a.ckpt = os.path.join(ROOT, "releases/P8W7_push01_fric05/model_2800.pt")
        if a.out == os.path.join(PKG, "eval/golden/isaac_golden.npz"):
            a.out = os.path.join(PKG, "eval/golden/isaac_golden_two.npz")
    if a.task == "wbr_switch":
        a.ckpt = os.path.join(ROOT, "releases/P7L_liedown/model_300.pt")
        if a.out == os.path.join(PKG, "eval/golden/isaac_golden.npz"):
            a.out = os.path.join(PKG, "eval/golden/isaac_golden_lie.npz")

    sys.argv = ["dump_golden", "--task", a.task, "--headless",
                "--num_envs=1", "--sim_device", "cpu", "--pipeline", "cpu"]

    patch_nominal(a.task)
    args = get_args()
    env, _ = task_registry.make_env(name=args.task, args=args)
    assert_nominal(env)
    env.add_noise = False

    if a.task == "wbr_switch":
        env.sw_task[0] = 0                     # 【S2c 趴下】reset 前定任务：起点=2W 站姿、相位 HOLD2W
    pol = load_policy(a.ckpt, env.num_obs, env.num_actions)
    pol2 = load_policy(a.ckpt2, 161, 6) if a.task == "wbr_switch" else None   # w3 槽=161 维两轮网
    rec = TorqueRecorder(env)

    scenarios = [
        ("stand", 0.0, 0.0, 0.19),
        ("walk03", 0.3, 0.0, 0.21),
        ("walk06", 0.6, 0.0, 0.21),
        ("height023", 0.0, 0.0, 0.23),
        ("height0175", 0.0, 0.0, 0.175),
        ("height0245", 0.0, 0.0, 0.245),
        ("yaw05", 0.0, 0.5, 0.19),
    ]
    if a.task == "wbr_switch":
        # 【S2c 趴下】reset 后 playback_set_mode(HOLD2W, task=0)，相位机自主推进到 HOLD4W；
        #   sw_done 触发 reset ⇒ run_scenario 用 sw_done 提前停（正常收卷，不计异常）。
        scenarios = [("lie", 0.0, 0.0, 0.19)]
    elif a.task == "wbr_flat":
        # 【S2 两轮】站立/行走/高度档/转向（h 量程 [0.20, 0.35]，站姿 0.28）。
        #   高度档用**过渡协议**：0.28 站稳 1 s 后切目标高度——瞬切会触发终止（0.38 s 就 reset）。
        scenarios = [
            ("stand", 0.0, 0.0, 0.28),
            ("walk03", 0.3, 0.0, 0.28),
            ("walk06", 0.6, 0.0, 0.28),
            ("h020", 0.0, 0.0, 0.28),
            ("h035", 0.0, 0.0, 0.28),
            ("yaw05", 0.3, 0.5, 0.28),
            ("jump10", 0.0, 0.0, 0.28),
        ]
        h_switch = {"h020": (0.20, 100), "h035": (0.35, 100)}
        jump_scenarios = {"jump10": 0.10}      # 【S2b】站立装填跳 0.10 m（0.28 站姿、settled 门自然满足）
    else:
        h_switch = {}
        jump_scenarios = {}
    if a.task == "wbr_switch":
        h_switch = {}
        jump_scenarios = {}
    if a.matrix:
        # 【S1 命令矩阵】vx 全幅＋负向 / 大 yaw 双向 / 组合高度。push 推扰只在 MuJoCo 侧做能力项
        #   （Isaac 侧速度注入需 set_actor_root_state 专线，与本工具录 golden 协议不同，不混入）。
        scenarios = [
            ("m10", -1.0, 0.0, 0.21),
            ("m06", -0.6, 0.0, 0.21),
            ("v09", 0.9, 0.0, 0.21),
            ("v10", 1.0, 0.0, 0.21),
            ("yn05", 0.3, -0.5, 0.19),
            ("yn10", 0.3, -1.0, 0.19),
            ("y10", 0.3, 1.0, 0.19),
            ("c05h245", 0.5, 0.0, 0.245),
            ("c05h175", 0.5, 0.0, 0.175),
        ]
        if a.out == os.path.join(PKG, "eval/golden/isaac_golden.npz"):
            a.out = os.path.join(PKG, "eval/golden/isaac_golden_matrix.npz")
    if a.only:
        keep = {s.strip() for s in a.only.split(",")}
        scenarios = [s for s in scenarios if s[0] in keep]
        assert scenarios, f"--only 没匹配到场景：{a.only}"
        if not a.matrix and len(keep) != len(a.only.split(",")):
            pass
    result = {}
    for name, vx, yaw, h in scenarios:
        h2, sw = h_switch.get(name, (None, None))
        jmp = jump_scenarios.get(name, 0.0)
        result[name] = run_scenario(env, pol, rec, name, vx, yaw, h, a.steps, h2=h2, switch=sw or 100, jump=jmp, pol2=pol2)
        print(f"  [ok] {name}: {a.steps} steps  z_end={result[name]['root'][-1][2]:.4f} "
              f"vx_end={result[name]['root'][-1][7]:.3f}", flush=True)

    meta = {
        "task": args.task,
        "ckpt": os.path.relpath(a.ckpt, ROOT),
        "obs_dim": int(env.num_obs),
        "act_dim": int(env.num_actions),
        "dof_dim": int(env.num_dof),
        "dof_names": list(env.dof_names),
        "action_dof_names": ([env.dof_names[i] for i in env.action_dof_indices.tolist()]
                             if hasattr(env, "action_dof_indices") else list(env.dof_names)),
        "body_names": list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])),
        "scenarios": {nm: {"vx": vx, "yaw": yaw, "h": h, "steps": int(a.steps)}
                      for nm, vx, yaw, h in scenarios},
        "step_dt": float(env.dt),
        "decimation": int(env.cfg.control.decimation),
    }
    flat = {}
    for sc, d in result.items():
        for k, v in d.items():
            if k != "meta":
                flat[f"{sc}/{k}"] = v
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, meta=np.asarray(json.dumps(meta, ensure_ascii=False)), **flat)
    print(f"[ok] {a.out}")
    print(f"     scenarios={list(result)}  steps={a.steps}")


if __name__ == "__main__":
    main()
