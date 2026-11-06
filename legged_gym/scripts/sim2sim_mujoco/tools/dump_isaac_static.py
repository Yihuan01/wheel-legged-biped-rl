# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : dump_isaac_static.py
# Purpose   : 从 Isaac Gym（CPU 管线，1 env）导出四轮静态契约：关节序/默认角/增益/力矩上限/
#             关节阻尼与限位、刚体质量-质心-惯量、索引集合，供 MuJoCo 侧逐项对账。
#             运行环境：conda activate isaacgym01（需 LD_LIBRARY_PATH=$CONDA_PREFIX/lib）
# ==============================================================================

import argparse
import json
import os
import sys

import numpy as np

# isaacgym 必须先于 torch / legged_gym 导入；`import *` 只能放模块级（py3.8）。
import isaacgym  # noqa: E402,F401
import torch  # noqa: E402,F401
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.envs.wbr.flat4.wbr_flat4_config import WBRFlat4Cfg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(PKG, "../../.."))


def patch_nominal(task="wbr_flat4"):
    """建 env 前把 DR/噪声/延迟全部关掉：静态契约必须是名义值。（两轮/四轮共用 _isaac_common）"""
    from legged_gym.scripts.sim2sim_mujoco.tools._isaac_common import patch_nominal as _p
    _p(task)


def mat33(m):
    try:
        return [[float(m.x.x), float(m.x.y), float(m.x.z)],
                [float(m.y.x), float(m.y.y), float(m.y.z)],
                [float(m.z.x), float(m.z.y), float(m.z.z)]]
    except Exception:
        return np.asarray(m, dtype=float).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PKG, "eval/golden/isaac_static.json"))
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--task", default="wbr_flat4", choices=["wbr_flat4", "wbr_flat"],
                    help="两轮 wbr_flat 用 v1 资产（无从动轮），默认写 isaac_static_two.json")
    a = ap.parse_args()
    if a.task == "wbr_flat" and a.out == os.path.join(PKG, "eval/golden/isaac_static.json"):
        a.out = os.path.join(PKG, "eval/golden/isaac_static_two.json")

    sys.argv = ["dump_static", "--task", a.task, "--headless",
                f"--num_envs={a.num_envs}", "--sim_device", "cpu", "--pipeline", "cpu"]

    patch_nominal(a.task)
    args = get_args()
    env, _ = task_registry.make_env(name=args.task, args=args)

    dp = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    bp = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])

    data = {
        "task": args.task,
        "issac_pipeline": {"sim_device": args.sim_device, "pipeline": args.pipeline},
        "num_obs": int(env.num_obs),
        "num_actions": int(env.num_actions),
        "num_dof": int(env.num_dof),
        "num_bodies": int(env.num_bodies),
        "dt": float(env.dt),
        "decimation": int(env.cfg.control.decimation),
        "episode_length_s": float(env.cfg.env.episode_length_s),
        "dof_names": list(env.dof_names),
        "body_names": list(body_names),
        "action_dof_names": ([env.dof_names[i] for i in env.action_dof_indices.tolist()]
                             if hasattr(env, "action_dof_indices") else list(env.dof_names)),
        "action_dof_indices": ([int(x) for x in env.action_dof_indices.tolist()]
                               if hasattr(env, "action_dof_indices") else list(range(int(env.num_dof)))),
        "wheel_indices": [int(x) for x in env.wheel_indices.tolist()],
        "leg_indices": ([int(x) for x in env.leg_indices.tolist()]
                        if hasattr(env, "leg_indices") else []),
        "caster_indices": ([int(env.caster_l_idx), int(env.caster_r_idx)]
                           if hasattr(env, "caster_l_idx") else []),
        "feet_indices": [int(x) for x in env.feet_indices.tolist()],
        "caster_body_indices": ([int(x) for x in env.caster_body_indices.tolist()]
                                if hasattr(env, "caster_body_indices") else []),
        "termination_contact_indices": [int(x) for x in env.termination_contact_indices.tolist()],
        "default_dof_pos": [float(x) for x in env.default_dof_pos[0].tolist()],
        "p_gains": [float(x) for x in env.p_gains.tolist()],
        "d_gains": [float(x) for x in env.d_gains.tolist()],
        "torque_limits": [float(x) for x in env.torque_limits.tolist()],
        "dof_pos_limits_soft": env.dof_pos_limits.tolist(),
        "dof_pos_limits_raw": {
            "lower": [float(x) for x in dp["lower"]],
            "upper": [float(x) for x in dp["upper"]],
        },
        "dof_velocity_limits": [float(x) for x in dp["velocity"]],
        "dof_effort_limits": [float(x) for x in dp["effort"]],
        "dof_damping": [float(x) for x in dp["damping"]],
        "dof_armature": [float(x) for x in dp["armature"]],
        "dof_friction": [float(x) for x in dp["friction"]],
        "bodies": [
            {"name": n, "mass": float(p.mass),
             "com": [float(p.com.x), float(p.com.y), float(p.com.z)],
             "inertia": mat33(p.inertia)}
            for n, p in zip(body_names, bp)
        ],
        "control": {
            "leg_action_scale": float(env.cfg.control.leg_action_scale),
            "wheel_action_scale": float(env.cfg.control.wheel_action_scale),
            "wheel_acceleration_filter_alpha": float(env.cfg.control.wheel_acceleration_filter_alpha),
            "wheel_target_accel_limit": float(env.cfg.control.wheel_target_accel_limit),
        },
        "normalization": {
            "clip_obs": float(env.cfg.normalization.clip_observations),
            "clip_actions": float(env.cfg.normalization.clip_actions),
        },
        "obs_scales": {k: float(getattr(env.obs_scales, k))
                       for k in dir(env.obs_scales) if not k.startswith("_")},
        "command_ranges": {k: list(v) for k, v in env.cfg.commands.ranges.__dict__.items()
                           if not k.startswith("_")},
        "jump_ranges": {"jump_height": list(env.cfg.commands.ranges.jump_height)},
        "initial_root_pos": [float(x) for x in env.base_init_state.tolist()],
        "PITCH_4W": float(getattr(env, "PITCH_4W", 0.0)),
        "history": {"enable": bool(env.cfg.env.enable_joint_state_history),
                    "length": int(env.cfg.env.joint_state_history_length)},
    }

    # 名义性断言：DR 必须全是退化值
    assert not env.add_noise, "add_noise 未关"
    assert float(env._dr_pd.max()) == 1.0 and float(env._dr_pd.min()) == 1.0, "pd DR 未关"
    assert float(env._dr_motor.max()) == 1.0, "motor DR 未关"

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    print(f"[ok] {a.out}")
    print(f"     dof={data['num_dof']} obs={data['num_obs']} act={data['num_actions']} bodies={data['num_bodies']}")
    print(f"     dof_names={data['dof_names']}")
    print(f"     torque_limits={data['torque_limits']}")
    print(f"     damping={data['dof_damping']}")
    print(f"     total mass={sum(b['mass'] for b in data['bodies']):.5f} kg")


if __name__ == "__main__":
    main()
