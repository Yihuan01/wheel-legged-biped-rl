# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : make_contract.py
# Purpose   : 冻结部署契约：以交付包的 config.json 为准，叠加只存在于代码里的观测常量
#             （obs_scales / clip / 布局），生成 assets/deploy_contract.json 并记录来源 sha；
#             两边不一致直接报错，防止"代码已改、权重还是老口径"的静默错。
# ==============================================================================

import argparse
import hashlib
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(PKG, "../../.."))

RELEASE = "releases/P7F3e_yaw1_dr15"

# 只存在于代码里的常量（源：legged_gym/envs/base/legged_robot_config.py、
# wbr_flat/wbr_flat_config.py、wbr_flat4.py 的 compute_observations）
CODE_CONST = {
    "obs_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05,
                   "joint_state_history": 0.25, "yaw_command": 1.0,
                   "roll_command": 4.0, "height_measurements": 5.0},
    "clip_obs": 100.0,
    "clip_actions": 5.0,
    "yaw_sign": -1.0,
    "history_length": 10,
    "action_dof_names": ["left_hip_joint", "left_knee_joint",
                         "right_hip_joint", "right_knee_joint",
                         "left_wheel_joint", "right_wheel_joint"],
    "obs_layout": {
        "0:3": "base_lin_vel * lin_vel",
        "3:6": "base_ang_vel * ang_vel",
        "6:9": "projected_gravity",
        "9:12": "commands[0:3] * [lin,lin,ang]; [2] = yaw_sign * yaw * yaw_command",
        "12": "jump_cmd / jump_hi",
        "13:15": "[ (h - h_mid)/h_half , roll * roll_command ]",
        "15:21": "dof_err[action_dof] * dof_pos (wheel err = 0)",
        "21:27": "dof_vel[action_dof] * dof_vel",
        "27:33": "clipped action",
        "33:93": "dof_pos_err_history(10x6) * dof_pos * joint_state_history",
        "93:153": "dof_vel_history(10x6) * dof_vel * joint_state_history",
        "153:159": "phase one-hot 6",
        "159": "off_knee / off_limit",
        "160": "air progress (0 at phase 0)",
        "161:163": "caster contacts [L,R]",
        "163": "tilt margin / 0.05 clamped to [-1,1]",
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=os.path.join(ROOT, RELEASE))
    ap.add_argument("--out", default=os.path.join(PKG, "assets/deploy_contract.json"))
    a = ap.parse_args()

    cfg_path = os.path.join(a.release, "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))

    ctl = cfg["control"]
    ranges = cfg["commands"]["ranges"]
    h_lo, h_hi = ranges["base_height"]

    # —— 一致性闸门：交付包与代码常量必须对得上 ——
    assert cfg["env"]["num_observations"] == 164, cfg["env"]["num_observations"]
    assert cfg["env"]["num_actions"] == 6
    assert cfg["asset"]["file"].endswith("wbr_v2.urdf"), cfg["asset"]["file"]
    assert float(ctl["leg_action_scale"]) == 0.5
    assert float(ctl["wheel_action_scale"]) == 3.0
    assert float(ctl["wheel_acceleration_filter_alpha"]) == 0.8
    assert int(ctl["decimation"]) == 2
    assert list(ctl["stiffness"].values())[0] == 20.0
    assert float(h_lo) == 0.175 and float(h_hi) == 0.245, (h_lo, h_hi)
    assert float(ranges["lin_vel_x"][1]) == 1.0, ranges["lin_vel_x"]
    assert float(cfg["normalization"]["clip_actions"]) == CODE_CONST["clip_actions"]

    # 关节级参数按 dof 序展开（dof 序以 Isaac 静态导出为准；这里是 Isaac 的 URDF 序）
    dof_names = ["left_hip_joint", "left_knee_joint", "left_knee_caster_joint", "left_wheel_joint",
                 "right_hip_joint", "right_knee_joint", "right_knee_caster_joint", "right_wheel_joint"]
    default = cfg["init_state"]["default_joint_angles"]
    stiff, damp = ctl["stiffness"], ctl["damping"]

    def gain(name, table):
        for key, val in table.items():
            if key == "caster":
                continue
            if key in name:
                return float(val)
        raise KeyError(name)

    contract = {
        "task": "wbr_flat4",
        "asset_urdf": cfg["asset"]["file"],
        "obs_dim": 164,
        "act_dim": 6,
        "dof_dim": 8,
        "dof_names": dof_names,
        "action_dof_names": CODE_CONST["action_dof_names"],
        "default_dof_pos": [float(default[n]) for n in dof_names],
        "kps": [float(stiff["caster"]) if "caster" in n else gain(n, stiff) for n in dof_names],
        "kds": [float(damp["caster"]) if "caster" in n else gain(n, damp) for n in dof_names],
        "torque_limits": [1.0 if "caster" in n else 23.7 for n in dof_names],
        "leg_action_scale": float(ctl["leg_action_scale"]),
        "wheel_action_scale": float(ctl["wheel_action_scale"]),
        "wheel_accel_filter_alpha": float(ctl["wheel_acceleration_filter_alpha"]),
        "wheel_target_accel_limit": float(ctl["wheel_target_accel_limit"]),
        "clip_actions": CODE_CONST["clip_actions"],
        "clip_obs": CODE_CONST["clip_obs"],
        "obs_scales": CODE_CONST["obs_scales"],
        "history_length": CODE_CONST["history_length"],
        "yaw_sign": CODE_CONST["yaw_sign"],
        "command_ranges": {k: list(v) for k, v in ranges.items()},
        "height_mid": 0.5 * (float(h_lo) + float(h_hi)),
        # ⚠ 代码里是 max(半宽, 0.05) 的地板（wbr_flat4.compute_observations），不是裸半宽：
        #   F3 区间 [0.175,0.245] 半宽 0.035 < 0.05 ⇒ 实际用 0.05。改这里等于换观测语义。
        "height_half": max(0.5 * (float(h_hi) - float(h_lo)), 0.05),
        "jump_hi": max(float(x) for x in ranges["jump_height"]),
        "dt_physics": float(cfg["sim"]["dt"]),
        "decimation": int(ctl["decimation"]),
        "control_dt": float(cfg["sim"]["dt"]) * int(ctl["decimation"]),
        "initial_root_pos": list(cfg["init_state"]["pos"]),
        "obs_layout": CODE_CONST["obs_layout"],
        "sources": {
            "release": RELEASE,
            "release_config_sha256": sha256(cfg_path),
            "release_model": "model_11800.pt",
            "git_sha": git_sha(),
            "code_constant_origin": "legged_robot_config.py / wbr_flat_config.py / wbr_flat4.py",
        },
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(contract, f, ensure_ascii=False, indent=1)
    print(f"[ok] {a.out}")
    print(f"     height mid={contract['height_mid']} half={contract['height_half']}")
    print(f"     jump_hi={contract['jump_hi']}  control_dt={contract['control_dt']}")
    print(f"     sources git={contract['sources']['git_sha']} "
          f"cfg_sha={contract['sources']['release_config_sha256'][:12]}")


if __name__ == "__main__":
    main()
