# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : obs_builder.py
# Purpose   : 四轮 164 维观测的部署侧唯一实现：按名字取 6 个动作关节（排除从动轮）、
#             轮位置误差清零、关节历史滞后一拍、末尾四轮块（从动轮接触＋倾翻裕度）。
#             顺序与 wbr_flat4.compute_observations 逐位一致，任何一个块顺序变了都会静默降智。
# ==============================================================================

import numpy as np
import torch

BLOCKS = (("base", 0, 33), ("history", 33, 153), ("jump", 153, 161), ("four_wheel", 161, 164))


class ObsBuilder:
    """【S3 uni 链】dof 载体与 obs 布局解耦：dof_names/default 可传 8 dof 载体（v2 链全程连续），
    obs 布局（尾块）由 self.c["obs_dim"] 决定，链段切换只换 self.c——历史与基准不中断。"""

    def __init__(self, contract, dof_names=None, default_dof_pos=None):
        c = contract
        self.c = c
        self.names = list(dof_names or c["dof_names"])
        self.default = torch.tensor(default_dof_pos or c["default_dof_pos"], dtype=torch.float32)
        self.adi = [self.names.index(n) for n in c["action_dof_names"]]
        self.wheel_in6 = [i for i, n in enumerate(c["action_dof_names"]) if "wheel" in n]
        self.obs_dim = int(c["obs_dim"])     # ⚠ uni 链换段时会被 self.c 同步覆盖（见 build）
        self.off_limit = float(c.get("jump_off_limit", 0.0))
        self.sw_state = {"ph": 0, "task": 0, "uep": 0.0}   # 【S2c 趴下】switch 尾块状态（SimLoop 注入）
        self.reset()

    def set_contract(self, contract):
        """【S3 uni 链】运行时换契约：布局/动作序/default 全量跟随（历史不动）。
        动作序必须等于**策略训练侧**的顺序，换段时 adi 不更新会把动作投到错的关节。"""
        self.c = contract
        self.adi = [self.names.index(n) for n in contract["action_dof_names"]]
        self.wheel_in6 = [i for i, n in enumerate(contract["action_dof_names"]) if "wheel" in n]
        self.default = torch.tensor(contract["default_dof_pos"], dtype=torch.float32)
        self.off_limit = float(contract.get("jump_off_limit", 0.0))
        self.obs_dim = int(contract["obs_dim"])

    def reset(self):
        L = int(self.c["history_length"])
        self.hist_pos = torch.zeros(L, 6, dtype=torch.float32)
        self.hist_vel = torch.zeros(L, 6, dtype=torch.float32)

    # ------------------------------------------------------------------ 观测
    def build(self, *, lin_vel, ang_vel, gravity, commands, dof_pos, dof_vel, last_action,
              caster_contacts, tip_margin, phase=0, off_knee=0.0, prog=0.0):
        s = self.c["obs_scales"]
        c = self.c
        commands = torch.as_tensor(commands, dtype=torch.float32)
        dof_pos = torch.as_tensor(dof_pos, dtype=torch.float32)
        dof_vel = torch.as_tensor(dof_vel, dtype=torch.float32)

        cmd = commands[:3] * torch.tensor([s["lin_vel"], s["lin_vel"], s["ang_vel"]])
        cmd[2] = float(c["yaw_sign"]) * commands[2] * s["yaw_command"]
        aux = torch.tensor([(float(commands[5]) - c["height_mid"]) / c["height_half"],
                            float(commands[6]) * s["roll_command"]])

        derr = dof_pos - self.default
        derr6 = derr[self.adi].clone()
        derr6[self.wheel_in6] = 0.0

        onehot = torch.zeros(6)
        if 0 <= int(phase) < 6:
            onehot[int(phase)] = 1.0
        off_col = float(off_knee) / self.off_limit if self.off_limit > 0 else 0.0
        tip = torch.clamp(torch.as_tensor(float(tip_margin)) / 0.05, -1.0, 1.0).reshape(1)

        blocks = [
            # （173 趴下契约的 switch 尾块在末尾统一追加，见下方 obs_dim==173 分支）
            torch.as_tensor(lin_vel, dtype=torch.float32) * s["lin_vel"],
            torch.as_tensor(ang_vel, dtype=torch.float32) * s["ang_vel"],
            torch.as_tensor(gravity, dtype=torch.float32),
            cmd,
            torch.tensor([float(commands[4]) / c["jump_hi"]]),
            aux,
            derr6 * s["dof_pos"],
            dof_vel[self.adi] * s["dof_vel"],
            torch.as_tensor(last_action, dtype=torch.float32),
            self.hist_pos.reshape(-1) * s["dof_pos"] * s["joint_state_history"],
            self.hist_vel.reshape(-1) * s["dof_vel"] * s["joint_state_history"],
            onehot,
            torch.tensor([off_col, float(prog)]),
        ]
        self.obs_dim = int(self.c.get("obs_dim", 164))   # 【S3 uni】换段只换 self.c ⇒ 这里动态生效
        od = self.obs_dim
        if od in (164, 173):     # 四轮/趴下：caster 2＋tip 1 尾块（趴下 173＝164＋switch 9）
            blocks += [torch.as_tensor(caster_contacts, dtype=torch.float32).reshape(2), tip]
        if od == 173:            # 【S2c 趴下】switch 尾块 9＝相位 one-hot 7＋task 符号＋UEP 俯仰
            st = self.c.get("switch_tail", {})
            nph = int(st.get("num_phases", 7))
            ph = int(self.sw_state.get("ph", st.get("ph_start", 3)))
            onehot = torch.zeros(nph)
            if 0 <= ph < nph:
                onehot[ph] = 1.0
            blocks += [onehot,
                       torch.tensor([float(self.sw_state.get("task", 0)) * 2.0 - 1.0]),
                       torch.tensor([float(np.clip(self.sw_state.get("uep", 0.0), -1.0, 1.0))])]
        obs = torch.cat(blocks)
        assert obs.numel() == self.obs_dim, (obs.numel(), self.obs_dim)
        return torch.clip(obs, -c["clip_obs"], c["clip_obs"])

    # ------------------------------------------------------------------ 历史
    def push(self, dof_pos, dof_vel):
        """控制步末尾调用：历史只收当前拍，且轮位置误差必须清零（否则网络能感知轮子绝对角）。"""
        dof_pos = torch.as_tensor(dof_pos, dtype=torch.float32)
        dof_vel = torch.as_tensor(dof_vel, dtype=torch.float32)
        derr = dof_pos - self.default
        derr6 = derr[self.adi].clone()
        derr6[self.wheel_in6] = 0.0
        self.hist_pos = torch.cat([self.hist_pos[1:], derr6.unsqueeze(0)], dim=0)
        self.hist_vel = torch.cat([self.hist_vel[1:], dof_vel[self.adi].unsqueeze(0)], dim=0)
