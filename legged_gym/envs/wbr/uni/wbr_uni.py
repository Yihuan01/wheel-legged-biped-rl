# ==============================================================================
# Date      : 2026-10-26
# Author    : SYH
# File      : wbr_uni.py
# Purpose   : 统一多模式环境：统一相位枚举 + FSM 路由 + 多策略观测拼接，单 env 内完成
#             四轮 / 起立 / 两轮 / 跳跃 / 趴下的切换。只做回放，不训练。
# ==============================================================================

import torch

from legged_gym.envs.wbr.switch.wbr_switch import WBRSwitchMob
from legged_gym.envs.wbr.switch.wbr_switch_config import (
    SP_HOLD4W, SP_RISE, SP_HANDOVER, SP_HOLD2W, SP_LOWER, SP_TILT, SP_ABSORB,
)
from .phase import (
    PH_IDLE2W, PH_JUMP1, PH_JUMP5, PH_HOLD4W, PH_RISE, PH_HANDOVER,
    PH_LOWER, PH_TILT, PH_ABSORB,
)
from .obs_builder import build_w12_obs as _build_w12_obs, w12_action_to_env as _w12_action_to_env
from .rise_script import RiseScript, SUB_DONE, SUB_NAMES as _SUB_NAMES


# switch 相位(0-6) → 统一相位(0-11)。起立/趴下过渡共用同一套，方向由 sw_task 区分。
# 两轮行走（switch 无独立相，用 HOLD2W 表示）→ IDLE2W(0)；其余一一对应到 6 之后。
_SP_TO_UNI = {
    SP_HOLD4W: PH_HOLD4W,      # 6
    SP_RISE: PH_RISE,          # 7
    SP_HANDOVER: PH_HANDOVER,  # 8
    SP_HOLD2W: PH_IDLE2W,      # 0（两轮行走/站立 = IDLE2W）
    SP_LOWER: PH_LOWER,        # 9
    SP_TILT: PH_TILT,          # 10
    SP_ABSORB: PH_ABSORB,      # 11
}


class WBRUniMob(WBRSwitchMob):
    """统一多模式程序：四轮 / 起立 / 趴下 / 两轮（R81/W12），一套相位机切换。"""

    # ---------------------------------------------------------------- 统一相位
    def uni_phase(self):
        """当前统一相位（标量，取 env0；回放时各 env 同相）。"""
        return _SP_TO_UNI.get(int(self.ph[0].item()), PH_IDLE2W)

    # ---------------------------------------------------------------- 观测路由
    def build_obs_for_mode(self, mode):
        """按控制模式取该策略的观测。
        mode: 'FOURW'→164 / 'RISE'|'LOWER'→173 / 'TWOW'→R81 的 161。
        """
        if mode == 'FOURW':
            return self.obs_buf[:, :164]
        if mode == 'TWOW':
            return self.build_w12_obs()
        return self.obs_buf[:, :173]          # 起立/趴下：173

    def build_w12_obs(self):
        """两轮控制器 R81 的 161 维观测（wbr_v1 布局，从 173 基础块重排）。"""
        return _build_w12_obs(self.obs_buf)

    def w12_action_to_env(self, act_w12):
        """R81 输出动作（wbr_v1 序）→ env 动作（wbr4 序）。"""
        return _w12_action_to_env(act_w12)

    # ---------------------------------------------------------------- P7T 相位脚本起立
    def init_rise_script(self, cfg=None):
        """构造 P7T 脚本实例（`standup_mode='script'` 时用）。crouch/2W/4W 姿态取自本 env。"""
        # 4W 腿姿必须在**构造时冻结**：脚本每步会把 default_dof_pos 写成目标，若每步再读
        # default_dof_pos 当起点，① 静稳段就会漂走（起点被自己的输出覆盖）。
        self._rise_pose_4w = self.default_dof_pos[0, self.leg_dof].detach().clone()
        self._rise_script = RiseScript(
            self.num_envs, self.device, cfg=cfg,
            crouch_pose=[1.5391, -2.2722, -1.5391, 2.2722],
            pose_2w=self.pose_2w.detach().cpu().tolist(),
            pose_4w=self._rise_pose_4w.detach().cpu().tolist(),
        )
        self._rise_wheel_torque_override = None
        self._rise_gain_mult = torch.ones(self.num_envs, device=self.device)
        self.rise_sub = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rise_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._rise_script

    def rise_script_step(self):
        """推进一拍 P7T 脚本：写腿目标(default_dof_pos)、轮力矩覆盖、腿 P 增益倍率。

        返回 dict（含 sub/done 供上层判交接）。仅在 standup_mode='script' 且处于 RISE 时调用。
        """
        rs = getattr(self, "_rise_script", None)
        if rs is None:
            rs = self.init_rise_script()
        out = rs.step(pitch=self._body_pitch(), z=self.root_states[:, 2], dt=self.dt,
                      pose_4w=self._rise_pose_4w)
        # 脚本给的是绝对关节角 ⇒ 写进 default_dof_pos（PD 目标 = default + 动作）。
        # default_dof_pos 形状 [1, num_dof]、全 env 共享，只写第 0 行；写 [n, 4] 会与广播冲突。
        self.default_dof_pos[0, self.leg_dof] = out["pose"][0]
        # 轮力矩覆盖（绕过裁剪，见 wbr4._compute_torques 的注入点）
        self._rise_wheel_torque_override = out["wheel_torque"]
        # 腿 P 增益倍率（伸腿/抬升段 ×boost）
        base_p = self._rise_p_nom
        self.p_gains[self.leg_dof] = base_p * out["gain_mult"][0]
        self._rise_gain_mult = out["gain_mult"]
        self.rise_sub = out["sub"]
        self.rise_done = out["done"]
        return out

    def rise_script_reset(self, env_ids=None):
        rs = getattr(self, "_rise_script", None)
        if rs is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        rs.reset(env_ids)
        self.rise_sub[env_ids] = 0
        self.rise_done[env_ids] = False

    def rise_script_clear(self):
        """离开 RISE（交接/中止）时清掉注入，恢复名义腿增益。"""
        self._rise_wheel_torque_override = None
        if hasattr(self, "_rise_p_nom"):
            self.p_gains[self.leg_dof] = self._rise_p_nom
        self._rise_gain_mult = torch.ones(self.num_envs, device=self.device)

    def set_rise_p_nominal(self):
        """记录名义腿 P 增益（脚本 boost 的基准），在进入脚本起立前调用一次。"""
        self._rise_p_nom = self.p_gains[self.leg_dof].clone()

    def rise_sub_name(self):
        try:
            return _SUB_NAMES.get(int(self.rise_sub[0].item()), "?")
        except Exception:
            return "?"
