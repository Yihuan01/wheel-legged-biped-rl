# ==============================================================================
# Date      : 2026-10-25
# Author    : SYH
# File      : wbr_switch.py
# Purpose   : 起立 / 趴下模式切换环境：单个 env 承载四轮与两轮两套策略，按 7 个相位
#             （SP_HOLD4W → SP_ABSORB）推进脚本过渡，观测 173 维。
# ==============================================================================

import torch
from isaacgym.torch_utils import torch_rand_float
from isaacgym import gymtorch

from legged_gym.envs.wbr.flat4.wbr_flat4 import WBRFlat4Mob
from legged_gym.envs.wbr.switch.wbr_switch_config import (
    SP_HOLD4W, SP_RISE, SP_HANDOVER, SP_HOLD2W, SP_LOWER, SP_TILT, SP_ABSORB,
    NUM_SW_PHASES,
)


class WBRSwitchMob(WBRFlat4Mob):
    """起立/趴下模式切换环境。"""

    BASE_OBS_DIM = 164                       # 四轮基础观测宽度（switch 块在其后追加 9 维）

    # ------------------------------------------------------------------ 初始化
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._init_switch_buffers()

    def _init_switch_buffers(self):
        n, dev = self.num_envs, self.device
        sw = self.cfg.rewards.switch
        self.sw = sw
        # ── 回放（play02 switch 档）专用开关，训练时保持默认 ──
        #   playback_hold：True ⇒ 相位机不推进、不结算、不 reset，相位与参考姿态由 play02 每步写入。
        self.playback_hold = False
        # 默认姿态张量（4W / 2W），顺序 [hip_l, knee_l, hip_r, knee_r]（左右镜像）
        self.pose_4w = torch.tensor(
            [sw.q4w_hip, sw.q4w_knee, -sw.q4w_hip, -sw.q4w_knee], device=dev)
        self.pose_2w = torch.tensor(
            [sw.q2w_hip, sw.q2w_knee, -sw.q2w_hip, -sw.q2w_knee], device=dev)
        # 腿 dof 索引（与动作前 4 维顺序一致）与动作缩放
        self.leg_dof = self.action_dof_indices[:4]                      # [4]
        self.leg_scale = self._act_scale_vec[:4]                        # [4]

        # 相位机状态
        self.sw_task = torch.ones(n, dtype=torch.long, device=dev)      # 1=起立, 0=趴下
        self.ph = torch.zeros(n, dtype=torch.long, device=dev)          # 复用父类字段名
        self.ph_t = torch.zeros(n, device=dev)
        self.ref_pose = self.pose_4w.unsqueeze(0).repeat(n, 1).clone()  # [n,4] 当前参考姿态
        self.sw_uep = torch.zeros(n, device=dev)                        # UEP 偏差代理（俯仰）
        self.sw_done = torch.zeros(n, dtype=torch.bool, device=dev)     # 本回合任务已完成→待 reset
        # 事件锁存 / 确认计数
        self.ev_up_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ev_set_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ev_up_fresh = torch.zeros(n, dtype=torch.bool, device=dev)  # 仅事件确认那一拍 True
        self.ev_set_fresh = torch.zeros(n, dtype=torch.bool, device=dev)
        self.up_cnt = torch.zeros(n, dtype=torch.long, device=dev)
        self.set_cnt = torch.zeros(n, dtype=torch.long, device=dev)
        # 冲击锁存（D2 从动轮触地那一刻的法向接近速度）
        self.v_impact = torch.zeros(n, device=dev)
        self.caster_was_air = torch.ones(n, dtype=torch.bool, device=dev)
        # 反向课程：本回合起点在 4W→2W 之间的推进比例（0=纯 4W 起点）
        self.start_frac = torch.zeros(n, device=dev)
        # 自动课程状态（起点比例 cur_frac 随成功率调整；0=完整 4W 起立，1=近 2W）
        self.cur_frac = float(self.sw.reverse_curriculum_frac)
        self._cur_eps = 0
        self._cur_succ = 0
        self._init_w3_baseline()

    def _init_w3_baseline(self):
        """加载 W3 作为起立轮平衡基线（u_base 的轮部分）。通道按关节名重排。"""
        import os
        self._w3_net = None
        if not getattr(self.sw, "residual_base", True):
            return
        path = self.sw.w3_model
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), path)
        if not os.path.exists(path):
            print(f"[wbr_switch] W3 基线缺失，跳过：{path}")
            return
        from rsl_rl.modules import ActorCritic
        net = ActorCritic(161, 161, 6, actor_hidden_dims=[512, 256, 128],
                          critic_hidden_dims=[512, 256, 128], activation='elu').to(self.device)
        net.load_state_dict(torch.load(path, map_location=self.device)['model_state_dict'])
        net.eval()
        self._w3_net = net
        # wbr4 6 通道动作序 [lhip,lknee,rhip,rknee,lwheel,rwheel] 在 W3 序 [lhip,lknee,lwheel,rhip,rknee,rwheel] 中的位置
        # 以及 W3 的轮通道（w3 序 2=l_wheel, 5=r_wheel）
        self._w3_lwheel = 2
        self._w3_rwheel = 5
        print(f"[wbr_switch] W3 基线已加载：{path}")

    def _build_w3_obs(self):
        """把当前 164 维基础观测转成 W3 的 161 维（6 通道按关节名重排，丢弃 caster2/tip1）。"""
        o = self.obs_buf[:, :self.BASE_OBS_DIM]
        perm = [0, 1, 4, 2, 3, 5]   # wbr4 序 -> w3 序
        def r6(x):
            return x[..., perm]
        def rh(x):
            s = x.shape
            return x.view(*s[:-1], s[-1] // 6, 6)[..., perm].reshape(*s)
        return torch.cat([o[:, :15], r6(o[:, 15:21]), r6(o[:, 21:27]), r6(o[:, 27:33]),
                          rh(o[:, 33:93]), rh(o[:, 93:153]), o[:, 153:161]], dim=-1)

    # ------------------------------------------------------ 回放接口（play02 switch 档用）
    def playback_set_mode(self, env_ids, ph, task=1, ph_t=0.0):
        """直接设定相位(0..6)/任务(1=起立,0=趴下)/相位计时，供 play02 手动驱动相位机。"""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        self.ph[env_ids] = int(ph)
        self.ph_t[env_ids] = float(ph_t)
        self.sw_task[env_ids] = int(task)

    def playback_reset(self):
        """把全部 env 复位到当前 play 起点（起立：完整 4W 起点、相位 HOLD4W），并重新计算观测。"""
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_idx(env_ids)
        self.compute_observations()
        return self.obs_buf

    # ------------------------------------------------------------------ 复位
    def reset_idx(self, env_ids):
        if len(env_ids) > 0:
            self._update_curriculum(env_ids)   # 先用旧 sw_done 记账，再重置
            self._reset_switch_state(env_ids)
        super().reset_idx(env_ids)   # 父类清相位/历史/命令

    def _update_curriculum(self, env_ids):
        """按本批将 reset 的回合统计成功率，调整课程起点比例 cur_frac。"""
        sw = self.sw
        if not getattr(sw, "curriculum", False):
            return
        n = len(env_ids)
        succ = int(self.sw_done[env_ids].sum())
        self._cur_eps += n
        self._cur_succ += succ
        if self._cur_eps >= int(sw.cur_window):
            rate = self._cur_succ / max(self._cur_eps, 1)
            if rate > sw.cur_rate_hi:
                self.cur_frac = max(0.0, self.cur_frac - sw.cur_step_down)
            elif rate < sw.cur_rate_lo:
                self.cur_frac = min(1.0, self.cur_frac + sw.cur_step_up)
            self.extras.setdefault("curriculum", {})
            self.extras["curriculum"]["succ_rate"] = rate
            self.extras["curriculum"]["cur_frac"] = self.cur_frac
            self._cur_eps = 0
            self._cur_succ = 0

    def _reset_switch_state(self, env_ids):
        """按任务采样 4W / 2W 起点，并把相位机置于对应起始相。"""
        m = len(env_ids)
        standup = torch.rand(m, device=self.device) < self.sw.standup_ratio
        self.sw_task[env_ids] = standup.long()
        # 起点相位：起立从 HOLD4W 起，趴下从 HOLD2W 起
        self.ph[env_ids] = torch.where(standup,
                                       torch.full((m,), SP_HOLD4W, dtype=torch.long, device=self.device),
                                       torch.full((m,), SP_HOLD2W, dtype=torch.long, device=self.device))
        self.ph_t[env_ids] = 0.0
        self.sw_done[env_ids] = False
        self.ev_up_latch[env_ids] = False
        self.ev_set_latch[env_ids] = False
        self.ev_up_fresh[env_ids] = False
        self.ev_set_fresh[env_ids] = False
        self.up_cnt[env_ids] = 0
        self.set_cnt[env_ids] = 0
        self.v_impact[env_ids] = 0.0
        self.caster_was_air[env_ids] = True
        self.sw_uep[env_ids] = 0.0
        # 反向课程：起点推进比例 frac ∈ [0, cur_frac]（混合稳定 4W 起点与近 2W 起点）
        frac = self.cur_frac * torch.rand(m, device=self.device)
        self.start_frac[env_ids] = frac
        # 参考姿态＝起点姿态：4W 姿态按 frac 向 2W 推进（smoothstep 取形）
        w0 = frac * frac * (3.0 - 2.0 * frac)
        pose0 = (1 - w0).unsqueeze(1) * self.pose_4w.unsqueeze(0) + w0.unsqueeze(1) * self.pose_2w.unsqueeze(0)
        self.ref_pose[env_ids] = torch.where(
            standup.unsqueeze(1), pose0, self.pose_2w.unsqueeze(0))

    def _reset_root_states(self, env_ids):
        """躯干水平(pitch=0)，按任务+反向课程给起点高度（起立沿走廊 0.19→2W / 趴下 0.28）。"""
        if len(env_ids) == 0:
            return
        # 构造期 _init_buffers 末尾会先触发一次复位（早于 _init_switch_buffers）⇒ 回退到父类
        if not hasattr(self, "sw_task"):
            return super()._reset_root_states(env_ids)
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 3:7] = self.reset_rot
        # 起立：z 由 start_frac 沿走廊（与姿态同用 smoothstep）插值；趴下：z2w
        f = self.start_frac[env_ids]
        w0 = f * f * (3.0 - 2.0 * f)                       # 与 _reset_dofs 姿态同口径
        z_stand = self.sw.z4w + w0 * (self.sw.z2w - self.sw.z4w) + 0.005   # +5mm 净空，避免复位嵌入
        z = torch.where(self.sw_task[env_ids] == 1, z_stand,
                        torch.full((len(env_ids),), self.sw.z2w + 0.005, device=self.device))
        self.root_states[env_ids, 2] = z
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.03, 0.03, (len(env_ids), 6), device=self.device)
        ids32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids32), len(ids32))

    def _reset_dofs(self, env_ids):
        """按任务+反向课程把腿摆到起点姿态（起立沿走廊 / 趴下 2W），轮速清零。"""
        if len(env_ids) == 0:
            return
        # 构造期回退：缓冲区未就绪时用父类（四轮默认）复位
        if not hasattr(self, "leg_dof"):
            return super()._reset_dofs(env_ids)
        self.dof_pos[env_ids] = self.default_dof_pos
        m = len(env_ids)
        noise = torch_rand_float(-0.02, 0.02, (m, 4), device=self.device)
        f = self.start_frac[env_ids]
        w0 = f * f * (3.0 - 2.0 * f)
        pose_stand = (1 - w0).unsqueeze(1) * self.pose_4w.unsqueeze(0) + w0.unsqueeze(1) * self.pose_2w.unsqueeze(0)
        pose = torch.where(self.sw_task[env_ids].bool().unsqueeze(1),
                           pose_stand, self.pose_2w.unsqueeze(0)) + noise
        self.dof_pos[env_ids.unsqueeze(1), self.leg_dof.unsqueeze(0)] = pose
        self.dof_vel[env_ids] = 0.0
        self.wheel_target_vel[env_ids] = 0.0
        self.last_wheel_control_vel[env_ids] = 0.0
        self.filtered_wheel_acceleration[env_ids] = 0.0
        if hasattr(self, "_caster_air_cnt"):
            self._caster_air_cnt[env_ids] = 0
        ids32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(ids32), len(ids32))

    # ------------------------------------------------------------------ 相位机
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()   # 命令重采样等（enable_jump=False ⇒ 跳过跳跃机）
        self._update_switch_phase()

    def _update_switch_phase(self):
        """事件驱动相位机：推进相位、算参考姿态、结算事件。每控制步恰好调用一次。"""
        sw = self.sw
        dt = self.dt
        self.ph_t += dt

        if self.playback_hold:
            # 回放：只刷新诊断量，不推进 / 不结算 / 不 reset；相位与参考姿态由 play02 每步写入。
            self.sw_uep = self._body_pitch()
            self.ev_up_fresh = torch.zeros_like(self.ev_up_fresh)
            self.ev_set_fresh = torch.zeros_like(self.ev_set_fresh)
            return

        pitch = self._body_pitch()
        wheels_c = self.contact_forces[:, self.feet_indices, 2] > 0.1     # [n,2]
        n_wheel = wheels_c.sum(dim=1)
        casters_c = self._caster_contacts()                                # [n,2] bool
        n_caster = casters_c.sum(dim=1)
        speed = torch.norm(self.root_states[:, 7:9], dim=1)
        yawrate = torch.abs(self.root_states[:, 11])                      # 世界 z 角速度近似
        self.sw_uep = pitch                                                # UEP 代理（俯仰）

        # ---- 各相位判定 ----
        aligned = torch.mean(torch.abs(
            self.dof_pos[:, self.leg_dof] - self.ref_pose), dim=1) < sw.align_tol
        settled2w = (torch.abs(pitch) < sw.upright_pitch_tol) & (n_wheel == 2) & (n_caster == 0) \
                    & (speed < sw.settled_v_tol)
        settled4w = (n_wheel == 2) & (n_caster == 2) & (speed < sw.settled_v_tol) \
                    & (yawrate < sw.settled_w_tol) & (torch.abs(pitch) < sw.upright_pitch_tol)
        caster_touch = casters_c.any(dim=1)

        # 事件确认计数（连续 N 拍）
        self.up_cnt = torch.where(settled2w, self.up_cnt + 1, torch.zeros_like(self.up_cnt))
        self.set_cnt = torch.where(settled4w, self.set_cnt + 1, torch.zeros_like(self.set_cnt))
        ev_up = self.up_cnt >= sw.ev_confirm_steps
        ev_set = self.set_cnt >= sw.ev_confirm_steps

        # ---- 参考姿态（默认＝当前起点姿态，逐相位覆盖）----
        # 起立 U1：起点姿态(start_frac) → 2W；趴下 D1：2W→4W。smoothstep 节奏（起停缓）
        s_rise = torch.clamp(self.ph_t / max(sw.t_rise, 1e-6), 0.0, 1.0)
        s_lower = torch.clamp(self.ph_t / max(sw.t_lower, 1e-6), 0.0, 1.0)
        w_rise = s_rise * s_rise * (3.0 - 2.0 * s_rise)
        w_lower = s_lower * s_lower * (3.0 - 2.0 * s_lower)
        base_w = self.start_frac.unsqueeze(1)                       # [n,1] 起点推进比例
        w_eff = base_w + (1.0 - base_w) * w_rise.unsqueeze(1)       # 起点→2W
        rise_ref = (1.0 - w_eff) * self.pose_4w.unsqueeze(0) + w_eff * self.pose_2w.unsqueeze(0)
        lower_ref = (1 - w_lower).unsqueeze(1) * self.pose_2w + w_lower.unsqueeze(1) * self.pose_4w

        # ---- 逐相位推进（用 where 组合，避免 per-env 循环）----
        p = self.ph
        is_standup = self.sw_task == 1
        # 起立链：U0 对齐(起点保持 t_align 后即转起) → U1 转起 → U2 交接
        to_rise = is_standup & (p == SP_HOLD4W) & (self.ph_t >= sw.t_align)
        to_handover = is_standup & (p == SP_RISE) & (self.ph_t >= sw.t_rise)
        to_hold2w = is_standup & (p == SP_HANDOVER) & ev_up
        # 趴下链
        to_lower = (~is_standup) & (p == SP_HOLD2W) & settled2w & (self.ph_t >= sw.t_align)
        to_tilt = (~is_standup) & (p == SP_LOWER) & (self.ph_t >= sw.t_lower)
        to_absorb = (~is_standup) & (p == SP_TILT) & caster_touch
        to_hold4w = (~is_standup) & (p == SP_ABSORB) & ev_set

        # 应用转移（就地改 p/ph_t）
        def _go(mask, newp):
            self.ph = torch.where(mask, torch.full_like(self.ph, newp), self.ph)
            self.ph_t = torch.where(mask, torch.zeros_like(self.ph_t), self.ph_t)
        _go(to_rise, SP_RISE)
        _go(to_handover, SP_HANDOVER)
        _go(to_hold2w, SP_HOLD2W)
        _go(to_lower, SP_LOWER)
        _go(to_tilt, SP_TILT)
        _go(to_absorb, SP_ABSORB)
        _go(to_hold4w, SP_HOLD4W)

        # 成功事件（fresh 只在确认那一拍，用于一次性结算）
        self.ev_up_fresh = to_hold2w
        self.ev_set_fresh = to_hold4w
        # 【甲】起立到 HOLD2W 后**继续保持** hold_after_upright 秒再结算 reset（让策略学"站住"）。
        #   趴下(to_hold4w)保持原"完成即 reset"。
        hold_steps = int(round(float(getattr(sw, "hold_after_upright", 0.0)) / dt))
        done_now = torch.zeros_like(self.sw_done)
        done_now |= to_hold4w
        if hold_steps > 0:
            hold_ok = (self.ph == SP_HOLD2W) & (self.ph_t >= float(sw.t_handover) + hold_steps * dt)
            done_now |= hold_ok
        else:
            done_now |= to_hold2w
        self.sw_done |= done_now   # 任务完成 → 触发 reset（见 check_termination）

        # 冲击锁存：D2/TILT 段从动轮由离地→触地的接近速度
        just_touch = caster_touch & self.caster_was_air
        self.v_impact = torch.where(just_touch & (~is_standup),
                                    torch.clamp(-self.root_states[:, 9], min=0.0), self.v_impact)
        self.caster_was_air = ~caster_touch

        # ---- 参考姿态按当前相位重算（在推进之后）----
        p = self.ph
        start_pose = (1.0 - base_w) * self.pose_4w.unsqueeze(0) + base_w * self.pose_2w.unsqueeze(0)
        ref = self.ref_pose.clone()
        ref = torch.where((p == SP_RISE).unsqueeze(1), rise_ref, ref)
        ref = torch.where((p == SP_LOWER).unsqueeze(1), lower_ref, ref)
        ref = torch.where((p == SP_HOLD4W).unsqueeze(1), start_pose, ref)
        ref = torch.where((p == SP_HOLD2W).unsqueeze(1), self.pose_2w.unsqueeze(0), ref)
        ref = torch.where((p == SP_HANDOVER).unsqueeze(1), self.pose_2w.unsqueeze(0), ref)
        ref = torch.where((p == SP_TILT).unsqueeze(1), self.pose_4w.unsqueeze(0), ref)
        ref = torch.where((p == SP_ABSORB).unsqueeze(1), self.pose_4w.unsqueeze(0), ref)
        self.ref_pose = ref

    # ------------------------------------------------------------------ 残差注入
    def step(self, actions):
        """腿 PD 目标 = 参考姿态 + 策略残差；轮基线 = W3 平衡输出 + pitch-PD（residual_base）。
        策略动作是 Δ（腿在参考姿态上、轮在 W3 平衡上叠加）。residual_base=False 时逐位等价纯策略。"""
        if getattr(self.sw, "force_symmetry", False):
            # 【对称起立】右腿动作 = 左腿动作镜像（a_hip_r=−a_hip_l, a_knee_r=−a_knee_l），起立全程双腿对称。
            actions = actions.clone()
            actions[:, 2] = -actions[:, 0]
            actions[:, 3] = -actions[:, 1]
        if getattr(self.sw, "residual_base", True):
            n = self.num_envs
            actions = actions.clone()
            # 腿：参考姿态 → 动作量纲偏移
            base_pose = self.ref_pose                                    # [n,4]
            dflt = self.default_dof_pos[:, self.leg_dof]                 # [n,4]
            actions[:, :4] = actions[:, :4] + (base_pose - dflt) / self.leg_scale
            # 轮：W3 平衡输出 + pitch-PD 辅助（仅起立相关相位；其余相位不加）
            if self._w3_net is not None:
                with torch.no_grad():
                    a_w3 = self._w3_net.act_inference(self._build_w3_obs())   # [n,6] w3 序
                pitch = self._body_pitch()
                prate = self.base_ang_vel[:, 1]
                corr = (self.sw.base_kp_pitch * pitch + self.sw.base_kd_pitch * prate) \
                    / self.cfg.control.wheel_action_scale        # rad/s → 动作单位
                w_l = a_w3[:, self._w3_lwheel] + corr            # 动作单位
                w_r = a_w3[:, self._w3_rwheel] + corr
                active = ((self.ph == SP_RISE) | (self.ph == SP_HANDOVER)
                          | (self.ph == SP_HOLD2W)).float()
                actions[:, 4] = actions[:, 4] + active * w_l
                actions[:, 5] = actions[:, 5] + active * w_r
        return super().step(actions)

    # ------------------------------------------------------------------ 终止
    def check_termination(self):
        if self.playback_hold:
            # 回放：屏蔽摔倒/超时/任务完成复位。play02 靠 isaac 侧外部复位（按 R），
            #   否则起立完成(sw_done)或一次擦地都会把机器人瞬移回起点、打断回放。
            self.reset_buf = torch.zeros_like(self.reset_buf)
            self.time_out_buf = torch.zeros_like(self.time_out_buf)
            return
        super().check_termination()          # 摔倒类（base/thigh/roll/pitch）
        self.reset_buf |= self.sw_done       # 任务完成 → reset

    # ------------------------------------------------------------------ 观测
    def _get_noise_scale_vec(self, cfg):
        """基础观测(164)加噪；switch 块(9，相位 one-hot/任务符号/UEP)不加噪 ⇒ 只取前 164。"""
        return super()._get_noise_scale_vec(cfg)[:self.BASE_OBS_DIM]

    def compute_observations(self):
        super().compute_observations()       # 164 维（含跳跃块常量 + caster + tip）
        # switch 块 9 维：相位 one-hot(7) + 任务(1) + UEP 偏差代理(1)
        phase_onehot = torch.stack(
            [(self.ph == k).float() for k in range(NUM_SW_PHASES)], dim=1)     # [n,7]
        task_col = (self.sw_task.float() * 2.0 - 1.0).unsqueeze(1)             # 起立+1/趴下-1
        uep_col = torch.clamp(self.sw_uep, -1.0, 1.0).unsqueeze(1)
        self.obs_buf = torch.cat((self.obs_buf, phase_onehot, task_col, uep_col), dim=-1)

    # ------------------------------------------------------------------ 奖励
    def _reward_sw_align(self):
        """对齐起点姿态（保持起点，软跟踪；不覆盖上升奖励）。起立 U0 与趴下 D0 各自对齐。"""
        p = self.ph
        active = (p == SP_HOLD4W) | (p == SP_HOLD2W)
        err = torch.mean(torch.abs(self.dof_pos[:, self.leg_dof] - self.ref_pose), dim=1)
        return torch.clamp(1.0 - err / 0.5, min=0.0) * active.float()

    def _reward_sw_height(self):
        """升体：base_z 从 z4w→z2w 的归一化进度（clamp 0~1，正向密集）。仅起立回合生效。"""
        z = self.root_states[:, 2]
        prog = torch.clamp((z - self.sw.z4w) / (self.sw.z2w - self.sw.z4w), 0.0, 1.0)
        return prog * (self.sw_task == 1).float()

    def _reward_sw_goal_state(self):
        """【核心】目标态密集：双从动轮离地 ∧ 双驱动轮触地 ∧ |pitch|小 ∧ 低速，连乘。仅起立回合生效。"""
        casters_off = (~self._caster_contacts().any(dim=1)).float()
        wheels_on = (self.contact_forces[:, self.feet_indices, 2] > 0.1).all(dim=1).float()
        pitch_ok = torch.exp(-torch.square(self._body_pitch()) / (2 * 0.12 ** 2))
        v = torch.norm(self.root_states[:, 7:9], dim=1)
        slow = torch.exp(-torch.square(v) / (2 * 0.4 ** 2))
        return casters_off * wheels_on * pitch_ok * slow * (self.sw_task == 1).float()

    def _reward_sw_caster_lift(self):
        """抬从动轮：从动轮离地（两从动轮都离地=1）。仅起立回合生效。"""
        casters = self._caster_contacts()
        return (~casters.any(dim=1)).float() * (self.sw_task == 1).float()

    def _reward_sw_keep_wheel(self):
        """保持双驱动轮触地（不跳起/不打滑悬空）。"""
        wheels = self.contact_forces[:, self.feet_indices, 2] > 0.1
        return wheels.all(dim=1).float()

    def _reward_sw_pitch_zero(self):
        """俯仰归零：exp(−pitch²/2σ²)（温和 σ，允许起立必要瞬态）。"""
        return torch.exp(-torch.square(self._body_pitch()) / (2 * self.sw.pitch_sigma ** 2))

    def _reward_sw_symmetry(self):
        """【甲】左右腿对称：髋左右反号相等、膝左右反号相等（消除"一前一后"）。
        leg_dof 顺序 [hip_l, knee_l, hip_r, knee_r]；理想对称＝(hip_l + hip_r)→0、(knee_l + knee_r)→0。
        全程生效（起立过程与站定都要对称）；σ 取宽（0.6 rad≈34°）保证当前大偏差(~64°)也有梯度，能真正拉回。"""
        dp = self.dof_pos[:, self.leg_dof]                 # [n,4]
        e_hip = dp[:, 0] + dp[:, 2]                        # hip_l + hip_r
        e_knee = dp[:, 1] + dp[:, 3]                       # knee_l + knee_r
        return torch.exp(-(torch.square(e_hip) + torch.square(e_knee)) / (2 * 0.6 ** 2))

    def _reward_sw_still(self):
        """【甲】站定静止：抑制残余线速/偏航角速度，让"站定"判据(速度<tol)真正达成、从而能进并稳住 HOLD2W。
        从 HANDOVER 起就生效（不能只在 HOLD2W：策略在 HANDOVER 慢不下来就永远进不了 HOLD2W，still 恒 0 没梯度）。
        σ 取宽（v 0.6 / ω 0.8）保证 0.4m/s 量级也有梯度。"""
        v = torch.norm(self.root_states[:, 7:9], dim=1)
        w = torch.abs(self.root_states[:, 11])
        val = torch.exp(-torch.square(v) / (2 * 0.6 ** 2)) * torch.exp(-torch.square(w) / (2 * 0.8 ** 2))
        active = ((self.ph == SP_HANDOVER) | (self.ph == SP_HOLD2W)).float() * (self.sw_task == 1).float()
        return val * active

    def _reward_sw_ev_upright(self):
        """起立成功一次性结算（站定闸确认那一拍给 1）。"""
        return self.ev_up_fresh.float()

    def _reward_sw_ev_settled4(self):
        """趴下成功一次性结算（暂不做，权重 0）。"""
        return self.ev_set_fresh.float()

    def _reward_sw_impact(self):
        """从动轮触地冲击速度 hinge 罚（超过 0.3 m/s 的部分）。"""
        return torch.clamp(self.v_impact - 0.3, min=0.0)

    # ── 趴下（2W→4W）密集塑形：均仅趴下回合(sw_task==0)生效 ──
    def _reward_sw_lower(self):
        """降体：base_z 从 z2w→z4w 的归一化进度（正向密集，鼓励趴下）。"""
        z = self.root_states[:, 2]
        prog = torch.clamp((self.sw.z2w - z) / (self.sw.z2w - self.sw.z4w), 0.0, 1.0)
        return prog * (self.sw_task == 0).float()

    def _reward_sw_caster_touch(self):
        """探地：从动轮触地（两从动轮都触地=1），鼓励放下从动轮。"""
        return self._caster_contacts().all(dim=1).float() * (self.sw_task == 0).float()

    def _reward_sw_goal_4w(self):
        """【核心】四轮目标态：双从动轮触地 ∧ 双驱动轮触地 ∧ |pitch|小 ∧ 低速，连乘。仅趴下回合生效。"""
        casters_on = self._caster_contacts().all(dim=1).float()
        wheels_on = (self.contact_forces[:, self.feet_indices, 2] > 0.1).all(dim=1).float()
        pitch_ok = torch.exp(-torch.square(self._body_pitch()) / (2 * 0.12 ** 2))
        v = torch.norm(self.root_states[:, 7:9], dim=1)
        slow = torch.exp(-torch.square(v) / (2 * 0.4 ** 2))
        return casters_on * wheels_on * pitch_ok * slow * (self.sw_task == 0).float()
