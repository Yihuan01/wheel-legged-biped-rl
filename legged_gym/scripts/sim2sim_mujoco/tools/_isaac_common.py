# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : _isaac_common.py
# Purpose   : Isaac 侧采集脚本共用件：把域随机化/噪声/延迟打到名义值，并断言确实打到了，
#             保证 golden 与静态契约只反映训练名义口径。
# ==============================================================================

from legged_gym.envs.wbr.flat4.wbr_flat4_config import WBRFlat4Cfg
from legged_gym.envs.wbr.flat.wbr_flat_config import WBRFlatCfg
from legged_gym.envs.wbr.switch.wbr_switch_config import WBRSwitchCfg


def _patch_cfg(cfg):
    """名义化一套任务 config（四轮/两轮共用：域随机化字段为基类共有的同名字段）。"""
    cfg.noise.add_noise = False
    dr = cfg.domain_rand
    dr.enable_dr_sim = False
    dr.randomize_friction = False
    dr.randomize_base_mass = False
    dr.push_robots = False
    dr.action_delay_max_steps = 0
    dr.obs_delay_max_steps = 0
    dr.pd_gain_range = [1.0, 1.0]
    dr.motor_strength_range = [1.0, 1.0]
    dr.com_offset_range = [0.0, 0.0]
    dr.imu_gyro_bias_range = [0.0, 0.0]
    dr.imu_acc_bias_range = [0.0, 0.0]
    dr.gravity_offset_range = [0.0, 0.0]
    if hasattr(dr, "caster_joint_damp_range"):        # v1 两轮无从动轮，字段可能不存在
        dr.caster_joint_damp_range = [0.02, 0.02]
    # ⚠ 现码 resampling_time=1.0 s（交付包是 10 s）：采集时会把指令每 100 步重随机一次，
    #   必须钉死，否则 4 s 的 golden 里指令会中途变（yaw05 的指令在第 100 步被清 0）。
    cfg.commands.resampling_time = 1.0e6
    cfg.commands.smooth_commands = False


def patch_nominal(task="wbr_flat4"):
    _patch_cfg({"wbr_flat4": WBRFlat4Cfg, "wbr_flat": WBRFlatCfg, "wbr_switch": WBRSwitchCfg}[task])


def assert_nominal(env):
    assert not env.add_noise, "add_noise 未关"
    assert float(env._dr_pd.max()) == 1.0 and float(env._dr_pd.min()) == 1.0, "pd DR 未关"
    assert float(env._dr_motor.max()) == 1.0 and float(env._dr_motor.min()) == 1.0, "motor DR 未关"
    assert float(env._dr_gyro_bias.abs().max()) == 0.0, "gyro bias 未关"
    assert float(env._dr_pg_bias.abs().max()) == 0.0, "gravity/acc bias 未关"
    assert getattr(env, "_dr_obs_delay", None) is None or int(env._dr_obs_delay.max()) == 0
    assert getattr(env, "_dr_act_delay", None) is None or int(env._dr_act_delay.max()) == 0
