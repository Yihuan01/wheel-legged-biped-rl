# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

# isaacgym 必须先于 torch 导入（gymdeps 会拒绝 torch-first 的进程），
# 所以 .base.legged_robot 要排在任何 utils/环境子模块之前。
from .base.legged_robot import LeggedRobot
from legged_gym.utils.task_registry import task_registry

# ── 两轮：平地行走 / 跳跃 ──────────────────────────────────────────────────────
from .wbr.flat.wbr_flat import WBRFlatMob
from .wbr.flat.wbr_flat_config import WBRFlatCfg, WBRFlatCfgPPO
task_registry.register("wbr_flat", WBRFlatMob, WBRFlatCfg(), WBRFlatCfgPPO())

# ── 两轮：随机起伏地形（观测仍 161 维，不放高程）──────────────────────────────
#   地形由 terrain/*_gen.py 离线生成为 .npz，训练侧只读。
#   wbr_demo / wbr_combo / wbr_elements 是同一份策略的演示与验收场景。
from .wbr.flat.wbr_terrain_config import (WBRTerrainCfg, WBRTerrainCfgPPO,
                                          WBRDemoCfg, WBRDemoCfgPPO,
                                          WBRComboCfg, WBRComboCfgPPO,
                                          WBRElementsCfg, WBRElementsCfgPPO)
task_registry.register("wbr_terrain", WBRFlatMob, WBRTerrainCfg(), WBRTerrainCfgPPO())
task_registry.register("wbr_demo", WBRFlatMob, WBRDemoCfg(), WBRDemoCfgPPO())
task_registry.register("wbr_combo", WBRFlatMob, WBRComboCfg(), WBRComboCfgPPO())
task_registry.register("wbr_elements", WBRFlatMob, WBRElementsCfg(), WBRElementsCfgPPO())

# ── 四轮：静稳构型行驶 ────────────────────────────────────────────────────────
from .wbr.flat4.wbr_flat4 import WBRFlat4Mob
from .wbr.flat4.wbr_flat4_config import WBRFlat4Cfg, WBRFlat4CfgPPO
task_registry.register("wbr_flat4", WBRFlat4Mob, WBRFlat4Cfg(), WBRFlat4CfgPPO())

# ── 模式切换：起立 / 趴下 ─────────────────────────────────────────────────────
from .wbr.switch.wbr_switch import WBRSwitchMob
from .wbr.switch.wbr_switch_config import WBRSwitchCfg, WBRSwitchCfgPPO
task_registry.register("wbr_switch", WBRSwitchMob, WBRSwitchCfg(), WBRSwitchCfgPPO())

# ── 统一结构：相位枚举 + FSM + 多模式环境（只回放，不训练）────────────────────
from .wbr.uni.wbr_uni import WBRUniMob
from .wbr.uni.wbr_uni_config import WBRUniCfg, WBRUniCfgPPO
task_registry.register("wbr_uni", WBRUniMob, WBRUniCfg(), WBRUniCfgPPO())
