# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : com_model.py
# Purpose   : 四轮系统的质心前向解算（机体系偏移），供"纵向倾翻裕度"观测使用。
#             ⚠ 本文件是 legged_gym/envs/wbr/flat4/wbr_flat4.py 的**逐字副本**（该环境文件
#             import isaacgym，部署环境不能直接导入）。副本与原文的一致性由 golden 对账的
#             obs 通道 163 把关；训练停后在环境侧做一次纯搬移即可消掉双正文。
#             源文件 sha1 记录在 assets/deploy_contract.json 的 sources 里。
# ==============================================================================

import torch

COM_M = {"base": 2.5418, "thigh": 0.80815,
         "shank_l": 0.93138, "shank_r": 0.92691, "wheel": 0.55336,
         "caster": 0.05}
COM_G = {"base": (4.4472e-05, 1.156e-04, 0.010416),
         "thigh_l": (0.15659, 3.2005e-04, 0.02025),
         "thigh_r": (0.15659, 1.9029e-04, 0.02025),
         "shank_l": (0.12156, 4.8719e-03, 1.996e-02),
         "shank_r": (0.12114, 5.1863e-03, 2.0146e-02),
         "wheel_l": (0.0, 0.0, 0.025381), "wheel_r": (0.0, 0.0, 0.025381)}
COM_CASTER_L = (-0.02939, 0.05000, 0.01983)
COM_CASTER_R = (-0.02939, -0.05000, 0.01983)
COM_TOTAL = (COM_M["base"] + 2 * COM_M["thigh"] + COM_M["shank_l"]
             + COM_M["shank_r"] + 2 * COM_M["wheel"] + 2 * COM_M["caster"])  # 7.22311 kg
COM_HIP_Y, COM_KNEE_O, COM_WHEEL_O = 0.0936, (0.2, 0.0, 0.0403), (0.18, 0.0, 0.0403)


def _com_rot_z(q):
    """[n] 的绕自身 +z 转角 → [n,3,3]。三行按 dim=1 堆叠（按 -1 堆会得到转置，踩过）。"""
    c, s = torch.cos(q), torch.sin(q)
    o, z = torch.ones_like(q), torch.zeros_like(q)
    return torch.stack([torch.stack([c, -s, z], -1),
                        torch.stack([s, c, z], -1),
                        torch.stack([z, z, o], -1)], 1)


def _com_hip_pre(side, dev):
    """髋关节固定 rpy（左 (−π/2,π/2,0)、右 (π/2,π/2,0)）；cos(±π/2)=0 ⇒ 直接写常量。"""
    sgn = 1.0 if side == "r" else -1.0
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -sgn], [0.0, sgn, 0.0]],
                      dtype=torch.float32, device=dev)
    ry = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
                      dtype=torch.float32, device=dev)
    return ry @ rx


def _com_ev(R, v):
    """[n,3,3] 作用到常量 [3] 或逐环境 [n,3]，返回 [n,3]。"""
    if v.dim() == 1:
        return torch.einsum("nij,j->ni", R, v.to(R.device))
    return torch.einsum("nij,nj->ni", R, v)


def com_constants(dev):
    """质心 FK 需要的常量件（每设备建一次）。"""
    def mk(v):
        return torch.tensor(v, dtype=torch.float32, device=dev)
    return {"base": mk(COM_G["base"]), "o_knee": mk(COM_KNEE_O),
            "hip_l": mk((0.0, COM_HIP_Y, 0.0)), "hip_r": mk((0.0, -COM_HIP_Y, 0.0)),
            "pre_l": _com_hip_pre("l", dev), "pre_r": _com_hip_pre("r", dev),
            "o_wheel_wh": mk([a + b for a, b in zip(COM_WHEEL_O, COM_G["wheel_l"])]),
            "o_caster_l": mk(COM_CASTER_L), "o_caster_r": mk(COM_CASTER_R),
            **{k: mk(COM_G[k]) for k in ("thigh_l", "thigh_r", "shank_l", "shank_r")}}


def com_body_offset(c, hip_l, hip_r, knee_l, knee_r):
    """机体系质心偏移 [n,3]；c 是 `com_constants` 的常量件。"""
    n = hip_l.shape[0]
    acc = c["base"].expand(n, 3).clone() * COM_M["base"]
    for s, qh, qk in (("l", hip_l, knee_l), ("r", hip_r, knee_r)):
        th = c["pre_" + s] @ _com_rot_z(qh)
        sh = th @ _com_rot_z(qk)
        p_th = c["hip_" + s] + _com_ev(th, c["thigh_" + s])
        p_sh = c["hip_" + s] + _com_ev(th, c["o_knee"]) + _com_ev(sh, c["shank_" + s])
        p_wh = c["hip_" + s] + _com_ev(th, c["o_knee"]) + _com_ev(sh, c["o_wheel_wh"])
        p_ca = c["hip_" + s] + _com_ev(th, c["o_knee"]) + _com_ev(sh, c["o_caster_" + s])
        acc = acc + COM_M["thigh"] * p_th + COM_M["shank_" + s] * p_sh \
            + COM_M["wheel"] * p_wh + COM_M["caster"] * p_ca
    return acc / COM_TOTAL
