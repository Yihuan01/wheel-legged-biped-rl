# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : policy.py
# Purpose   : rsl_rl checkpoint 的 actor 直载（纯 torch，无训练框架依赖）：
#             512-256-128 ELU MLP，只取 actor.* 权重，加载后做形状与有限性检查。
# ==============================================================================

import hashlib

import torch
import torch.nn as nn

# 【CPU 占用】单环境 512-256-128 的小网络，torch 默认会拉满 OpenMP/MKL 线程池
#   （实测 9 个线程各 90%+、进程 700%+ CPU，100 Hz 推理反而被线程调度拖慢）。
#   限到 1 线程：推理更快、主线程能及时响应 SIGINT/桥请求、停止不再迟钝。
#   只在导入本模块（部署侧）时设置，不影响训练/采集进程。
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Policy:
    def __init__(self, ckpt, obs_dim, act_dim=6, hidden=(512, 256, 128), device="cpu"):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        sd = {k[len("actor."):]: v for k, v in ck["model_state_dict"].items()
              if k.startswith("actor.")}
        assert sd, f"{ckpt} 里没有 actor.* 权重"

        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ELU()]
            last = h
        layers.append(nn.Linear(last, act_dim))
        self.model = nn.Sequential(*layers).to(device)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ckpt = str(ckpt)
        self.sha256 = _sha256(ckpt)
        self.iter = int(ck.get("iter", -1))
        assert self.model[0].in_features == obs_dim, self.model[0].in_features
        assert all(torch.isfinite(p).all() for p in self.model.parameters())

    @torch.no_grad()
    def act(self, obs):
        x = obs if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.model(x.float())

    def describe(self):
        n = sum(p.numel() for p in self.model.parameters())
        return {"ckpt": self.ckpt, "sha256": self.sha256, "iter": self.iter,
                "obs_dim": self.obs_dim, "act_dim": self.act_dim, "params": int(n)}
