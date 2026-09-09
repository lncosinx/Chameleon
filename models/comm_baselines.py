"""
SQ3 外部通信基线（CommNet / DIAL / ATOC / TarMAC / IC3Net / DGN）。

**为什么要自己实现而不是跑原作者代码**：这些方法的原始实现各自绑死了不同的环境、骨干网络
和训练范式（有的是 A3C、有的是 DQN、有的连观测都不同）。直接拿原仓库跑 = 同时变了通信机制、
骨干、训练算法三样东西，得出的差异说明不了"是通信机制的功劳"。本文件把它们**全部移植到与
MADCL 完全相同的骨干上**（同一残差 CNN 编码器、同一 GRU 时间核、同一动作头、同一离线 IL 训练、
同一 FOV 通信半径），**唯一变量 = 通信模块**。这是唯一能支撑"我们的通信机制更好"这一主张的比法。

统一接口（与 models/communication.py 的 BiIntentAttention 对齐，可直接互换）：
    forward_sequence(intents (B,T,N,D), positions (B,T,N,2)) -> (comm (B,T,N,D), info dict)
    reset_state()

公平性约定（所有臂一致）：
  - 邻居候选集 = FOV 内（0 < 距离 ≤ comm_radius），与 INVARIANT 10 相同；
  - 消息维度、参数量级尽量对齐（都用 comm_dim）；
  - info 里都报 avg_neighbors / edge_flip_rate / avg_active_neighbors，
    以便画"通信量-性能"Pareto 与稳定性曲线。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def _neighbor_mask(positions: torch.Tensor, comm_radius: float) -> torch.Tensor:
    """(B,T,N,2) → (B,T,N,N) bool：FOV 内邻居（不含自己）。"""
    diff = positions.unsqueeze(3) - positions.unsqueeze(2)          # (B,T,N,N,2)
    dist = torch.norm(diff.float(), dim=-1)
    return (dist <= comm_radius) & (dist > 1e-6)


def _edge_flip_rate(edges: torch.Tensor, prev_last: Optional[torch.Tensor]) -> float:
    """相邻时间步之间通信边集合的变化率（越低越稳定）。"""
    seq = torch.cat([prev_last.unsqueeze(1), edges], dim=1) if prev_last is not None else edges
    if seq.shape[1] <= 1:
        return 0.0
    a, b = seq[:, 1:], seq[:, :-1]
    flips = (a != b).float().sum(dim=(-1, -2))
    union = (a | b).float().sum(dim=(-1, -2)).clamp(min=1)
    return float((flips / union).mean().item())


class _BaseComm(nn.Module):
    """共用：邻居掩码、统计量、跨调用状态（用于 EdgeFlipRate 的跨段连续性）。"""

    def __init__(self, comm_radius: int = 4):
        super().__init__()
        self.comm_radius = comm_radius
        self._prev_edge_set: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int = None):
        self._prev_edge_set = None

    def _stats(self, edges: torch.Tensor) -> dict:
        prev = self._prev_edge_set
        if prev is not None and prev.shape != edges.shape[:1] + edges.shape[2:]:
            prev = None
        efr = _edge_flip_rate(edges, prev)
        self._prev_edge_set = edges[:, -1].detach()
        deg = edges.float().sum(-1)
        return {"edge_flip_rate": efr,
                "avg_neighbors": float(deg.mean().item()),
                "avg_active_neighbors": float(deg.mean().item())}

    def forward(self, intents, positions):
        c, info = self.forward_sequence(intents.unsqueeze(1), positions.unsqueeze(1))
        return c.squeeze(1), info


# ===========================================================================
# CommNet (Sukhbaatar et al., NeurIPS'16)
#   c_i = H · mean_{j∈N(i)} h_j —— 无选择、无权重，所有可见邻居等权平均。
# ===========================================================================
class CommNetComm(_BaseComm):
    def __init__(self, intent_dim=128, comm_dim=128, comm_radius=4, **_):
        super().__init__(comm_radius)
        self.h_proj = nn.Linear(intent_dim, comm_dim)
        self.out_proj = nn.Linear(comm_dim, intent_dim)

    def forward_sequence(self, intents, positions):
        nb = _neighbor_mask(positions, self.comm_radius)                 # (B,T,N,N)
        h = self.h_proj(intents)                                          # (B,T,N,Dc)
        w = nb.float()
        denom = w.sum(-1, keepdim=True).clamp(min=1.0)
        c = torch.einsum('btnm,btmd->btnd', w / denom, h)
        c = self.out_proj(c) * (w.sum(-1, keepdim=True) > 0).float()
        return c, self._stats(nb)


# ===========================================================================
# IC3Net (Singh et al., ICLR'19)
#   CommNet + 每体一个**学习到的二值门**决定"这一步我要不要广播"。
#   训练期用 straight-through Bernoulli（前向硬采样、反向走 sigmoid 梯度）。
# ===========================================================================
class IC3NetComm(_BaseComm):
    def __init__(self, intent_dim=128, comm_dim=128, comm_radius=4, **_):
        super().__init__(comm_radius)
        self.h_proj = nn.Linear(intent_dim, comm_dim)
        self.gate = nn.Linear(intent_dim, 1)
        # ★ 门初始化为"默认打开"（bias=+1 ⇒ p≈0.73）。随机初始化下 sigmoid 常落在 0.5 以下，
        # 推断期 (p>0.5) 硬门会让这个臂**开局完全不通信**，梯度也难把它推开 —— 那是给基线
        # 人为设障，比较就不公平了。门控类方法的常规做法就是偏置初始化为开。
        nn.init.constant_(self.gate.bias, 1.0)
        self.out_proj = nn.Linear(comm_dim, intent_dim)

    def forward_sequence(self, intents, positions):
        nb = _neighbor_mask(positions, self.comm_radius)
        p_gate = torch.sigmoid(self.gate(intents)).squeeze(-1)           # (B,T,N) 发送概率
        if self.training:
            hard = torch.bernoulli(p_gate)
            g = hard + p_gate - p_gate.detach()                           # straight-through
        else:
            g = (p_gate > 0.5).float()
        h = self.h_proj(intents)
        w = nb.float() * g.unsqueeze(2)                                   # 只有开门的体才发出
        denom = w.sum(-1, keepdim=True).clamp(min=1.0)
        c = torch.einsum('btnm,btmd->btnd', w / denom, h)
        c = self.out_proj(c) * (w.sum(-1, keepdim=True) > 0).float()
        info = self._stats(w > 0)
        info["gate_open_frac"] = float(g.mean().item())
        return c, info


# ===========================================================================
# TarMAC (Das et al., ICML'19)
#   发送方发 (signature k_j, value v_j)，接收方发 query q_i，
#   注意力 = softmax over FOV 内邻居（**dense**：所有可见邻居都拿到非零权重）。
# ===========================================================================
class TarMACComm(_BaseComm):
    def __init__(self, intent_dim=128, comm_dim=128, comm_radius=4, sig_dim=16, **_):
        super().__init__(comm_radius)
        self.q = nn.Linear(intent_dim, sig_dim)
        self.k = nn.Linear(intent_dim, sig_dim)
        self.v = nn.Linear(intent_dim, comm_dim)
        self.out_proj = nn.Linear(comm_dim, intent_dim)
        self.scale = sig_dim ** -0.5

    def forward_sequence(self, intents, positions):
        nb = _neighbor_mask(positions, self.comm_radius)
        logits = torch.matmul(self.q(intents), self.k(intents).transpose(-1, -2)) * self.scale
        logits = logits.masked_fill(~nb, float('-inf'))
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0) * nb.any(-1, keepdim=True).float()
        c = self.out_proj(torch.einsum('btnm,btmd->btnd', attn, self.v(intents)))
        return c, self._stats(attn > 1e-6)


# ===========================================================================
# DIAL (Foerster et al., NeurIPS'16)
#   离散化消息 + DRU（训练期加噪 → 二值化；推断期硬阈值），接收方求和。
# ===========================================================================
class DIALComm(_BaseComm):
    def __init__(self, intent_dim=128, comm_dim=128, comm_radius=4, msg_bits=16,
                 noise_std=0.5, **_):
        super().__init__(comm_radius)
        self.msg_head = nn.Linear(intent_dim, msg_bits)
        self.msg_in = nn.Linear(msg_bits, comm_dim)
        self.out_proj = nn.Linear(comm_dim, intent_dim)
        self.noise_std = noise_std

    def forward_sequence(self, intents, positions):
        nb = _neighbor_mask(positions, self.comm_radius)
        m = self.msg_head(intents)                                        # (B,T,N,bits)
        if self.training:      # DRU：加噪后 sigmoid（噪声逼近二值化，梯度可传）
            m = torch.sigmoid(m + torch.randn_like(m) * self.noise_std)
        else:
            m = (m > 0).float()
        w = nb.float()
        agg = torch.einsum('btnm,btmd->btnd', w, m)                       # 接收方求和
        c = self.out_proj(self.msg_in(agg)) * (w.sum(-1, keepdim=True) > 0).float()
        return c, self._stats(nb)


# ===========================================================================
# ATOC (Jiang & Lu, NeurIPS'18)
#   注意力单元决定"是否发起通信"；发起者与其 FOV 内邻居组成通信组，
#   组内用双向 LSTM 整合信息（原文用 BiLSTM 作 communication channel）。
# ===========================================================================
class ATOCComm(_BaseComm):
    def __init__(self, intent_dim=128, comm_dim=128, comm_radius=4, group_size=6, **_):
        super().__init__(comm_radius)
        self.initiator = nn.Linear(intent_dim, 1)
        nn.init.constant_(self.initiator.bias, 1.0)   # 同 IC3Net：发起器默认开，见其注释
        self.h_proj = nn.Linear(intent_dim, comm_dim)
        self.channel = nn.LSTM(comm_dim, comm_dim // 2, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(comm_dim, intent_dim)
        self.group_size = group_size

    def forward_sequence(self, intents, positions):
        B, T, N, D = intents.shape
        nb = _neighbor_mask(positions, self.comm_radius)
        p_init = torch.sigmoid(self.initiator(intents)).squeeze(-1)       # (B,T,N)
        init = (p_init > 0.5).float()
        if self.training:
            init = torch.bernoulli(p_init) + p_init - p_init.detach()

        # 通信组：发起者 + 其最近的 group_size 个 FOV 内邻居
        dist = torch.norm((positions.unsqueeze(3) - positions.unsqueeze(2)).float(), dim=-1)
        dist = dist.masked_fill(~nb, float('inf'))
        k = min(self.group_size, N)
        idx = dist.topk(k, dim=-1, largest=False).indices                 # (B,T,N,k)
        valid = torch.gather(nb.float(), -1, idx)                         # 真实存在的邻居

        h = self.h_proj(intents)                                          # (B,T,N,Dc)
        Dc = h.shape[-1]
        gathered = torch.gather(h.unsqueeze(2).expand(B, T, N, N, Dc), 3,
                                idx.unsqueeze(-1).expand(B, T, N, k, Dc))
        seq = torch.cat([h.unsqueeze(3), gathered * valid.unsqueeze(-1)], dim=3)  # 自己+组员
        out, _ = self.channel(seq.reshape(B * T * N, k + 1, Dc))
        c = out[:, 0].reshape(B, T, N, Dc)                                # 取自己位置的输出
        c = self.out_proj(c) * init.unsqueeze(-1)                         # 没发起则不通信
        edges = (valid > 0) & (init.unsqueeze(-1) > 0)
        info = self._stats(torch.zeros_like(nb).scatter(-1, idx, edges))
        info["initiate_frac"] = float(init.mean().item())
        return c, info


# ===========================================================================
# DGN (Jiang et al., ICLR'20)
#   多头点积注意力图卷积，邻居集 = k 近邻（原文固定 k，**hard top-k**），
#   与我们的"软稀疏+迟滞"形成直接对照（正是 INVARIANT 10 警告的闪烁来源）。
# ===========================================================================
class DGNComm(_BaseComm):
    def __init__(self, intent_dim=128, comm_dim=128, comm_radius=4, num_heads=4, topk=3, **_):
        super().__init__(comm_radius)
        self.q = nn.Linear(intent_dim, comm_dim)
        self.k_proj = nn.Linear(intent_dim, comm_dim)
        self.v = nn.Linear(intent_dim, comm_dim)
        self.out_proj = nn.Linear(comm_dim, intent_dim)
        self.num_heads = num_heads
        self.head_dim = comm_dim // num_heads
        self.topk = topk

    def forward_sequence(self, intents, positions):
        B, T, N, _ = intents.shape
        nb = _neighbor_mask(positions, self.comm_radius)
        dist = torch.norm((positions.unsqueeze(3) - positions.unsqueeze(2)).float(), dim=-1)
        dist = dist.masked_fill(~nb, float('inf'))
        k = min(self.topk, N)
        idx = dist.topk(k, dim=-1, largest=False).indices
        hard = torch.zeros_like(nb).scatter(-1, idx, True) & nb           # hard top-k 邻居集

        logits = torch.matmul(self.q(intents), self.k_proj(intents).transpose(-1, -2))
        logits = logits * (self.head_dim ** -0.5)
        logits = logits.masked_fill(~hard, float('-inf'))
        attn = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        attn = attn * hard.any(-1, keepdim=True).float()
        c = self.out_proj(torch.einsum('btnm,btmd->btnd', attn, self.v(intents)))
        return c, self._stats(hard)


COMM_REGISTRY = {
    "commnet": CommNetComm,
    "ic3net": IC3NetComm,
    "tarmac": TarMACComm,
    "dial": DIALComm,
    "atoc": ATOCComm,
    "dgn": DGNComm,
}


def build_comm_baseline(name: str, **kwargs) -> nn.Module:
    if name not in COMM_REGISTRY:
        raise ValueError(f"未知通信基线 {name!r}，可选：{sorted(COMM_REGISTRY)}")
    return COMM_REGISTRY[name](**kwargs)
