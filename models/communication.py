"""
通信模块：FOV 内 top-k 双向意图注意力。

约束（INVARIANT 10）：
- comm_radius=4 < obs_radius=5
- 仅与 FOV 内可见邻居通信
- top-k=6 截断，抑制拓扑雪崩
- 批量稀疏注意力，禁止 per-agent Python 循环
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiIntentAttention(nn.Module):
    """
    双向意图注意力层。

    为每个 agent 聚合其 top-k 最近可见邻居的意图向量。

    输入:
        intents:    (B, N, D)  各体意图向量
        positions:  (B, N, 2)  各体全局坐标 (row, col)

    输出:
        comm_features: (B, N, D)  通信聚合后的特征
        attn_weights:  (B, N, K)  注意力权重（用于监控）
    """

    def __init__(
        self,
        intent_dim: int = 128,
        comm_dim: int = 64,
        comm_radius: int = 4,
        comm_topk: int = 6,
        num_heads: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.comm_radius = comm_radius
        self.comm_topk = comm_topk
        self.num_heads = num_heads
        self.temperature = temperature
        self.head_dim = comm_dim // num_heads

        # 意图 → query/key/value
        self.q_proj = nn.Linear(intent_dim, comm_dim)
        self.k_proj = nn.Linear(intent_dim, comm_dim)
        self.v_proj = nn.Linear(intent_dim, comm_dim)
        self.out_proj = nn.Linear(comm_dim, intent_dim)

    def forward(
        self,
        intents: torch.Tensor,    # (B, N, D_intent)
        positions: torch.Tensor,  # (B, N, 2)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = intents.shape

        # ---- 构建邻接矩阵：基于 comm_radius ----
        # pos_diff (B, N, N, 2)
        pos_diff = positions.unsqueeze(1) - positions.unsqueeze(2)
        dist = torch.norm(pos_diff.float(), dim=-1)  # (B, N, N)

        # 邻居：距离 ≤ comm_radius 且非自身
        is_neighbor = (dist <= self.comm_radius) & (dist > 1e-6)  # (B, N, N)

        # top-k 截断：对每个 agent 只保留最近的 k 个邻居
        # 将非邻居的距离设为 inf，然后 topk
        dist_masked = dist.clone()
        dist_masked[~is_neighbor] = float('inf')

        # top-k 索引
        _, topk_indices = torch.topk(
            dist_masked, k=min(self.comm_topk, N), dim=-1, largest=False
        )  # (B, N, K)

        # ---- 收集 top-k 邻居的意图 ----
        K_actual = topk_indices.shape[-1]

        # gather: (B, N, K, D)
        intent_gathered = batched_index_select(intents, topk_indices)

        # ---- 多头注意力 ----
        Q = self.q_proj(intents).view(B, N, self.num_heads, self.head_dim)  # (B,N,H,d)
        K = self.k_proj(intent_gathered.view(B * N, K_actual, D))
        K = K.view(B, N, K_actual, self.num_heads, self.head_dim)
        V = self.v_proj(intent_gathered.view(B * N, K_actual, D))
        V = V.view(B, N, K_actual, self.num_heads, self.head_dim)

        # attention scores: Q (B,N,H,d), K (B,N,K,H,d) → (B,N,H,K)
        # Q: (B,N,H,d) → (B,N,H,1,d), K: (B,N,K,H,d)
        scores = torch.einsum('bnhid,bnkhi->bnhk', Q.unsqueeze(3), K)
        scores = scores / (self.head_dim ** 0.5 * self.temperature)

        # 对无邻居的 agent，scores 全为 0（K_actual=0 时不进入此路径）
        weights = F.softmax(scores, dim=-1)  # (B, N, H, K)

        # 加权聚合: (B, N, H, d)
        attended = torch.einsum('bnhk,bnkhi->bnhi', weights, V)
        attended = attended.reshape(B, N, self.num_heads * self.head_dim)

        comm_features = self.out_proj(attended)

        # 无邻居的 agent: 通信特征 = 0（通过 mask 实现）
        has_neighbor = is_neighbor.any(dim=-1).float().unsqueeze(-1)  # (B, N, 1)
        comm_features = comm_features * has_neighbor

        # 用于监控的注意力权重（取每头平均）
        attn_weights = weights.mean(dim=2)  # (B, N, K)

        return comm_features, attn_weights


def batched_index_select(
    x: torch.Tensor,  # (B, N, D)
    indices: torch.Tensor,  # (B, N, K)
) -> torch.Tensor:
    """
    批量 index_select：对每个 batch 和每个 anchor，从 x 中选 indices[b,n,:]。

    Args:
        x: (B, N, D) 源张量。
        indices: (B, N, K) 索引（long）。

    Returns:
        gathered: (B, N, K, D)
    """
    B, N, D = x.shape
    _, _, K = indices.shape

    # 将 x 展平为 (B*N, D)，indices 加 offset
    offset = torch.arange(B, device=x.device).view(B, 1, 1) * N  # (B, 1, 1)
    flat_indices = (indices + offset).reshape(-1)  # (B*N*K,)
    flat_x = x.reshape(B * N, D)  # (B*N, D)

    gathered = flat_x[flat_indices]  # (B*N*K, D)
    gathered = gathered.reshape(B, N, K, D)
    return gathered
