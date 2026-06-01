"""
通信模块：FOV 内 软稀疏注意力(entmax) + 迟滞(hysteresis) 双向意图注意力。

约束（INVARIANT 10 修订版）：
- comm_radius=4 < obs_radius=5
- 仅与 FOV 内可见邻居通信
- 邻居选择用 **软稀疏注意力(entmax)+迟滞(τ_out < τ_in)**，
  **不用 hard top-k 做选择**——hard top-k 引起通信图闪烁
- comm_topk=6 仅作**算力上限**（非零邻居超额时封顶 GAT 计算量）
- 监控 EdgeFlipRate 防闪烁
- 批量稀疏注意力，禁止 per-agent Python 循环
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# =========================================================================
# entmax (α-entmax) — 软稀疏 attention
# =========================================================================

def entmax(logits: torch.Tensor, alpha: float = 1.5, dim: int = -1,
           n_iter: int = 25) -> torch.Tensor:
    """
    α-entmax：softmax (α=1) 与 sparsemax (α=2) 之间的稀疏归一化。

    实现基于 Peters et al. (2019) "Sparse Sequence-to-Sequence Models"
    的 bisection 算法。α ∈ (1, 2) 时产生稀疏输出。

    Args:
        logits: (..., N) 输入 logits
        alpha: 稀疏度参数，1=softmax, 2=sparsemax, 默认 1.5
        dim: 归一化维度
        n_iter: bisection 迭代次数

    Returns:
        probs: (..., N) 稀疏概率分布，sum=1, 部分值为 0
    """
    if alpha == 1.0:
        return F.softmax(logits, dim=dim)

    if alpha == 2.0:
        return _sparsemax(logits, dim=dim)

    # α ∈ (1, 2): bisection on τ
    # entmax 概率: p_i = [(α-1)·(logits_i - τ)]_+^{1/(α-1)}
    shape = logits.shape
    logits_flat = logits.reshape(-1, shape[dim])
    B, N = logits_flat.shape

    # 处理 -inf：用一个大负数替换 -inf 以避免 bisection 中的 NaN
    # 但 -inf 最终应映射到概率 0
    safe_logits = logits_flat.clone()
    safe_logits[torch.isneginf(safe_logits)] = -1e10  # 足够小，entmax 中权重 → 0

    # 处理全 -inf 的行（无合法候选）→ 返回均匀分布
    all_neg_inf = torch.all(torch.isneginf(logits_flat), dim=1)  # (B,)
    if all_neg_inf.any():
        uniform = torch.ones_like(logits_flat) / N
        safe_logits[all_neg_inf] = 0.0
    else:
        uniform = None

    # bisection bounds（基于安全 logits）
    tau_min = safe_logits.min(dim=1).values - 1.0
    tau_max = safe_logits.max(dim=1).values

    exponent = 1.0 / (alpha - 1.0)
    scale = alpha - 1.0

    for _ in range(n_iter):
        tau = (tau_min + tau_max) / 2.0
        margin = scale * (safe_logits - tau.unsqueeze(1))
        p = torch.clamp(margin, min=0.0).pow(exponent)
        sum_p = p.sum(dim=1)
        gt_one = sum_p > 1.0
        tau_min = torch.where(gt_one, tau, tau_min)
        tau_max = torch.where(gt_one, tau_max, tau)

    # Final computation
    tau = (tau_min + tau_max) / 2.0
    margin = scale * (safe_logits - tau.unsqueeze(1))
    p = torch.clamp(margin, min=0.0).pow(exponent)
    sum_p = p.sum(dim=1, keepdim=True).clamp(min=1e-12)
    p = p / sum_p

    # 对全 -inf 的行替换为均匀分布（通信中最终会被 has_neighbor mask 清零）
    if uniform is not None:
        p = torch.where(all_neg_inf.unsqueeze(1), uniform, p)

    return p.reshape(shape)


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax (α=2): 产生稀疏概率分布。

    Reference: Martins & Astudillo (2016)
    """
    shape = logits.shape
    logits_flat = logits.reshape(-1, shape[dim])
    B, N = logits_flat.shape

    # Sort descending
    z_sorted, _ = torch.sort(logits_flat, dim=1, descending=True)

    # Cumulative sum
    z_cumsum = torch.cumsum(z_sorted, dim=1)  # (B, N)

    # k(z) = max{k : 1 + k*z_k > sum_{j≤k} z_j}
    # threshold = (cumsum - 1) / k_range
    k_range = torch.arange(1, N + 1, device=logits.device, dtype=logits.dtype)
    threshold = (z_cumsum - 1.0) / k_range  # (B, N)

    # Find k: the largest index where z_k > threshold
    k_mask = z_sorted > threshold  # (B, N)
    k = k_mask.sum(dim=1).clamp(min=1)  # (B,)

    # τ = threshold at index k-1
    tau = threshold[torch.arange(B, device=logits.device), k - 1]  # (B,)

    # p_i = max(z_i - τ, 0)
    p = torch.clamp(logits_flat - tau.unsqueeze(1), min=0.0)

    return p.reshape(shape)


# =========================================================================
# 双向意图注意力（软稀疏 + 迟滞）
# =========================================================================

class BiIntentAttention(nn.Module):
    """
    双向意图注意力层（entmax 稀疏选择 + hysteresis 防闪烁）。

    输入:
        intents:    (B, N, D)  各体意图向量
        positions:  (B, N, 2)  各体全局坐标 (row, col)

    输出:
        comm_features: (B, N, D)  通信聚合后的特征
        info:          dict with attn_weights, edge_flip_rate, etc.
    """

    def __init__(
        self,
        intent_dim: int = 128,
        comm_dim: int = 64,
        comm_radius: int = 4,
        comm_topk_cap: int = 6,       # 仅算力上限
        num_heads: int = 4,
        # 软稀疏参数
        entmax_alpha: float = 1.5,    # 1=softmax, 2=sparsemax, 默认 1.5
        # 迟滞参数
        tau_in: float = 0.05,         # 进入活跃集的注意力阈值
        tau_out: float = 0.02,        # 退出活跃集的阈值 (τ_out < τ_in)
        stickiness_bonus: float = 0.1,  # 已活跃邻居的加分
        # 平滑参数
        ema_decay: float = 0.0,       # 跨步 EMA 平滑（0=不平滑）
    ):
        super().__init__()
        self.comm_radius = comm_radius
        self.comm_topk_cap = comm_topk_cap
        self.num_heads = num_heads
        self.entmax_alpha = entmax_alpha
        self.tau_in = tau_in
        self.tau_out = tau_out
        self.stickiness_bonus = stickiness_bonus
        self.ema_decay = ema_decay
        self.head_dim = comm_dim // num_heads

        # 意图 → query/key/value
        self.q_proj = nn.Linear(intent_dim, comm_dim)
        self.k_proj = nn.Linear(intent_dim, comm_dim)
        self.v_proj = nn.Linear(intent_dim, comm_dim)
        self.out_proj = nn.Linear(comm_dim, intent_dim)

        # 跨时间步状态（由外部管理）
        self._active_mask: Optional[torch.Tensor] = None  # (B, N, N) bool
        self._prev_edge_set: Optional[torch.Tensor] = None  # (B, N, N) bool
        self._ema_c: Optional[torch.Tensor] = None  # (B, N, D)

    def reset_state(self, batch_size: int = None):
        """重置跨 step 状态。"""
        self._active_mask = None
        self._prev_edge_set = None
        self._ema_c = None

    def forward(
        self,
        intents: torch.Tensor,    # (B, N, D_intent)
        positions: torch.Tensor,  # (B, N, 2)
    ) -> Tuple[torch.Tensor, dict]:
        B, N, D = intents.shape
        device = intents.device

        # ---- Step 1: 计算候选邻居（FOV 内 + 距离约束）----
        pos_diff = positions.unsqueeze(1) - positions.unsqueeze(2)
        dist = torch.norm(pos_diff.float(), dim=-1)  # (B, N, N)

        # 邻居候选：距离 ≤ comm_radius 且非自身
        is_candidate = (dist <= self.comm_radius) & (dist > 1e-6)  # (B, N, N)

        # ---- Step 2: 计算注意力 logits（用于邻居选择的预备分数）----
        Q_raw = self.q_proj(intents)  # (B, N, comm_dim)
        K_raw = self.k_proj(intents)  # (B, N, comm_dim)

        # 注意力分数：Q·K^T / √comm_dim
        # 注意：scale 基于 Q/K 的维度 comm_dim，而非输入的 intent_dim（B6 修复）
        comm_dim = Q_raw.shape[-1]
        scale = comm_dim ** -0.5
        attn_logits = torch.bmm(Q_raw, K_raw.transpose(1, 2)) * scale  # (B, N, N)

        # 非候选邻居 → -inf（entmax 中权重 → 0）
        attn_logits = attn_logits.masked_fill(~is_candidate, float('-inf'))

        # ---- Step 3: 迟滞（hysteresis） ----
        # 对已在活跃集中的邻居施加 stickiness bonus
        if self._active_mask is not None and self._active_mask.shape == is_candidate.shape:
            attn_logits = attn_logits + self.stickiness_bonus * self._active_mask.float()

        # ---- Step 4: 软稀疏注意力（entmax） ----
        # 对每个 anchor agent 的候选邻居做 entmax
        # attn_logits 中有 -inf（非候选），entmax 会自动将其压为 0
        attn_weights = entmax(attn_logits, alpha=self.entmax_alpha, dim=-1)  # (B, N, N)

        # 无候选邻居的行：entmax 数值 fallback 会返回均匀分布，强制归零
        has_any_cand = is_candidate.any(dim=-1, keepdim=True).float()  # (B, N, 1)
        attn_weights = attn_weights * has_any_cand

        # ---- Step 5: 迟滞阈值过滤 ----
        # τ_in: 之前不活跃的邻居，权重 < τ_in → 置 0
        # τ_out: 之前活跃的邻居，权重 < τ_out → 置 0
        if self._active_mask is not None and self._active_mask.shape == is_candidate.shape:
            # 混合阈值：活跃邻居用 τ_out，非活跃用 τ_in
            threshold = torch.where(self._active_mask, self.tau_out, self.tau_in)
            attn_weights = attn_weights.masked_fill(attn_weights < threshold, 0.0)

            # 重新归一化
            sum_w = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            attn_weights = attn_weights / sum_w

        # 更新活跃集（用于下一步迟滞）
        # 凡是通过了迟滞阈值过滤后权重 > 0 的边，均视为活跃（B5 修复）
        # 用 tau_in 而非 > 0 会让 tau_out <= w < tau_in 的边在下一步被当成"非活跃"
        # 从而失去迟滞保护，退化为每步都重新用 tau_in 判断，与设计矛盾
        self._active_mask = (attn_weights > 0).detach()

        # ---- Step 6: 算力上限封顶（comm_topk_cap）----
        # 如果非零邻居仍超过 cap，保留权重最高的 k 个
        nonzero_count = (attn_weights > 0).sum(dim=-1)  # (B, N)
        needs_cap = nonzero_count > self.comm_topk_cap  # (B, N)

        if needs_cap.any():
            # 对需要封顶的 agent，保留 top-k 权重，其余置 0
            topk_vals, topk_idx = torch.topk(
                attn_weights, k=self.comm_topk_cap, dim=-1
            )
            capped = torch.zeros_like(attn_weights)
            capped.scatter_(-1, topk_idx, topk_vals)
            # 只对需要封顶的 agent 替换
            need_mask = needs_cap.unsqueeze(-1)  # (B, N, 1)
            attn_weights = torch.where(need_mask, capped, attn_weights)
            # 重新归一化封顶的
            sum_w = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            attn_weights = torch.where(need_mask, attn_weights / sum_w, attn_weights)

        # ---- Step 7: EdgeFlipRate 监控 ----
        # 公式（设计文档§5.7）：|E^t △ E^{t+1}| / |E^t ∪ E^{t+1}|
        current_edge_set = (attn_weights > 0).detach()  # (B, N, N)
        if self._prev_edge_set is not None and self._prev_edge_set.shape == current_edge_set.shape:
            flips = (current_edge_set != self._prev_edge_set).float().sum()
            union = (current_edge_set | self._prev_edge_set).float().sum().clamp(min=1)
            edge_flip_rate = (flips / union).item()
        else:
            edge_flip_rate = 0.0
        self._prev_edge_set = current_edge_set

        # ---- Step 8: 多头注意力聚合 ----
        V_raw = self.v_proj(intents)  # (B, N, D_comm)
        Q = Q_raw.view(B, N, self.num_heads, self.head_dim)
        V = V_raw.view(B, N, self.num_heads, self.head_dim)

        # 用软稀疏权重做加权聚合（不是标准 QKV attention，而是 weight-based）
        # attn_weights: (B, N, N) — 每体对其邻居的权重
        # 扩展到多头：attn_weights_heads: (B, N, N, H)
        attn_weights_heads = attn_weights.unsqueeze(-1).expand(-1, -1, -1, self.num_heads)

        # 加权和：(B, N, H, d_head)
        attended = torch.einsum('bnmh,bmhd->bnhd', attn_weights_heads, V)
        attended = attended.reshape(B, N, self.num_heads * self.head_dim)

        comm_features = self.out_proj(attended)

        # ---- Step 9: 无活跃邻居的 agent：通信特征 → 0 ----
        # 用实际活跃权重（迟滞过滤后 > 0）而非原始候选集判断，
        # 避免候选存在但权重全被过滤时 out_proj bias 泄漏（设计文档§2.5）
        has_active = (attn_weights > 0).any(dim=-1).float().unsqueeze(-1)  # (B, N, 1)
        comm_features = comm_features * has_active

        # ---- Step 10: 跨步 EMA 平滑（可选） ----
        if self.ema_decay > 0 and self._ema_c is not None:
            self._ema_c = self.ema_decay * self._ema_c + (1 - self.ema_decay) * comm_features.detach()
            comm_features = self._ema_c
        elif self.ema_decay > 0:
            self._ema_c = comm_features.detach()

        # 诊断信息
        avg_neighbors = nonzero_count.float().mean().item()
        info = {
            "attn_weights": attn_weights.detach(),
            "edge_flip_rate": edge_flip_rate,
            "avg_neighbors": avg_neighbors,
            "entmax_alpha": self.entmax_alpha,
        }

        return comm_features, info
