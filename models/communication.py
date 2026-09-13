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
    # AMP 安全：bisection 用 -1e10 代替 -inf 且含 pow/clamp，在 fp16 下会溢出为 inf/NaN，
    # 强制 fp32 计算（autocast 区域内也保持稳定）。
    logits = logits.float()

    if alpha == 1.0:
        # ★ 2026-09-13 bug 修复：softmax 分支原来直接 F.softmax，**没处理"整行全 -inf"**
        # （FOV 内一个邻居都没有的体）。那种行 softmax 出来是 NaN，一路污染到 loss=nan，
        # 把 SQ3 的 dense-softmax 消融臂整条训废（而它正是"软稀疏 vs 稠密"的关键对照）。
        # α≠1 的 bisection 路径本来就显式做了均匀兜底，这里补齐同样的语义：
        # 无候选行给均匀分布，下游 has_any_cand 掩码会把它清零。
        all_neg = torch.all(torch.isneginf(logits), dim=dim, keepdim=True)
        safe = torch.where(all_neg, torch.zeros_like(logits), logits)
        return F.softmax(safe, dim=dim)

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

    # 与 α=1 / α∈(1,2) 两条路径同样的 -inf 处理（见 entmax 里的 bug 注释）：
    # -inf 换成大负数参与排序/cumsum（否则 cumsum 出 -inf/NaN），整行全 -inf 的
    # （FOV 内无邻居）给均匀分布，下游掩码会清零。
    all_neg = torch.all(torch.isneginf(logits_flat), dim=1, keepdim=True)
    logits_flat = torch.where(torch.isneginf(logits_flat),
                              torch.full_like(logits_flat, -1e10), logits_flat)
    logits_flat = torch.where(all_neg, torch.zeros_like(logits_flat), logits_flat)

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
        # SQ3 消融：用 **hard top-k 做邻居选择**（而非只作算力上限）。INVARIANT 10 断言这会
        # 引起通信图闪烁（边界排名交换→边反复通断），本开关就是为了把该断言做成可证伪的实验。
        hard_topk_select: bool = False,
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
        self.hard_topk_select = hard_topk_select
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
        """单步入口（保持向后兼容；内部走 forward_sequence 的 T=1 路径）。"""
        comm, info = self.forward_sequence(
            intents.unsqueeze(1), positions.unsqueeze(1)
        )
        return comm.squeeze(1), info

    def forward_sequence(
        self,
        intents: torch.Tensor,    # (B, T, N, D_intent)
        positions: torch.Tensor,  # (B, T, N, 2)
    ) -> Tuple[torch.Tensor, dict]:
        """
        向量化整段序列的通信（替代逐 t Python 循环）。

        重活全部跨 (B,T) 批量化：候选/距离、Q/K logits、entmax、topk 封顶、多头聚合。
        唯一保留的串行依赖是迟滞 active-mask 递推（active[t] 依赖 active[t-1]），
        但其每步只是廉价 elementwise（masked_fill+renorm，无 entmax/matmul），launch 开销极小。

        语义保持（与逐 t 版一致）：FOV 内候选、entmax 软稀疏、τ_in/τ_out 迟滞阈值、
        active 在 topk 前更新、EdgeFlip 在 topk 后统计、无活跃邻居→通信特征 0。
        **唯一改动**：原"预-entmax stickiness logit 加分"（可选项）会使 entmax 无法批量化，
        改为在迟滞 scan 内对 active 边做"后-entmax 权重加分"（同样制造切换惯性，且廉价）。
        stickiness_bonus=0 时与原实现数值等价。

        跨调用迟滞：self._active_mask / self._prev_edge_set 在段首读、段尾写，
        故 T=1 的在线推断逐步调用仍维持跨步迟滞（与原行为一致）。
        """
        B, T, N, D = intents.shape

        # ===== 并行段（跨 B,T 一次算完）=====
        pos_diff = positions.unsqueeze(3) - positions.unsqueeze(2)   # (B,T,N,N,2)
        dist = torch.norm(pos_diff.float(), dim=-1)                  # (B,T,N,N)
        is_candidate = (dist <= self.comm_radius) & (dist > 1e-6)    # (B,T,N,N)

        Q_raw = self.q_proj(intents)   # (B,T,N,comm_dim)
        K_raw = self.k_proj(intents)
        comm_dim = Q_raw.shape[-1]
        scale = comm_dim ** -0.5
        attn_logits = torch.matmul(Q_raw, K_raw.transpose(-1, -2)) * scale  # (B,T,N,N)
        attn_logits = attn_logits.masked_fill(~is_candidate, float('-inf'))

        # entmax 一次性批量（最大热点：逐 t 67ms → 批量 3ms）
        attn_all = entmax(attn_logits, alpha=self.entmax_alpha, dim=-1)     # (B,T,N,N)
        has_any_cand = is_candidate.any(dim=-1, keepdim=True).float()       # (B,T,N,1)
        attn_all = attn_all * has_any_cand

        # ===== 串行段（仅迟滞 active 递推；廉价 elementwise）=====
        if self.hard_topk_select:
            # 消融臂：直接按注意力权重取 top-k 作为邻居集合，不做迟滞、不做软稀疏。
            # 与主臂的唯一区别就是"选择方式"，其余（候选集、聚合、统计）完全一致。
            k = min(self.comm_topk_cap, N)
            topv, topi = torch.topk(attn_all, k=k, dim=-1)
            W = torch.zeros_like(attn_all).scatter_(-1, topi, topv)
            W = W * (attn_all > 0).float()          # 候选集外的位置不能被选进来
            W = W / W.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            self._active_mask = (W[:, -1] > 0).detach()
            return self._aggregate(W, intents, B, T, N)

        prev_active = (self._active_mask
                       if (self._active_mask is not None
                           and self._active_mask.shape == (B, N, N))
                       else None)
        w_list = []
        for t in range(T):
            w = attn_all[:, t]                                    # (B,N,N)
            if prev_active is not None:
                # 后-entmax stickiness（替代原预-entmax 加分；sb=0 时无影响）
                if self.stickiness_bonus > 0:
                    w = w + self.stickiness_bonus * prev_active.float()
                threshold = torch.where(prev_active, self.tau_out, self.tau_in)
                w = w.masked_fill(w < threshold, 0.0)
                w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            prev_active = (w > 0).detach()                        # active 在 topk 前更新（B5）
            w_list.append(w)
        self._active_mask = prev_active.detach() if prev_active is not None else (w_list[-1] > 0).detach()
        W_thr = torch.stack(w_list, dim=1)                        # (B,T,N,N)

        # ===== 并行段：topk 封顶 =====
        nonzero_count = (W_thr > 0).sum(dim=-1)                   # (B,T,N)
        needs_cap = nonzero_count > self.comm_topk_cap
        if bool(needs_cap.any()):
            topk_vals, topk_idx = torch.topk(W_thr, k=self.comm_topk_cap, dim=-1)
            capped = torch.zeros_like(W_thr).scatter_(-1, topk_idx, topk_vals)
            capped = capped / capped.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            W = torch.where(needs_cap.unsqueeze(-1), capped, W_thr)
        else:
            W = W_thr

        return self._aggregate(W, intents, B, T, N)

    def _aggregate(self, W, intents, B, T, N):
        """把权重矩阵 W (B,T,N,N) 聚合成通信特征 + 统计量（主臂/消融臂共用，保证口径一致）。"""
        nonzero_count = (W > 0).sum(dim=-1)
        # ===== 并行段：多头聚合 =====
        V_raw = self.v_proj(intents)                              # (B,T,N,comm_dim)
        V = V_raw.view(B, T, N, self.num_heads, self.head_dim)
        attended = torch.einsum('btnm,btmhd->btnhd', W, V)        # 权重跨头共享
        attended = attended.reshape(B, T, N, self.num_heads * self.head_dim)
        comm_features = self.out_proj(attended)                   # (B,T,N,intent_dim)

        has_active = (W > 0).any(dim=-1, keepdim=True).float()    # (B,T,N,1)
        comm_features = comm_features * has_active

        # 实际生效的通信度数（topk 封顶后，W>0），才是真实带宽/消息量口径；
        # 上面的 nonzero_count 是 topk 前的 FOV 候选数，会高估通信量。
        active_neighbors = (W > 0).sum(dim=-1)                    # (B,T,N)

        # ===== EdgeFlipRate（topk 后；跨调用边界用 _prev_edge_set）=====
        cur_edges = (W > 0).detach()                              # (B,T,N,N)
        if (self._prev_edge_set is not None
                and self._prev_edge_set.shape == (B, N, N)):
            seq_edges = torch.cat([self._prev_edge_set.unsqueeze(1), cur_edges], dim=1)
        else:
            seq_edges = cur_edges
        if seq_edges.shape[1] > 1:
            a, b = seq_edges[:, 1:], seq_edges[:, :-1]
            flips = (a != b).float().sum(dim=(-1, -2))
            union = (a | b).float().sum(dim=(-1, -2)).clamp(min=1)
            edge_flip_rate = (flips / union).mean().item()
        else:
            edge_flip_rate = 0.0
        self._prev_edge_set = cur_edges[:, -1]

        # 注：avg_neighbors 现在与 avg_active_neighbors 同义，都是**topk 封顶后**的真实度数
        # （即真实带宽/消息量口径）。封顶前的 FOV 候选数会高估通信量，不再作为主指标上报。
        info = {
            "edge_flip_rate": edge_flip_rate,
            "avg_neighbors": nonzero_count.float().mean().item(),
            "avg_active_neighbors": active_neighbors.float().mean().item(),  # topk 封顶后的真实度数
            "entmax_alpha": self.entmax_alpha,
        }
        return comm_features, info
