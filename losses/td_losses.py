"""
TD 损失：1-step 与 n-step。

L_TD  = (Q(s,a) - y_1)²,   y_1 = r + γ·(1-done)·V_target(s')
L_n   = (Q(s,a) - y_n)²,   y_n = Σ_{l=0}^{n-1} γ^l·r_{t+l} + γ^n·V_target(s_{t+n})
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                  weights: torch.Tensor = None) -> torch.Tensor:
    """加权 MSE：weights=None 时退化为普通 mean。weights 为 0/1 时屏蔽无效样本。"""
    se = (pred - target) ** 2
    if weights is None:
        return se.mean()
    weights = weights.to(se.dtype)
    denom = weights.sum().clamp(min=1.0)
    return (se * weights).sum() / denom


def one_step_td_loss(
    q_values: torch.Tensor,        # (..., num_actions)
    actions: torch.Tensor,         # (...,) long, 动作索引
    rewards: torch.Tensor,         # (...,)
    dones: torch.Tensor,           # (...,) bool or float
    v_target: torch.Tensor,        # (...,) V_target(s') = mellowmax target
    gamma: float = 0.99,
    weights: torch.Tensor = None,  # (...,) 0/1 或权重，None=全部计入
) -> torch.Tensor:
    """
    1-step TD loss。支持 (B, num_actions) 或 (B, N, num_actions)。

    weights：per-sample 权重（如 valid 掩码）。0 表示该样本不计入 loss。

    Returns:
        loss: scalar
    """
    # Q(s, a) — gather chosen actions
    q_sa = q_values.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # (...,)

    # TD target: r + γ * (1 - done) * V(s')
    done_mask = (~dones.bool()).float() if dones.dtype == torch.bool else (1.0 - dones)
    td_target = rewards + gamma * done_mask * v_target  # (...,)

    return _weighted_mse(q_sa, td_target.detach(), weights)


def n_step_td_loss(
    q_values: torch.Tensor,        # (B, num_actions) — 时刻 t 的 Q(s_t,·)
    actions: torch.Tensor,         # (B,) — 时刻 t 的动作
    rewards_seq: torch.Tensor,     # (B, n) — r_t, r_{t+1}, ..., r_{t+n-1}
    dones_seq: torch.Tensor,       # (B, n) — done flags
    v_target_n: torch.Tensor,      # (B,) — V_target(s_{t+n})
    gamma: float = 0.99,
    n_step: int = 5,
    weights: torch.Tensor = None,  # (B,) 0/1 或权重，None=全部计入
) -> torch.Tensor:
    """
    n-step TD loss。

    y_n = r_t + γ·(1-d_t)·r_{t+1} + γ²·(1-d_t)·(1-d_{t+1})·r_{t+2} + ...
          + γ^n · (1-d_{t+n-1}) · ... · (1-d_t) · V_target(s_{t+n})

    done 掩码语义：done[l]=1 表示在收集 r[l] 后 episode 结束，
    因此 r[l] 本身仍被计入，但 r[l+1:] 以及 V_target 的 bootstrap 归零。

    Args:
        q_values:   Q(s_t, ·)，shape (B, num_actions)。
        actions:    时刻 t 的动作，shape (B,)。
        rewards_seq: n 步奖励序列，shape (B, n)。
        dones_seq:  n 步 done 序列，shape (B, n)。
        v_target_n: V(s_{t+n})，shape (B,)。
        gamma:      折扣因子。
        n_step:     n 步数（与 rewards_seq.shape[1] 一致）。

    Returns:
        loss: scalar
    """
    # 兼容旧调用格式 (B, N, n) → 展平成 (B*N, n)
    if rewards_seq.dim() == 3:
        B_orig, N_orig, n = rewards_seq.shape
        rewards_seq = rewards_seq.reshape(B_orig * N_orig, n)
        dones_seq   = dones_seq.reshape(B_orig * N_orig, n)
        v_target_n  = v_target_n.reshape(B_orig * N_orig)
        if q_values.dim() == 3:
            q_values = q_values.reshape(B_orig * N_orig, -1)
        if actions.dim() == 2:
            actions = actions.reshape(B_orig * N_orig)
        if weights is not None and weights.dim() == 2:
            weights = weights.reshape(B_orig * N_orig)

    B, n = rewards_seq.shape
    device = rewards_seq.device

    discounts = gamma ** torch.arange(n, device=device, dtype=torch.float32)  # (n,)

    # 将 done 转为 float (1=done, 0=not done)
    if dones_seq.dtype == torch.bool:
        dones_f = dones_seq.float()
    else:
        dones_f = dones_seq.float()

    # not_done[l] = 1 − done[l]
    not_done = 1.0 - dones_f  # (B, n)

    # survival[l]: 从步 0 到步 l-1 都没有 done（即 r[l] 是否应该 bootstrap）
    # r[0] 始终纳入；r[l](l>=1) 纳入当且仅当 done[0..l-1] 全为 False
    # = cumprod(not_done, shift by 1)，第 0 位恒为 1
    # 实现：survival[:, 0] = 1；survival[:, l] = prod(not_done[:, 0:l])
    survival = torch.ones(B, n, device=device)
    if n > 1:
        survival[:, 1:] = torch.cumprod(not_done[:, :-1], dim=1)

    discounted_rewards = rewards_seq * discounts.unsqueeze(0) * survival  # (B, n)
    return_sum = discounted_rewards.sum(dim=-1)  # (B,)

    # V_target 的 bootstrap：当且仅当全程未 done 时纳入
    bootstrap_mask = survival[:, -1] * not_done[:, -1]          # (B,) — 最后一步也未 done
    v_term = (gamma ** n) * v_target_n * bootstrap_mask

    td_target_n = (return_sum + v_term).detach()

    q_sa = q_values.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    return _weighted_mse(q_sa, td_target_n, weights)
