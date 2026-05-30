"""
TD 损失：1-step 与 n-step。

L_TD  = (Q(s,a) - y_1)²,   y_1 = r + γ·(1-done)·V_target(s')
L_n   = (Q(s,a) - y_n)²,   y_n = Σ_{l=0}^{n-1} γ^l·r_{t+l} + γ^n·V_target(s_{t+n})
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def one_step_td_loss(
    q_values: torch.Tensor,        # (..., num_actions)
    actions: torch.Tensor,         # (...,) long, 动作索引
    rewards: torch.Tensor,         # (...,)
    dones: torch.Tensor,           # (...,) bool or float
    v_target: torch.Tensor,        # (...,) V_target(s') = mellowmax target
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    1-step TD loss。支持 (B, num_actions) 或 (B, N, num_actions)。

    Returns:
        loss: scalar
    """
    # Q(s, a) — gather chosen actions
    q_sa = q_values.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # (...,)

    # TD target: r + γ * (1 - done) * V(s')
    done_mask = (~dones.bool()).float() if dones.dtype == torch.bool else (1.0 - dones)
    td_target = rewards + gamma * done_mask * v_target  # (...,)

    loss = F.mse_loss(q_sa, td_target.detach())
    return loss


def n_step_td_loss(
    q_values: torch.Tensor,        # (B, N, num_actions) — 时刻 t 的 Q(s_t,·)
    actions: torch.Tensor,         # (B, N) — 时刻 t 的动作
    rewards_seq: torch.Tensor,     # (B, N, n) — r_t, r_{t+1}, ..., r_{t+n-1}
    dones_seq: torch.Tensor,       # (B, N, n) — done flags
    v_target_n: torch.Tensor,      # (B, N) — V_target(s_{t+n})
    gamma: float = 0.99,
    n_step: int = 5,
) -> torch.Tensor:
    """
    n-step TD loss。

    y_n = r_t + γ·r_{t+1} + ... + γ^{n-1}·r_{t+n-1} + γ^n·V_target(s_{t+n})

    Args:
        q_values: Q(s_t, a)。(B, N, num_actions)
        actions: 时刻 t 的动作。
        rewards_seq: n 步奖励序列。
        dones_seq: n 步 done 序列。
        v_target_n: V(s_{t+n})。
        gamma: 折扣因子。
        n_step: n 步数。

    Returns:
        loss: scalar
    """
    B, N, n = rewards_seq.shape
    device = rewards_seq.device

    # 计算 n-step return
    discounts = gamma ** torch.arange(n, device=device, dtype=torch.float32)  # (n,)
    discount_factor = gamma ** n

    # 累积折扣奖励
    done_mask = (~dones_seq.bool()).float() if dones_seq.dtype == torch.bool else (1.0 - dones_seq)

    # 累计：对每一步 l，如果之前某步 done，则该步及之后 = 0
    cum_done = torch.cumprod(done_mask, dim=-1)  # (B, N, n)
    # 第 0 步的 done mask 不在此序列内
    # 简化：假设 dones_seq 已正确截断
    discounted_rewards = rewards_seq * (discounts.view(1, 1, n))  # (B, N, n)
    # 将 n-step done 隔离
    discounted_rewards = discounted_rewards * cum_done

    return_sum = discounted_rewards.sum(dim=-1)  # (B, N)

    # V_target 项
    # 如果序列中任何步 done，V_target 项归零
    any_done = 1.0 - done_mask.prod(dim=-1)  # (B, N) — 有一个 done 则归零
    v_term = discount_factor * v_target_n * (1.0 - any_done)  # (B, N)

    td_target_n = return_sum + v_term  # (B, N)

    q_sa = q_values.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

    loss = F.mse_loss(q_sa, td_target_n.detach())
    return loss
