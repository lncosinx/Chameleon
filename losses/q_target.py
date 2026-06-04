"""
Masked Mellowmax Target（数值安全版本）。

V_target = α · ( logsumexp( Q(s', a') / α ) - log|A_legal| )

INVARIANT 7: 非法动作以 -inf 隔离在求导路径外。
INVARIANT 12: 默认 α=1.0，调参在 {0.5, 1, 2}，**勿用 0.1**。
"""
from __future__ import annotations

import torch


def masked_mellowmax(
    q_values: torch.Tensor,        # (..., num_actions)
    action_masks: torch.Tensor,    # (..., num_actions) bool, True=合法
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    数值安全的 masked mellowmax。

    Args:
        q_values: Q 值。
        action_masks: 合法动作掩码（True=合法）。
        alpha: 温度参数，越小越接近 hard max。

    Returns:
        v_target: (...,) mellowmax 值。
    """
    # AMP 安全：mellowmax 含 -inf 掩码 + logsumexp，在 fp16 下易出 NaN/精度问题，
    # 强制 fp32 计算（即使在 autocast 区域内也保持数值稳定）。
    q_values = q_values.float()

    # 非法动作 → -inf，不在 logsumexp 中贡献
    safe_q = q_values / alpha
    safe_q = safe_q.masked_fill(~action_masks, float('-inf'))

    # logsumexp over last dim
    logsumexp = torch.logsumexp(safe_q, dim=-1)  # (...,)

    # 合法动作数量
    valid_counts = action_masks.float().sum(dim=-1).clamp(min=1)  # (...,)

    v_target = alpha * (logsumexp - torch.log(valid_counts))

    return v_target


def masked_mellowmax_target(
    online_q: torch.Tensor,        # (B, N, num_actions) 在线网络 Q(s,·)
    target_q: torch.Tensor,        # (B, N, num_actions) target 网络 Q(s',·)
    action_masks: torch.Tensor,    # (B, N, num_actions) bool, True=合法
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    构建 mellowmax TD target 所需的 V(s')。

    通常配合 TD loss 使用：
        y = r + γ * (1 - done) * V_target(s')
        L_TD = (Q(s,a) - y)²

    Args:
        online_q: 在线网络 Q(s,·)，仅用于 shape 对齐（实际 target 只用 target_q）。
        target_q: target 网络 Q(s',·)。
        action_masks: s' 时刻的合法动作掩码。
        alpha: mellowmax 温度。

    Returns:
        v_target: (B, N) mellowmax V(s')。
    """
    return masked_mellowmax(target_q, action_masks, alpha)
