"""
DQfD 总损失。

L = λ_td · L_TD + λ_n · L_n + λ_margin · L_margin + λ_l2 · ||θ||²

INVARIANT 11: margin_excluded 样本剔除出 margin loss（TD 项保留）。
              wait 动作 margin 按比例降权。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.q_target import masked_mellowmax, masked_double_q_target, masked_clipped_double_q, masked_max
from losses.td_losses import one_step_td_loss, n_step_td_loss


def margin_loss(
    q_values: torch.Tensor,         # (B, N, num_actions)
    expert_actions: torch.Tensor,   # (B, N) long
    margin: float = 0.5,
    margin_excluded: torch.Tensor = None,  # (B, N) bool, True=exclude from margin
    margin_weights: torch.Tensor = None,   # (B, N) float, wait 降权
) -> torch.Tensor:
    """
    专家大间隔损失。

    L_margin = max_a [Q(s,a) + ℓ(a_E, a)] - Q(s, a_E)
    ℓ(a_E, a) = 0 if a == a_E else margin

    Args:
        q_values: (B, N, num_actions) Q 值。
        expert_actions: (B, N) 专家动作。
        margin: 间隔值 m。
        margin_excluded: (B, N) True = 剔除出 margin loss。
        margin_weights: (B, N) per-sample 权重。

    Returns:
        loss: scalar
    """
    B_flat = q_values.shape[0]
    A = q_values.shape[-1]
    device = q_values.device

    # 如果输入是 (B, N, A)，展平为 (B*N, A)
    if q_values.dim() == 3:
        B, N, A_orig = q_values.shape
        q_values = q_values.reshape(B * N, A)
        expert_actions = expert_actions.reshape(B * N)
        if margin_excluded is not None:
            margin_excluded = margin_excluded.reshape(B * N)
        if margin_weights is not None:
            margin_weights = margin_weights.reshape(B * N)
        B_flat = B * N

    # Q(s, a_E)
    q_expert = q_values.gather(dim=-1, index=expert_actions.unsqueeze(-1)).squeeze(-1)  # (B_flat,)

    # max_a [Q(s,a) + ℓ(a_E, a)]
    # 构造 margin 矩阵：非专家动作 + margin
    margin_mat = torch.full((B_flat, A), margin, device=device)
    margin_mat.scatter_(-1, expert_actions.unsqueeze(-1), 0.0)  # 专家动作 = 0
    q_margin = q_values + margin_mat  # (B_flat, A)
    max_other = q_margin.max(dim=-1).values  # (B_flat,)

    loss_per_sample = F.relu(max_other - q_expert)  # (B_flat,)

    # 应用权重
    if margin_weights is not None:
        loss_per_sample = loss_per_sample * margin_weights

    # 剔除歧义样本
    if margin_excluded is not None:
        mask = (~margin_excluded).float()
        loss_per_sample = loss_per_sample * mask
        # 避免除零
        valid_count = mask.sum() + 1e-8
    else:
        valid_count = B_flat

    return loss_per_sample.sum() / valid_count


def dqfd_total_loss(
    q_values: torch.Tensor,            # (B, N, num_actions) 在线 Q(s,·)
    target_q_next: torch.Tensor,       # (B, N, num_actions) target Q(s',·)
    actions: torch.Tensor,             # (B, N) long, 实际采取的动作
    rewards: torch.Tensor,             # (B, N)
    dones: torch.Tensor,               # (B, N) bool
    action_masks_next: torch.Tensor,   # (B, N, num_actions) s' 时刻合法动作掩码
    expert_actions: torch.Tensor,      # (B, N) long, 专家动作
    online_q_next: torch.Tensor = None,  # (B, N, num_actions) 在线 Q(s',·)；给定则用 Double DQN target
    q_values2: torch.Tensor = None,      # (B, N, num_actions) 第二在线头 Q(s,·)；clipped double-Q 双头 TD
    target_q2_next: torch.Tensor = None, # (B, N, num_actions) 第二 target 头 Q(s',·)；给定则 clipped(min) target
    # n-step
    rewards_seq: torch.Tensor = None,  # (B, N, n)
    dones_seq: torch.Tensor = None,    # (B, N, n)
    v_target_n: torch.Tensor = None,   # (B, N)
    # margin
    margin_excluded: torch.Tensor = None,  # (B, N) bool
    margin_weights: torch.Tensor = None,   # (B, N) float
    # validity / sample weighting (finished agents → 0，不计入 TD)
    sample_weights: torch.Tensor = None,   # (B*N,) 或与 q_values 前缀对齐的 0/1 权重
    # hyperparameters
    gamma: float = 0.99,
    n_step: int = 5,
    mellowmax_alpha: float = 1.0,
    margin: float = 0.5,
    lambda_td: float = 1.0,
    lambda_nstep: float = 1.0,
    lambda_margin: float = 1.0,
    lambda_l2: float = 1e-5,
    parameters: list = None,  # for L2 regularization
) -> tuple[torch.Tensor, dict]:
    """
    DQfD 总损失。

    Returns:
        total_loss: scalar
        info: dict with individual loss components
    """
    info = {}

    # ---- 1-step TD ----
    # bootstrap 默认 masked max（标准 Q-learning，取贪婪动作值，避免 mellowmax 把 margin
    # 压低的非专家动作平均进来导致小 Q 时负向发散）。clipped/double-Q 仅在显式传参时启用。
    if target_q2_next is not None:
        v_target = masked_clipped_double_q(target_q_next, target_q2_next,
                                           action_masks_next, alpha=mellowmax_alpha)
    elif online_q_next is not None:
        v_target = masked_double_q_target(online_q_next, target_q_next, action_masks_next)
    else:
        v_target = masked_max(target_q_next, action_masks_next)
    loss_td = one_step_td_loss(q_values, actions, rewards, dones, v_target, gamma,
                               weights=sample_weights)
    # clipped double-Q：第二头也向同一 clipped target 训 1-step TD
    if q_values2 is not None:
        loss_td = loss_td + one_step_td_loss(q_values2, actions, rewards, dones, v_target, gamma,
                                             weights=sample_weights)
    info["loss_td"] = loss_td.item()

    # ---- n-step TD ----
    if rewards_seq is not None and dones_seq is not None and v_target_n is not None:
        # 对齐 q_values / actions 与 rewards_seq 的 leading dim
        # （调用方可能已提前裁剪为 B*(T-n)，此时需要同步裁剪 q/a）
        n_seq = rewards_seq.shape[0] if rewards_seq.dim() == 2 else rewards_seq.shape[0] * rewards_seq.shape[1]
        q_n = q_values[:n_seq] if q_values.shape[0] > n_seq else q_values
        a_n = actions[:n_seq] if actions.shape[0] > n_seq else actions
        loss_n = n_step_td_loss(
            q_n, a_n, rewards_seq, dones_seq,
            v_target_n, gamma, n_step,
        )
        info["loss_nstep"] = loss_n.item()
    else:
        loss_n = torch.tensor(0.0, device=q_values.device)
        info["loss_nstep"] = 0.0

    # ---- Margin loss ----
    loss_margin = margin_loss(
        q_values, expert_actions, margin,
        margin_excluded=margin_excluded,
        margin_weights=margin_weights,
    )
    info["loss_margin"] = loss_margin.item()

    # ---- L2 regularization ----
    if parameters is not None and lambda_l2 > 0:
        l2_reg = sum(p.pow(2.0).sum() for p in parameters)
    else:
        l2_reg = torch.tensor(0.0, device=q_values.device)
    info["l2_reg"] = l2_reg.item() if torch.is_tensor(l2_reg) else l2_reg

    # ---- Total ----
    total = (
        lambda_td * loss_td
        + lambda_nstep * loss_n
        + lambda_margin * loss_margin
        + lambda_l2 * l2_reg
    )
    info["loss_total"] = total.item()

    # ---- Q 量级诊断（监控高估/发散，设计文档 §5 / line 607 要求）----
    # q_values=在线 Q(s,·)；v_target=mellowmax 自举值。Q 无界上爬即为高估发散信号。
    with torch.no_grad():
        info["q_max"]        = q_values.max().item()
        info["q_mean"]       = q_values.mean().item()
        info["vtarget_mean"] = v_target.mean().item()

    return total, info
