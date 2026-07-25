"""
离线 IL 损失（v3）：对专家动作做 masked 交叉熵。取代 DQfD（TD/n-step/margin/mellowmax）。

- 非法动作在 logits 上已被 forward 屏蔽为 -inf（softmax 后概率 0）；专家动作恒合法。
- 按 valid 屏蔽 finished agent（零权）。
- margin_excluded（对称让行下专家动作不唯一）样本可选降权，缓解标签噪声。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def il_cross_entropy_loss(
    logits: torch.Tensor,           # (B,T,N,A) 或 (M,A)
    expert_actions: torch.Tensor,   # (B,T,N) 或 (M,) long
    valid: Optional[torch.Tensor] = None,        # (B,T,N)/(M,) bool/float，finished=0
    margin_excluded: Optional[torch.Tensor] = None,  # (B,T,N)/(M,) bool，对称让行样本
    excluded_weight: float = 0.3,   # margin_excluded 样本权重（<1 降权）
    label_smoothing: float = 0.0,
    wait_weight: float = 1.0,       # wait 动作(0) 样本权重：专家数据 wait 占 ~50%，纯 CE 会
                                    # 塌缩到到处等待。<1 降权(≈逆频率 0.23~0.3) 打破多数类主导。
    wait_action: int = 0,           # INVARIANT 1：wait 在第 0 位
) -> Tuple[torch.Tensor, Dict]:
    A = logits.shape[-1]
    logits_flat = logits.reshape(-1, A)
    act_flat = expert_actions.reshape(-1).long()

    # 逐样本权重
    if valid is not None:
        w = valid.reshape(-1).float()
    else:
        w = torch.ones(act_flat.shape[0], device=logits.device)
    if margin_excluded is not None:
        w = w * torch.where(margin_excluded.reshape(-1).bool(),
                            torch.full_like(w, excluded_weight), torch.ones_like(w))
    # wait 类别平衡：对专家动作=wait 的样本按 wait_weight 降权（对齐 DQfD 的 wait 降权，
    # INVARIANT 11）。wait_weight=1.0 时为无操作，保持旧行为。
    if wait_weight != 1.0:
        is_wait = (act_flat == wait_action)
        w = w * torch.where(is_wait, torch.full_like(w, wait_weight), torch.ones_like(w))

    # 只在有效行算 CE：finished/invalid agent 的动作可能全非法(-inf)→CE=nan，
    # 且 nan×0=nan 会污染求和。按 w>0 取有效行，从根上避免 nan。
    keep = w > 0
    if not bool(keep.any()):
        zero = logits_flat.sum() * 0.0   # 保持计算图、可反传的 0
        return zero, {"loss_il": 0.0, "acc_expert": 0.0}
    lf, af, wf = logits_flat[keep], act_flat[keep], w[keep]
    ce = F.cross_entropy(lf, af, reduction="none", label_smoothing=label_smoothing)
    denom = wf.sum().clamp_min(1.0)
    loss = (ce * wf).sum() / denom

    with torch.no_grad():
        pred = lf.argmax(dim=-1)
        acc = ((pred == af).float() * wf).sum() / denom
        # 预测里 wait 占比：塌缩到到处等待时会飙到接近 1，是最直接的诊断
        wait_pred_frac = (pred == wait_action).float().mean()
    return loss, {"loss_il": loss.item(), "acc_expert": acc.item(),
                  "wait_pred_frac": wait_pred_frac.item()}
