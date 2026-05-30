#!/usr/bin/env python3
"""
Phase 4 验收：损失函数。

T4.1: Masked Mellowmax — 非法动作无 NaN, α↓ 时 V 趋近 max
T4.2: TD + DQfD — margin loss 推动 Q(s,a_E) 高于其他, margin_excluded 不进 margin
"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.q_target import masked_mellowmax, masked_mellowmax_target
from losses.td_losses import one_step_td_loss, n_step_td_loss
from losses.dqfd_losses import margin_loss, dqfd_total_loss
from models.biintent_drqn import BiIntentGATDRQN


def test_mellowmax_stability():
    """T4.1.1: 含 -inf 的 mellowmax 无 NaN。"""
    print("\n--- T4.1.1: Mellowmax stability ---")

    q = torch.tensor([
        [1.0, 0.5, 0.3, float('-inf'), float('-inf')],
        [2.0, float('-inf'), float('-inf'), float('-inf'), float('-inf')],
    ])  # (2, 5)
    mask = torch.tensor([
        [True, True, True, False, False],
        [True, False, False, False, False],
    ])

    for alpha in [0.5, 1.0, 2.0]:
        v = masked_mellowmax(q, mask, alpha=alpha)
        assert not torch.isnan(v).any(), f"α={alpha}: NaN detected!"
        assert not torch.isinf(v).any(), f"α={alpha}: Inf detected!"
        print(f"  α={alpha}: V = {v.tolist()} ✓")


def test_mellowmax_monotonic():
    """T4.1.2: α 减小时 V 单调趋近 max（语义方向）。"""
    print("\n--- T4.1.2: Mellowmax monotonicity ---")

    q = torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.2]])  # max=3.0
    mask = torch.ones(1, 5, dtype=torch.bool)

    v_large = masked_mellowmax(q, mask, alpha=2.0)
    v_med = masked_mellowmax(q, mask, alpha=1.0)
    v_small = masked_mellowmax(q, mask, alpha=0.5)

    # α 越小 → 越接近 hard max (3.0)
    print(f"  α=2.0: V={v_large.item():.4f}")
    print(f"  α=1.0: V={v_med.item():.4f}")
    print(f"  α=0.5: V={v_small.item():.4f}")
    print(f"  hard max = 3.0")

    # 应该 2.0 < 1.0 < 0.5 越接近 3.0
    assert v_large < v_med < v_small, (
        f"Expected V(α=2.0) < V(α=1.0) < V(α=0.5), "
        f"got {v_large.item():.3f}, {v_med.item():.3f}, {v_small.item():.3f}"
    )
    print(f"  [PASS] α↓ → V↑ → approaches max ✓")


def test_td_loss():
    """T4.2.1: 1-step TD 可计算 + 可反传。"""
    print("\n--- T4.2.1: 1-step TD loss ---")

    q = torch.randn(2, 4, 5, requires_grad=True)
    actions = torch.randint(0, 5, (2, 4))
    rewards = torch.randn(2, 4)
    dones = torch.zeros(2, 4, dtype=torch.bool)
    v_target = torch.randn(2, 4)

    loss = one_step_td_loss(q, actions, rewards, dones, v_target)
    loss.backward()
    assert q.grad is not None
    print(f"  TD loss = {loss.item():.4f}, grad norm = {q.grad.norm():.4f} ✓")


def test_margin_loss():
    """T4.2.2: margin loss 推动 Q(s,a_E) 高于其他合法动作。"""
    print("\n--- T4.2.2: Margin loss ---")

    q = torch.tensor([[
        [1.0, 0.5, 0.3, 0.8, 0.2],  # expert=2 (0.3), max_other=1.0
        [2.0, 1.0, 3.0, 0.5, 0.1],  # expert=0 (2.0), max_other=3.0 → margin loss
    ]]).float()  # (1, 2, 5)
    expert_a = torch.tensor([[2, 0]]).long()  # (1, 2)

    loss = margin_loss(q, expert_a, margin=0.5)
    # Agent 0: Q(expert=0.3) + 0.5=0.8 < max_other=1.0, so loss > 0
    # Agent 1: Q(expert=2.0), max(Q+margin) = max(2.0, 1.5, 3.5, 1.0, 0.6) = 3.5
    #          margin loss = max(0, 3.5 - 2.0) = 1.5
    assert loss.item() > 0, f"Margin loss should be > 0, got {loss.item()}"
    print(f"  Margin loss = {loss.item():.4f} ✓")


def test_margin_exclusion():
    """T4.2.3: margin_excluded 样本不进 margin loss（通过梯度验证）。"""
    print("\n--- T4.2.3: Margin exclusion ---")

    q = torch.tensor([[
        [1.0, 0.5, 0.3, 0.8, 0.2],
        [1.0, 0.5, 0.3, 0.8, 0.2],
    ]]).float().requires_grad_(True)
    expert_a = torch.tensor([[2, 2]]).long()
    # agent 0 margin_excluded, agent 1 正常
    excluded = torch.tensor([[True, False]])

    loss = margin_loss(q, expert_a, margin=0.5, margin_excluded=excluded)
    loss.backward()

    # agent 1 的 grad 应非零（参与 loss），agent 0 应为零或接近零
    grad_norms = q.grad.norm(dim=-1)  # (1, 2)
    print(f"  Grad norms: agent0={grad_norms[0,0]:.6f}, agent1={grad_norms[0,1]:.6f}")
    assert grad_norms[0, 0] < 1e-6, (
        f"Excluded agent should have zero grad, got {grad_norms[0,0]}"
    )
    assert grad_norms[0, 1] > 1e-6, (
        f"Non-excluded agent should have non-zero grad"
    )
    print(f"  [PASS] Excluded agent grad=0, included agent grad>0 ✓")


def test_dqfd_total():
    """T4.2.4: DQfD 总损失可计算。"""
    print("\n--- T4.2.4: DQfD total loss ---")

    B, N, A = 2, 4, 5
    q = torch.randn(B, N, A, requires_grad=True)
    target_q = torch.randn(B, N, A)
    actions = torch.randint(0, A, (B, N))
    rewards = torch.randn(B, N)
    dones = torch.zeros(B, N, dtype=torch.bool)
    masks = torch.ones(B, N, A, dtype=torch.bool)
    expert_a = torch.randint(0, A, (B, N))

    total, info = dqfd_total_loss(
        q_values=q,
        target_q_next=target_q,
        actions=actions,
        rewards=rewards,
        dones=dones,
        action_masks_next=masks,
        expert_actions=expert_a,
        parameters=list(torch.nn.Linear(2, 2).parameters()),  # dummy
    )

    total.backward()
    print(f"  Total loss = {total.item():.4f}")
    for k, v in info.items():
        print(f"    {k}: {v:.4f}")
    print(f"  [PASS] DQfD total loss OK ✓")


def main():
    print("=" * 60)
    print("  Chameleon — Phase 4 验收")
    print("=" * 60)

    all_ok = True
    tests = [
        test_mellowmax_stability,
        test_mellowmax_monotonic,
        test_td_loss,
        test_margin_loss,
        test_margin_exclusion,
        test_dqfd_total,
    ]
    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"\n  [FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("  Phase 4 验收结果: 全部通过 ✓")
    else:
        print("  Phase 4 验收结果: 存在失败项 ✗")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
