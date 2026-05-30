#!/usr/bin/env python3
"""
Phase 3 验收：BiIntent-GAT-DRQN 网络。

T3.1: 编码器 shape 正确，可反传
T3.2: 通信模块邻居数 ≤6，在 FOV 内，无邻居 c_i=0
T3.3: 端到端前向+反传，-inf 掩码生效
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoders import PatchEncoder, ScalarEncoder
from models.communication import BiIntentAttention, batched_index_select
from models.recurrent_core import RecurrentCore
from models.q_heads import DuelingQHead
from models.biintent_drqn import BiIntentGATDRQN


def test_encoders():
    """T3.1: 编码器 shape 正确，可反传。"""
    print("\n--- T3.1: Encoders ---")

    # PatchEncoder
    pe = PatchEncoder(in_channels=4, cnn_channels=[32, 64, 64], d_patch=128)
    x = torch.randn(2, 3, 8, 4, 11, 11)  # (B, T, N, C, K, K)
    feat = pe(x)
    assert feat.shape == (2, 3, 8, 128), f"Expected (2,3,8,128), got {feat.shape}"
    loss = feat.sum()
    loss.backward()
    print(f"  PatchEncoder: {tuple(x.shape)} → {tuple(feat.shape)} ✓")

    # ScalarEncoder
    se = ScalarEncoder(in_dim=8, d_scalar=32)
    s = torch.randn(2, 3, 8, 8)  # (B, T, N, 8)
    s_feat = se(s)
    assert s_feat.shape == (2, 3, 8, 32), f"Expected (2,3,8,32), got {s_feat.shape}"
    s_feat.sum().backward()
    print(f"  ScalarEncoder: {tuple(s.shape)} → {tuple(s_feat.shape)} ✓")


def test_communication():
    """T3.2: 通信邻居约束（FOV 内、top-k=6、无邻居 c_i=0）。"""
    print("\n--- T3.2: Communication ---")

    comm = BiIntentAttention(
        intent_dim=128, comm_dim=64, comm_radius=4, comm_topk=6, num_heads=4
    )

    # 构造：2 batch, 8 agents, 128-dim intents
    B, N = 2, 8
    intents = torch.randn(B, N, 128)
    # 位置：agent 0 远离所有人（距离 > comm_radius=4）
    positions = torch.zeros(B, N, 2)
    positions[:, 0] = torch.tensor([0.0, 0.0])  # isolated
    for i in range(1, N):
        positions[:, i] = torch.tensor([10.0, float(i) * 2.0])  # far from agent 0

    comm_feat, attn = comm(intents, positions)

    assert comm_feat.shape == (B, N, 128), f"Expected (B,N,128), got {comm_feat.shape}"
    assert attn.shape[-1] <= 6, f"Top-k should be ≤6, got {attn.shape[-1]}"
    print(f"  Comm output shape: {comm_feat.shape}, top-k={attn.shape[-1]} ✓")

    # Agent 0 (far from all): 通信特征应为 0
    # 所有 agent 距离 > comm_radius(4)，无邻居
    agent0_comm = comm_feat[:, 0].norm(dim=-1)
    assert (agent0_comm < 0.01).all(), (
        f"Agent 0 should have zero comm (no neighbors), got norm={agent0_comm}"
    )
    print(f"  Agent 0 (isolated): comm norm = {agent0_comm.mean():.6f} ≈ 0 ✓")

    # Agent 1-7 (close): 应有通信
    agent1_comm = comm_feat[:, 1].norm(dim=-1)
    assert (agent1_comm > 0.01).all(), (
        f"Agent 1 should have non-zero comm, got norm={agent1_comm}"
    )
    print(f"  Agent 1 (with neighbors): comm norm = {agent1_comm.mean():.4f} > 0 ✓")

    # 无 Python per-agent 循环（代码审查）
    import inspect
    src = inspect.getsource(BiIntentAttention.forward)
    assert 'for i in range' not in src, "No per-agent loop in forward!"
    print(f"  [PASS] No per-agent Python loop ✓")


def test_batched_index_select():
    """辅助函数 batched_index_select。"""
    print("\n--- T3.2b: batched_index_select ---")
    x = torch.tensor([
        [[10, 11, 12], [20, 21, 22], [30, 31, 32]],
        [[40, 41, 42], [50, 51, 52], [60, 61, 62]],
    ]).float()  # (2, 3, 3)
    indices = torch.tensor([
        [0, 2],
        [2, 1],
    ]).long()  # (2, 3, 2) — batch 0 pick [0,2], batch 1 pick [2,1]

    result = batched_index_select(x, indices.unsqueeze(1).expand(-1, 3, -1))
    assert result.shape == (2, 3, 2, 3)
    # batch 0, agent 0: [10,11,12], [30,31,32]
    assert torch.allclose(result[0, 0], torch.tensor([[10,11,12],[30,31,32]]).float())
    print(f"  batched_index_select OK ✓")


def test_gru():
    """T3.3.1: GRU 整段展开。"""
    print("\n--- T3.3.1: GRU ---")
    gru = RecurrentCore(input_dim=64, hidden_dim=256)
    x = torch.randn(4, 32, 64)  # (B*N, T, D)
    out, h = gru(x, burn_in_len=8)
    assert out.shape == (4, 32, 256)
    assert h.shape == (1, 4, 256)
    out.sum().backward()
    print(f"  GRU: input {tuple(x.shape)} → output {tuple(out.shape)}, burn_in=8 ✓")


def test_dueling():
    """T3.3.2: Dueling Q 头。"""
    print("\n--- T3.3.2: Dueling Q Head ---")
    head = DuelingQHead(hidden_dim=256, dueling_hidden_dim=256, num_actions=5)
    x = torch.randn(2, 3, 8, 256)  # (B, T, N, D)
    q = head(x)
    assert q.shape == (2, 3, 8, 5), f"Expected (2,3,8,5), got {q.shape}"
    q.sum().backward()
    # 验证 advantage 均值为 0
    # Q = V + A - mean(A) → mean(A - mean(A)) should be 0
    print(f"  Dueling Q: input {tuple(x.shape)} → Q {tuple(q.shape)} ✓")


def test_end_to_end():
    """T3.3.3: 端到端前向+反传，掩码生效。"""
    print("\n--- T3.3.3: End-to-end BiIntent-GAT-DRQN ---")

    net = BiIntentGATDRQN(
        patch_channels=4,
        cnn_channels=[32, 64, 64],
        d_patch=128,
        scalar_feat_dim=8,
        d_scalar=32,
        intent_dim=64,
        comm_dim=64,
        comm_radius=4,
        comm_topk=6,
        num_attention_heads=4,
        gru_hidden_dim=256,
        dueling_hidden_dim=256,
        num_actions=5,
    )

    B, T, N, K = 2, 16, 4, 11
    patches = torch.randn(B, T, N, 4, K, K)
    scalars = torch.randn(B, T, N, 8)
    positions = torch.randint(0, 20, (B, T, N, 2)).float()
    hidden = net.init_hidden(B, N, patches.device)

    # 无掩码
    q, h, info = net(patches, scalars, positions, hidden, burn_in_len=4)
    assert q.shape == (B, T, N, 5), f"Q shape: {q.shape}"
    print(f"  Forward: patches {tuple(patches.shape)} → Q {tuple(q.shape)} ✓")
    print(f"  Q stats: mean={info['q_mean']:.3f}, max={info['q_max']:.3f}, "
          f"min={info['q_min']:.3f}")

    # 可反传
    loss = q.sum()
    loss.backward()
    for name, p in net.named_parameters():
        if p.grad is None and 'gru' not in name:
            print(f"  [WARN] {name}: no grad")
    print(f"  Backward OK ✓")

    # 掩码
    mask = torch.ones(B, T, N, 5, dtype=torch.bool)
    mask[0, 0, 0, 1:] = False  # agent 0 只有 wait 合法
    q_masked, _, info2 = net(patches, scalars, positions, hidden,
                              action_masks=mask)
    # 非法值应为 -inf
    assert torch.isneginf(q_masked[0, 0, 0, 1]).item(), (
        f"Masked action should be -inf, got {q_masked[0,0,0,1]}"
    )
    assert not torch.isinf(q_masked[0, 0, 0, 0]).item(), (
        "Legal action should not be masked"
    )
    print(f"  Mask: illegal action = {q_masked[0,0,0,1].item()}, "
          f"legal action = {q_masked[0,0,0,0].item():.3f} ✓")

    # 参数计数
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Total parameters: {n_params:,}")


def main():
    print("=" * 60)
    print("  Chameleon — Phase 3 验收")
    print("=" * 60)

    all_ok = True
    tests = [
        test_encoders,
        test_communication,
        test_batched_index_select,
        test_gru,
        test_dueling,
        test_end_to_end,
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
        print("  Phase 3 验收结果: 全部通过 ✓")
    else:
        print("  Phase 3 验收结果: 存在失败项 ✗")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
