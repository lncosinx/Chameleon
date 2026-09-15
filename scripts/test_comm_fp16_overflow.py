#!/usr/bin/env python3
"""回归测试：AMP(fp16) 下通信注意力分数不得溢出成 inf/NaN。

2026-09-15 事故：`attn_logits = matmul(Q, K^T) * scale` 把缩放放在 matmul **之后**，
fp16 矩阵乘先累加 comm_dim 个未缩放乘积。容量放大臂（comm_dim 192、激活训到 ~155）
第一次越过 fp16 上限 65504 → +inf → entmax 里 inf-inf=NaN → 整条训练报废，
而 GradScaler 只会一路砍 scale（坏的是前向不是梯度），救不回来。

本测试直接喂**大幅度 intents**（模拟训练若干步后的激活），断言：
  1. 注意力 logits 有限（除了 -inf 掩码位，那是 INVARIANT 7 要求的）
  2. entmax 输出无 NaN
  3. 通信输出无 NaN
对 comm_dim = 128 / 192 / 256 各测一遍——小网只是侥幸没越界，不能只测小网。
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.communication import BiIntentAttention


def check(comm_dim: int, scale_act: float, device: str = "cuda:0") -> bool:
    torch.manual_seed(0)
    B, T, N, D = 4, 8, 48, comm_dim
    comm = BiIntentAttention(
        intent_dim=D, comm_dim=comm_dim, comm_radius=4, comm_topk_cap=6,
        entmax_alpha=1.5, tau_in=0.05, tau_out=0.02, stickiness_bonus=0.1,
        num_heads=4,
    ).to(device)
    # 模拟"训练若干步后"的激活幅度（事故现场实测 absmax≈155）
    intents = (torch.randn(B, T, N, D, device=device) * scale_act)
    positions = torch.randint(0, 20, (B, T, N, 2), device=device).float()

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        out, info = comm.forward_sequence(intents, positions)

    ok = True
    if torch.isnan(out).any():
        print(f"    ✗ comm_dim={comm_dim} act={scale_act}: 通信输出含 "
              f"{int(torch.isnan(out).sum())} 个 NaN"); ok = False
    if torch.isinf(out).any():
        print(f"    ✗ comm_dim={comm_dim} act={scale_act}: 通信输出含 inf"); ok = False
    if ok:
        print(f"    ✓ comm_dim={comm_dim:<4} 激活幅度×{scale_act:<5} → 输出有限 "
              f"(absmax={out.abs().max().item():.4g}, 平均邻居 {info['avg_neighbors']:.2f})")
    return ok


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA（本测试针对 fp16 autocast）"); return 1
    print("AMP fp16 通信溢出回归测试")
    all_ok = True
    for cd in (128, 192, 256):
        for act in (1.0, 50.0, 155.0, 400.0):   # 155 = 事故现场实测幅度
            all_ok &= check(cd, act)
    print("\n全部通过 ✓" if all_ok else "\n存在失败 ✗")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
