#!/usr/bin/env python3
"""官方 CUDA 选择性扫描 vs 纯 PyTorch 参考实现的数值对拍（前向 + 反向）。

背景（2026-09-20）：本仓库的 MambaBlock 原为纯 PyTorch 选择性扫描（旧服务器装不上官方扩展）。
现装上官方 mamba-ssm 预编译 wheel 后接入其 kernel 作为**训练快路径**。两条路径必须数值等价，
否则此前小规模骨干 A/B（Mamba 12 档全胜 GRU）的结论无法平移到用 kernel 训练的模型上。

对拍内容：同一份权重与输入，比较
  1. 前向输出；2. 对输入与各参数的梯度；3. 推断态 last_state。
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.temporal_core import MambaBlock, _SELECTIVE_SCAN


def run(d_model=320, d_state=16, expand=2, B=8, L=32, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    blk = MambaBlock(d_model, d_state=d_state, expand=expand).cuda()
    x = torch.randn(B, L, d_model, device="cuda", dtype=dtype, requires_grad=True)

    outs = {}
    for use_kernel in (True, False):
        blk.use_kernel = use_kernel
        blk.zero_grad(set_to_none=True)
        xi = x.detach().clone().requires_grad_(True)
        y, cs, ss = blk(xi)
        y.square().mean().backward()
        outs[use_kernel] = (y.detach(), ss.detach(),
                            xi.grad.detach(),
                            {n: p.grad.detach().clone() for n, p in blk.named_parameters()})
    yk, ssk, gxk, gpk = outs[True]
    yr, ssr, gxr, gpr = outs[False]
    rel = lambda a, b: ((a - b).abs().max() / b.abs().max().clamp_min(1e-8)).item()
    worst_p = max(((n, rel(gpk[n], gpr[n])) for n in gpk if gpr[n] is not None),
                  key=lambda kv: kv[1])
    print(f"  d_model={d_model:4d} d_state={d_state:3d} L={L:3d} {str(dtype).split('.')[-1]:8s} | "
          f"前向 {rel(yk, yr):.2e} | last_state {rel(ssk, ssr):.2e} | "
          f"d输入 {rel(gxk, gxr):.2e} | 最差参数梯度 {worst_p[0]} {worst_p[1]:.2e}")
    return max(rel(yk, yr), rel(ssk, ssr), rel(gxk, gxr), worst_p[1])


def main():
    if _SELECTIVE_SCAN is None:
        print("官方 kernel 不可用，跳过（纯 PyTorch 路径仍可用）"); return 0
    print("官方 CUDA 核 vs 纯 PyTorch 参考实现（相对误差 = max|Δ| / max|ref|）")
    worst = 0.0
    for kw in (dict(d_model=320, d_state=16),          # 7.45M 档实际配置
               dict(d_model=192, d_state=16),          # 2.16M 档实际配置
               dict(d_model=320, d_state=8),
               dict(d_model=192, d_state=16, L=8),
               dict(d_model=192, d_state=16, B=32, seed=3)):
        worst = max(worst, run(**kw))
    ok = worst < 2e-4
    print(f"\n最大相对误差 {worst:.2e} → {'通过 ✓（两条路径数值等价）' if ok else '不通过 ✗'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
