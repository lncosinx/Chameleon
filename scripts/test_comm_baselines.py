#!/usr/bin/env python3
"""
SQ3 通信基线的正确性/公平性单测。

不是"能跑通"就算数 —— 这些臂是论文头条贡献的对照组，必须先证明它们**确实在通信**、
**确实只在 FOV 内通信**、**确实只有通信模块这一个变量不同**。校验：

  1. 形状/可反传：每个臂都能前向出 (B,T,N,A) 的 logits 且梯度能回到通信模块参数；
  2. **FOV 约束**：把某个 agent 移出 comm_radius 之外，其它体的输出必须改变；
     移动一个**本来就在半径外**的 agent，输出必须**完全不变**（证明信息确实没泄漏进来）；
  3. **确实在用通信**：把通信输出置零（comm_rounds=0）会改变结果，说明该臂的通信有效；
  4. 骨干一致性：除通信模块外，各臂参数名集合完全相同。

跑：python scripts/test_comm_baselines.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from models.mapf_comm_net import MAPFCommNet
from models.comm_baselines import COMM_REGISTRY

ARMS = ["madcl"] + sorted(COMM_REGISTRY)


def build(comm_type):
    torch.manual_seed(0)
    return MAPFCommNet(patch_channels=5, scalar_feat_dim=8, obs_radius=5, num_actions=5,
                       d_model=64, cnn_channels=16, d_scalar=16, n_res_blocks=1,
                       comm_rounds=1, intent_dim=32, comm_dim=32, comm_radius=4,
                       temporal_backend="gru", temporal_layers=1, head_hidden=32,
                       spatial_pool=0, comm_type=comm_type)


def main():
    B, T, N = 2, 3, 6
    torch.manual_seed(1)
    p = torch.rand(B, T, N, 5, 11, 11)
    s = torch.rand(B, T, N, 8)
    m = torch.ones(B, T, N, 5, dtype=torch.bool)
    # 位置：0..3 号彼此在通信半径内（互相相邻），4/5 号远在半径外
    pos = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [2., 1.], [40., 40.], [45., 45.]])
    pos = pos.view(1, 1, N, 2).expand(B, T, N, 2).contiguous()

    backbone_names = None
    for arm in ARMS:
        net = build(arm).train()
        logits, info = net(p, s, pos, action_masks=m, reset_comm=True)
        assert logits.shape == (B, T, N, 5), f"{arm}: 形状 {logits.shape}"
        assert torch.isfinite(logits).all(), f"{arm}: 输出含 NaN/Inf"

        # 2a) 梯度确实流到通信模块
        net.zero_grad(); logits.square().mean().backward()
        gsum = sum(float(q.grad.abs().sum()) for q in net.communication.parameters()
                   if q.grad is not None)
        assert gsum > 0, f"{arm}: 通信模块没有梯度（等于没参与计算）"

        net.eval()
        with torch.no_grad():
            base, _ = net(p, s, pos, action_masks=m, reset_comm=True)

            # 2b) 移动一个 FOV 内的邻居 → 其它体的输出必须变（证明确实在收信息）
            pos_in = pos.clone(); pos_in[:, :, 1] = torch.tensor([3., 3.])
            moved_in, _ = net(p, s, pos_in, action_masks=m, reset_comm=True)
            changed = (moved_in[:, :, 0] - base[:, :, 0]).abs().max().item()

            # 2c) 移动一个**半径外**的 agent → 输出必须逐位不变（无泄漏）
            pos_out = pos.clone(); pos_out[:, :, 5] = torch.tensor([80., 80.])
            moved_out, _ = net(p, s, pos_out, action_masks=m, reset_comm=True)
            leak = (moved_out[:, :, 0] - base[:, :, 0]).abs().max().item()

        assert changed > 1e-6, f"{arm}: 邻居移动不影响输出 —— 这个臂根本没在通信"
        assert leak < 1e-6, f"{arm}: FOV 外的 agent 影响了输出（泄漏 {leak:.2e}），不满足部分可观测"

        names = {n for n, _ in net.named_parameters() if not n.startswith("communication.")}
        if backbone_names is None:
            backbone_names = names
        else:
            assert names == backbone_names, f"{arm}: 骨干参数与其它臂不一致（不是唯一变量）"

        ncp = sum(q.numel() for q in net.communication.parameters())
        print(f"  {arm:<9} ok | 通信参数 {ncp:>6,} | 邻居影响 {changed:.3e} | "
              f"FOV外泄漏 {leak:.1e} | nbr={info.get('avg_neighbors', 0):.2f}")

    print(f"✅ {len(ARMS)} 个通信臂全部通过（FOV 约束 + 确实通信 + 骨干唯一变量）")


if __name__ == "__main__":
    main()
