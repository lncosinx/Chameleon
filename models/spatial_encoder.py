"""
空间编码器（v3）：单步 11×11 FOV patch + 8 维标量 → agent 嵌入 d_model。

轻量残差 CNN：对 11×11 这种小 FOV，3×3 conv 几层感受野已盖满，无需深堆；加 1 个残差块
改善梯度流（近零成本）。空间编码不是瓶颈（瓶颈是时间记忆=Mamba + 协调=通信），故刻意保持轻。
channels_last 仅 CUDA 上启用（小卷积 tensor-core 友好）。仅含局部信息（INVARIANT 2）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)


class ResidualCNNEncoder(nn.Module):
    def __init__(
        self,
        patch_channels: int = 4,
        scalar_feat_dim: int = 8,
        d_model: int = 192,
        cnn_channels: int = 64,
        d_scalar: int = 32,
        n_res_blocks: int = 1,
        obs_radius: int = 5,
        spatial_pool: int = 0,   # 0/None = 全展平（保留完整空间）；n>0 = AdaptiveAvgPool2d(n)
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(patch_channels, cnn_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[_ResBlock(cnn_channels) for _ in range(n_res_blocks)])

        # ★ 2026-07-22：原为 AdaptiveAvgPool2d(1)（把 11×11 全局平均成 1×1）。
        # conv 平移等变 + 全局平均 ⇒ 输出对平移近似**不变**：模型分不清"障碍在左"还是"在右"，
        # FOV 空间布局被彻底丢弃，唯一方向信息只剩标量里的目标方向向量。这也把 cost2go 的
        # **空间梯度**（哪个邻格更近目标）平均掉了，正是它的价值所在。
        # 改为默认全展平，保留完整 K×K 空间结构。
        K = 2 * obs_radius + 1
        if spatial_pool and spatial_pool > 0:
            self.pool = nn.AdaptiveAvgPool2d(spatial_pool)
            spatial_dim = cnn_channels * spatial_pool * spatial_pool
        else:
            self.pool = nn.Identity()
            spatial_dim = cnn_channels * K * K
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_feat_dim, d_scalar), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Linear(spatial_dim + d_scalar, d_model)

    def forward(self, patch: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        """patch (M,C,K,K), scalar (M,scalar_feat_dim) → (M, d_model)"""
        if patch.is_cuda:
            patch = patch.contiguous(memory_format=torch.channels_last)
        h = self.stem(patch)
        h = self.res(h)
        h = self.pool(h).flatten(1)                # (M, cnn_channels)
        s = self.scalar_mlp(scalar)                # (M, d_scalar)
        return self.fuse(torch.cat([h, s], dim=-1))  # (M, d_model)
