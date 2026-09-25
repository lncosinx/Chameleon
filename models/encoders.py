"""
编码器模块。

- PatchEncoder: CNN 对 (4, K, K) patch 提取特征 → d_patch
- ScalarEncoder: MLP 编码 8 维标量特征 (方向+动作+到达) → d_scalar
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchEncoder(nn.Module):
    """
    CNN patch 编码器。

    输入: (B, 4, K, K) 或 (B, T, 4, K, K)
    输出: (B, d_patch) 或 (B, T, d_patch)
    """

    def __init__(
        self,
        in_channels: int = 4,
        cnn_channels: list = None,
        d_patch: int = 128,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 64]

        layers = []
        prev_ch = in_channels
        for ch in cnn_channels:
            layers.extend([
                nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch
        self.conv = nn.Sequential(*layers)

        # 计算 flatten 尺寸（需要 K 已知）
        # 这里用 adaptive pooling 避免硬编码 K
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(cnn_channels[-1] * 4 * 4, d_patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., 4, K, K) — 支持任意 batch 维度前缀。

        Returns:
            feat: (..., d_patch)
        """
        orig_shape = x.shape
        # 展平所有 batch 维度
        if x.dim() > 4:
            B_flat = 1
            for d in orig_shape[:-3]:
                B_flat *= d
            x = x.reshape(B_flat, *orig_shape[-3:])
        else:
            B_flat = orig_shape[0]

        # channels_last (NHWC)：小卷积在 GPU tensor core 上更高效（实测 patch CNN
        # fwd 44→21ms、fwd+bwd 114→59ms，~1.9×；patch_encoder 占整网 forward ~83%）。
        # 纯内存布局变更、数学等价（cudnn 改用 NHWC 卷积算法，低位 fp 误差 ~1e-5）。
        # 仅 CUDA 上启用；下游 reshape 按逻辑 C-order 展平，结果与 NCHW 一致。
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        feat = self.conv(x)           # (B_flat, C, K, K)
        feat = self.pool(feat)        # (B_flat, C, 4, 4)
        feat = feat.reshape(B_flat, -1)
        feat = self.fc(feat)          # (B_flat, d_patch)

        # 恢复 batch 维度
        if len(orig_shape) > 4:
            feat = feat.reshape(*orig_shape[:-3], -1)

        return feat


class ScalarEncoder(nn.Module):
    """
    标量特征编码器。

    输入: (..., 8) — [目标方向(2) + 上一步动作 one-hot(5) + 到达(1)]
    输出: (..., d_scalar)
    """

    def __init__(self, in_dim: int = 8, d_scalar: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_scalar),
            nn.ReLU(inplace=True),
            nn.Linear(d_scalar, d_scalar),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
