"""
观测分词器（v3）：把 `(4,K,K)` patch + `(8,)` 标量观测转成 token 序列，喂给 Transformer。

token 布局（每体每步独立）：
    [CLS] + [scalar token] + [K*K 个 cell token]
- cell token：11×11 FOV 每格的 4 通道值线性嵌入 + 可学习位置嵌入（保留空间结构）。
- scalar token：8 维标量（目标方向/上一步动作/到达）线性嵌入。
- CLS token：可学习；Transformer 输出该位作为 agent 嵌入（聚合整个 FOV）。

设计依据：MAPF-GPT/LC-MAPF 式 tokenized 观测——Transformer 吃 token 是 GPU 友好的大矩阵，
取代旧 CNN 小卷积（吃不满 tensor core、是旧训练大头）。仅含局部信息（INVARIANT 2）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ObsTokenizer(nn.Module):
    def __init__(
        self,
        patch_channels: int = 4,
        scalar_feat_dim: int = 8,
        obs_radius: int = 5,
        d_model: int = 192,
    ):
        super().__init__()
        self.K = 2 * obs_radius + 1
        self.n_cells = self.K * self.K
        self.d_model = d_model

        self.cell_embed = nn.Linear(patch_channels, d_model)       # 每格 4 通道 → d_model
        self.cell_pos = nn.Parameter(torch.zeros(self.n_cells, d_model))  # 格位置嵌入
        self.scalar_embed = nn.Linear(scalar_feat_dim, d_model)    # 标量 → d_model
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))        # CLS

        nn.init.normal_(self.cell_pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    @property
    def n_tokens(self) -> int:
        return 1 + 1 + self.n_cells   # CLS + scalar + cells

    def forward(self, patches: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (M, C, K, K)
            scalars: (M, scalar_feat_dim)
        Returns:
            tokens: (M, n_tokens, d_model)
        """
        M = patches.shape[0]
        # cell tokens: (M,C,K,K) → (M, K*K, C) → 嵌入
        cells = patches.flatten(2).transpose(1, 2)          # (M, K*K, C)
        cell_tok = self.cell_embed(cells) + self.cell_pos   # (M, K*K, d_model)

        scalar_tok = self.scalar_embed(scalars).unsqueeze(1)  # (M, 1, d_model)
        cls_tok = self.cls.expand(M, -1, -1)                  # (M, 1, d_model)

        return torch.cat([cls_tok, scalar_tok, cell_tok], dim=1)  # (M, n_tokens, d_model)
