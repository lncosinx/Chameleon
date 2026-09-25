"""
Transformer 骨干（v3）：对一体的 token 序列做自注意力，取 CLS 位输出为 agent 嵌入。

取代旧 `PatchEncoder(CNN) + GRU`。每体独立前向、参数全共享（去中心化）。
无时间递归——记忆由观测自带（足迹通道）+ 通信承担；时间稳定性由通信迟滞保证。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        d_model: int = 192,
        n_layers: int = 3,
        n_heads: int = 3,
        d_ff: int = 768,
        dropout: float = 0.0,
        norm_first: bool = True,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (M, n_tokens, d_model)
        Returns:
            agent_emb: (M, d_model)  —— CLS 位输出
        """
        h = self.encoder(tokens)        # (M, n_tokens, d_model)
        return self.norm(h[:, 0])       # CLS
