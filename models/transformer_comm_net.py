"""
v3 网络组装：Tokenizer → Transformer 骨干 → 软稀疏/迟滞通信（保留）→ 动作 logits。

取代 BiIntentGATDRQN（CNN+GRU+GAT+Dueling Q）。关键不变量：
- 去中心化、参数共享，每体只用自身局部观测（INVARIANT 2）；不构造 Q_tot。
- 通信只与 FOV 内邻居、软稀疏 entmax + 迟滞（INVARIANT 10，v3 头条贡献）——直接复用
  models/communication.BiIntentAttention（架构无关）。
- 输出动作 logits（5），配离线 IL 交叉熵（非 Q 值）。

前向接口与旧 net 对齐 (B,T,N,...) 以便复用数据管线/通信迟滞；无 GRU hidden。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict

from models.obs_tokenizer import ObsTokenizer
from models.transformer_backbone import TransformerBackbone
from models.communication import BiIntentAttention


class TransformerCommNet(nn.Module):
    def __init__(
        self,
        patch_channels: int = 4,
        scalar_feat_dim: int = 8,
        obs_radius: int = 5,
        num_actions: int = 5,
        # transformer
        d_model: int = 192,
        n_layers: int = 3,
        n_heads: int = 3,
        d_ff: int = 768,
        dropout: float = 0.0,
        norm_first: bool = True,
        # communication
        comm_rounds: int = 1,
        intent_dim: int = 128,
        comm_dim: int = 128,
        comm_radius: int = 4,
        comm_topk_cap: int = 6,
        entmax_alpha: float = 1.5,
        tau_in: float = 0.05,
        stickiness_bonus: float = 0.1,
        ema_decay: float = 0.0,
        num_attention_heads: int = 4,
        # head
        head_hidden: int = 128,
    ):
        super().__init__()
        self.tokenizer = ObsTokenizer(patch_channels, scalar_feat_dim, obs_radius, d_model)
        self.backbone = TransformerBackbone(d_model, n_layers, n_heads, d_ff, dropout, norm_first)

        self.comm_rounds = comm_rounds
        self.comm_in = nn.Linear(d_model, intent_dim)       # agent 嵌入 → 通信意图
        self.comm_out = nn.Linear(intent_dim, d_model)      # 通信特征 → 残差回 d_model
        self.communication = BiIntentAttention(
            intent_dim=intent_dim, comm_dim=comm_dim,
            comm_radius=comm_radius, comm_topk_cap=comm_topk_cap,
            num_heads=num_attention_heads, entmax_alpha=entmax_alpha,
            tau_in=tau_in, stickiness_bonus=stickiness_bonus,
            ema_decay=ema_decay,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, num_actions),
        )
        self.num_actions = num_actions

    def reset_comm_state(self):
        """episode 边界重置通信迟滞状态（推断逐步调用时由外部在 episode 起点调）。"""
        self.communication.reset_state()

    def forward(
        self,
        patches: torch.Tensor,     # (B, T, N, C, K, K)
        scalars: torch.Tensor,     # (B, T, N, scalar_feat_dim)
        positions: torch.Tensor,   # (B, T, N, 2) 仅供通信算邻居距离，不进决策
        action_masks: Optional[torch.Tensor] = None,  # (B,T,N,5) bool，True=合法
        reset_comm: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        B, T, N = patches.shape[:3]
        C, K = patches.shape[3], patches.shape[4]

        if reset_comm:
            self.communication.reset_state()

        # ---- 1. 每体 token 化 + Transformer → agent 嵌入 ----
        M = B * T * N
        tok = self.tokenizer(patches.reshape(M, C, K, K), scalars.reshape(M, -1))
        emb = self.backbone(tok).reshape(B, T, N, -1)        # (B,T,N,d_model)

        # ---- 2. 通信（软稀疏+迟滞，保留）：R 轮残差消息传递 ----
        info: Dict = {}
        for _ in range(self.comm_rounds):
            intent = self.comm_in(emb)                       # (B,T,N,intent_dim)
            comm_feat, info = self.communication.forward_sequence(intent, positions)
            emb = emb + self.comm_out(comm_feat)             # 残差

        # ---- 3. 动作 logits ----
        logits = self.head(emb)                              # (B,T,N,5)
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks, float("-inf"))  # 非法动作屏蔽

        info = {
            "edge_flip_rate": float(info.get("edge_flip_rate", 0.0)),
            "avg_neighbors": float(info.get("avg_neighbors", 0.0)),
        }
        return logits, info
