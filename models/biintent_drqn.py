"""
BiIntent-GAT-DRQN：完整网络组装。

前向流程：
    patch (4,K,K) → PatchEncoder → patch_feat
    scalar (8,)   → ScalarEncoder → scalar_feat
    fused = patch_feat ⊕ scalar_feat → intent
    intent → BiIntentAttention (FOV内 top-k) → comm_feat
    fused ⊕ comm_feat → GRU (整段展开) → hidden
    hidden → DuelingQHead → Q (5,)

共享参数去中心化：每体独立前向，参数全共享。
不使用 QMIX，不构造 Q_tot。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from models.encoders import PatchEncoder, ScalarEncoder
from models.communication import BiIntentAttention
from models.recurrent_core import RecurrentCore
from models.q_heads import DuelingQHead


class BiIntentGATDRQN(nn.Module):
    """
    BiIntent-GAT-DRQN 共享参数去中心化网络。

    用法:
        net = BiIntentGATDRQN(cfg)
        q_values, hidden, info = net(
            patches, scalars, positions, hidden_init,
            burn_in_len=8,
        )
    """

    def __init__(
        self,
        # Patch encoder
        patch_channels: int = 4,
        cnn_channels: list = None,
        d_patch: int = 128,
        # Scalar encoder
        scalar_feat_dim: int = 8,
        d_scalar: int = 32,
        # Communication
        intent_dim: int = 64,
        comm_dim: int = 64,
        comm_radius: int = 4,
        comm_topk_cap: int = 6,          # 仅算力上限
        num_attention_heads: int = 4,
        entmax_alpha: float = 1.5,       # 软稀疏度
        tau_in: float = 0.05,            # 迟滞进入阈值
        tau_out: float = 0.02,           # 迟滞退出阈值
        stickiness_bonus: float = 0.1,   # 活跃邻居加分
        ema_decay: float = 0.0,          # 跨步 EMA 平滑
        # GRU
        gru_hidden_dim: int = 256,
        # Dueling Q
        dueling_hidden_dim: int = 256,
        num_actions: int = 5,
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64, 64]

        self.patch_encoder = PatchEncoder(
            in_channels=patch_channels,
            cnn_channels=cnn_channels,
            d_patch=d_patch,
        )
        self.scalar_encoder = ScalarEncoder(
            in_dim=scalar_feat_dim,
            d_scalar=d_scalar,
        )

        self.fused_dim = d_patch + d_scalar  # 128 + 32 = 160

        # 意图投影
        self.intent_proj = nn.Linear(self.fused_dim, intent_dim)

        # 双向意图注意力（软稀疏 + 迟滞）
        self.communication = BiIntentAttention(
            intent_dim=intent_dim,
            comm_dim=comm_dim,
            comm_radius=comm_radius,
            comm_topk_cap=comm_topk_cap,
            num_heads=num_attention_heads,
            entmax_alpha=entmax_alpha,
            tau_in=tau_in,
            tau_out=tau_out,
            stickiness_bonus=stickiness_bonus,
            ema_decay=ema_decay,
        )

        # 通信后的特征维度 = fused_dim + intent_dim（通信输出维度 = intent_dim）
        gru_input_dim = self.fused_dim + intent_dim

        self.gru = RecurrentCore(
            input_dim=gru_input_dim,
            hidden_dim=gru_hidden_dim,
        )

        self.q_head = DuelingQHead(
            hidden_dim=gru_hidden_dim,
            dueling_hidden_dim=dueling_hidden_dim,
            num_actions=num_actions,
        )

        # 缓存超参
        self.num_actions = num_actions
        self.gru_hidden_dim = gru_hidden_dim

    def forward(
        self,
        patches: torch.Tensor,       # (B, T, N, 4, K, K)
        scalars: torch.Tensor,       # (B, T, N, 8)
        positions: torch.Tensor,     # (B, T, N, 2)
        hidden_init: Optional[torch.Tensor] = None,  # (1, B*N, gru_hidden_dim)
        burn_in_len: int = 0,
        action_masks: Optional[torch.Tensor] = None,  # (B, T, N, 5) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            patches:    (B, T, N, 4, K, K) 观测 patches。
            scalars:    (B, T, N, 8) 标量特征。
            positions:  (B, T, N, 2) 全局坐标 (row, col) — 仅用于计算邻居距离，
                        **不进 Q 头决策**（通信只选邻居，不泄露全局地图）。
            hidden_init: 初始隐状态。
            burn_in_len: R2D2 burn-in 步数。
            action_masks: 合法动作掩码，True=合法。非法位填 -inf。

        Returns:
            q_values:   (B, T, N, 5) 动作价值。
            h_final:    (1, B*N, gru_hidden_dim) 最终隐状态。
            info:       dict 含 attn_weights 等诊断信息。
        """
        B, T, N = patches.shape[:3]
        device = patches.device

        # ---- 1. 编码 ----
        # patches: (B, T, N, 4, K, K) → (B*T*N, d_patch)
        patches_flat = patches.reshape(B * T * N, *patches.shape[-3:])
        patch_feat = self.patch_encoder(patches_flat)       # (B*T*N, d_patch)
        patch_feat = patch_feat.reshape(B, T, N, -1)

        # scalars: (B, T, N, 8) → (B, T, N, d_scalar)
        scalar_feat = self.scalar_encoder(scalars)

        # fused: (B, T, N, d_patch + d_scalar)
        fused = torch.cat([patch_feat, scalar_feat], dim=-1)

        # ---- 2. 通信（逐时间步，带迟滞状态） ----
        intent = self.intent_proj(fused)  # (B, T, N, intent_dim)

        # 重置通信模块的跨时间步状态
        self.communication.reset_state(batch_size=B)

        comm_list = []
        edge_flip_rates = []
        avg_neighbors_list = []
        for t in range(T):
            intent_t = intent[:, t]        # (B, N, intent_dim)
            pos_t = positions[:, t]        # (B, N, 2)
            comm_t, comm_info = self.communication(intent_t, pos_t)
            comm_list.append(comm_t)
            edge_flip_rates.append(comm_info.get("edge_flip_rate", 0.0))
            avg_neighbors_list.append(comm_info.get("avg_neighbors", 0.0))

        comm_feat = torch.stack(comm_list, dim=1)  # (B, T, N, intent_dim)

        # ---- 3. GRU ----
        # 拼接 fused + comm → (B*N, T, gru_input_dim)
        gru_input = torch.cat([fused, comm_feat], dim=-1)  # (B, T, N, fused_dim+intent_dim)
        gru_input = gru_input.permute(0, 2, 1, 3)          # (B, N, T, D)
        gru_input = gru_input.reshape(B * N, T, -1)         # (B*N, T, D)

        gru_output, h_final = self.gru(
            gru_input, hidden_init, burn_in_len=burn_in_len
        )  # gru_output: (B*N, T, gru_hidden_dim)

        # ---- 4. Dueling Q Head ----
        gru_output = gru_output.reshape(B, N, T, -1)        # (B, N, T, D)
        gru_output = gru_output.permute(0, 2, 1, 3)          # (B, T, N, D)
        q_values = self.q_head(gru_output)                   # (B, T, N, 5)

        # ---- 5. 动作掩码 ----
        if action_masks is not None:
            # INVARIANT 7: 非法动作 = -inf
            q_values = q_values.masked_fill(~action_masks, float('-inf'))

        # 诊断信息
        info = {
            "q_mean": q_values.detach().mean().item(),
            "q_max": q_values.detach().max().item(),
            "q_min": q_values.detach().min().item(),
            "edge_flip_rate": float(np.mean(edge_flip_rates)) if edge_flip_rates else 0.0,
            "avg_neighbors": float(np.mean(avg_neighbors_list)) if avg_neighbors_list else 0.0,
        }

        return q_values, h_final, info

    def init_hidden(self, batch_size: int, num_agents: int,
                    device: torch.device) -> torch.Tensor:
        """初始化 GRU 隐状态。"""
        return self.gru.init_hidden(batch_size * num_agents, device)
