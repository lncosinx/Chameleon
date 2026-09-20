"""
v3 网络组装：残差 CNN 空间编码 → 软稀疏/迟滞通信（保留，头条）→ Mamba 时间核 → 动作 logits。

结构沿用旧 BiIntent-GAT-DRQN 的"编码→通信→时间→头"骨架，但：
  PatchEncoder(CNN) 保留思路（换轻量残差）；GRU → Mamba/SSM（治长程记忆）；
  Dueling Q 头 → 动作 logits（配离线 IL 交叉熵）。
去中心化、参数共享，每体只用局部观测（INVARIANT 2）；通信只与 FOV 内邻居（INVARIANT 10）。

前向 (B,T,N,...) → logits (B,T,N,5)：
  空间编码每个 (b,t,n) 独立 → 通信在 (b,t) 内跨 agent（迟滞跨 t）→ 时间核在 (b,n) 内跨 t。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from models.spatial_encoder import ResidualCNNEncoder
from models.communication import BiIntentAttention
from models.comm_baselines import build_comm_baseline
from models.temporal_core import TemporalCore


class MAPFCommNet(nn.Module):
    def __init__(
        self,
        patch_channels: int = 4,
        scalar_feat_dim: int = 8,
        obs_radius: int = 5,
        num_actions: int = 5,
        # 空间编码
        d_model: int = 192,
        cnn_channels: int = 64,
        d_scalar: int = 32,
        n_res_blocks: int = 1,
        # 通信
        comm_rounds: int = 1,
        intent_dim: int = 128,
        comm_dim: int = 128,
        comm_radius: int = 4,
        comm_topk_cap: int = 6,
        entmax_alpha: float = 1.5,
        tau_in: float = 0.05,
        tau_out: float = 0.02,
        stickiness_bonus: float = 0.1,
        ema_decay: float = 0.0,
        num_attention_heads: int = 4,
        hard_topk_select: bool = False,   # SQ3 消融：hard top-k 做选择（预期引起边闪烁）
        # 时间核
        temporal_backend: str = "mamba",
        temporal_checkpoint: bool = False,
        temporal_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        mamba_expand: int = 2,
        # 头
        head_hidden: int = 128,
        # 空间池化：0=全展平（保留空间布局，默认）；n>0=AdaptiveAvgPool2d(n)
        spatial_pool: int = 0,
        # 通信消息内容："feature"=观测特征（原）；"action"=特征+通信前动作分布（显式意图）
        intent_mode: str = "feature",
        # SQ3 通信消融轴："madcl"=本文的软稀疏entmax+迟滞；其余见 models/comm_baselines.py
        # （commnet/ic3net/tarmac/dial/atoc/dgn）。**只换通信模块，其它一切不变**。
        comm_type: str = "madcl",
    ):
        super().__init__()
        self.spatial = ResidualCNNEncoder(patch_channels, scalar_feat_dim, d_model,
                                          cnn_channels, d_scalar, n_res_blocks,
                                          obs_radius=obs_radius, spatial_pool=spatial_pool)
        self.comm_rounds = comm_rounds
        # intent_mode：通信里"意图"到底是什么。
        #   "feature"（原实现）= 编码后的观测特征，邻居收到的是对方的**观测摘要**；
        #   "action"          = 观测特征 **+ 通信前的初步动作分布**，邻居收到的是对方
        #                       "下一步想往哪走"——这才是字面意义上的意图。
        # 动机：窄走廊里决定性的信息就是对面那个体的行进方向；让它显式出现在消息里，
        # 而不是指望网络从观测摘要里自己解码出来（我们只有 ~600 万步数据，不是 10 亿）。
        self.intent_mode = intent_mode
        if intent_mode == "action":
            self.pre_head = nn.Linear(d_model, num_actions)
            self.comm_in = nn.Linear(d_model + num_actions, intent_dim)
        else:
            self.pre_head = None
            self.comm_in = nn.Linear(d_model, intent_dim)
        self.comm_out = nn.Linear(intent_dim, d_model)
        self.comm_type = comm_type
        if comm_type == "madcl":
            self.communication = BiIntentAttention(
                intent_dim=intent_dim, comm_dim=comm_dim, comm_radius=comm_radius,
                comm_topk_cap=comm_topk_cap, num_heads=num_attention_heads,
                entmax_alpha=entmax_alpha, tau_in=tau_in, tau_out=tau_out,
                stickiness_bonus=stickiness_bonus, ema_decay=ema_decay,
                hard_topk_select=hard_topk_select,
            )
        else:
            self.communication = build_comm_baseline(
                comm_type, intent_dim=intent_dim, comm_dim=comm_dim,
                comm_radius=comm_radius, num_heads=num_attention_heads)
        self.temporal = TemporalCore(d_model, backend=temporal_backend,
                                     n_layers=temporal_layers, d_state=d_state,
                                     d_conv=d_conv, expand=mamba_expand,
                                     checkpoint=temporal_checkpoint)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, num_actions),
        )
        self.num_actions = num_actions

    def reset_comm_state(self):
        """episode 边界重置通信迟滞 + 时间核推断态（rollout 在 episode 起点调）。"""
        self.communication.reset_state()
        self.temporal.reset_state()

    def forward(
        self,
        patches: torch.Tensor,     # (B, T, N, C, K, K)
        scalars: torch.Tensor,     # (B, T, N, scalar_feat_dim)
        positions: torch.Tensor,   # (B, T, N, 2)
        action_masks: Optional[torch.Tensor] = None,
        reset_comm: bool = False,
        stateful: bool = False,    # True=推断逐步（跨调用携带通信迟滞 + Mamba 状态）
    ) -> Tuple[torch.Tensor, Dict]:
        B, T, N = patches.shape[:3]
        C, K = patches.shape[3], patches.shape[4]
        if reset_comm:
            self.communication.reset_state()
            self.temporal.reset_state()

        # ---- 1. 空间编码（每体每步独立）----
        M = B * T * N
        e = self.spatial(patches.reshape(M, C, K, K), scalars.reshape(M, -1))
        e = e.reshape(B, T, N, -1)                          # (B,T,N,d_model)

        # ---- 2. 通信（软稀疏+迟滞，保留）：R 轮残差消息传递 ----
        # 迟滞是跨"时间步"的稳定机制，不是跨"轮"——R 轮 refinement 都应以"上一时间步的
        # active 集"为迟滞先验，仅最后一轮推进持久状态。故进 round 循环前 snapshot 迟滞态，
        # 每轮（除最后）跑完恢复 snapshot，只留最后一轮写入。comm_rounds=1 时无影响。
        info: Dict = {}
        c = self.communication
        has_hyst = hasattr(c, "_active_mask")     # 只有 madcl 的 BiIntentAttention 有迟滞态
        snap = (c._active_mask, c._prev_edge_set, c._ema_c) if has_hyst else None
        for r in range(self.comm_rounds):
            if r > 0 and has_hyst:
                c._active_mask, c._prev_edge_set, c._ema_c = snap
            if self.pre_head is not None:
                # 通信前的初步动作分布（对非法动作屏蔽后 softmax），与特征拼接成消息。
                pre = self.pre_head(e)
                if action_masks is not None:
                    pre = pre.masked_fill(~action_masks, float("-inf"))
                pre = torch.softmax(pre, dim=-1)
                pre = torch.nan_to_num(pre, nan=0.0)
                intent = self.comm_in(torch.cat([e, pre], dim=-1))
            else:
                intent = self.comm_in(e)
            comm_feat, info = c.forward_sequence(intent, positions)
            e = e + self.comm_out(comm_feat)

        # ---- 3. 时间核（每体跨 t；Mamba/SSM）----
        seq = e.permute(0, 2, 1, 3).reshape(B * N, T, -1)   # (B*N, T, d_model)
        h = self.temporal(seq, stateful=stateful).reshape(B, N, T, -1).permute(0, 2, 1, 3)

        # ---- 4. 动作 logits ----
        logits = self.head(h)
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks, float("-inf"))

        info = {
            "edge_flip_rate": float(info.get("edge_flip_rate", 0.0)),
            "avg_neighbors": float(info.get("avg_neighbors", 0.0)),
        }
        return logits, info
