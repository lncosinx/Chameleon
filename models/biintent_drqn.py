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
        comm_rounds: int = 1,            # 通信层数（GAT 轮数）：0=无通信, 1/2/3=多轮消息传递
        # GRU
        gru_hidden_dim: int = 256,
        # Dueling Q
        dueling_hidden_dim: int = 256,
        num_actions: int = 5,
        # 性能（§4.2）
        use_torch_compile: bool = False,
        # 消融：False=禁用通信（comm_feat 置零，架构/参数量不变，仅去通信信号）
        comm_enabled: bool = True,
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64, 64]

        self.patch_encoder = PatchEncoder(
            in_channels=patch_channels,
            cnn_channels=cnn_channels,
            d_patch=d_patch,
        )

        # torch.compile：只编译 patch encoder（B·T·N 次 CNN 前向是最大算力热点，
        # 且无状态、形状固定 → 编译友好）。不编译整网，因通信模块带跨步可变迟滞
        # 状态 + 逐 t Python 循环，整网编译会 graph-break 或捕获错误状态（§4.2 风险说明）。
        # reduce-overhead 启用 CUDA graphs 砍 launch 开销；失败则回退默认 mode，再回退 eager。
        if use_torch_compile:
            self.patch_encoder = self._try_compile(self.patch_encoder)
        self.scalar_encoder = ScalarEncoder(
            in_dim=scalar_feat_dim,
            d_scalar=d_scalar,
        )

        self.fused_dim = d_patch + d_scalar  # 128 + 32 = 160

        # 意图投影
        self.intent_proj = nn.Linear(self.fused_dim, intent_dim)

        # 双向意图注意力（软稀疏 + 迟滞）。多轮通信 = 堆叠多个 BiIntentAttention。
        # 第一层保留属性名 `self.communication`（兼容 reset/消融代码 + comm_rounds≤1 主路径）；
        # 第 2..comm_rounds 层放 extra_comm_layers，层间用 intent_refine 残差刷新意图。
        def _make_comm():
            return BiIntentAttention(
                intent_dim=intent_dim, comm_dim=comm_dim, comm_radius=comm_radius,
                comm_topk_cap=comm_topk_cap, num_heads=num_attention_heads,
                entmax_alpha=entmax_alpha, tau_in=tau_in, tau_out=tau_out,
                stickiness_bonus=stickiness_bonus, ema_decay=ema_decay,
            )
        # 始终构造第一层 → comm_rounds=0(无通信)与>=1 的参数量基本一致，便于公平消融。
        self.communication = _make_comm()
        n_extra = max(0, comm_rounds - 1)
        self.extra_comm_layers = nn.ModuleList([_make_comm() for _ in range(n_extra)])
        # 层间意图刷新：把上一轮通信输出(intent_dim) 残差加回意图，作为下一轮输入。
        self.intent_refine = nn.ModuleList(
            [nn.Linear(intent_dim, intent_dim) for _ in range(n_extra)]
        )
        self.comm_rounds = comm_rounds

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
        self.comm_enabled = comm_enabled
        self.intent_dim = intent_dim

    @staticmethod
    def _try_compile(module: nn.Module) -> nn.Module:
        """尝试 torch.compile（reduce-overhead→default→eager 逐级回退），不让编译失败中断训练。"""
        if not hasattr(torch, "compile"):
            return module
        for mode in ("reduce-overhead", "default"):
            try:
                compiled = torch.compile(module, mode=mode)
                print(f"[torch.compile] patch_encoder compiled (mode={mode})")
                return compiled
            except Exception as e:  # noqa: BLE001
                print(f"[torch.compile] mode={mode} failed ({type(e).__name__}); trying next")
        print("[torch.compile] all modes failed; running eager")
        return module

    def reset_comm_state(self):
        """
        重置通信模块的跨时间步迟滞状态。
        应在每个 episode 开始时调用一次，而非在每次 forward 时调用。

        B7 修复：原先在 forward() 内调用 reset_state()，导致推断时每步都清空
        迟滞状态（_active_mask/_prev_edge_set），EdgeFlipRate 防闪烁机制失效。
        现改为由调用方在 episode 边界显式调用本方法。
        """
        self._reset_all_comm()

    def _reset_all_comm(self):
        """重置所有通信层（含多轮）的迟滞状态。"""
        self.communication.reset_state()
        for layer in self.extra_comm_layers:
            layer.reset_state()

    def forward(
        self,
        patches: torch.Tensor,       # (B, T, N, 4, K, K)
        scalars: torch.Tensor,       # (B, T, N, 8)
        positions: torch.Tensor,     # (B, T, N, 2)
        hidden_init: Optional[torch.Tensor] = None,  # (1, B*N, gru_hidden_dim)
        burn_in_len: int = 0,
        action_masks: Optional[torch.Tensor] = None,  # (B, T, N, 5) bool
        reset_comm: bool = False,    # 是否在本次 forward 前重置通信状态（训练序列开始时设 True）
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
            reset_comm: True = 在本次 forward 前先重置通信迟滞状态（用于训练序列
                        起始或 episode 开始的单步推断）。推断时逐步调用不应设 True，
                        否则迟滞状态每步清空，防闪烁失效（B7）。

        Returns:
            q_values:   (B, T, N, 5) 动作价值。
            h_final:    (1, B*N, gru_hidden_dim) 最终隐状态。
            info:       dict 含 attn_weights 等诊断信息。
        """
        B, T, N = patches.shape[:3]
        device = patches.device

        # 仅在显式要求时重置通信状态（训练 batch 起始或 episode 起始）
        if reset_comm:
            self._reset_all_comm()

        # ---- 1. 编码 ----
        patches_flat = patches.reshape(B * T * N, *patches.shape[-3:])
        patch_feat = self.patch_encoder(patches_flat)       # (B*T*N, d_patch)
        patch_feat = patch_feat.reshape(B, T, N, -1)

        scalar_feat = self.scalar_encoder(scalars)          # (B, T, N, d_scalar)

        fused = torch.cat([patch_feat, scalar_feat], dim=-1)  # (B, T, N, fused_dim)

        # ---- 2. 通信（多轮 GAT；向量化整段，迟滞 active 递推内部串行，重活批量化）----
        # 有效轮数：comm_enabled=False 等价 comm_rounds=0（无通信，仅去信号，参数量不变）。
        eff_rounds = self.comm_rounds if self.comm_enabled else 0
        if eff_rounds <= 0:
            # 消融：无通信——comm_feat 置零（GRU 输入维度不变，仅去通信信号）
            B_, T_, N_ = fused.shape[:3]
            comm_feat = fused.new_zeros(B_, T_, N_, self.intent_dim)
            edge_flip_rates, avg_neighbors_list, active_neighbors_list = [0.0], [0.0], [0.0]
        else:
            intent = self.intent_proj(fused)  # (B, T, N, intent_dim)
            comm_feat, comm_info = self.communication.forward_sequence(intent, positions)
            edge_flip_rates = [comm_info.get("edge_flip_rate", 0.0)]
            avg_neighbors_list = [comm_info.get("avg_neighbors", 0.0)]
            active_neighbors_list = [comm_info.get("avg_active_neighbors", 0.0)]
            # 第 2..eff_rounds 层：用上一轮输出残差刷新意图后再传一轮消息。
            for r in range(eff_rounds - 1):
                intent = intent + self.intent_refine[r](comm_feat)
                comm_feat, comm_info = self.extra_comm_layers[r].forward_sequence(intent, positions)
                edge_flip_rates.append(comm_info.get("edge_flip_rate", 0.0))
                avg_neighbors_list.append(comm_info.get("avg_neighbors", 0.0))
                active_neighbors_list.append(comm_info.get("avg_active_neighbors", 0.0))

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
            "avg_active_neighbors": float(np.mean(active_neighbors_list)) if active_neighbors_list else 0.0,
        }

        return q_values, h_final, info

    def init_hidden(self, batch_size: int, num_agents: int,
                    device: torch.device) -> torch.Tensor:
        """初始化 GRU 隐状态。"""
        return self.gru.init_hidden(batch_size * num_agents, device)
