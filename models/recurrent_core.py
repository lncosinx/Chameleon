"""
DRQN 循环记忆核（GRU）。

支持：
- 整段序列一次性展开（cudnn 加速）
- R2D2 burn-in：burn_in_len 步仅前向重建隐状态
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RecurrentCore(nn.Module):
    """
    GRU 循环核。

    输入: (B, T, N, D_in)
    输出: (B, T, N, D_hidden), hidden_state

    支持 burn-in：burn_in_len 步仅前向不回传。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,  # (B*N, T, D_in) 或 (B, T, N, D_in)
        hidden: Optional[torch.Tensor] = None,  # (1, B*N, D_hidden)
        burn_in_len: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征序列。
            hidden: 初始隐状态，默认全零。
            burn_in_len: burn-in 步数（仅前向，不回传梯度）。

        Returns:
            outputs: (B*N, T, D_hidden) 全部时间步的输出。
            h_final: (1, B*N, D_hidden) 最终隐状态。
        """
        if burn_in_len > 0 and burn_in_len < x.shape[1]:
            # burn-in: 前 burn_in_len 步用来重建隐状态，不参与后续 loss
            with torch.no_grad():
                x_burn = x[:, :burn_in_len]
                _, hidden = self.gru(x_burn, hidden)
            x_train = x[:, burn_in_len:]
            outputs_train, h_final = self.gru(x_train, hidden)
            # 补齐 outputs 形状（burn 部分填 0）
            outputs_burn = torch.zeros(
                x.shape[0], burn_in_len, self.hidden_dim,
                device=x.device, dtype=x.dtype,
            )
            outputs = torch.cat([outputs_burn, outputs_train], dim=1)
        else:
            outputs, h_final = self.gru(x, hidden)

        return outputs, h_final

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """初始化全零隐状态。"""
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)
