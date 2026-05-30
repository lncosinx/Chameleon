"""
Dueling Q 头。

Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))

输出 5 维 Q 值（对应 wait/up/down/left/right）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DuelingQHead(nn.Module):
    """
    Dueling Q 头：V(s) + Advantage(s, a) - mean(Advantage).

    输入: (..., D_hidden)
    输出: (..., num_actions=5)
    """

    def __init__(self, hidden_dim: int = 256, dueling_hidden_dim: int = 256,
                 num_actions: int = 5):
        super().__init__()
        self.num_actions = num_actions

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, dueling_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(dueling_hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, dueling_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(dueling_hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., hidden_dim)

        Returns:
            q: (..., num_actions)
        """
        v = self.value_stream(x)           # (..., 1)
        a = self.advantage_stream(x)       # (..., num_actions)
        # Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=-1, keepdim=True))
        return q
