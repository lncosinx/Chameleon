"""
时间记忆核（v3）：跨时间步聚合每体特征，治"大图长程记不住"（局部 FOV 的固有短板）。

默认 backend='mamba'：纯 PyTorch 的 selective SSM（Mamba/S6 核心——输入相关 Δ/B/C 选择性门控
+ 因果深度卷积 + 选择性扫描）。**不依赖 mamba-ssm CUDA 扩展**（torch cu130+nvcc12.6 装不上）。
backend 'gru'/'lstm' 作零依赖兜底/对照。

两种调用：
- 训练（整段）：forward(x, stateful=False)，状态从 0 起，每段独立（与通信 reset 对齐）。
- 推断（闭环 rollout 逐步）：reset_state() 后每步 forward(x, stateful=True)，**跨步携带**
  conv 历史 + SSM 状态（Mamba）/ RNN 隐状态（gru/lstm）。否则逐步状态归零=无记忆、与训练不符。
逐步与整段在同一序列上**数值等价**（见 scripts smoke 校验）。
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor,
                conv_state: Optional[torch.Tensor] = None,
                ssm_state: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x (B,L,d_model) → (y (B,L,d_model), new_conv_state, new_ssm_state)。
        conv_state (B,d_inner,d_conv-1) / ssm_state (B,d_inner,d_state)；None=从 0 起。"""
        B, L, _ = x.shape
        x_, z = self.in_proj(x).chunk(2, dim=-1)          # (B,L,d_inner) ×2
        conv_in = x_.transpose(1, 2)                       # (B,d_inner,L)
        w, b = self.conv1d.weight, self.conv1d.bias

        # 统一用"左侧零/状态前置 + padding=0"实现因果卷积：conv_state=None 时用零前置，
        # 与训练的 padding=kpad[:L] 因果卷积数值等价；推断时前置上一步保存的 kpad 列历史。
        # new_conv_state 恒为固定 kpad 列（L<kpad 也成立，因 full 长度 kpad+L≥kpad）。
        kpad = self.d_conv - 1
        if kpad > 0:
            if conv_state is None:
                conv_state = conv_in.new_zeros(B, self.d_inner, kpad)
            full = torch.cat([conv_state, conv_in], dim=-1)             # (B,d_inner,kpad+L)
            conv_out = F.conv1d(full, w, b, padding=0, groups=self.d_inner)  # length L
            new_conv_state = full[..., -kpad:].detach()
        else:
            conv_out = F.conv1d(conv_in, w, b, padding=0, groups=self.d_inner)
            new_conv_state = conv_in.new_zeros(B, self.d_inner, 0)
        xc = self.act(conv_out.transpose(1, 2))            # (B,L,d_inner)

        dbl = self.x_proj(xc)
        dt, Bp, Cp = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                  # (B,L,d_inner)
        A = -torch.exp(self.A_log.float())                 # (d_inner,d_state)

        if ssm_state is None:
            h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=torch.float32)
        else:
            h = ssm_state.float()
        xc_f, dt_f, Bp_f, Cp_f = xc.float(), dt.float(), Bp.float(), Cp.float()
        ys = []
        for t in range(L):
            dA = torch.exp(dt_f[:, t].unsqueeze(-1) * A)                        # (B,d_inner,d_state)
            dBx = dt_f[:, t].unsqueeze(-1) * Bp_f[:, t].unsqueeze(1) * xc_f[:, t].unsqueeze(-1)
            h = dA * h + dBx
            ys.append((h * Cp_f[:, t].unsqueeze(1)).sum(-1))                    # (B,d_inner)
        y = torch.stack(ys, dim=1).to(x.dtype) + xc * self.D
        y = y * self.act(z)
        return self.out_proj(y), new_conv_state, h.detach()


class TemporalCore(nn.Module):
    def __init__(self, d_model: int, backend: str = "mamba", n_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.backend = backend
        self.n_layers = n_layers
        if backend == "mamba":
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand)
                                         for _ in range(n_layers)])
        elif backend in ("gru", "lstm"):
            rnn = nn.GRU if backend == "gru" else nn.LSTM
            self.rnn = rnn(d_model, d_model, num_layers=n_layers, batch_first=True)
        else:
            raise ValueError(f"unknown temporal backend: {backend}")
        self.reset_state()

    def reset_state(self):
        """清推断态（下次 stateful forward 从 0 起）。每 episode 起点调。"""
        self._mamba_state = [(None, None)] * self.n_layers   # (conv_state, ssm_state) per layer
        self._rnn_state = None

    def forward(self, x: torch.Tensor, stateful: bool = False) -> torch.Tensor:
        """(BN, T, D) → (BN, T, D)。stateful=True 时跨调用携带状态（推断逐步用）。"""
        if self.backend == "mamba":
            for i, (norm, blk) in enumerate(zip(self.norms, self.blocks)):
                cs, ss = self._mamba_state[i] if stateful else (None, None)
                out, cs, ss = blk(norm(x), cs, ss)
                if stateful:
                    self._mamba_state[i] = (cs, ss)
                x = x + out                                  # pre-norm 残差
            return x
        # gru/lstm
        st = self._rnn_state if stateful else None
        out, st = self.rnn(x, st)
        if stateful:
            self._rnn_state = (st[0].detach(), st[1].detach()) if isinstance(st, tuple) else st.detach()
        return out
