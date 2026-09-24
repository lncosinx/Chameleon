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

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _load_selective_scan():
    """加载 mamba-ssm 的官方 CUDA 选择性扫描核；不可用则返回 None（回落到纯 PyTorch 实现）。

    ★ 为什么要绕过 `import mamba_ssm`：该包的 __init__ 会连带导入 transformers / huggingface_hub
      等重依赖（只为它的 LM 封装用），而我们只需要最内层的 selective_scan_fn。这里给父包塞桩，
      只加载 ops.selective_scan_interface，避免为一个 kernel 把整套 LM 依赖拖进环境。
    ★ 历史：旧服务器 torch cu130 + nvcc12.6 装不上官方扩展，故本文件先写了纯 PyTorch 版；
      现机器 torch 2.8.0+cu128 与 nvcc 12.8 匹配，官方预编译 wheel 可直接安装（2026-09-20）。
      纯 PyTorch 版保留：既是无扩展环境的兜底，也是 kernel 的数值对拍基准
      （scripts/test_mamba_kernel.py）。
    """
    import os, sys, types, importlib
    # ★ 幂等性：本函数第一次成功后会在 sys.modules 留下**桩** mamba_ssm（只有 __path__、
    #   没有 __file__）。若第二次调用仍走 `import mamba_ssm` 分支，会拿到那个桩 →
    #   os.path.dirname(_m.__file__) 抛 AttributeError → 回退 find_spec，而已导入模块的
    #   __spec__ 是 None → base=None → **误报"内核不可用"**。2026-09-24 据此误判过一次
    #   （当时训练其实正在走内核）。故先直接复用已加载好的子模块。
    _done = sys.modules.get("mamba_ssm.ops.selective_scan_interface")
    if _done is not None and hasattr(_done, "selective_scan_fn"):
        return _done.selective_scan_fn
    try:
        import mamba_ssm as _m
        base = os.path.dirname(_m.__file__)
    except Exception:
        try:
            import importlib.util as _u
            spec = _u.find_spec("mamba_ssm")
            base = os.path.dirname(spec.origin) if spec and spec.origin else None
        except Exception:
            base = None
        if not base:
            return None
    try:
        for name, path in [("mamba_ssm", base), ("mamba_ssm.ops", base + "/ops"),
                           ("mamba_ssm.utils", base + "/utils"),
                           ("mamba_ssm.ops.triton", base + "/ops/triton")]:
            if name not in sys.modules or not hasattr(sys.modules[name], "__path__"):
                m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
        return importlib.import_module("mamba_ssm.ops.selective_scan_interface").selective_scan_fn
    except Exception:
        return None


_SELECTIVE_SCAN = _load_selective_scan()


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_init: str = "torch_default"):
        """dt_init: 'torch_default'（**默认，保持历史行为**，`dt_proj` 用 nn.Linear 默认初始化）
        | 'official'（官方 Mamba 初始化）。

        ★ 背景（SCHEDULE 已定论 18，2026-09-24）：默认初始化下 softplus(dt_proj(·)) ≈ 0.70，
          最慢通道 dA≈0.50（半衰期 1 步），训练 12000 步后 dt_base 仍为 0.70 纹丝不动 ⇒
          选择性扫描坍缩成 `y_i ≈ dt_i·conv4(x)_i·⟨B,C⟩`（⟨B,C⟩ 是跨通道共享的单标量），
          SSM 实际无记忆、d_state 形同虚设。官方初始化把 dt 放到 log-uniform[1e-3,1e-1]
          （半衰期 ≈35 步）。**不改默认值**：改了会让全部历史 Mamba 结果不可比（见已定论 19）。
        """
        super().__init__()
        self.dt_init = dt_init
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
        # False = 强制走纯 PyTorch 参考实现。环境变量 MADCL_NO_SSM_KERNEL=1 可全局关闭，
        # 用途：与**内核安装之前**训练/评测的臂做同实现对拍（内核装于 2026-09-20 20:54，
        # 早于此的臂如 smamba 全部走参考实现）。
        self.use_kernel = os.environ.get("MADCL_NO_SSM_KERNEL") != "1"
        if dt_init == "official":
            # 官方 Mamba：weight 按 dt_rank**-0.5 均匀初始化；bias 置为 inverse-softplus(dt)，
            # dt ~ log-uniform[dt_min, dt_max]，使初始时间常数覆盖 1~1000 步量级。
            dt_min, dt_max, dt_floor = 1e-3, 1e-1, 1e-4
            std = self.dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -std, std)
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                           + math.log(dt_min)).clamp(min=dt_floor)
            with torch.no_grad():
                self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        elif dt_init != "torch_default":
            raise ValueError(f"unknown dt_init: {dt_init}")

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

        # ---- 快路径：官方 CUDA 选择性扫描（仅训练用）----
        # 官方核**不接受非零初始状态**，故只在 ssm_state is None（训练整段，h0=0）时启用；
        # 推断逐步（L=1 且需跨步带状态）仍走下面的参考实现——那条本来就只有一步，不是热点。
        # 参数化逐项对应：u=xc, delta=dt(已 softplus), A=-exp(A_log), B=Bp, C=Cp, D, z 门控，
        # 融合后同为 y=(scan + u·D)·silu(z)。数值对拍见 scripts/test_mamba_kernel.py。
        if (_SELECTIVE_SCAN is not None and ssm_state is None
                and x.is_cuda and self.use_kernel):
            y = _SELECTIVE_SCAN(
                xc.transpose(1, 2).contiguous().float(),      # u     (B,d_inner,L)
                dt.transpose(1, 2).contiguous().float(),      # delta (B,d_inner,L)
                A,                                            # A     (d_inner,d_state)
                Bp.transpose(1, 2).contiguous().float(),      # B     (B,d_state,L)
                Cp.transpose(1, 2).contiguous().float(),      # C     (B,d_state,L)
                self.D.float(), z.transpose(1, 2).contiguous().float(),
                None, False, True)
            y, last_state = y
            y = y.transpose(1, 2).to(x.dtype)
            return self.out_proj(y), new_conv_state, last_state.detach()

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
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 checkpoint: bool = False, dt_init: str = "torch_default",
                 final_norm: bool = False):
        super().__init__()
        self.backend = backend
        self.n_layers = n_layers
        # 梯度检查点：纯 PyTorch 的选择性扫描会把每个时间步的 (BN,d_inner,d_state) 中间量
        # 全部留给反向（d_model=320/T=32/N=64 下实测需 >29GB，31.4GB 卡直接 OOM）。
        # 开启后反向时重算该段，显存大降、算力约 +30%，语义不变——这样 batch/seq 不用改，
        # 才能与已有 GRU 结果（同 batch/seq/数据/步数）直接对比。
        self.checkpoint = checkpoint
        # final_norm（已定论 18 §4 的修法臂，2026-09-24）：pre-norm 残差栈出口本无归一化，
        # 两层 `x = x + out` 无界累加使时间核增益随训练单调上涨（2k:1.30 → 8k:1.53 → 12k:2.05），
        # 掉分窗口与增益加速窗口重合。默认 False = 保持历史行为。
        self.final_norm = nn.LayerNorm(d_model) if (final_norm and backend == "mamba") else None
        if backend == "mamba":
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand, dt_init)
                                         for _ in range(n_layers)])
        elif backend in ("gru", "lstm"):
            rnn = nn.GRU if backend == "gru" else nn.LSTM
            self.rnn = rnn(d_model, d_model, num_layers=n_layers, batch_first=True)
        elif backend in ("gru_prenorm", "lstm_prenorm"):
            # 归因臂（2026-09-24）：把 mamba 分支的**块结构**（pre-norm + 逐层残差）原样套到
            # RNN 上，唯一变量 = 递归核是 SSM 还是 GRU。动机见 SCHEDULE 已定论 18：
            # 此前所有 Mamba-vs-GRU 对比里，mamba 分支带 LayerNorm+残差而 gru 分支是裸 nn.GRU，
            # "SSM 优于 GRU" 与 "pre-norm 残差优于无" 两个变量混在一起。
            # 参数量与裸 nn.GRU(num_layers=n) 相同（堆 n 个单层 GRU），只多 n 个 LayerNorm。
            rnn = nn.GRU if backend.startswith("gru") else nn.LSTM
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.rnns = nn.ModuleList([rnn(d_model, d_model, num_layers=1, batch_first=True)
                                       for _ in range(n_layers)])
        else:
            raise ValueError(f"unknown temporal backend: {backend}")
        self.reset_state()

    def reset_state(self):
        """清推断态（下次 stateful forward 从 0 起）。每 episode 起点调。"""
        self._mamba_state = [(None, None)] * self.n_layers   # (conv_state, ssm_state) per layer
        self._rnn_state = None
        self._rnn_states = [None] * self.n_layers            # gru_prenorm/lstm_prenorm 逐层状态

    def forward(self, x: torch.Tensor, stateful: bool = False) -> torch.Tensor:
        """(BN, T, D) → (BN, T, D)。stateful=True 时跨调用携带状态（推断逐步用）。"""
        if self.backend == "mamba":
            for i, (norm, blk) in enumerate(zip(self.norms, self.blocks)):
                cs, ss = self._mamba_state[i] if stateful else (None, None)
                if self.checkpoint and self.training and not stateful and x.requires_grad:
                    out, cs, ss = torch.utils.checkpoint.checkpoint(
                        blk, norm(x), cs, ss, use_reentrant=False)
                else:
                    out, cs, ss = blk(norm(x), cs, ss)
                if stateful:
                    self._mamba_state[i] = (cs, ss)
                x = x + out                                  # pre-norm 残差
            return self.final_norm(x) if self.final_norm is not None else x
        if self.backend in ("gru_prenorm", "lstm_prenorm"):
            for i, (norm, rnn) in enumerate(zip(self.norms, self.rnns)):
                st = self._rnn_states[i] if stateful else None
                out, st = rnn(norm(x), st)
                if stateful:
                    self._rnn_states[i] = ((st[0].detach(), st[1].detach())
                                           if isinstance(st, tuple) else st.detach())
                x = x + out                                  # pre-norm 残差（与 mamba 分支一致）
            return x
        # gru/lstm
        st = self._rnn_state if stateful else None
        out, st = self.rnn(x, st)
        if stateful:
            self._rnn_state = (st[0].detach(), st[1].detach()) if isinstance(st, tuple) else st.detach()
        return out
