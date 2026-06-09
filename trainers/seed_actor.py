"""
阶段三 SEED-RL：GPU 批量推理 actor（同步批量版）。

把 N_env 个并行 env 当作 batch 维（B=N_env），每步以 T=1 调用共享网络**一次**前向，
得到全部 env×agent 的动作。actor 是**有状态**的：
- per-(env,agent) GRU hidden：(1, N_env*N, H)，跨段步维护，段首 reset。
- 通信迟滞状态 _active_mask/_prev_edge_set：形状 (B,N,N)，B=N_env 天然每 env 一份，
  段首由 reset_comm_state() 清空（与单进程逐步推断一致）。

actor 网络是冻结副本（不回传梯度），权重由 learner 每步 sync（lag≤1，INVARIANT 9）。
本 actor 跑在 GPU1（actor_gpu），learner 在 GPU0；二者在同步设计下不重叠执行。
"""
from __future__ import annotations

import numpy as np
import torch

from models.biintent_drqn import BiIntentGATDRQN


class SeedActor:
    def __init__(self, net: BiIntentGATDRQN, device: torch.device, use_amp: bool = False):
        self.net = net.to(device)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.device = device
        self.use_amp = bool(use_amp) and device.type == "cuda"
        self.hidden = None

    @torch.no_grad()
    def load_weights(self, state_dict: dict):
        """从 learner 同步权重（跨设备 copy_ 自动处理）。"""
        self.net.load_state_dict(state_dict)

    def reset(self, n_env: int, n_agents: int):
        """段首重置：per-(env,agent) GRU hidden + 通信迟滞状态。"""
        self.hidden = self.net.init_hidden(n_env, n_agents, self.device)
        self.net.reset_comm_state()

    @torch.no_grad()
    def step(self, patches: np.ndarray, scalars: np.ndarray,
             positions: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """
        一步批量前向。输入均为 (N_env, N, ...) numpy（来自共享内存切片）。

        Returns:
            actions: (N_env, N) int64，masked argmax。
        """
        def _to(x, dtype):
            return torch.from_numpy(np.ascontiguousarray(x)).to(dtype).unsqueeze(1).to(self.device)

        p = _to(patches, torch.float32)      # (B,1,N,4,K,K)
        s = _to(scalars, torch.float32)      # (B,1,N,8)
        pos = _to(positions, torch.float32)  # (B,1,N,2)
        m = _to(masks, torch.bool)           # (B,1,N,5)

        with torch.autocast("cuda", enabled=self.use_amp):
            q_values, self.hidden, _ = self.net(
                p, s, pos, self.hidden, action_masks=m,
            )
        # (B,1,N,5) → (B,N)
        actions = q_values.squeeze(1).argmax(dim=-1)
        return actions.to(torch.int64).cpu().numpy()
