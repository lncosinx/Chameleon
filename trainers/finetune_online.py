"""
阶段三：目标地图低滞后近 on-policy 微调。

- θ ← θ*（从 Reptile meta checkpoint 初始化）
- 同步段采样：GPU0 learner / GPU1 推理 actor
- 丢弃 lag > 1 的样本（INVARIANT 9）
- LaCAM 专家介入概率退火 0.5 → 0.02
- margin 权重退火 1.0 → 0（末期退化为纯 Q 学习）
- 周期性 target 同步（标准 DQN，INVARIANT 8）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from copy import deepcopy

from models.biintent_drqn import BiIntentGATDRQN
from losses.dqfd_losses import dqfd_total_loss
from buffers.recent_online_buffer import RecentOnlineBuffer


class FinetuneOnlineTrainer:
    """
    低滞后近 on-policy 微调训练器。

    用法:
        trainer = FinetuneOnlineTrainer(net, cfg)
        trainer.load_meta_checkpoint("checkpoints/meta_init.pt")
        for segment in range(num_segments):
            samples = rollout_segment(env, net)  # 同步段采样
            metrics = trainer.train_on_segment(samples)
    """

    def __init__(
        self,
        net: BiIntentGATDRQN,
        config: dict,
        device: torch.device = None,
    ):
        self.net = net
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # Target network（周期性同步）
        self.target_net = deepcopy(self.net)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

        # 超参
        self.rollout_segment_len = config.get("rollout_segment_len", 32)
        self.max_policy_version_lag = config.get("max_policy_version_lag", 1)
        self.reuse_epochs_per_segment = config.get("reuse_epochs_per_segment", 1)
        self.gamma = config.get("gamma", 0.99)
        self.mellowmax_alpha = config.get("mellowmax_alpha", 1.0)
        self.margin = config.get("margin", 0.5)

        # 退火参数
        self.expert_prob_start = config.get("expert_prob_start", 0.5)
        self.expert_prob_end = config.get("expert_prob_end", 0.02)
        self.expert_decay_steps = config.get("expert_decay_steps", 200000)
        self.lambda_margin_start = config.get("lambda_margin_start", 1.0)
        self.lambda_margin_end = config.get("lambda_margin_end", 0.0)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-5)

        # Buffer
        self.buffer = RecentOnlineBuffer(
            capacity=10000,
            max_policy_version_lag=self.max_policy_version_lag,
        )

        self.train_step = 0
        self.target_sync_interval = 100  # 每 100 步同步 target

    def load_meta_checkpoint(self, path: str):
        """加载 Reptile meta checkpoint。"""
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.target_net.load_state_dict(ckpt["net"])
        print(f"Loaded meta checkpoint from {path}")

    def expert_prob(self) -> float:
        """当前训练步的 LaCAM 专家介入概率。"""
        if self.train_step >= self.expert_decay_steps:
            return self.expert_prob_end
        progress = self.train_step / self.expert_decay_steps
        return self.expert_prob_start + (self.expert_prob_end - self.expert_prob_start) * progress

    def lambda_margin(self) -> float:
        """当前 margin loss 权重。"""
        if self.train_step >= self.expert_decay_steps:
            return self.lambda_margin_end
        progress = self.train_step / self.expert_decay_steps
        return self.lambda_margin_start + (self.lambda_margin_end - self.lambda_margin_start) * progress

    def add_rollout_samples(self, samples: list, policy_version: int):
        """添加同步段采样数据。"""
        self.buffer.add_batch(samples, policy_version)

    def train_on_segment(
        self, batch_size: int = 32,
        sequence_len: int = None,
        burn_in_len: int = 8,
    ) -> Dict[str, float]:
        """
        在最新 segment 数据上以序列模式训练（DRQN + R2D2 burn-in）。

        Args:
            batch_size:   序列条数（B）。
            sequence_len: 每条序列长度，默认 = rollout_segment_len。
            burn_in_len:  R2D2 burn-in 步数（INVARIANT 13）。

        Returns:
            metrics dict
        """
        self.net.train()
        if sequence_len is None:
            sequence_len = self.rollout_segment_len

        # 获取符合 lag 约束的样本（单步样本列表）
        valid_samples = self.buffer.get_trainable_samples()
        total_needed = batch_size * sequence_len
        if len(valid_samples) < total_needed:
            return {"loss_total": 0.0, "error": "not enough samples"}

        # 随机采样起始索引，组装序列 batch
        max_start = len(valid_samples) - sequence_len
        start_indices = np.random.randint(0, max_start + 1, size=batch_size)

        K = valid_samples[0].get("obs_patch", np.zeros((4, 11, 11))).shape[-1]
        obs_patches_np = np.zeros((batch_size, sequence_len, 4, K, K), dtype=np.float32)
        obs_scalars_np = np.zeros((batch_size, sequence_len, 8), dtype=np.float32)
        actions_np     = np.zeros((batch_size, sequence_len), dtype=np.int64)
        rewards_np     = np.zeros((batch_size, sequence_len), dtype=np.float32)
        dones_np       = np.zeros((batch_size, sequence_len), dtype=bool)
        masks_np       = np.ones((batch_size, sequence_len, 5), dtype=bool)
        next_masks_np  = np.ones((batch_size, sequence_len, 5), dtype=bool)
        expert_np      = np.zeros((batch_size, sequence_len), dtype=np.int64)
        excl_np        = np.zeros((batch_size, sequence_len), dtype=bool)
        wt_np          = np.ones((batch_size, sequence_len), dtype=np.float32)

        for b, si in enumerate(start_indices):
            for t in range(sequence_len):
                s = valid_samples[si + t]
                obs_patches_np[b, t] = s.get("obs_patch", np.zeros((4, K, K)))
                obs_scalars_np[b, t] = s.get("obs_scalar", np.zeros(8))
                actions_np[b, t]     = s.get("action_expert", 0)
                rewards_np[b, t]     = s.get("reward", 0.0)
                dones_np[b, t]       = s.get("done", False)
                masks_np[b, t]       = s.get("action_mask", np.ones(5, dtype=bool))
                next_masks_np[b, t]  = s.get("next_action_mask",
                    valid_samples[si + t + 1].get("action_mask", np.ones(5, dtype=bool))
                    if si + t + 1 < len(valid_samples) else np.ones(5, dtype=bool))
                expert_np[b, t]      = s.get("action_expert", 0)
                excl_np[b, t]        = s.get("margin_excluded", True)  # 自游走默认 True
                wt_np[b, t]          = s.get("margin_weight", 0.0)

        def _t(x, dtype=None):
            t = torch.from_numpy(np.asarray(x))
            return t.to(dtype).to(self.device) if dtype else t.to(self.device)

        obs_patches  = _t(obs_patches_np)                     # (B, T, 4, K, K)
        obs_scalars  = _t(obs_scalars_np)                     # (B, T, 8)
        actions      = _t(actions_np, dtype=torch.long)       # (B, T)
        rewards      = _t(rewards_np)                         # (B, T)
        dones        = _t(dones_np, dtype=torch.bool)         # (B, T)
        masks        = _t(masks_np, dtype=torch.bool)         # (B, T, 5)  — s_t 掩码（margin）
        next_masks   = _t(next_masks_np, dtype=torch.bool)    # (B, T, 5)  — s_{t+1} 掩码（TD）
        expert_acts  = _t(expert_np, dtype=torch.long)
        excl         = _t(excl_np, dtype=torch.bool)
        weights      = _t(wt_np)

        B, T = obs_patches.shape[:2]
        positions = torch.zeros(B, T, 1, 2, device=self.device)
        hidden    = self.net.init_hidden(B, 1, self.device)
        h_target  = self.target_net.init_hidden(B, 1, self.device)

        # 在线网络前向（带 burn-in，INVARIANT 13）
        q_online, _, _ = self.net(
            obs_patches.unsqueeze(2),   # (B, T, 1, 4, K, K)
            obs_scalars.unsqueeze(2),   # (B, T, 1, 8)
            positions,
            hidden,
            burn_in_len=burn_in_len,
            action_masks=masks.unsqueeze(2),
        )  # (B, T, 1, 5)

        with torch.no_grad():
            q_target_out, _, _ = self.target_net(
                obs_patches.unsqueeze(2),
                obs_scalars.unsqueeze(2),
                positions,
                h_target,
                burn_in_len=burn_in_len,
            )  # (B, T, 1, 5)

        q_online_flat  = q_online.reshape(B * T, 5)
        q_target_flat  = q_target_out.reshape(B * T, 5)
        act_flat       = actions.reshape(B * T)
        rew_flat       = rewards.reshape(B * T)
        done_flat      = dones.reshape(B * T)
        nmask_flat     = next_masks.reshape(B * T, 5)
        exp_flat       = expert_acts.reshape(B * T)
        excl_flat      = excl.reshape(B * T)
        wt_flat        = weights.reshape(B * T)

        lam_margin = self.lambda_margin()

        total_loss, loss_info = dqfd_total_loss(
            q_values=q_online_flat,
            target_q_next=q_target_flat,
            actions=act_flat,
            rewards=rew_flat,
            dones=done_flat,
            action_masks_next=nmask_flat,
            expert_actions=exp_flat,
            margin_excluded=excl_flat,
            margin_weights=wt_flat,
            gamma=self.gamma,
            mellowmax_alpha=self.mellowmax_alpha,
            margin=self.margin,
            lambda_td=1.0,
            lambda_nstep=0.0,
            lambda_margin=lam_margin,
            lambda_l2=1e-5,
            parameters=list(self.net.parameters()),
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optimizer.step()

        self.train_step += 1

        # 周期性 target 同步
        if self.train_step % self.target_sync_interval == 0:
            self._sync_target()
            self.buffer.sync()

        # 退火后丢弃过期样本
        self.buffer.discard_old()

        return {
            "loss_total": loss_info.get("loss_total", 0.0),
            "loss_td": loss_info.get("loss_td", 0.0),
            "loss_margin": loss_info.get("loss_margin", 0.0),
            "expert_prob": self.expert_prob(),
            "lambda_margin": lam_margin,
            "train_step": self.train_step,
            "buffer_size": len(self.buffer),
        }

    def _sync_target(self):
        """周期性 target 网络同步（INVARIANT 8：阶段三用标准 DQN target 同步）。"""
        self.target_net.load_state_dict(self.net.state_dict())

    def state_dict(self) -> dict:
        return {
            "net": self.net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step,
        }

    def load_state_dict(self, state: dict):
        self.net.load_state_dict(state["net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.train_step = state.get("train_step", 0)
