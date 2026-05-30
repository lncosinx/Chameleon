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

    def train_on_segment(self, batch_size: int = 32) -> Dict[str, float]:
        """
        在最新 segment 数据上训练。

        Returns:
            metrics dict
        """
        self.net.train()

        # 获取符合 lag 约束的样本
        valid_samples = self.buffer.get_trainable_samples()
        if len(valid_samples) < batch_size:
            return {"loss_total": 0.0, "error": "not enough samples"}

        # 随机采样 batch
        indices = np.random.choice(len(valid_samples), batch_size, replace=False)
        batch_samples = [valid_samples[i] for i in indices]

        # 转为 tensors（简化：使用 collator）
        from buffers.batch_collator import collate_samples_to_tensors
        batch = collate_samples_to_tensors(batch_samples, sequence_len=1, device=self.device)

        # 前向（在线网络 + target 网络）
        # 简化单步训练模式
        obs_patches = batch["obs_patches"].squeeze(1)  # (B, 4, K, K)
        obs_scalars = batch["obs_scalars"].squeeze(1)
        actions = batch["actions"].squeeze(1)
        rewards = batch["rewards"].squeeze(1)
        dones = batch["dones"].squeeze(1)
        masks = batch["action_masks"].squeeze(1)
        expert_acts = batch["expert_actions"].squeeze(1)

        # 添加 agent 和 time 维度
        B = obs_patches.shape[0]
        positions = torch.zeros(B, 1, 1, 2, device=self.device)
        hidden = self.net.init_hidden(B, 1, self.device)

        q_online, _, _ = self.net(
            obs_patches.unsqueeze(1).unsqueeze(2),  # (B, 1, 1, 4, K, K)
            obs_scalars.unsqueeze(1).unsqueeze(2),   # (B, 1, 1, 8)
            positions,
            hidden,
        )

        with torch.no_grad():
            q_target, _, _ = self.target_net(
                obs_patches.unsqueeze(1).unsqueeze(2),
                obs_scalars.unsqueeze(1).unsqueeze(2),
                positions,
                hidden,
            )

        # 展平
        q_online_flat = q_online.reshape(B, 5)
        q_target_flat = q_target.reshape(B, 5)

        lam_margin = self.lambda_margin()

        total_loss, loss_info = dqfd_total_loss(
            q_values=q_online_flat,
            target_q_next=q_target_flat,
            actions=actions.long(),
            rewards=rewards,
            dones=dones,
            action_masks_next=masks,
            expert_actions=expert_acts.long(),
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
