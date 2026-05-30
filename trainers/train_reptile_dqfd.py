"""
DQfD + Reptile 联合训练器（单 GPU）。

Reptile 元学习：外层更新 θ ← θ + ε·(θ_inner - θ)
内层 K=5 步串行 SGD，target 冻结于 θ_meta。
DQfD 总损失：L = λ_td·L_TD + λ_n·L_n + λ_margin·L_margin + λ_l2·||θ||²

INVARIANT 8: target 在 inner loop 内冻结于 θ_meta
INVARIANT 13: burn_in_len=8 步 R2D2 burn-in
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import time
import copy
from typing import Dict, Optional, Tuple
from copy import deepcopy

from models.biintent_drqn import BiIntentGATDRQN
from losses.dqfd_losses import dqfd_total_loss
from losses.q_target import masked_mellowmax
from trainers.scheduler import MapScheduler, expert_ratio
from utils.checkpoint import save_checkpoint, load_checkpoint


class ReptileDQfDTrainer:
    """
    单 GPU Reptile + DQfD 训练器。

    用法:
        trainer = ReptileDQfDTrainer(net, cfg)
        for meta_iter in range(total_iters):
            metrics = trainer.meta_step(expert_buffer, selfplay_buffer, env_factory)
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

        # 超参
        self.inner_lr = config.get("inner_lr", 3e-4)
        self.meta_lr = config.get("meta_lr", 0.05)
        self.inner_steps = config.get("inner_steps", 5)
        self.batch_size = config.get("batch_size", 32)
        self.sequence_len = config.get("sequence_len", 32)
        self.burn_in_len = config.get("burn_in_len", 8)
        self.gamma = config.get("gamma", 0.99)
        self.n_step = config.get("n_step", 5)
        self.mellowmax_alpha = config.get("mellowmax_alpha", 1.0)
        self.margin = config.get("margin", 0.5)
        self.lambda_td = config.get("lambda_td", 1.0)
        self.lambda_nstep = config.get("lambda_nstep", 1.0)
        self.lambda_margin = config.get("lambda_margin", 1.0)
        self.lambda_l2 = config.get("lambda_l2", 1e-5)

        # 课程参数
        self.expert_ratio_start = config.get("expert_ratio_start", 0.9)
        self.expert_ratio_end = config.get("expert_ratio_end", 0.5)
        self.expert_ratio_decay_iters = config.get("expert_ratio_decay_iters", 8000)

        # optimizer（outer loop 用 SGD，inner 用 Adam）
        self.meta_optimizer = torch.optim.SGD(self.net.parameters(), lr=self.meta_lr)

        # 地图调度器
        self.scheduler = MapScheduler(config.get("map_schedule", None))

        # 统计
        self.meta_iter = 0
        self.total_meta_iters = config.get("meta_iterations", 20000)

    def meta_step(
        self,
        expert_buffer,      # ExpertDemoBuffer
        selfplay_buffer,    # SelfPlayBuffer
        env_factory,        # callable: (map_type, seed) → PogemaWrapper
        seed: int = None,
    ) -> Dict[str, float]:
        """
        执行一次 meta-iteration：采样 task → K 步 inner loop → outer update。

        Returns:
            metrics dict
        """
        # 保存 meta 参数 θ
        theta_meta = {k: v.clone() for k, v in self.net.state_dict().items()}

        # 采样地图类型
        map_type = self.scheduler.sample(
            self.meta_iter, self.total_meta_iters, adaptive=False
        )

        # 采样任务（生成环境 + 专家数据或自游走）
        task_seed = seed or (100000 + self.meta_iter)
        exp_ratio = expert_ratio(
            self.meta_iter, self.total_meta_iters,
            self.expert_ratio_start, self.expert_ratio_end,
            self.expert_ratio_decay_iters,
        )

        # ---- Inner loop (K steps) ----
        # θ_target 冻结于 θ_meta（INVARIANT 8）
        target_net = deepcopy(self.net)
        target_net.load_state_dict(theta_meta)
        target_net.eval()
        for p in target_net.parameters():
            p.requires_grad = False

        inner_optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.inner_lr
        )

        inner_losses = []
        for k in range(self.inner_steps):
            # 采样混合 batch
            n_expert = int(self.batch_size * exp_ratio)
            n_selfplay = self.batch_size - n_expert

            # 从 expert buffer 采样
            try:
                expert_batch = expert_buffer.sample(
                    batch_size=max(1, n_expert // self.sequence_len),
                    sequence_len=self.sequence_len,
                    burn_in_len=self.burn_in_len,
                )
            except (ValueError, IndexError):
                # buffer 不够大，减少用量或跳过
                expert_batch = None

            # 简化：直接用 expert_batch 训练
            if expert_batch is not None:
                loss, info = self._inner_step(
                    expert_batch, target_net, inner_optimizer
                )
                inner_losses.append(info.get("loss_total", 0.0))

        # ---- Outer update ----
        # θ ← θ + ε · (θ_inner - θ_meta)
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                meta_param = theta_meta[name]
                # Reptile: θ = θ + ε*(θ - θ_meta)
                # 由于我们现在是 θ = θ_inner（inner loop 已在 θ 上做 SGD），
                # 所以更新为 θ_meta + ε*(θ - θ_meta)
                param.copy_(meta_param + self.meta_lr * (param - meta_param))

        self.meta_iter += 1

        metrics = {
            "meta_iter": self.meta_iter,
            "map_type": map_type,
            "expert_ratio": exp_ratio,
            "inner_loss_mean": np.mean(inner_losses) if inner_losses else 0.0,
        }

        return metrics

    def _inner_step(
        self,
        batch: dict,
        target_net: BiIntentGATDRQN,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[torch.Tensor, dict]:
        """单步 inner loop SGD。"""
        self.net.train()

        # 搬数据到 GPU（兼容 numpy array 和 tensor）
        def _to_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                t = x
            else:
                t = torch.from_numpy(np.asarray(x))
            if dtype is not None:
                t = t.to(dtype)
            return t.to(self.device)

        obs_patches = _to_tensor(batch["obs_patches"])  # (B, T, 4, K, K)
        obs_scalars = _to_tensor(batch["obs_scalars"])
        actions = _to_tensor(batch["actions_expert"], dtype=torch.long)
        rewards = _to_tensor(batch["rewards"])
        dones = _to_tensor(batch["dones"], dtype=torch.bool)
        action_masks = _to_tensor(batch["action_masks"], dtype=torch.bool)
        expert_actions = _to_tensor(batch["actions_expert"], dtype=torch.long)
        margin_excluded = batch.get("margin_excluded")
        if margin_excluded is not None:
            margin_excluded = _to_tensor(margin_excluded, dtype=torch.bool)
        margin_weights = batch.get("margin_weights")
        if margin_weights is not None:
            margin_weights = _to_tensor(margin_weights)

        # 需要 positions 用于通信——从 patch 推导（简化：用零向量，通信暂时退化）
        B, T = obs_patches.shape[:2]
        N = 1  # batch 中是 per-agent 还是 per-episode?
        # 实际 batch 格式：每个样本是单人 (4,K,K)，batch 是 (B, T, ...)
        # 通信模块需要 (B, T, N, 2) positions
        positions = torch.zeros(B, T, 1, 2, device=self.device)

        # 前向（在线网络）
        hidden = self.net.init_hidden(B, 1, self.device)
        q_online, _, q_info = self.net(
            obs_patches.unsqueeze(2),     # (B, T, 1, 4, K, K)
            obs_scalars.unsqueeze(2),     # (B, T, 1, 8)
            positions,
            hidden,
            burn_in_len=self.burn_in_len,
        )  # (B, T, 1, 5)

        # 前向（target 网络）
        with torch.no_grad():
            # next_obs 需要从 batch 获取，这里简化为用当前 obs
            q_target, _, _ = target_net(
                obs_patches.unsqueeze(2),
                obs_scalars.unsqueeze(2),
                positions,
                hidden,
                burn_in_len=self.burn_in_len,
            )

        # 展平时间维
        q_online_flat = q_online.reshape(B * T, 5)       # (B*T, 5)
        q_target_flat = q_target.reshape(B * T, 5)       # (B*T, 5)
        actions_flat = actions.reshape(B * T)              # (B*T,)
        rewards_flat = rewards.reshape(B * T)
        dones_flat = dones.reshape(B * T)
        masks_flat = action_masks.reshape(B * T, 5)
        expert_flat = expert_actions.reshape(B * T)
        margin_excl_flat = margin_excluded.reshape(B * T) if margin_excluded is not None else None
        margin_w_flat = margin_weights.reshape(B * T) if margin_weights is not None else None

        # DQfD 总损失
        total_loss, loss_info = dqfd_total_loss(
            q_values=q_online_flat,
            target_q_next=q_target_flat,
            actions=actions_flat,
            rewards=rewards_flat,
            dones=dones_flat,
            action_masks_next=masks_flat,
            expert_actions=expert_flat,
            margin_excluded=margin_excl_flat,
            margin_weights=margin_w_flat,
            gamma=self.gamma,
            mellowmax_alpha=self.mellowmax_alpha,
            margin=self.margin,
            lambda_td=self.lambda_td,
            lambda_nstep=self.lambda_nstep,
            lambda_margin=self.lambda_margin,
            lambda_l2=self.lambda_l2,
            parameters=list(self.net.parameters()),
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        optimizer.step()

        return total_loss, loss_info

    def state_dict(self) -> dict:
        return {
            "net": self.net.state_dict(),
            "meta_optimizer": self.meta_optimizer.state_dict(),
            "meta_iter": self.meta_iter,
        }

    def load_state_dict(self, state: dict):
        self.net.load_state_dict(state["net"])
        self.meta_optimizer.load_state_dict(state["meta_optimizer"])
        self.meta_iter = state.get("meta_iter", 0)
