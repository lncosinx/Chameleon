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
            # 课程式混合比例（设计文档§3.3.1）
            n_expert   = max(1, int(self.batch_size * exp_ratio))
            n_selfplay = self.batch_size - n_expert

            # 从 expert buffer 采样序列
            expert_batch = None
            n_expert_seqs = max(1, n_expert // self.sequence_len)
            try:
                expert_batch = expert_buffer.sample(
                    batch_size=n_expert_seqs,
                    sequence_len=self.sequence_len,
                    burn_in_len=self.burn_in_len,
                )
            except (ValueError, IndexError):
                pass

            # 从 selfplay buffer 采样单条样本并构成 batch
            selfplay_batch = None
            if selfplay_buffer is not None and len(selfplay_buffer) >= n_selfplay and n_selfplay > 0:
                sp_samples = selfplay_buffer.sample(
                    n=n_selfplay,
                    rng=np.random.default_rng(self.meta_iter * 1000 + k),
                )
                if sp_samples:
                    from buffers.batch_collator import collate_samples_to_tensors
                    selfplay_batch = collate_samples_to_tensors(
                        sp_samples, sequence_len=1,
                        device=self.device, pin_memory=False,
                    )

            # 至少需要 expert_batch 才能训练（margin loss 依赖专家动作）
            if expert_batch is not None:
                loss, info = self._inner_step(
                    expert_batch, target_net, inner_optimizer,
                    aux_batch=selfplay_batch,
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
        aux_batch: dict = None,  # 自游走 batch（只做 TD loss，无 margin）
    ) -> Tuple[torch.Tensor, dict]:
        """单步 inner loop SGD。"""
        self.net.train()

        def _to_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                t = x
            else:
                t = torch.from_numpy(np.asarray(x))
            if dtype is not None:
                t = t.to(dtype)
            return t.to(self.device)

        obs_patches      = _to_tensor(batch["obs_patches"])       # (B, T, 4, K, K)
        obs_scalars      = _to_tensor(batch["obs_scalars"])        # (B, T, 8)
        next_obs_patches = _to_tensor(batch["next_obs_patches"])   # (B, T, 4, K, K)
        next_obs_scalars = _to_tensor(batch["next_obs_scalars"])   # (B, T, 8)
        actions          = _to_tensor(batch["actions_expert"], dtype=torch.long)
        rewards          = _to_tensor(batch["rewards"])            # (B, T)
        dones            = _to_tensor(batch["dones"], dtype=torch.bool)
        action_masks     = _to_tensor(batch["action_masks"], dtype=torch.bool)      # s_t 掩码
        # s_{t+1} 掩码用于 mellowmax target；若 buffer 未提供则回退到 action_masks（I2 修复）
        if "next_action_masks" in batch:
            next_action_masks = _to_tensor(batch["next_action_masks"], dtype=torch.bool)
        else:
            next_action_masks = action_masks
        expert_actions   = _to_tensor(batch["actions_expert"], dtype=torch.long)
        margin_excluded  = batch.get("margin_excluded")
        if margin_excluded is not None:
            margin_excluded = _to_tensor(margin_excluded, dtype=torch.bool)
        margin_weights = batch.get("margin_weights")
        if margin_weights is not None:
            margin_weights = _to_tensor(margin_weights)

        B, T = obs_patches.shape[:2]
        # 每个 batch 样本是单体序列（per-agent），N=1；通信在多体前向时生效
        positions      = torch.zeros(B, T, 1, 2, device=self.device)
        next_positions = torch.zeros(B, T, 1, 2, device=self.device)

        # ---- 在线网络：Q(s, ·) ----
        hidden = self.net.init_hidden(B, 1, self.device)
        q_online, _, _ = self.net(
            obs_patches.unsqueeze(2),      # (B, T, 1, 4, K, K)
            obs_scalars.unsqueeze(2),      # (B, T, 1, 8)
            positions,
            hidden,
            burn_in_len=self.burn_in_len,
            action_masks=action_masks.unsqueeze(2),  # (B, T, 1, 5)
        )  # (B, T, 1, 5)

        # ---- target 网络：Q(s', ·) ——必须用 next_obs ----
        with torch.no_grad():
            next_hidden = target_net.init_hidden(B, 1, self.device)
            q_target, _, _ = target_net(
                next_obs_patches.unsqueeze(2),
                next_obs_scalars.unsqueeze(2),
                next_positions,
                next_hidden,
                burn_in_len=self.burn_in_len,
            )  # (B, T, 1, 5)

        # 展平 (B, T, 1, 5) → (B*T, 5)
        q_online_flat     = q_online.reshape(B * T, 5)
        q_target_flat     = q_target.reshape(B * T, 5)
        actions_flat      = actions.reshape(B * T)
        rewards_flat      = rewards.reshape(B * T)
        dones_flat        = dones.reshape(B * T)
        masks_flat        = action_masks.reshape(B * T, 5)       # s_t 掩码（margin loss 用）
        next_masks_flat   = next_action_masks.reshape(B * T, 5)  # s_{t+1} 掩码（mellowmax 用）
        expert_flat       = expert_actions.reshape(B * T)
        margin_excl_flat  = margin_excluded.reshape(B * T) if margin_excluded is not None else None
        margin_w_flat     = margin_weights.reshape(B * T) if margin_weights is not None else None

        # ---- DQfD 1-step TD + Margin 损失 ----
        # n-step 损失单独计算（其样本数 B*(T-n) ≠ B*T，不能合并进 dqfd_total_loss）
        total_loss, loss_info = dqfd_total_loss(
            q_values=q_online_flat,
            target_q_next=q_target_flat,
            actions=actions_flat,
            rewards=rewards_flat,
            dones=dones_flat,
            action_masks_next=next_masks_flat,  # I2 修复：使用 s_{t+1} 掩码
            expert_actions=expert_flat,
            rewards_seq=None,       # n-step 在下方单独计算
            dones_seq=None,
            v_target_n=None,
            margin_excluded=margin_excl_flat,
            margin_weights=margin_w_flat,
            gamma=self.gamma,
            n_step=self.n_step,
            mellowmax_alpha=self.mellowmax_alpha,
            margin=self.margin,
            lambda_td=self.lambda_td,
            lambda_nstep=0.0,       # 由下方单独项负责
            lambda_margin=self.lambda_margin,
            lambda_l2=self.lambda_l2,
            parameters=list(self.net.parameters()),
        )

        # ---- n-step TD 损失（滑动窗口，仅前 T-n 步可计算）----
        n = self.n_step
        if T > n and self.lambda_nstep > 0:
            # 滑动窗口：rewards_seq (B, T-n, n)
            rewards_seq = torch.stack(
                [rewards[:, i:T - n + i] for i in range(n)], dim=-1
            )  # (B, T-n, n)
            dones_seq = torch.stack(
                [dones[:, i:T - n + i].float() for i in range(n)], dim=-1
            )  # (B, T-n, n)
            # V_target(s_{t+n})：使用 t+n 时刻的 next 掩码（action_masks[:, n:]）
            v_target_n = masked_mellowmax(
                q_target.reshape(B, T, 5)[:, n:],     # (B, T-n, 5)
                action_masks[:, n:],                   # (B, T-n, 5)
                alpha=self.mellowmax_alpha,
            )  # (B, T-n)

            # 正确展平：(B, T-n, 5) → (B*(T-n), 5)，按 batch-major 顺序
            # 不能直接取 q_online_flat[:B*(T-n)]（包含 batch0 所有 T 步而非各 batch 前 T-n 步）
            q_nstep  = q_online.reshape(B, T, 5)[:, :T - n, :].reshape(B * (T - n), 5)
            a_nstep  = actions.reshape(B, T)[:, :T - n].reshape(B * (T - n))
            rs_flat  = rewards_seq.reshape(B * (T - n), n)
            ds_flat  = dones_seq.reshape(B * (T - n), n)
            vt_flat  = v_target_n.reshape(B * (T - n))

            from losses.td_losses import n_step_td_loss
            loss_n = n_step_td_loss(q_nstep, a_nstep, rs_flat, ds_flat, vt_flat,
                                    gamma=self.gamma, n_step=n)
            total_loss = total_loss + self.lambda_nstep * loss_n
            loss_info["loss_nstep"] = loss_n.item()
        else:
            loss_info["loss_nstep"] = 0.0

        # ---- 自游走 batch 补充 TD（课程式混合，无 margin）----
        if aux_batch is not None:
            try:
                aux_obs   = _to_tensor(aux_batch["obs_patches"])
                aux_sc    = _to_tensor(aux_batch["obs_scalars"])
                aux_acts  = _to_tensor(aux_batch["actions"], dtype=torch.long)
                aux_rews  = _to_tensor(aux_batch["rewards"])
                aux_dones = _to_tensor(aux_batch["dones"], dtype=torch.bool)
                _amask_np = aux_batch.get("action_masks")
                aux_masks = (_to_tensor(_amask_np, dtype=torch.bool)
                             if _amask_np is not None
                             else torch.ones(*aux_acts.shape, 5, dtype=torch.bool, device=self.device))

                B_sp = aux_obs.shape[0]
                T_sp = aux_obs.shape[1] if aux_obs.dim() == 5 else 1
                pos_sp = torch.zeros(B_sp, T_sp, 1, 2, device=self.device)
                h_sp   = self.net.init_hidden(B_sp, 1, self.device)
                h_t_sp = target_net.init_hidden(B_sp, 1, self.device)

                # (B_sp, T_sp, 4, K, K) → unsqueeze N=1
                obs_in = aux_obs.unsqueeze(-4) if aux_obs.dim() == 4 else aux_obs
                sc_in  = aux_sc.unsqueeze(-3)  if aux_sc.dim()  == 3 else aux_sc

                with torch.no_grad():
                    q_t_sp, _, _ = target_net(obs_in, sc_in, pos_sp, h_t_sp)

                q_sp, _, _ = self.net(obs_in, sc_in, pos_sp, h_sp)

                # 使用模块级导入（不在此处重复 import，避免遮蔽外层绑定）
                mask_sp_flat = aux_masks.reshape(-1, 5)
                v_sp = masked_mellowmax(q_t_sp.reshape(-1, 5), mask_sp_flat,
                                        alpha=self.mellowmax_alpha)
                from losses.td_losses import one_step_td_loss as _1step_loss
                sp_td_loss = _1step_loss(
                    q_sp.reshape(-1, 5), aux_acts.reshape(-1),
                    aux_rews.reshape(-1), aux_dones.reshape(-1), v_sp, self.gamma,
                )
                total_loss = total_loss + self.lambda_td * sp_td_loss
                loss_info["loss_sp_td"] = sp_td_loss.item()
            except Exception:
                pass  # 自游走 batch 格式不兼容时静默跳过

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        optimizer.step()

        return total_loss, loss_info

    def state_dict(self) -> dict:
        return {
            "net": self.net.state_dict(),
            "meta_iter": self.meta_iter,
        }

    def load_state_dict(self, state: dict):
        self.net.load_state_dict(state["net"])
        self.meta_iter = state.get("meta_iter", 0)
