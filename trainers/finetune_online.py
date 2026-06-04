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
from trainers.train_reptile_dqfd import dqfd_sequence_loss
from buffers.recent_online_buffer import RecentOnlineBuffer, SEGMENT_KEYS


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
        self.burn_in_len = config.get("burn_in_len", 8)
        self.max_policy_version_lag = config.get("max_policy_version_lag", 1)
        self.reuse_epochs_per_segment = config.get("reuse_epochs_per_segment", 1)
        self.gamma = config.get("gamma", 0.99)
        self.mellowmax_alpha = config.get("mellowmax_alpha", 1.0)
        self.margin = config.get("margin", 0.5)
        self.grad_clip = config.get("grad_clip", 10.0)

        # 退火参数。expert_decay_steps 单位为「训练 segment 数」（train_step 每段 +1），
        # 应取 --num-segments 的 ~80%，退火才能在训练过程中真正完成。
        self.expert_prob_start = config.get("expert_prob_start", 0.5)
        self.expert_prob_end = config.get("expert_prob_end", 0.02)
        self.expert_decay_steps = config.get("expert_decay_steps", 4000)
        self.lambda_margin_start = config.get("lambda_margin_start", 1.0)
        self.lambda_margin_end = config.get("lambda_margin_end", 0.0)

        # Optimizer（lr 进 config，勿硬编码，CLAUDE.md 开发约定）
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=config.get("lr", 3.0e-5)
        )

        # AMP（§4.2）
        _par = config.get("parallel", {}) or {}
        self.use_amp = bool(config.get("use_amp", _par.get("use_amp", False)))
        self.use_amp = self.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Buffer
        self.buffer = RecentOnlineBuffer(
            capacity=config.get("buffer_capacity_segments", 128),
            max_policy_version_lag=self.max_policy_version_lag,
        )

        self.train_step = 0
        # 周期性 target 同步（标准 DQN，INVARIANT 8）；单位为训练 segment 数
        self.target_sync_interval = config.get("target_sync_interval", 100)

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

    def add_rollout_samples(self, segment: dict, policy_version: int):
        """添加一段同步采样 rollout（per-timestep × N-agent segment dict）。"""
        self.buffer.add_segment(segment, policy_version)

    def _collate_windows(self, segs: list, batch_size: int, seq_len: int,
                         rng: np.random.Generator):
        """
        从 trainable segments 抽取 B 条长度 seq_len 的序列窗口，组成 (B,T,N,...) batch。

        窗口不跨 segment 边界（每个 segment 是单段同步 rollout，即一个 episode 片段）。
        """
        # 只用长度足够的 segment；都不够则取最长者并把 seq_len 收缩到它
        usable = [s for s in segs if s["actions_expert"].shape[0] >= seq_len]
        if not usable:
            longest = max(segs, key=lambda s: s["actions_expert"].shape[0])
            seq_len = int(longest["actions_expert"].shape[0])
            usable = [s for s in segs if s["actions_expert"].shape[0] >= seq_len]

        N = usable[0]["actions_expert"].shape[1]
        out = {k: [] for k in SEGMENT_KEYS}
        for _ in range(batch_size):
            seg = usable[int(rng.integers(0, len(usable)))]
            T_seg = seg["actions_expert"].shape[0]
            start = int(rng.integers(0, T_seg - seq_len + 1))
            for k in SEGMENT_KEYS:
                out[k].append(np.asarray(seg[k])[start:start + seq_len])
        batch = {k: np.stack(out[k], axis=0) for k in SEGMENT_KEYS}  # (B,T,N,...)
        return batch, seq_len, N

    def train_on_segment(
        self, batch_size: int = 32,
        sequence_len: int = None,
        burn_in_len: int = None,
    ) -> Dict[str, float]:
        """
        在 lag 内的 segment 上以多体序列模式训练（DRQN + R2D2 burn-in）。

        与阶段二共享 dqfd_sequence_loss：传入真实多体坐标 → 通信参数获得梯度；
        target 用周期同步网络 + next_obs/next_positions；finished agent 由 valid 屏蔽。
        margin 权重随 self.lambda_margin() 退火（末期→纯近 on-policy Q 学习）。

        lag 约束（INVARIANT 9）：用 self.train_step 作为唯一策略版本计数器，
        每段采集时以当前 train_step 打标，训练时丢弃 train_step − version > max_lag 的段。
        reuse_epochs_per_segment：对当前可训练段做多次梯度更新（近 on-policy 默认 1 次）。
        """
        self.net.train()
        if sequence_len is None:
            sequence_len = self.rollout_segment_len
        if burn_in_len is None:
            burn_in_len = self.burn_in_len

        # lag 过滤：以 train_step 为统一版本（与 add_rollout_samples 打标一致）
        segs = self.buffer.get_trainable_segments(learner_version=self.train_step)
        if not segs or self.buffer.num_timesteps() < sequence_len:
            return {"loss_total": 0.0, "error": "not enough samples"}

        def _t(x, dtype=None):
            t = torch.from_numpy(np.asarray(x))
            return t.to(dtype).to(self.device) if dtype else t.to(self.device)

        rng = np.random.default_rng()
        lam_margin = self.lambda_margin()
        last_info = {}

        # 近 on-policy：每段做 reuse_epochs_per_segment 次梯度更新（默认 1）
        for _ in range(max(1, self.reuse_epochs_per_segment)):
            batch, T_eff, N = self._collate_windows(segs, batch_size, sequence_len, rng)
            total_loss, loss_info = dqfd_sequence_loss(
                self.net, self.target_net,
                obs_patches=_t(batch["obs_patches"]),
                obs_scalars=_t(batch["obs_scalars"]),
                positions=_t(batch["positions"]),
                next_obs_patches=_t(batch["next_obs_patches"]),
                next_obs_scalars=_t(batch["next_obs_scalars"]),
                next_positions=_t(batch["next_positions"]),
                actions=_t(batch["actions_expert"], dtype=torch.long),
                rewards=_t(batch["rewards"]),
                dones=_t(batch["dones"], dtype=torch.bool),
                valid=_t(batch["valid"], dtype=torch.bool),
                action_masks=_t(batch["action_masks"], dtype=torch.bool),
                next_action_masks=_t(batch["next_action_masks"], dtype=torch.bool),
                margin_excluded=_t(batch["margin_excluded"], dtype=torch.bool),
                margin_weights=_t(batch["margin_weights"]),
                gamma=self.gamma, n_step=1, mellowmax_alpha=self.mellowmax_alpha,
                margin=self.margin, lambda_td=1.0, lambda_nstep=0.0,
                lambda_margin=lam_margin, lambda_l2=1e-5,
                burn_in_len=burn_in_len, parameters=list(self.net.parameters()),
                use_amp=self.use_amp,
            )
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            last_info = loss_info

        # train_step = 策略版本，每段（每次同步采样）+1
        self.train_step += 1

        # 周期性 target 同步（标准 DQN，INVARIANT 8）
        if self.train_step % self.target_sync_interval == 0:
            self._sync_target()

        # 按统一版本（train_step）丢弃 lag 超限的 segment
        self.buffer.discard_old(learner_version=self.train_step)

        return {
            "loss_total": last_info.get("loss_total", 0.0),
            "loss_td": last_info.get("loss_td", 0.0),
            "loss_margin": last_info.get("loss_margin", 0.0),
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
