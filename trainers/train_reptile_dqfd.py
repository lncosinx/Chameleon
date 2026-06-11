"""
DQfD + Reptile 联合训练器。

- ReptileDQfDTrainer   — 单 GPU 版
- DualGPUReptileTrainer — 双 GPU task-parallel 版（§4.4/§4.5）

Reptile 外层更新 θ ← θ + ε · avg_j(θ_inner_j − θ)
内层 K=5 步串行 SGD，target 冻结于 θ_meta（INVARIANT 8）。
DQfD 总损失：L = λ_td·L_TD + λ_n·L_n + λ_margin·L_margin + λ_l2·||θ||²
"""
from __future__ import annotations

import threading
import torch
import torch.nn as nn
import numpy as np
import copy
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

from models.biintent_drqn import BiIntentGATDRQN
from losses.dqfd_losses import dqfd_total_loss
from losses.q_target import masked_mellowmax, masked_double_q_target, masked_clipped_double_q, masked_max
from losses.td_losses import n_step_td_loss
from trainers.scheduler import MapScheduler, expert_ratio
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.profiler import Profiler


def dqfd_sequence_loss(
    net: "BiIntentGATDRQN",
    target_net: "BiIntentGATDRQN",
    *,
    obs_patches: torch.Tensor,        # (B, T, N, 4, K, K)
    obs_scalars: torch.Tensor,        # (B, T, N, 8)
    positions: torch.Tensor,          # (B, T, N, 2)  真实坐标（finished→远哨兵）
    next_obs_patches: torch.Tensor,
    next_obs_scalars: torch.Tensor,
    next_positions: torch.Tensor,
    actions: torch.Tensor,            # (B, T, N) long
    rewards: torch.Tensor,            # (B, T, N)
    dones: torch.Tensor,              # (B, T, N) bool
    valid: torch.Tensor,              # (B, T, N) bool —— finished agent=False，零权屏蔽
    action_masks: torch.Tensor,       # (B, T, N, 5) bool —— s_t 掩码
    next_action_masks: torch.Tensor,  # (B, T, N, 5) bool —— s_{t+1} 掩码
    margin_excluded: torch.Tensor = None,  # (B, T, N) bool
    margin_weights: torch.Tensor = None,   # (B, T, N) float
    expert_actions: torch.Tensor = None,   # (B, T, N) long，默认=actions
    gamma: float = 0.99,
    n_step: int = 5,
    mellowmax_alpha: float = 1.0,
    margin: float = 0.5,
    lambda_td: float = 1.0,
    lambda_nstep: float = 1.0,
    lambda_margin: float = 1.0,
    lambda_l2: float = 1e-5,
    burn_in_len: int = 8,
    parameters: list = None,
    use_amp: bool = False,
    reward_scale: float = 1.0,
):
    """
    共享的多体序列 DQfD 损失（阶段二 inner loop 与阶段三微调复用）。

    关键点（详见 buffers/expert_demo_buffer.py、CLAUDE.md INVARIANT 8/13）：
    - 前向传入真实多体坐标，通信邻居在 FOV 内生效 → 通信参数获得梯度；
    - 训练时不传 action_masks（避免 Q(s,a_expert)=-inf）；reset_comm=True 序列起始重置迟滞；
    - target 用 next_obs + next_positions；
    - (B,T,N) 折叠为有效序列批 (B*N, T)，GRU/TD/n-step 对每体独立；
    - finished agent 由 valid→sample_weights/margin 屏蔽（零贡献）。
    - use_amp：在 autocast 下跑前向+loss（entmax/mellowmax 内部强制 fp32 保数值稳定）；
      backward/scaler 由调用方负责。

    Returns: (total_loss, loss_info)
    """
    device = obs_patches.device
    B, T, N = obs_patches.shape[:3]
    if expert_actions is None:
        expert_actions = actions

    amp_enabled = use_amp and device.type == "cuda"
    _amp_ctx = torch.autocast(device_type=device.type, enabled=amp_enabled)

    net.train()
    with _amp_ctx:
        return _dqfd_sequence_loss_inner(
            net, target_net, device, B, T, N,
            obs_patches, obs_scalars, positions,
            next_obs_patches, next_obs_scalars, next_positions,
            actions, rewards, dones, valid, action_masks, next_action_masks,
            margin_excluded, margin_weights, expert_actions,
            gamma, n_step, mellowmax_alpha, margin,
            lambda_td, lambda_nstep, lambda_margin, lambda_l2, burn_in_len, parameters,
            reward_scale,
        )


def _dqfd_sequence_loss_inner(
    net, target_net, device, B, T, N,
    obs_patches, obs_scalars, positions,
    next_obs_patches, next_obs_scalars, next_positions,
    actions, rewards, dones, valid, action_masks, next_action_masks,
    margin_excluded, margin_weights, expert_actions,
    gamma, n_step, mellowmax_alpha, margin,
    lambda_td, lambda_nstep, lambda_margin, lambda_l2, burn_in_len, parameters,
    reward_scale=1.0,
):
    """dqfd_sequence_loss 的计算体（已在 autocast 上下文内）。"""
    hidden = net.init_hidden(B, N, device)
    q_online, _, _ = net(
        obs_patches, obs_scalars, positions, hidden,
        burn_in_len=burn_in_len, action_masks=None, reset_comm=True,
    )  # (B, T, N, 5)

    with torch.no_grad():
        next_hidden = target_net.init_hidden(B, N, device)
        q_target, _, _ = target_net(
            next_obs_patches, next_obs_scalars, next_positions, next_hidden,
            burn_in_len=burn_in_len, reset_comm=True,
        )  # (B, T, N, 5)

    # 折叠 (B,T,N,*) → (B*N, T, *)
    def _bn_t(x):
        x = x.transpose(1, 2).contiguous()
        return x.reshape(B * N, T, *x.shape[3:])

    q_bn      = _bn_t(q_online)
    qt_bn     = _bn_t(q_target)
    act_bn    = _bn_t(actions)
    rew_bn    = _bn_t(rewards)
    if reward_scale != 1.0:
        rew_bn = rew_bn * reward_scale   # reward 降尺度（专家+自游走同处同系数）→ Q→c·Q，策略不变
    done_bn   = _bn_t(dones)
    valid_bn  = _bn_t(valid)
    mask_bn   = _bn_t(action_masks)
    nmask_bn  = _bn_t(next_action_masks)
    expert_bn = _bn_t(expert_actions)
    mexcl_bn  = _bn_t(margin_excluded) if margin_excluded is not None else None
    mwt_bn    = _bn_t(margin_weights)  if margin_weights  is not None else None

    BN = B * N
    q_online_flat   = q_bn.reshape(BN * T, 5)
    q_target_flat   = qt_bn.reshape(BN * T, 5)
    actions_flat    = act_bn.reshape(BN * T)
    rewards_flat    = rew_bn.reshape(BN * T)
    dones_flat      = done_bn.reshape(BN * T)
    valid_flat      = valid_bn.reshape(BN * T).float()
    next_masks_flat = nmask_bn.reshape(BN * T, 5)
    expert_flat     = expert_bn.reshape(BN * T)

    margin_excl_flat = mexcl_bn.reshape(BN * T) if mexcl_bn is not None else None
    valid_bool_flat  = valid_bn.reshape(BN * T).bool()
    if margin_excl_flat is not None:
        margin_excl_flat = margin_excl_flat | (~valid_bool_flat)
    else:
        margin_excl_flat = ~valid_bool_flat
    margin_w_flat = (mwt_bn.reshape(BN * T) if mwt_bn is not None
                     else torch.ones(BN * T, device=device))
    margin_w_flat = margin_w_flat * valid_flat

    total_loss, loss_info = dqfd_total_loss(
        q_values=q_online_flat,
        target_q_next=q_target_flat,
        actions=actions_flat,
        rewards=rewards_flat,
        dones=dones_flat,
        action_masks_next=next_masks_flat,
        expert_actions=expert_flat,
        rewards_seq=None, dones_seq=None, v_target_n=None,
        margin_excluded=margin_excl_flat,
        margin_weights=margin_w_flat,
        sample_weights=valid_flat,
        gamma=gamma, n_step=n_step, mellowmax_alpha=mellowmax_alpha, margin=margin,
        lambda_td=lambda_td, lambda_nstep=0.0,   # n-step 单独计算
        lambda_margin=lambda_margin, lambda_l2=lambda_l2,
        parameters=parameters,
    )

    # n-step TD（滑动窗口，仅前 T-n 步）
    n = n_step
    if T > n and lambda_nstep > 0:
        rewards_seq = torch.stack([rew_bn[:, i:T - n + i] for i in range(n)], dim=-1)
        dones_seq   = torch.stack([done_bn[:, i:T - n + i].float() for i in range(n)], dim=-1)
        # qt_bn[k]=Q_target(s_{k+1}); bootstrap 状态 s_n..s_{T-1} ↔ qt_bn[:, n-1:T-1]
        # masked max bootstrap（标准 Q-learning）：取贪婪动作值，避免 mellowmax 拖负。
        v_target_n = masked_max(
            qt_bn[:, n - 1:T - 1], mask_bn[:, n:],
        )
        q_nstep = q_bn[:, :T - n, :].reshape(BN * (T - n), 5)
        a_nstep = act_bn[:, :T - n].reshape(BN * (T - n))
        rs_flat = rewards_seq.reshape(BN * (T - n), n)
        ds_flat = dones_seq.reshape(BN * (T - n), n)
        vt_flat = v_target_n.reshape(BN * (T - n))
        w_nstep = valid_bn[:, :T - n].reshape(BN * (T - n)).float()
        loss_n = n_step_td_loss(q_nstep, a_nstep, rs_flat, ds_flat, vt_flat,
                                gamma=gamma, n_step=n, weights=w_nstep)
        total_loss = total_loss + lambda_nstep * loss_n
        loss_info["loss_nstep"] = loss_n.item()
    else:
        loss_info["loss_nstep"] = 0.0

    # loss_total 必须含 n-step（dqfd_total_loss 内以 lambda_nstep=0 计，n-step 在此单独累加）；
    # 否则报告的 loss 漏掉主导项，严重误导对收敛/发散的判断。
    loss_info["loss_total"] = total_loss.item()

    # reward 诊断（post reward_scale，按 valid 过滤）：PBRS shaping 每步~+1，return~30-80，
    # ×scale 后即合法 Q 尺度，便于判断 Qmax 是真高估还是合法大 return。
    with torch.no_grad():
        _vm = valid_bn.bool()
        _rv = rew_bn[_vm] if _vm.any() else rew_bn
        loss_info["r_mean"] = _rv.mean().item()
        loss_info["r_max"]  = _rv.max().item()

    return total_loss, loss_info


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
        # meta_lr 线性衰减：兼容旧版单值写法（若只有 meta_lr 则 start=end=该值）
        _legacy_lr = config.get("meta_lr", None)
        self.meta_lr_start = config.get("meta_lr_start", _legacy_lr if _legacy_lr is not None else 0.01)
        self.meta_lr_end   = config.get("meta_lr_end",   _legacy_lr if _legacy_lr is not None else 0.001)
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
        self.reward_scale = config.get("reward_scale", 1.0)
        self.grad_clip = config.get("grad_clip", 10.0)

        # AMP（§4.2）：autocast 前向/loss + GradScaler。键可在顶层或 parallel 下。
        _par = config.get("parallel", {}) or {}
        self.use_amp = bool(config.get("use_amp", _par.get("use_amp", False)))
        self.use_amp = self.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # 课程参数
        self.expert_ratio_start = config.get("expert_ratio_start", 0.9)
        self.expert_ratio_end = config.get("expert_ratio_end", 0.5)
        self.expert_ratio_decay_iters = config.get("expert_ratio_decay_iters", 8000)
        self.bc_iters = config.get("bc_iters", 0)   # 两阶段 BC 步数；er 衰减时钟偏移它

        # 地图调度器
        self.scheduler = MapScheduler(config.get("map_schedule", None))

        # 统计
        self.meta_iter = 0
        self.total_meta_iters = config.get("meta_iterations", 20000)

        # 性能诊断
        self.profiler          = Profiler() if config.get("debug", False) else None
        self._profile_interval = config.get("profile_interval", 200)

    def _current_meta_lr(self) -> float:
        """线性衰减 meta_lr：meta_lr_start → meta_lr_end。"""
        t = self.meta_iter / max(1, self.total_meta_iters - 1)
        return self.meta_lr_start + (self.meta_lr_end - self.meta_lr_start) * t

    def run_inner_loop(
        self,
        expert_buffer,
        selfplay_buffer,
        theta_meta_cpu: Dict[str, torch.Tensor],
        exp_ratio_val: float,
        buffer_lock=None,       # threading.Lock 或 None（单 GPU 不需要）
        seed: int = None,
        pre_batches: Optional[List] = None,  # 预采样 expert batch 列表（DualGPU 优化路径）
    ) -> Dict:
        """
        在当前 GPU 上执行 K 步 inner loop，返回 θ_inner（CPU state dict）。

        不做 outer update——由调用方（meta_step 或 DualGPUReptileTrainer）负责。
        这是 task-parallel 的执行单元：每个 GPU 在各自线程中独立调用此方法。

        Args:
            theta_meta_cpu:  起始参数（CPU tensors），每次 meta-iter 重置。
            exp_ratio_val:   当前专家混合比例。
            buffer_lock:     多线程时保护 buffer 采样的锁；None = 单线程不加锁。
            seed:            RNG 种子（保证两卡采样不同 batch）。

        Returns:
            dict with keys:
              theta_inner: CPU state dict (θ after K inner steps)
              loss_mean:   inner loop 平均 loss
        """
        rng = np.random.default_rng(seed)
        ctx = buffer_lock if buffer_lock is not None else nullcontext()

        # 将 θ_meta 加载到本 GPU
        self.net.load_state_dict(
            {k: v.to(self.device) for k, v in theta_meta_cpu.items()}
        )
        # cuDNN GRU 要求权重连续；load_state_dict 后显式调用以消除警告和潜在性能损失
        self.net.gru.gru.flatten_parameters()

        # target 冻结于 θ_meta（INVARIANT 8）
        target_net = deepcopy(self.net)
        target_net.gru.gru.flatten_parameters()   # deepcopy 后重新打包 cuDNN 权重
        target_net.eval()
        for p in target_net.parameters():
            p.requires_grad = False

        inner_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.inner_lr)

        inner_losses = []
        q_max_vals = []
        q_mean_vals = []
        td_vals = []
        nstep_vals = []
        margin_vals = []
        l2_vals = []
        rmean_vals = []
        rmax_vals = []
        for k in range(self.inner_steps):
            n_expert   = max(1, int(self.batch_size * exp_ratio_val))
            n_selfplay = self.batch_size - n_expert

            # 优先使用预采样 batch（DualGPUReptileTrainer 主线程预采，消除锁竞争）
            # 回退：thread-safe buffer 采样（单 GPU 路径或未预采样时）
            if pre_batches is not None and k < len(pre_batches):
                expert_batch = pre_batches[k]
            else:
                expert_batch = None
                with ctx:
                    try:
                        expert_batch = expert_buffer.sample(
                            batch_size=max(1, n_expert),
                            sequence_len=self.sequence_len,
                            burn_in_len=self.burn_in_len,
                            rng=rng,
                        )
                    except (ValueError, IndexError):
                        pass

            selfplay_batch = None
            if selfplay_buffer is not None and len(selfplay_buffer) >= n_selfplay and n_selfplay > 0:
                with ctx:
                    sp_segs = selfplay_buffer.sample(n=n_selfplay, rng=rng)
                if sp_segs:
                    from buffers.batch_collator import collate_selfplay_segments
                    selfplay_batch = collate_selfplay_segments(sp_segs)  # (B,T,N,...) numpy，_inner_step 搬设备

            if expert_batch is not None:
                loss, info = self._inner_step(
                    expert_batch, target_net, inner_optimizer,
                    aux_batch=selfplay_batch,
                    expert_weight=exp_ratio_val,
                )
                inner_losses.append(info.get("loss_total", 0.0))
                q_max_vals.append(info.get("q_max", 0.0))
                q_mean_vals.append(info.get("q_mean", 0.0))
                td_vals.append(info.get("loss_td", 0.0))
                nstep_vals.append(info.get("loss_nstep", 0.0))
                margin_vals.append(info.get("loss_margin", 0.0))
                l2_vals.append(info.get("l2_reg", 0.0))
                rmean_vals.append(info.get("r_mean", 0.0))
                rmax_vals.append(info.get("r_max", 0.0))

        # 返回 θ_inner（CPU，避免 GPU 内存在跨线程传递时的 CUDA 上下文问题）
        theta_inner = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        _avg = lambda xs: float(np.mean(xs)) if xs else 0.0
        return {
            "theta_inner": theta_inner,
            "loss_mean": float(np.mean(inner_losses)) if inner_losses else 0.0,
            "q_max": max(q_max_vals) if q_max_vals else 0.0,
            "q_mean": _avg(q_mean_vals),
            "loss_td": _avg(td_vals),
            "loss_nstep": _avg(nstep_vals),
            "loss_margin": _avg(margin_vals),
            "l2_reg": _avg(l2_vals),
            "r_mean": _avg(rmean_vals),
            "r_max": max(rmax_vals) if rmax_vals else 0.0,
        }

    def meta_step(
        self,
        expert_buffer,
        selfplay_buffer,
        env_factory=None,
        seed: int = None,
    ) -> Dict[str, float]:
        """
        执行一次 meta-iteration（单 GPU）：
        run_inner_loop → outer update。
        """
        theta_meta = {k: v.detach().cpu().clone()
                      for k, v in self.net.state_dict().items()}

        map_type = self.scheduler.sample(
            self.meta_iter, self.total_meta_iters, adaptive=False
        )
        exp_ratio_val = expert_ratio(
            self.meta_iter, self.total_meta_iters,
            self.expert_ratio_start, self.expert_ratio_end,
            self.expert_ratio_decay_iters, bc_offset=self.bc_iters,
        )

        _seg = self.profiler.segment if self.profiler else lambda _: nullcontext()

        with _seg("inner_loop"):
            result = self.run_inner_loop(
                expert_buffer=expert_buffer,
                selfplay_buffer=selfplay_buffer,
                theta_meta_cpu=theta_meta,
                exp_ratio_val=exp_ratio_val,
                buffer_lock=None,
                seed=seed or (100000 + self.meta_iter),
            )

        # Outer update: θ = θ_meta + ε·(θ_inner - θ_meta)
        cur_meta_lr = self._current_meta_lr()
        with _seg("outer_update"):
            with torch.no_grad():
                for name, param in self.net.named_parameters():
                    delta = result["theta_inner"][name].to(self.device) - theta_meta[name].to(self.device)
                    param.copy_(theta_meta[name].to(self.device) + cur_meta_lr * delta)

        self.meta_iter += 1
        if self.profiler and self.meta_iter % self._profile_interval == 0:
            import pprint
            print(f"\n[Profiler] meta_iter={self.meta_iter}")
            pprint.pprint(self.profiler.report())
            print()
        return {
            "meta_iter":       self.meta_iter,
            "map_type":        map_type,
            "expert_ratio":    exp_ratio_val,
            "inner_loss_mean": result["loss_mean"],
            "q_max":           result.get("q_max", 0.0),
            "q_mean":          result.get("q_mean", 0.0),
            "loss_td":         result.get("loss_td", 0.0),
            "loss_nstep":      result.get("loss_nstep", 0.0),
            "loss_margin":     result.get("loss_margin", 0.0),
            "l2_reg":          result.get("l2_reg", 0.0),
            "r_mean":          result.get("r_mean", 0.0),
            "r_max":           result.get("r_max", 0.0),
        }

    def _inner_step(
        self,
        batch: dict,
        target_net: BiIntentGATDRQN,
        optimizer: torch.optim.Optimizer,
        aux_batch: dict = None,      # 自游走段 batch（多体 (B,T,N,...)，走 dqfd_sequence_loss，关 margin）
        expert_weight: float = 1.0,  # 课程式混合权重 er：total = er·L_expert + (1-er)·L_selfplay
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

        def _patch_to_gpu(x):
            """patch 传 GPU 再转 float（uint8→/255）；省 CPU 转换 + 4× 传输量。"""
            t = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x))
            t = t.to(self.device, non_blocking=True)
            if t.dtype == torch.uint8:
                return t.float().div_(255.0)
            return t.float()

        obs_patches      = _patch_to_gpu(batch["obs_patches"])     # (B, T, N, 4, K, K) uint8→float/255
        obs_scalars      = _to_tensor(batch["obs_scalars"])        # (B, T, N, 8)
        positions        = _to_tensor(batch["positions"])          # (B, T, N, 2)
        next_obs_patches = _patch_to_gpu(batch["next_obs_patches"])# (B, T, N, 4, K, K) uint8→float/255
        next_obs_scalars = _to_tensor(batch["next_obs_scalars"])   # (B, T, N, 8)
        next_positions   = _to_tensor(batch["next_positions"])     # (B, T, N, 2)
        actions          = _to_tensor(batch["actions_expert"], dtype=torch.long)   # (B, T, N)
        rewards          = _to_tensor(batch["rewards"])            # (B, T, N)
        dones            = _to_tensor(batch["dones"], dtype=torch.bool)
        valid            = _to_tensor(batch["valid"], dtype=torch.bool)             # (B, T, N)
        action_masks     = _to_tensor(batch["action_masks"], dtype=torch.bool)      # s_t 掩码
        if "next_action_masks" in batch:
            next_action_masks = _to_tensor(batch["next_action_masks"], dtype=torch.bool)
        else:
            next_action_masks = action_masks
        expert_actions   = actions
        margin_excluded  = batch.get("margin_excluded")
        if margin_excluded is not None:
            margin_excluded = _to_tensor(margin_excluded, dtype=torch.bool)
        margin_weights = batch.get("margin_weights")
        if margin_weights is not None:
            margin_weights = _to_tensor(margin_weights)

        # ---- 多体序列 DQfD 损失（共享 helper：前向+折叠+1step+nstep+margin）----
        total_loss, loss_info = dqfd_sequence_loss(
            self.net, target_net,
            obs_patches=obs_patches, obs_scalars=obs_scalars, positions=positions,
            next_obs_patches=next_obs_patches, next_obs_scalars=next_obs_scalars,
            next_positions=next_positions,
            actions=actions, rewards=rewards, dones=dones, valid=valid,
            action_masks=action_masks, next_action_masks=next_action_masks,
            margin_excluded=margin_excluded, margin_weights=margin_weights,
            expert_actions=expert_actions,
            gamma=self.gamma, n_step=self.n_step, mellowmax_alpha=self.mellowmax_alpha,
            margin=self.margin, lambda_td=self.lambda_td, lambda_nstep=self.lambda_nstep,
            lambda_margin=self.lambda_margin, lambda_l2=self.lambda_l2,
            burn_in_len=self.burn_in_len, parameters=list(self.net.parameters()),
            use_amp=self.use_amp, reward_scale=self.reward_scale,
        )

        # ---- 自游走段：与专家同走 dqfd_sequence_loss，但关 margin（无专家监督）----
        # 自游走的价值就是覆盖策略实际会漂到的 OOD 脏状态，让 1-step+n-step TD 把那里
        # 被高估的 Q 压回真值（设计文档 §3.3.1 / line 397）。按 expert_ratio 课程混合。
        if aux_batch is not None:
            sp_patches      = _patch_to_gpu(aux_batch["obs_patches"])
            sp_scalars      = _to_tensor(aux_batch["obs_scalars"])
            sp_positions    = _to_tensor(aux_batch["positions"])
            sp_next_patches = _patch_to_gpu(aux_batch["next_obs_patches"])
            sp_next_scalars = _to_tensor(aux_batch["next_obs_scalars"])
            sp_next_pos     = _to_tensor(aux_batch["next_positions"])
            sp_actions      = _to_tensor(aux_batch["actions_expert"], dtype=torch.long)
            sp_rewards      = _to_tensor(aux_batch["rewards"])
            sp_dones        = _to_tensor(aux_batch["dones"], dtype=torch.bool)
            sp_valid        = _to_tensor(aux_batch["valid"], dtype=torch.bool)
            sp_masks        = _to_tensor(aux_batch["action_masks"], dtype=torch.bool)
            sp_next_masks   = _to_tensor(aux_batch["next_action_masks"], dtype=torch.bool)

            sp_loss, sp_info = dqfd_sequence_loss(
                self.net, target_net,
                obs_patches=sp_patches, obs_scalars=sp_scalars, positions=sp_positions,
                next_obs_patches=sp_next_patches, next_obs_scalars=sp_next_scalars,
                next_positions=sp_next_pos,
                actions=sp_actions, rewards=sp_rewards, dones=sp_dones, valid=sp_valid,
                action_masks=sp_masks, next_action_masks=sp_next_masks,
                expert_actions=sp_actions,
                gamma=self.gamma, n_step=self.n_step, mellowmax_alpha=self.mellowmax_alpha,
                margin=self.margin, lambda_td=self.lambda_td, lambda_nstep=self.lambda_nstep,
                lambda_margin=0.0,   # 自游走无专家动作监督，关 margin
                lambda_l2=0.0,       # L2 已在专家项对 self.net 计入，勿重复
                burn_in_len=self.burn_in_len, parameters=None,
                use_amp=self.use_amp, reward_scale=self.reward_scale,
            )
            # 课程式混合：expert 项 ×er，selfplay 项 ×(1-er)
            total_loss = expert_weight * total_loss + (1.0 - expert_weight) * sp_loss
            loss_info["loss_sp"]       = sp_loss.item()
            loss_info["loss_sp_td"]    = sp_info.get("loss_td", 0.0)
            loss_info["loss_sp_nstep"] = sp_info.get("loss_nstep", 0.0)
            loss_info["loss_total"]    = total_loss.item()

        # AMP：scaler.scale → backward → unscale → clip → step → update。
        # use_amp=False 时 GradScaler(enabled=False) 全为 no-op，与原路径等价。
        optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.scaler.step(optimizer)
        self.scaler.update()

        return total_loss, loss_info

    def state_dict(self) -> dict:
        return {
            "net": self.net.state_dict(),
            "meta_iter": self.meta_iter,
        }

    def load_state_dict(self, state: dict):
        self.net.load_state_dict(state["net"])
        self.meta_iter = state.get("meta_iter", 0)


# =============================================================================
# 双 GPU task-parallel Reptile 训练器（设计文档 §4.4 / §4.5）
# =============================================================================

class DualGPUReptileTrainer:
    """
    双 GPU task-parallel Reptile + DQfD 训练器。

    两卡各跑一个不同 task 的完整 K 步 inner loop（threading 并行），
    meta-iter 末平均增量做一次 outer 更新（非逐步 DDP）。

        GPU 0: task A  inner loop (K 步 DQfD)  ──┐
        GPU 1: task B  inner loop (K 步 DQfD)  ──┘→ avg Δ → outer update

    吞吐目标：相比单 GPU 接近翻倍，元梯度方差减半。
    CUDA 内核在 GPU kernel 期间释放 GIL，两卡真正并行。

    用法:
        trainer = DualGPUReptileTrainer(net, cfg, devices=["cuda:0","cuda:1"])
        for _ in range(total_iters):
            metrics = trainer.meta_step(expert_buffer, selfplay_buffer)
    """

    def __init__(
        self,
        net_template: BiIntentGATDRQN,
        cfg: dict,
        devices: List[str] = None,
    ):
        """
        Args:
            net_template: 网络实例（用作架构模板，内部 deepcopy 到各 GPU）。
            cfg:          train_reptile_dqfd.yaml 配置字典。
            devices:      GPU 设备列表，默认 ["cuda:0","cuda:1"]。
        """
        if devices is None:
            devices = ["cuda:0", "cuda:1"]
        self.devices = [torch.device(d) for d in devices]

        # 为每个 GPU 创建独立的 sub-trainer（各自持有一份网络）
        self.sub_trainers: List[ReptileDQfDTrainer] = []
        for device in self.devices:
            net_copy = deepcopy(net_template)
            self.sub_trainers.append(ReptileDQfDTrainer(net_copy, cfg, device=device))

        # θ_meta 统一保存在 CPU（outer update 的基准；不绑定特定 GPU）
        self._meta_state: Dict[str, torch.Tensor] = {
            k: v.detach().cpu().clone()
            for k, v in net_template.state_dict().items()
        }

        # 超参
        _legacy_lr = cfg.get("meta_lr", None)
        self.meta_lr_start = cfg.get("meta_lr_start", _legacy_lr if _legacy_lr is not None else 0.01)
        self.meta_lr_end   = cfg.get("meta_lr_end",   _legacy_lr if _legacy_lr is not None else 0.001)
        self.tasks_per_gpu = cfg.get("parallel", {}).get("tasks_per_gpu", cfg.get("tasks_per_gpu", 1))
        self.meta_iter = 0
        self.total_meta_iters = cfg.get("meta_iterations", 20000)
        self.expert_ratio_start       = cfg.get("expert_ratio_start",       0.9)
        self.expert_ratio_end         = cfg.get("expert_ratio_end",         0.5)
        self.expert_ratio_decay_iters = cfg.get("expert_ratio_decay_iters", 8000)
        self.bc_iters                 = cfg.get("bc_iters",                  0)

        self.scheduler = MapScheduler(cfg.get("map_schedule", None))

        # selfplay buffer 并发访问锁：多个 _run_gpu 线程共享同一 buffer，需要序列化采样
        self._buffer_lock = threading.Lock()

        # 异步预取：在当前 meta-iter 的 GPU inner loop 跑的同时，后台线程预采下一轮 batch。
        # expert_buffer 在阶段二是只读静态的，sample 用独立 rng → 并发采样安全。
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetched = None  # (key, all_pre_batches)

        # 性能诊断
        self.profiler          = Profiler() if cfg.get("debug", False) else None
        self._profile_interval = cfg.get("profile_interval", 200)

    # ------------------------------------------------------------------
    # 预采样 + 异步预取
    # ------------------------------------------------------------------

    def _presample(self, expert_buffer, meta_iter: int, seed) -> list:
        """采样某 meta-iter 全部 GPU×task×inner 的 expert batch。布局 [gpu][task][inner]。"""
        n_gpus = len(self.devices)
        tpg = self.tasks_per_gpu
        ref = self.sub_trainers[0]
        exp_ratio_val = expert_ratio(
            meta_iter, self.total_meta_iters,
            self.expert_ratio_start, self.expert_ratio_end,
            self.expert_ratio_decay_iters, bc_offset=self.bc_iters,
        )
        n_expert_seqs = max(1, int(ref.batch_size * exp_ratio_val))
        all_pre = []
        for gpu_idx in range(n_gpus):
            gpu_b = []
            for task_offset in range(tpg):
                task_idx = gpu_idx * tpg + task_offset
                task_seed = (seed or meta_iter * 1000) + task_idx * 99991
                rng = np.random.default_rng(task_seed)
                batches = []
                for _ in range(ref.inner_steps):
                    try:
                        batches.append(expert_buffer.sample(
                            batch_size=n_expert_seqs,
                            sequence_len=ref.sequence_len,
                            burn_in_len=ref.burn_in_len,
                            rng=rng,
                        ))
                    except (ValueError, IndexError):
                        batches.append(None)
                gpu_b.append(batches)
            all_pre.append(gpu_b)
        return all_pre

    def _get_presample(self, expert_buffer, meta_iter: int, seed) -> list:
        """取本轮预采样：命中上一轮后台预取则直接用，否则同步采样。"""
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None
        if self._prefetched is not None and self._prefetched[0] == (meta_iter, seed):
            batches = self._prefetched[1]
            self._prefetched = None
            return batches
        return self._presample(expert_buffer, meta_iter, seed)

    def _start_prefetch(self, expert_buffer, meta_iter: int, seed):
        """后台线程预采下一轮（与本轮 GPU inner loop 重叠；CUDA kernel 释放 GIL 时 CPU 采样得以并行）。"""
        key = (meta_iter, seed)
        def _work():
            self._prefetched = (key, self._presample(expert_buffer, meta_iter, seed))
        self._prefetch_thread = threading.Thread(target=_work, name="prefetch", daemon=True)
        self._prefetch_thread.start()

    def _current_meta_lr(self) -> float:
        """线性衰减 meta_lr：meta_lr_start → meta_lr_end。"""
        t = self.meta_iter / max(1, self.total_meta_iters - 1)
        return self.meta_lr_start + (self.meta_lr_end - self.meta_lr_start) * t

    @property
    def net(self) -> BiIntentGATDRQN:
        """返回 GPU0 的网络（用于 checkpoint 保存 / 评估接口兼容）。"""
        return self.sub_trainers[0].net

    def meta_step(
        self,
        expert_buffer,
        selfplay_buffer,
        env_factory=None,
        seed: int = None,
    ) -> Dict[str, float]:
        """
        执行一次 meta-iteration（双 GPU 并行）：

        1. 主线程顺序预采样两张 GPU 所需的全部 expert batch（消除跨线程锁竞争）
        2. 两线程在各自 GPU 上并行跑 K 步 inner loop（纯 GPU compute，无 Python 锁）
        3. 收集两份 θ_inner
        4. 平均增量，做 outer update
        5. 将新 θ_meta 同步回两卡
        """
        theta_meta = self._meta_state   # 当前 θ_meta（CPU tensors）
        n_gpus     = len(self.devices)
        tpg        = self.tasks_per_gpu   # 每卡串行任务数
        n_tasks    = n_gpus * tpg         # 本次 meta-iter 总 task 数

        exp_ratio_val = expert_ratio(
            self.meta_iter, self.total_meta_iters,
            self.expert_ratio_start, self.expert_ratio_end,
            self.expert_ratio_decay_iters, bc_offset=self.bc_iters,
        )

        # 采样 n_tasks 个地图类型（相邻偏移保证多样性）
        map_types = [
            str(self.scheduler.sample(self.meta_iter + i, self.total_meta_iters))
            for i in range(n_tasks)
        ]

        _seg = self.profiler.segment if self.profiler else lambda _: nullcontext()

        # ---- 预采样（异步预取）----
        # 本轮 batch：命中上一轮后台预取直接用；随后立即启动下一轮预取，与下方
        # GPU inner loop 重叠（patch 已是 uint8，转换在 GPU；此处只剩 gather，可被隐藏）。
        with _seg("presample"):
            all_pre_batches = self._get_presample(expert_buffer, self.meta_iter, seed)
        self._start_prefetch(expert_buffer, self.meta_iter + 1, seed)

        # ---- 并行 inner loop（每线程在各自 GPU 上串行跑 tasks_per_gpu 个任务）----
        results: Dict[int, List[Optional[Dict]]] = {i: [] for i in range(n_gpus)}
        errors:  Dict[int, Optional[Exception]]  = {i: None for i in range(n_gpus)}

        def _run_gpu(gpu_idx: int):
            try:
                gpu_results: List[Optional[Dict]] = []
                for task_offset in range(tpg):
                    task_idx  = gpu_idx * tpg + task_offset
                    task_seed = (seed or self.meta_iter * 1000) + task_idx * 99991
                    r = self.sub_trainers[gpu_idx].run_inner_loop(
                        expert_buffer   = expert_buffer,
                        selfplay_buffer = selfplay_buffer,
                        theta_meta_cpu  = theta_meta,
                        exp_ratio_val   = exp_ratio_val,
                        buffer_lock     = self._buffer_lock,  # selfplay 采样序列化
                        seed            = task_seed,
                        pre_batches     = all_pre_batches[gpu_idx][task_offset],
                    )
                    gpu_results.append(r)
                results[gpu_idx] = gpu_results
            except Exception as exc:
                errors[gpu_idx] = exc

        with _seg("inner_loop_wall"):
            threads = [threading.Thread(target=_run_gpu, args=(i,), name=f"inner-gpu{i}")
                       for i in range(n_gpus)]
            for t in threads: t.start()
            for t in threads: t.join()

        for idx, exc in errors.items():
            if exc is not None:
                raise RuntimeError(f"GPU {idx} inner loop failed") from exc

        # 展平所有任务结果，过滤采样失败项
        valid = [r for gpu_res in results.values() for r in gpu_res if r is not None]
        if not valid:
            self.meta_iter += 1
            return {"meta_iter": self.meta_iter, "inner_loss_mean": 0.0,
                    "map_types": map_types, "expert_ratio": exp_ratio_val, "num_tasks": 0}

        # ---- Outer update（CPU，节省显存；使用线性衰减 meta_lr）----
        cur_meta_lr = self._current_meta_lr()
        with _seg("outer_update"):
            new_meta_state: Dict[str, torch.Tensor] = {}
            with torch.no_grad():
                for name in theta_meta:
                    deltas = [r["theta_inner"][name] - theta_meta[name]
                              for r in valid if name in r["theta_inner"]]
                    avg_delta = torch.stack(deltas).mean(dim=0)
                    new_meta_state[name] = theta_meta[name] + cur_meta_lr * avg_delta
            self._meta_state = new_meta_state

        # 同步新 θ_meta 到所有 sub-trainer（各自 GPU）
        with _seg("sync_meta"):
            for trainer in self.sub_trainers:
                trainer.net.load_state_dict(
                    {k: v.to(trainer.device) for k, v in new_meta_state.items()}
                )

        self.meta_iter += 1
        if self.profiler and self.meta_iter % self._profile_interval == 0:
            import pprint
            print(f"\n[Profiler] meta_iter={self.meta_iter}")
            pprint.pprint(self.profiler.report())
            print()

        loss_vals = [r["loss_mean"] for r in valid if r["loss_mean"] > 0]
        q_max_vals  = [r.get("q_max", 0.0) for r in valid]
        q_mean_vals = [r.get("q_mean", 0.0) for r in valid]
        _avg = lambda key: float(np.mean([r.get(key, 0.0) for r in valid])) if valid else 0.0

        return {
            "meta_iter":       self.meta_iter,
            "map_types":       map_types,
            "expert_ratio":    exp_ratio_val,
            "inner_loss_mean": float(np.mean(loss_vals)) if loss_vals else 0.0,
            "q_max":           max(q_max_vals) if q_max_vals else 0.0,
            "q_mean":          float(np.mean(q_mean_vals)) if q_mean_vals else 0.0,
            "loss_td":         _avg("loss_td"),
            "loss_nstep":      _avg("loss_nstep"),
            "loss_margin":     _avg("loss_margin"),
            "l2_reg":          _avg("l2_reg"),
            "r_mean":          _avg("r_mean"),
            "r_max":           max((r.get("r_max", 0.0) for r in valid), default=0.0),
            "num_tasks":       len(valid),
            "meta_lr":         cur_meta_lr,
        }

    def state_dict(self) -> dict:
        """兼容 save_checkpoint：net 字段保存 θ_meta。"""
        return {
            "net":      self._meta_state,   # CPU state dict，可直接被 load_state_dict 使用
            "meta_iter": self.meta_iter,
        }

    def load_state_dict(self, state: dict):
        self._meta_state = {k: v.cpu() for k, v in state["net"].items()}
        self.meta_iter   = state.get("meta_iter", 0)
        for trainer in self.sub_trainers:
            trainer.net.load_state_dict(
                {k: v.to(trainer.device) for k, v in self._meta_state.items()}
            )
