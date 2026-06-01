"""Batch collator：构造训练 batch，支持 pinned memory + non_blocking 传输。"""
import numpy as np
import torch
from typing import Dict, List, Optional


def collate_samples_to_tensors(
    samples: List[dict],
    sequence_len: int = 32,
    device: torch.device = None,
    pin_memory: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    将样本列表整理为训练 batch。

    每个 sample 是单体单步字典（per-agent single-step），
    按 sequence_len 分组构成 (B, T, ...) 的 batch。

    Args:
        samples:      sample dict 列表，长度须为 B * sequence_len。
        sequence_len: 每条序列长度 T。
        device:       目标设备（None = 留在 CPU）。
        pin_memory:   是否 pin memory（与 non_blocking 传输配合，CLAUDE.md 性能策略）。

    Returns:
        dict of tensors，形状均含 (B, T) 前缀。
    """
    if len(samples) == 0:
        return {}

    # 推断 B，保证能整除
    total = len(samples)
    B = max(1, total // sequence_len)
    sequence_len = total // B  # 向下取整

    # 从第一条样本推断 patch 尺寸（不依赖 shape[0]，避免把 channels 当 N）
    sample0 = samples[0]
    patch = sample0.get("obs_patch")
    if patch is not None:
        patch_arr = np.asarray(patch)
        # 期望 (C, K, K)，C=4
        if patch_arr.ndim == 3:
            C, K, _ = patch_arr.shape
        else:
            C, K = 4, 11
    else:
        C, K = 4, 11

    batch = {
        "obs_patches":    np.zeros((B, sequence_len, C, K, K), dtype=np.float32),
        "obs_scalars":    np.zeros((B, sequence_len, 8),        dtype=np.float32),
        "actions":        np.zeros((B, sequence_len),           dtype=np.int64),
        "rewards":        np.zeros((B, sequence_len),           dtype=np.float32),
        "dones":          np.zeros((B, sequence_len),           dtype=bool),
        "action_masks":   np.ones( (B, sequence_len, 5),        dtype=bool),
        "next_action_masks": np.ones((B, sequence_len, 5),      dtype=bool),
        "expert_actions": np.zeros((B, sequence_len),           dtype=np.int64),
        "margin_excluded":np.zeros((B, sequence_len),           dtype=bool),
        "margin_weights": np.ones( (B, sequence_len),           dtype=np.float32),
    }

    for b in range(B):
        for t in range(sequence_len):
            idx = b * sequence_len + t
            if idx >= len(samples):
                break
            s = samples[idx]
            p = s.get("obs_patch", np.zeros((C, K, K), dtype=np.float32))
            batch["obs_patches"][b, t]     = np.asarray(p, dtype=np.float32)
            batch["obs_scalars"][b, t]     = np.asarray(s.get("obs_scalar",  np.zeros(8)), dtype=np.float32)
            batch["actions"][b, t]         = int(s.get("action_expert", 0))
            batch["rewards"][b, t]         = float(s.get("reward", 0.0))
            batch["dones"][b, t]           = bool(s.get("done", False))
            batch["action_masks"][b, t]    = np.asarray(s.get("action_mask",  np.ones(5, bool)), dtype=bool)
            batch["next_action_masks"][b, t] = np.asarray(
                s.get("next_action_mask", np.ones(5, bool)), dtype=bool
            )
            batch["expert_actions"][b, t]  = int(s.get("action_expert", 0))
            batch["margin_excluded"][b, t] = bool(s.get("margin_excluded", False))
            batch["margin_weights"][b, t]  = float(s.get("margin_weight", 1.0))

    # numpy → tensor
    out: Dict[str, torch.Tensor] = {}
    dtype_map = {
        "obs_patches":     torch.float32,
        "obs_scalars":     torch.float32,
        "actions":         torch.long,
        "rewards":         torch.float32,
        "dones":           torch.bool,
        "action_masks":    torch.bool,
        "next_action_masks": torch.bool,
        "expert_actions":  torch.long,
        "margin_excluded": torch.bool,
        "margin_weights":  torch.float32,
    }
    for key, arr in batch.items():
        t = torch.from_numpy(arr).to(dtype_map[key])
        if pin_memory and t.is_floating_point():
            t = t.pin_memory()
        out[key] = t

    if device is not None:
        out = {k: v.to(device, non_blocking=pin_memory) for k, v in out.items()}

    return out
