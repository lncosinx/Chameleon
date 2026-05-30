"""Batch collator：构造训练 batch，支持 pinned memory + non_blocking 传输。"""
import numpy as np
import torch
from typing import Dict, List


def collate_samples_to_tensors(
    samples: List[dict],
    sequence_len: int = 32,
    device: torch.device = None,
    pin_memory: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    将样本列表整理为训练 batch。

    Returns:
        dict of tensors with batch dimension.
    """
    B = len(samples) // sequence_len
    if B == 0:
        B = 1
        sequence_len = len(samples)

    N = samples[0].get("obs_patch", np.zeros((4, 11, 11))).shape[0] if "num_agents" not in samples[0] else 1
    # Infer N from sample
    if "obs_patch" in samples[0]:
        patch_shape = samples[0]["obs_patch"].shape
        # (4, K, K) for single agent or (N, 4, K, K)?
    K = 11

    batch = {
        "obs_patches": torch.zeros(B, sequence_len, 4, K, K),
        "obs_scalars": torch.zeros(B, sequence_len, 8),
        "actions": torch.zeros(B, sequence_len, dtype=torch.long),
        "rewards": torch.zeros(B, sequence_len),
        "dones": torch.zeros(B, sequence_len, dtype=torch.bool),
        "action_masks": torch.zeros(B, sequence_len, 5, dtype=torch.bool),
        "expert_actions": torch.zeros(B, sequence_len, dtype=torch.long),
        "margin_excluded": torch.zeros(B, sequence_len, dtype=torch.bool),
        "margin_weights": torch.ones(B, sequence_len),
    }

    for b in range(B):
        for t in range(sequence_len):
            idx = b * sequence_len + t
            if idx >= len(samples):
                break
            s = samples[idx]
            batch["obs_patches"][b, t] = torch.from_numpy(s.get("obs_patch", np.zeros((4, K, K))))
            batch["obs_scalars"][b, t] = torch.from_numpy(s.get("obs_scalar", np.zeros(8)))
            batch["actions"][b, t] = s.get("action_expert", 0)
            batch["rewards"][b, t] = s.get("reward", 0.0)
            batch["dones"][b, t] = s.get("done", False)
            batch["action_masks"][b, t] = torch.from_numpy(
                s.get("action_mask", np.ones(5, dtype=bool))
            )
            batch["expert_actions"][b, t] = s.get("action_expert", 0)
            batch["margin_excluded"][b, t] = s.get("margin_excluded", False)
            batch["margin_weights"][b, t] = s.get("margin_weight", 1.0)

    if device is not None:
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

    return batch
