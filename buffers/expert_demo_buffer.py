"""
专家演示回放缓冲。

存储 LaCAM 成功专家轨迹，每条样本含：
    obs_patch, obs_scalar, hidden_init,
    action_expert, reward, next_obs_patch, next_obs_scalar,
    done, action_mask, map_type, task_id, margin_excluded, margin_weight

支持按 sequence_len 采样定长序列（含 R2D2 burn-in）。
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque


class ExpertDemoBuffer:
    """
    专家演示回放缓冲（循环队列）。

    用法:
        buf = ExpertDemoBuffer(capacity=100000)
        buf.add(sample)  # 或 buf.add_batch(samples)
        batch = buf.sample(batch_size=32, sequence_len=32, burn_in_len=8)
    """

    def __init__(self, capacity: int = 500_000):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._rng = np.random.default_rng()

    def add(self, sample: dict):
        """添加单条样本。"""
        self._buffer.append(sample)

    def add_batch(self, samples: List[dict]):
        """批量添加样本。"""
        for s in samples:
            self._buffer.append(s)

    def __len__(self) -> int:
        return len(self._buffer)

    def sample(
        self,
        batch_size: int,
        sequence_len: int = 32,
        burn_in_len: int = 8,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, np.ndarray]:
        """
        采样一个 batch。

        返回 dict，字段：
            obs_patches:    (B, seq_len, 4, K, K)
            obs_scalars:    (B, seq_len, 8)
            actions_expert: (B, seq_len)
            rewards:        (B, seq_len)
            next_obs_patches:   (B, seq_len, 4, K, K)
            next_obs_scalars:   (B, seq_len, 8)
            dones:          (B, seq_len)
            action_masks:   (B, seq_len, 5)
            margin_excluded:(B, seq_len) bool
            margin_weights: (B, seq_len)

        注意：hidden_init 需外部维护（buffer 不管理 GRU 隐状态）。
              训练时用 burn_in_len 步重建隐状态。
        """
        if rng is None:
            rng = self._rng

        total_needed = batch_size * sequence_len
        if len(self._buffer) < total_needed:
            raise ValueError(
                f"Buffer has {len(self._buffer)} samples, "
                f"need at least {total_needed}"
            )

        # 随机采样起始索引
        max_start = len(self._buffer) - sequence_len
        indices = rng.integers(0, max_start, size=batch_size)

        # 收集 batch
        K = None
        for idx in indices:
            sample = self._buffer[idx]
            if K is None and "obs_patch" in sample:
                K = sample["obs_patch"].shape[1]

        batch = {
            "obs_patches": np.zeros((batch_size, sequence_len, 4, K or 11, K or 11), dtype=np.float32),
            "obs_scalars": np.zeros((batch_size, sequence_len, 8), dtype=np.float32),
            "actions_expert": np.zeros((batch_size, sequence_len), dtype=np.int64),
            "rewards": np.zeros((batch_size, sequence_len), dtype=np.float32),
            "next_obs_patches": np.zeros((batch_size, sequence_len, 4, K or 11, K or 11), dtype=np.float32),
            "next_obs_scalars": np.zeros((batch_size, sequence_len, 8), dtype=np.float32),
            "dones": np.zeros((batch_size, sequence_len), dtype=bool),
            "action_masks": np.zeros((batch_size, sequence_len, 5), dtype=bool),
            "next_action_masks": np.ones((batch_size, sequence_len, 5), dtype=bool),  # s_{t+1} 掩码
            "margin_excluded": np.zeros((batch_size, sequence_len), dtype=bool),
            "margin_weights": np.zeros((batch_size, sequence_len), dtype=np.float32),
        }

        for b, start_idx in enumerate(indices):
            for t in range(sequence_len):
                s = self._buffer[start_idx + t]
                batch["obs_patches"][b, t] = s.get("obs_patch", np.zeros((4, K or 11, K or 11)))
                batch["obs_scalars"][b, t] = s.get("obs_scalar", np.zeros(8))
                batch["actions_expert"][b, t] = s.get("action_expert", 0)
                batch["rewards"][b, t] = s.get("reward", 0.0)
                batch["next_obs_patches"][b, t] = s.get("next_obs_patch",
                    np.zeros((4, K or 11, K or 11)))
                batch["next_obs_scalars"][b, t] = s.get("next_obs_scalar", np.zeros(8))
                batch["dones"][b, t] = s.get("done", False)
                batch["action_masks"][b, t] = s.get("action_mask", np.ones(5, dtype=bool))
                # next_action_mask 优先从样本读取，回退到下一条样本的 action_mask
                if "next_action_mask" in s:
                    batch["next_action_masks"][b, t] = s["next_action_mask"]
                elif start_idx + t + 1 < len(self._buffer):
                    batch["next_action_masks"][b, t] = self._buffer[start_idx + t + 1].get(
                        "action_mask", np.ones(5, dtype=bool)
                    )
                # 若无法获取则保持默认全合法（全为 True）
                batch["margin_excluded"][b, t] = s.get("margin_excluded", False)
                batch["margin_weights"][b, t] = s.get("margin_weight", 1.0)

        return batch

    def get_stats(self) -> dict:
        """返回 buffer 统计信息。"""
        if not self._buffer:
            return {"size": 0}
        actions = [s.get("action_expert", 0) for s in self._buffer]
        wait_ratio = sum(1 for a in actions if a == 0) / max(len(actions), 1)
        margin_excluded_count = sum(1 for s in self._buffer if s.get("margin_excluded", False))
        return {
            "size": len(self._buffer),
            "wait_ratio": wait_ratio,
            "margin_excluded_ratio": margin_excluded_count / max(len(self._buffer), 1),
        }

    def clear(self):
        """清空 buffer。"""
        self._buffer.clear()
