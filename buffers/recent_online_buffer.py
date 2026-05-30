"""
近 on-policy 短窗口缓冲（阶段三）。

记录 policy_version, sample_time, task_id, is_expert_action。
仅当 v_learner - v_sample ≤ lag_max(=1) 时可用于训练，否则丢弃。

INVARIANT 9: 阶段三 max_policy_version_lag=1，同步段内权重冻结。
"""
from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple


class RecentOnlineBuffer:
    """短窗口同步式近 on-policy 缓冲。"""

    def __init__(
        self,
        capacity: int = 10_000,
        max_policy_version_lag: int = 1,
    ):
        self.capacity = capacity
        self.max_lag = max_policy_version_lag
        self._buffer: deque = deque(maxlen=capacity)
        self._current_policy_version: int = 0

    def add(self, sample: dict, policy_version: int):
        """添加样本。"""
        sample["policy_version"] = policy_version
        self._buffer.append(sample)

    def add_batch(self, samples: List[dict], policy_version: int):
        """批量添加。"""
        for s in samples:
            s["policy_version"] = policy_version
            self._buffer.append(s)

    def sync(self):
        """同步策略版本号（learner 更新后调用）。"""
        self._current_policy_version += 1

    def get_trainable_samples(
        self, learner_version: Optional[int] = None
    ) -> List[dict]:
        """
        获取可用于训练的样本（lag ≤ max_lag）。

        Args:
            learner_version: 当前 learner 版本号。默认用 buffer 内部版本。

        Returns:
            符合 lag 约束的样本列表。
        """
        if learner_version is None:
            learner_version = self._current_policy_version

        valid = []
        discarded = 0
        for s in self._buffer:
            lag = learner_version - s.get("policy_version", 0)
            if lag <= self.max_lag:
                valid.append(s)
            else:
                discarded += 1
        return valid

    def discard_old(self):
        """丢弃 lag 超限的样本。"""
        learner_version = self._current_policy_version
        new_buffer = deque(maxlen=self.capacity)
        for s in self._buffer:
            lag = learner_version - s.get("policy_version", 0)
            if lag <= self.max_lag:
                new_buffer.append(s)
        self._buffer = new_buffer

    def __len__(self) -> int:
        return len(self._buffer)

    def get_lag_distribution(self) -> Dict[int, int]:
        """返回 lag 分布（用于监控）。"""
        dist = {}
        for s in self._buffer:
            lag = self._current_policy_version - s.get("policy_version", 0)
            dist[lag] = dist.get(lag, 0) + 1
        return dist

    def clear(self):
        self._buffer.clear()
        self._current_policy_version = 0
