"""
近 on-policy 短窗口缓冲（阶段三）—— **per-timestep × N-agent 段（segment）** 格式。

每个 segment 是一段同步采样的 rollout，形状 (T, N, ...)，与专家 buffer 同构
（含 positions/valid，供通信模块计算邻居并回传梯度）。每段记 policy_version，
仅当 v_learner − v_sample ≤ lag_max(=1) 时可训练，否则丢弃。

INVARIANT 9: 阶段三 max_policy_version_lag=1，同步段内权重冻结。

为何按 segment（而非单体单步）存储：与阶段二同理，通信需要同一时刻多个共存
agent 的观测+坐标；逐体单步存储会让通信候选集恒空、参数永不更新。
"""
from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, List, Optional


# segment dict 必须包含的键（与 ExpertDemoBuffer.add_episode 同构）
SEGMENT_KEYS = [
    "obs_patches", "next_obs_patches", "obs_scalars", "next_obs_scalars",
    "positions", "next_positions", "actions_expert", "rewards",
    "dones", "valid", "action_masks", "next_action_masks",
    "margin_excluded", "margin_weights",
]


class RecentOnlineBuffer:
    """短窗口同步式近 on-policy 缓冲（按 segment 存储）。"""

    def __init__(
        self,
        capacity: int = 64,                 # 最多保留的 segment 数
        max_policy_version_lag: int = 1,
    ):
        self.capacity = capacity
        self.max_lag = max_policy_version_lag
        self._segments: deque = deque(maxlen=capacity)  # 每项 = (seg_dict, version)
        self._current_policy_version: int = 0

    def add_segment(self, seg: Dict[str, np.ndarray], policy_version: int):
        """添加一段 rollout（seg 为 (T,N,...) 数组 dict）。"""
        self._segments.append((seg, int(policy_version)))

    # 兼容旧接口名
    def add_batch(self, seg: Dict[str, np.ndarray], policy_version: int):
        self.add_segment(seg, policy_version)

    def sync(self):
        """同步策略版本号（learner 更新后调用）。"""
        self._current_policy_version += 1

    def get_trainable_segments(
        self, learner_version: Optional[int] = None
    ) -> List[Dict[str, np.ndarray]]:
        """返回 lag ≤ max_lag 的 segment 列表。"""
        lv = self._current_policy_version if learner_version is None else learner_version
        return [seg for seg, v in self._segments if lv - v <= self.max_lag]

    def discard_old(self, learner_version: Optional[int] = None):
        """丢弃 lag 超限的 segment。learner_version 默认用内部版本。"""
        lv = self._current_policy_version if learner_version is None else learner_version
        kept = deque(maxlen=self.capacity)
        for seg, v in self._segments:
            if lv - v <= self.max_lag:
                kept.append((seg, v))
        self._segments = kept

    def num_timesteps(self) -> int:
        """当前所有 segment 的总时间步数。"""
        return sum(int(seg["actions_expert"].shape[0]) for seg, _ in self._segments)

    def __len__(self) -> int:
        """segment 数量。"""
        return len(self._segments)

    def get_lag_distribution(self) -> Dict[int, int]:
        dist: Dict[int, int] = {}
        for _, v in self._segments:
            lag = self._current_policy_version - v
            dist[lag] = dist.get(lag, 0) + 1
        return dist

    def clear(self):
        self._segments.clear()
        self._current_policy_version = 0
