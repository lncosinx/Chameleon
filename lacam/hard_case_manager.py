"""
困难案例管理器。

hard case 来源：
- LaCAM 失败
- 策略 rollout 失败
- 高碰撞、高停滞、长时间未达标
- warehouse 拥堵、room 门口阻塞、maze 长廊往返

对困难任务提高在环境回放中被重采样的权重。
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class HardCaseManager:
    """
    困难案例管理器。

    用法:
        mgr = HardCaseManager()
        mgr.add_hard_case(map_type="maze", starts=..., goals=..., reason="deadlock")
        # 采样时
        weights = mgr.get_sampling_weights()
    """

    def __init__(self, max_cases: int = 10000):
        self.max_cases = max_cases
        self._cases: List[dict] = []
        self._type_counts: Dict[str, int] = defaultdict(int)
        self._reason_counts: Dict[str, int] = defaultdict(int)

    def add_hard_case(
        self,
        map_type: str,
        starts: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
        reason: str = "unknown",
        obstacles: Optional[np.ndarray] = None,
        metrics: Optional[dict] = None,
    ):
        """添加一个困难案例。"""
        case = {
            "map_type": map_type,
            "starts": [(int(s[0]), int(s[1])) for s in starts],
            "goals": [(int(g[0]), int(g[1])) for g in goals],
            "reason": reason,
            "num_agents": len(starts),
            "obstacles": obstacles,
            "metrics": metrics or {},
            "added_at_step": len(self._cases),
        }

        if len(self._cases) >= self.max_cases:
            # FIFO 淘汰
            self._cases.pop(0)

        self._cases.append(case)
        self._type_counts[map_type] += 1
        self._reason_counts[reason] += 1

    def add_from_rollout_failure(
        self,
        wrapper,  # PogemaWrapper
        reason: str,
        metrics: Optional[dict] = None,
    ):
        """从失败的 rollout 记录 hard case。"""
        self.add_hard_case(
            map_type=wrapper.map_type,
            starts=wrapper.starts or [],
            goals=wrapper.goals or [],
            reason=reason,
            obstacles=wrapper.global_grid.copy() if wrapper.global_grid is not None else None,
            metrics=metrics,
        )

    def get_sampling_weights(self) -> np.ndarray:
        """返回每个 case 的采样权重（困难案例权重更高）。"""
        if not self._cases:
            return np.array([])

        weights = np.ones(len(self._cases), dtype=np.float32)

        # 按 reason 加权
        reason_multipliers = {
            "lacam_failure": 3.0,
            "deadlock": 3.0,
            "high_collision": 2.0,
            "high_stagnation": 2.0,
            "warehouse_congestion": 2.5,
            "room_doorway_block": 2.5,
            "maze_往复": 2.5,
            "no_progress_long": 2.0,
            "unknown": 1.5,
        }
        for i, case in enumerate(self._cases):
            weights[i] *= reason_multipliers.get(case["reason"], 1.0)

        # 归一化
        weights /= weights.sum()
        return weights

    def sample(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> List[dict]:
        """按权重采样 n 个困难案例。"""
        if rng is None:
            rng = np.random.default_rng()
        if not self._cases:
            return []
        weights = self.get_sampling_weights()
        indices = rng.choice(len(self._cases), size=min(n, len(self._cases)),
                             p=weights, replace=False)
        return [self._cases[i] for i in indices]

    def get_stats(self) -> dict:
        """返回统计信息。"""
        return {
            "total_cases": len(self._cases),
            "by_type": dict(self._type_counts),
            "by_reason": dict(self._reason_counts),
        }

    def __len__(self) -> int:
        return len(self._cases)
