"""
轨迹→动作转换 + 可解释性过滤。

将 LaCAM 路径转为 per-step 动作序列，并按 INVARIANT 1 映射。
关键约束：
- 观测必须由 observation_builder 按局部可观测规则**重新生成**，不得用 LaCAM 全局状态
- 同一 (局部patch+足迹+目标方向) 下专家动作不唯一的样本标记 margin_excluded=True

INVARIANT 1: 动作映射 wait=0, up=1, down=2, left=3, right=4
INVARIANT 11: 对称让行等歧义样本剔除出 margin loss
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict


# INVARIANT 1: 位移 → 动作映射
DELTA_TO_ACTION = {
    (0, 0): 0,   # wait
    (-1, 0): 1,  # up
    (1, 0): 2,   # down
    (0, -1): 3,  # left
    (0, 1): 4,   # right
}


def path_to_actions(path: List[Tuple[int, int]]) -> List[int]:
    """
    将路径（位置序列）转换为动作序列。

    Args:
        path: [(r0,c0), (r1,c1), ..., (rT,cT)]

    Returns:
        actions: [a0, a1, ..., a_{T-1}]，长度 = len(path) - 1
    """
    actions = []
    for t in range(len(path) - 1):
        dr = path[t + 1][0] - path[t][0]
        dc = path[t + 1][1] - path[t][1]
        key = (int(dr), int(dc))
        if key not in DELTA_TO_ACTION:
            raise ValueError(
                f"Invalid displacement {key} at step {t}: "
                f"from {path[t]} to {path[t+1]}. "
                f"Only 4-directional moves + wait are allowed."
            )
        actions.append(DELTA_TO_ACTION[key])
    return actions


def actions_to_path(
    start: Tuple[int, int], actions: List[int]
) -> List[Tuple[int, int]]:
    """反向校验：将动作施加到起始位置，应还原路径。"""
    # 与 INVARIANT 1 匹配
    DELTAS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    path = [tuple(start)]
    r, c = start
    for a in actions:
        dr, dc = DELTAS[a]
        r, c = r + dr, c + dc
        path.append((r, c))
    return path


# ---------------------------------------------------------------------------
# 可解释性过滤
# ---------------------------------------------------------------------------

def compute_observation_hash(
    patch: np.ndarray,   # (4, K, K)
    scalar: np.ndarray,  # (8,)
) -> int:
    """
    为 (patch, scalar) 计算 hash，用于检测"同一观测下多专家动作"的歧义样本。

    对 float patch 做分桶量化以避免浮点精度导致 hash 不匹配。
    """
    # 量化 patch 到 0.05 精度（20 bins）
    patch_binned = (patch * 20).astype(np.int16)
    scalar_binned = (scalar * 20).astype(np.int16)
    combined = np.concatenate([patch_binned.ravel(), scalar_binned])
    return hash(combined.tobytes())


def mark_ambiguous_samples(
    samples: List[dict],  # each has "obs_hash" and "action_expert"
) -> List[dict]:
    """
    标记 margin_excluded：同一 obs_hash 下出现多个不同专家动作的样本。

    INVARIANT 11: 对称让行等歧义样本剔除出 margin loss（TD 项可保留）。
    """
    # obs_hash -> set of actions
    hash_to_actions: Dict[int, Set[int]] = defaultdict(set)
    for s in samples:
        hash_to_actions[s["obs_hash"]].add(s["action_expert"])

    for s in samples:
        s["margin_excluded"] = len(hash_to_actions[s["obs_hash"]]) > 1

    return samples


def compute_wait_ratio(actions: List[int]) -> float:
    """计算动作序列中 wait 的比例。"""
    if len(actions) == 0:
        return 0.0
    return sum(1 for a in actions if a == 0) / len(actions)


def get_wait_margin_weight(
    wait_ratio: float,
    base_weight: float = 1.0,
    min_weight: float = 0.3,
) -> float:
    """
    wait 过多的样本降低 margin 权重。

    INVARIANT 11: wait 动作 margin 按比例降权。
    """
    if wait_ratio <= 0.3:
        return base_weight
    # 线性衰减：0.3 → 1.0 wait_ratio 映射回 1.0 → 0.3
    w = base_weight - (wait_ratio - 0.3) * (base_weight - min_weight) / 0.7
    return max(min_weight, w)
