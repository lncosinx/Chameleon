"""
BFS 测地距离场（训练期特权信号）。

对每个 (map, goal) 在四连通可行图上 BFS 计算真实测地距离场。
缓存结果避免重复计算。不可达格填大常数。

**只用于环境计算 reward shaping，不进入任何 agent 的观测**（INVARIANT 3）。
"""
from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional


# 四连通方向：上、下、左、右
_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 不可达格常量（应大于 max_episode_steps）
UNREACHABLE_CONSTANT = 1_000_000


def _bfs_distance(
    grid: np.ndarray,  # (H, W), 0=free, 1=obstacle
    goal: Tuple[int, int],
) -> np.ndarray:
    """
    在四连通可行图上 BFS 计算测地距离场。

    Args:
        grid: 障碍物栅格，0=free，1=obstacle。
        goal: 目标坐标 (row, col)。

    Returns:
        dist: (H, W) int32，dist[r,c] = 从 (r,c) 到 goal 的 BFS 测地距离。
              不可达格 = UNREACHABLE_CONSTANT。
    """
    H, W = grid.shape
    dist = np.full((H, W), UNREACHABLE_CONSTANT, dtype=np.int32)

    gr, gc = int(goal[0]), int(goal[1])

    # 目标格若为障碍，仍计算（可能已被环境清除），但不从此 BFS
    if gr < 0 or gr >= H or gc < 0 or gc >= W:
        return dist
    if grid[gr, gc] != 0:
        # 目标在障碍上：罕见情况，返回全不可达
        return dist

    dist[gr, gc] = 0
    q = deque()
    q.append((gr, gc))

    while q:
        r, c = q.popleft()
        d_next = dist[r, c] + 1
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if grid[nr, nc] == 0 and dist[nr, nc] == UNREACHABLE_CONSTANT:
                    dist[nr, nc] = d_next
                    q.append((nr, nc))

    return dist


class PotentialFieldCache:
    """
    势函数缓存：对每个 (map_hash, goal) 缓存 BFS 距离场。
    每张地图每个 goal 只算一次，开销可忽略。

    用法:
        cache = PotentialFieldCache()
        d_bfs = cache.get(obstacles, goal)
        phi = -d_bfs  # 势函数
    """

    def __init__(self):
        self._cache: Dict[Tuple[int, int, int], np.ndarray] = {}

    @staticmethod
    def _make_key(grid: np.ndarray, goal: Tuple[int, int]) -> Tuple[int, int, int]:
        """生成缓存 key：(grid_hash, goal_row, goal_col)。"""
        grid_bytes = grid.tobytes() if isinstance(grid, np.ndarray) else grid
        grid_hash = hash(grid_bytes) if not isinstance(grid_bytes, int) else grid_bytes
        return (grid_hash, int(goal[0]), int(goal[1]))

    def get(self, grid: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
        """
        获取 (grid, goal) 的 BFS 距离场，命中缓存则直接返回。

        Args:
            grid: (H, W) 障碍物栅格。
            goal: 目标坐标 (row, col)。

        Returns:
            dist: (H, W) BFS 测地距离场。
        """
        key = self._make_key(grid, goal)
        if key not in self._cache:
            self._cache[key] = _bfs_distance(grid, goal)
        return self._cache[key]

    def clear(self):
        """清空缓存。"""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# 快捷函数
# ---------------------------------------------------------------------------

def compute_potential(
    grid: np.ndarray,
    goal: Tuple[int, int],
    cache: Optional[PotentialFieldCache] = None,
) -> np.ndarray:
    """
    计算势函数 Φ = -d_BFS(pos, goal)。

    Args:
        grid: (H, W) 障碍物栅格。
        goal: 目标坐标。
        cache: 可选缓存。

    Returns:
        phi: (H, W) 势函数值（负数；-UNREACHABLE_CONSTANT 表示不可达）。
    """
    if cache is not None:
        d = cache.get(grid, goal)
    else:
        d = _bfs_distance(grid, goal)
    return -d.astype(np.float32)
