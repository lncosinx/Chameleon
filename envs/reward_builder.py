"""
Reward 构建器：BFS 势函数 shaping（PBRS）。

R = 稀疏达标 - 每步惩罚 - 碰撞惩罚 - 无意义等待惩罚
    + λ_F · (γ·Φ(s') - Φ(s))

其中 Φ = -d_BFS(p_i, goal_i)，势函数 shaping 项为策略不变。

INVARIANT 3: BFS 距离是训练期特权信号，**只用于环境计算 reward，不进 policy**。
INVARIANT 4: 使用 BFS 势函数 shaping，**禁止**点积对齐/曼哈顿势函数。

性能策略（CLAUDE.md）：热路径禁止 per-agent/per-step Python 循环；
碰撞检测、到达检测、等待检测均用 NumPy 向量化实现。
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

from envs.potential_field import PotentialFieldCache, UNREACHABLE_CONSTANT


class RewardBuilder:
    """
    BFS-PBRS 奖励构建器。

    用法:
        rb = RewardBuilder(lambda_f=0.1, step_cost=0.01, collision_cost=0.1, wait_cost=0.05)
        rewards, info = rb.compute(
            obstacles, prev_agents_xy, agents_xy,
            targets_xy, prev_targets_xy, prev_actions, prev_dones
        )
    """

    def __init__(
        self,
        lambda_f: float = 0.1,
        goal_reward: float = 1.0,
        step_cost: float = 0.01,
        collision_cost: float = 0.1,
        wait_cost: float = 0.05,
        gamma: float = 0.99,
    ):
        self.lambda_f      = lambda_f
        self.goal_reward   = goal_reward
        self.step_cost     = step_cost
        self.collision_cost = collision_cost
        self.wait_cost     = wait_cost
        self.gamma         = gamma

        self._cache = PotentialFieldCache()

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def compute(
        self,
        obstacles: np.ndarray,                               # (H, W)
        prev_agents_xy: List[Tuple[int, int]],
        agents_xy: List[Tuple[int, int]],
        targets_xy: List[Tuple[int, int]],
        prev_targets_xy: Optional[List[Tuple[int, int]]] = None,
        prev_actions: Optional[np.ndarray] = None,           # (N,) int
        prev_dones: Optional[List[bool]] = None,
    ) -> Tuple[List[float], dict]:
        """
        计算每体的 PBRS reward（向量化）。
        """
        N = len(agents_xy)

        if prev_targets_xy is None:
            prev_targets_xy = targets_xy
        if prev_actions is None:
            prev_actions = np.zeros(N, dtype=np.int64)
        if prev_dones is None:
            prev_dones = [False] * N

        # 转为 numpy 数组（向量化基础）
        prev_pos = np.array(prev_agents_xy, dtype=np.int32)   # (N, 2)
        curr_pos = np.array(agents_xy, dtype=np.int32)         # (N, 2)
        goal_pos = np.array(targets_xy, dtype=np.int32)        # (N, 2)
        prev_goal = np.array(prev_targets_xy, dtype=np.int32)  # (N, 2)
        prev_act  = np.asarray(prev_actions, dtype=np.int32)   # (N,)
        done_mask = np.array(prev_dones, dtype=bool)           # (N,)
        active    = ~done_mask                                  # (N,)

        rewards = np.zeros(N, dtype=np.float32)

        # ---- 1. 稀疏达标奖励（向量化）----
        dist_curr = np.abs(curr_pos[:, 0] - goal_pos[:, 0]) + np.abs(curr_pos[:, 1] - goal_pos[:, 1])
        arrived   = (dist_curr < 1) & active           # 到达且上步未 done
        rewards   += arrived.astype(np.float32) * self.goal_reward

        # ---- 2. 每步惩罚 ----
        rewards -= active.astype(np.float32) * self.step_cost

        # ---- 3. 碰撞惩罚（向量化，O(N²) → 广播）----
        # curr_pos (N, 2)，计算两两相等的对数
        # 展开：pos_i (N, 1, 2) vs pos_j (1, N, 2)
        pos_i = curr_pos[:, np.newaxis, :]      # (N, 1, 2)
        pos_j = curr_pos[np.newaxis, :, :]      # (1, N, 2)
        same  = np.all(pos_i == pos_j, axis=-1) # (N, N)
        np.fill_diagonal(same, False)
        collision_counts = same.sum(axis=1)      # (N,) — 每体与多少其他体碰撞
        rewards -= collision_counts.astype(np.float32) * self.collision_cost * active

        # ---- 4. 无意义等待惩罚（向量化）----
        is_wait   = (prev_act == 0) & active
        not_at_goal = dist_curr >= 1
        meaningless_wait = is_wait & not_at_goal
        rewards -= meaningless_wait.astype(np.float32) * self.wait_cost

        # ---- 5. PBRS shaping（逐体查 BFS 表，必须按 agent 循环）----
        # BFS 查表本身是 O(1)（数组索引），循环仅遍历 N 个 goal 不同的智能体
        shaping = np.zeros(N, dtype=np.float32)
        for i in range(N):
            if done_mask[i]:
                continue
            phi_prev = self._phi(obstacles, (int(prev_pos[i, 0]), int(prev_pos[i, 1])),
                                 (int(prev_goal[i, 0]), int(prev_goal[i, 1])))
            phi_curr = self._phi(obstacles, (int(curr_pos[i, 0]), int(curr_pos[i, 1])),
                                 (int(goal_pos[i, 0]), int(goal_pos[i, 1])))
            if phi_prev > -UNREACHABLE_CONSTANT / 2 and phi_curr > -UNREACHABLE_CONSTANT / 2:
                shaping[i] = self.gamma * phi_curr - phi_prev

        rewards += self.lambda_f * shaping

        info = {
            "reward/goal_count":      int(arrived.sum()),
            "reward/collision_count": int((collision_counts > 0).sum()),
            "reward/wait_count":      int(meaningless_wait.sum()),
            "reward/shaping_mean":    float(shaping[active].mean()) if active.any() else 0.0,
            "reward/total_mean":      float(rewards.mean()),
        }

        return rewards.tolist(), info

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _phi(self, obstacles: np.ndarray, pos: Tuple[int, int],
              goal: Tuple[int, int]) -> float:
        """势函数 Φ(pos) = -d_BFS(pos, goal)。"""
        d = self._cache.get(obstacles, goal)
        r, c = int(pos[0]), int(pos[1])
        if 0 <= r < d.shape[0] and 0 <= c < d.shape[1]:
            dist = d[r, c]
        else:
            dist = UNREACHABLE_CONSTANT
        return float(-dist)

    def clear_cache(self):
        """清空 BFS 缓存（切换地图时调用）。"""
        self._cache.clear()
