"""
Reward 构建器：BFS 势函数 shaping（PBRS）。

R = 稀疏达标 - 每步惩罚 - 碰撞惩罚 - 无意义等待惩罚
    + λ_F · (γ·Φ(s') - Φ(s))

其中 Φ = -d_BFS(p_i, goal_i)，势函数 shaping 项为策略不变。

INVARIANT 3: BFS 距离是训练期特权信号，**只用于环境计算 reward，不进 policy**。
INVARIANT 4: 使用 BFS 势函数 shaping，**禁止**点积对齐/曼哈顿势函数。
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
        # 每步调用
        rewards, info = rb.compute(
            obstacles, prev_agents_xy, agents_xy,
            prev_goals_xy, goals_xy, prev_actions, prev_dones
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
        """
        Args:
            lambda_f: PBRS shaping 权重。
            goal_reward: 到达目标的正奖励。
            step_cost: 每步时间惩罚（鼓励高效路径）。
            collision_cost: 顶点冲突惩罚（两体占据同一格）。
            wait_cost: 无意义等待惩罚（无障碍时 wait）。
            gamma: 折扣因子。
        """
        self.lambda_f = lambda_f
        self.goal_reward = goal_reward
        self.step_cost = step_cost
        self.collision_cost = collision_cost
        self.wait_cost = wait_cost
        self.gamma = gamma

        self._cache = PotentialFieldCache()

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def compute(
        self,
        obstacles: np.ndarray,           # (H, W), 0=free, 1=obstacle
        prev_agents_xy: List[Tuple[int, int]],
        agents_xy: List[Tuple[int, int]],
        targets_xy: List[Tuple[int, int]],      # 当前目标（s' 之后的目标）
        prev_targets_xy: Optional[List[Tuple[int, int]]] = None,  # s 时的目标
        prev_actions: Optional[np.ndarray] = None,  # (N,) int
        prev_dones: Optional[List[bool]] = None,     # 上一步各体是否已 done
    ) -> Tuple[List[float], dict]:
        """
        计算每体的 PBRS reward。

        Args:
            obstacles: 全局障碍物栅格（训练期特权信号）。
            prev_agents_xy: 上一步各体位置 (s)。
            agents_xy: 当前各体位置 (s')。
            targets_xy: 当前各体目标（用于 Φ(s')）。
            prev_targets_xy: 上一步各体目标（用于 Φ(s)），默认同 targets_xy。
            prev_actions: 上一步动作，用于无意义等待检测。
            prev_dones: 上一步 done 标志。

        Returns:
            rewards: list of float, 每体奖励。
            info: dict with diagnostic fields.
        """
        N = len(agents_xy)

        if prev_targets_xy is None:
            prev_targets_xy = targets_xy

        if prev_actions is None:
            prev_actions = np.zeros(N, dtype=np.int64)

        if prev_dones is None:
            prev_dones = [False] * N

        rewards = np.zeros(N, dtype=np.float32)

        # ---- 1. 稀疏达标奖励 ----
        for i in range(N):
            pr, pc = prev_agents_xy[i]
            tr, tc = targets_xy[i]  # 使用当前目标（到达检查在 s'）
            dist_prev = abs(pr - tr) + abs(pc - tc)
            # 到达判定：距离从 >0 变为 0（或在目标格上）
            ar, ac = agents_xy[i]
            dist_curr = abs(ar - tr) + abs(ac - tc)
            if dist_curr < 0.5 and not prev_dones[i]:
                rewards[i] += self.goal_reward

        # ---- 2. 每步惩罚 ----
        for i in range(N):
            if not prev_dones[i]:
                rewards[i] -= self.step_cost

        # ---- 3. 碰撞惩罚 ----
        collision_count = 0
        for i in range(N):
            for j in range(i + 1, N):
                if agents_xy[i] == agents_xy[j]:
                    # 顶点冲突
                    rewards[i] -= self.collision_cost
                    rewards[j] -= self.collision_cost
                    collision_count += 1

        # ---- 4. 无意义等待惩罚 ----
        # 等待而在无障碍的情况下不移动
        wait_count = 0
        for i in range(N):
            if prev_actions[i] == 0 and not prev_dones[i]:
                ar, ac = agents_xy[i]
                tr, tc = targets_xy[i]
                # 检查是否已到达
                if abs(ar - tr) < 0.5 and abs(ac - tc) < 0.5:
                    continue  # 已到达，等待合理
                # 检查周围是否被障碍包围（无路可走则等待合理）
                # 简化：只要前方有至少一个可行方向，等待就无意义
                # 这里用宽松判定——只要不在目标上且周围有自由格，wait 就罚
                rewards[i] -= self.wait_cost
                wait_count += 1

        # ---- 5. PBRS shaping: F = γ·Φ(s') - Φ(s) ----
        shaping = np.zeros(N, dtype=np.float32)
        for i in range(N):
            if prev_dones[i]:
                continue
            phi_prev = self._phi(obstacles, prev_agents_xy[i], prev_targets_xy[i])
            phi_curr = self._phi(obstacles, agents_xy[i], targets_xy[i])

            # 不可达时跳过 shaping（避免无意义的大常数差值）
            if phi_prev <= -UNREACHABLE_CONSTANT / 2 or phi_curr <= -UNREACHABLE_CONSTANT / 2:
                shaping[i] = 0.0
            else:
                shaping[i] = self.gamma * phi_curr - phi_prev

        rewards += self.lambda_f * shaping

        info = {
            "reward/goal": float(np.sum([self.goal_reward for i in range(N)
                                         if not prev_dones[i]
                                         and abs(agents_xy[i][0] - targets_xy[i][0])
                                         + abs(agents_xy[i][1] - targets_xy[i][1]) < 0.5])),
            "reward/step_penalty": float(-self.step_cost * sum(1 for d in prev_dones if not d)),
            "reward/collision": float(-self.collision_cost * collision_count),
            "reward/wait_penalty": float(-self.wait_cost * wait_count),
            "reward/shaping_mean": float(np.mean(shaping)),
            "reward/total_mean": float(np.mean(rewards)),
        }

        return [float(r) for r in rewards], info

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _phi(self, obstacles: np.ndarray, pos: Tuple[int, int],
              goal: Tuple[int, int]) -> float:
        """
        势函数 Φ(pos) = -d_BFS(pos, goal)。
        不可达 = -UNREACHABLE_CONSTANT。
        """
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
