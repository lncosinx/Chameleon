"""
Reward 构建器：BFS 势函数 shaping（PBRS）。

R = 稀疏达标 - step_cost - collision_cost
    + lambda_f · clip(γ·Φ(s') − Φ(s), -shaping_clip, +shaping_clip)

两个重要修复：
1. goal 刷新（on_target: restart）时 Φ(s) 与 Φ(s') 基于不同目标，不可比，
   该步 shaping 置零（到达奖励由 goal_reward 单独处理）。
2. shaping 用 clip 截断极端值，防止多体 MAPF 中强制绕路产生习得性无助。
3. 移除 wait_cost：等待是拥挤 MAPF 的合理策略，不应额外惩罚。

INVARIANT 3: BFS 距离是训练期特权信号，只用于奖励，不进 policy。
INVARIANT 4: 使用 BFS 势函数 shaping，禁止点积对齐/曼哈顿势函数。
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

from envs.potential_field import PotentialFieldCache, UNREACHABLE_CONSTANT


class RewardBuilder:
    """
    BFS-PBRS 奖励构建器（可配置版）。

    用法:
        rb = RewardBuilder.from_config(cfg["reward"])
        rewards, info = rb.compute(
            obstacles, prev_agents_xy, agents_xy,
            targets_xy, prev_targets_xy, prev_actions, prev_dones
        )
    """

    def __init__(
        self,
        lambda_f: float = 1.0,
        shaping_clip: float = 5.0,
        goal_reward: float = 1.0,
        step_cost: float = 0.01,
        collision_cost: float = 0.1,
        gamma: float = 0.99,
    ):
        self.lambda_f      = lambda_f
        self.shaping_clip  = shaping_clip
        self.goal_reward   = goal_reward
        self.step_cost     = step_cost
        self.collision_cost = collision_cost
        self.gamma         = gamma

        self._cache = PotentialFieldCache()

    @classmethod
    def from_config(cls, cfg: dict) -> "RewardBuilder":
        """从 yaml config dict 构建（键名与 __init__ 参数一致）。"""
        return cls(
            lambda_f      = cfg.get("lambda_f",      1.0),
            shaping_clip  = cfg.get("shaping_clip",  5.0),
            goal_reward   = cfg.get("goal_reward",   1.0),
            step_cost     = cfg.get("step_cost",     0.01),
            collision_cost= cfg.get("collision_cost",0.1),
            gamma         = cfg.get("gamma",         0.99),
        )

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def compute(
        self,
        obstacles: np.ndarray,
        prev_agents_xy: List[Tuple[int, int]],
        agents_xy: List[Tuple[int, int]],
        targets_xy: List[Tuple[int, int]],
        prev_targets_xy: Optional[List[Tuple[int, int]]] = None,
        prev_actions: Optional[np.ndarray] = None,
        prev_dones: Optional[List[bool]] = None,
    ) -> Tuple[List[float], dict]:
        """计算每体的 PBRS reward（向量化）。"""
        N = len(agents_xy)

        if prev_targets_xy is None:
            prev_targets_xy = targets_xy
        if prev_actions is None:
            prev_actions = np.zeros(N, dtype=np.int64)
        if prev_dones is None:
            prev_dones = [False] * N

        prev_pos  = np.array(prev_agents_xy,  dtype=np.int32)
        curr_pos  = np.array(agents_xy,       dtype=np.int32)
        goal_pos  = np.array(targets_xy,      dtype=np.int32)
        prev_goal = np.array(prev_targets_xy, dtype=np.int32)
        done_mask = np.array(prev_dones,      dtype=bool)
        active    = ~done_mask

        rewards = np.zeros(N, dtype=np.float32)

        # ---- 1. 稀疏到达奖励 ----
        dist_curr = np.abs(curr_pos[:, 0] - goal_pos[:, 0]) + np.abs(curr_pos[:, 1] - goal_pos[:, 1])
        arrived   = (dist_curr < 1) & active
        rewards  += arrived.astype(np.float32) * self.goal_reward

        # ---- 2. 每步惩罚 ----
        rewards  -= active.astype(np.float32) * self.step_cost

        # ---- 3. 碰撞惩罚（向量化 O(N²)）----
        pos_i = curr_pos[:, np.newaxis, :]
        pos_j = curr_pos[np.newaxis, :, :]
        same  = np.all(pos_i == pos_j, axis=-1)
        np.fill_diagonal(same, False)
        collision_counts = same.sum(axis=1)
        rewards -= collision_counts.astype(np.float32) * self.collision_cost * active

        # ---- 4. PBRS shaping（关键修复）----
        shaping = np.zeros(N, dtype=np.float32)
        for i in range(N):
            if done_mask[i]:
                continue

            # goal 刷新检测：目标坐标变化 → 该步跨 goal，shaping 置零
            goal_changed = not (
                int(prev_goal[i, 0]) == int(goal_pos[i, 0]) and
                int(prev_goal[i, 1]) == int(goal_pos[i, 1])
            )
            if goal_changed:
                continue  # shaping[i] stays 0; goal_reward already handles arrival

            phi_prev = self._phi(
                obstacles,
                (int(prev_pos[i, 0]), int(prev_pos[i, 1])),
                (int(prev_goal[i, 0]), int(prev_goal[i, 1])),
            )
            phi_curr = self._phi(
                obstacles,
                (int(curr_pos[i, 0]), int(curr_pos[i, 1])),
                (int(goal_pos[i, 0]),  int(goal_pos[i, 1])),
            )

            unreachable_thresh = -UNREACHABLE_CONSTANT / 2
            if phi_prev > unreachable_thresh and phi_curr > unreachable_thresh:
                raw = self.gamma * phi_curr - phi_prev
                # clip 防止强制绕路（多体 MAPF）或 goal-restart 残差产生极端惩罚
                shaping[i] = float(np.clip(raw, -self.shaping_clip, self.shaping_clip))

        rewards += self.lambda_f * shaping

        info = {
            "reward/goal_count":      int(arrived.sum()),
            "reward/collision_count": int((collision_counts > 0).sum()),
            "reward/shaping_mean":    float(shaping[active].mean()) if active.any() else 0.0,
            "reward/total_mean":      float(rewards.mean()),
        }

        return rewards.tolist(), info

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _phi(self, obstacles: np.ndarray, pos: Tuple[int, int],
             goal: Tuple[int, int]) -> float:
        d = self._cache.get(obstacles, goal)
        r, c = int(pos[0]), int(pos[1])
        if 0 <= r < d.shape[0] and 0 <= c < d.shape[1]:
            return float(-d[r, c])
        return float(-UNREACHABLE_CONSTANT)

    def clear_cache(self):
        self._cache.clear()
