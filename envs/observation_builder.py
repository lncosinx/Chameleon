"""
观测构建器。

在 Pogema 原生 3 通道观测之上追加：
- channel 3: 时间衰减自身足迹（范围 [0, 0.8]，当前位置 = 1.0）
- 8 维标量特征：目标相对方向(2) + 上一步动作 one-hot(5) + 到达标志(1)

输出: patch (N, 4, K, K) + scalar (N, 8)

INVARIANT 2: 输入仅为局部观测+自身历史，**不含**全局地图/距离场/LaCAM。
INVARIANT 5: 4 通道，**不得**再加与原生 ch2 重复的目标投影通道。
INVARIANT 6: 目标相对方向是合法局部特征（智能体始终知道自身 goal）。
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class ObservationBuilder:
    """
    观测构建器：原生 3 通道 → 4 通道 patch + 8 维标量。

    用法:
        builder = ObservationBuilder(num_agents=8, obs_radius=5, decay_rho=0.9)
        # 每个 episode 开始调用 reset()
        builder.reset()
        for step in range(max_steps):
            patches, scalars = builder.build(obs_list, targets, agents, prev_actions)
            # patches:  (N, 4, K, K)  float32
            # scalars:  (N, 8)        float32
    """

    def __init__(
        self,
        num_agents: int,
        obs_radius: int = 5,
        decay_rho: float = 0.9,
        max_footprint_history: int = 200,
    ):
        """
        Args:
            num_agents: 智能体数量。
            obs_radius: 观测半径（K = 2*r + 1）。
            decay_rho: 足迹衰减因子 ρ ∈ (0, 1)。
            max_footprint_history: 每个智能体最多存储的历史位置数。
        """
        self.num_agents = num_agents
        self.obs_radius = obs_radius
        self.K = 2 * obs_radius + 1
        self.decay_rho = decay_rho
        self.max_footprint_history = max_footprint_history

        # 足迹历史：per-agent deque of (global_row, global_col, step)
        self._footprint_history: List[deque] = []

        # 当前步数
        self._step: int = 0

        self.reset()

    def reset(self):
        """每个 episode 开始时调用，清空足迹历史。"""
        self._step = 0
        self._footprint_history = [
            deque(maxlen=self.max_footprint_history) for _ in range(self.num_agents)
        ]

    def build(
        self,
        obs_list: List[np.ndarray],
        agents_xy: List[Tuple[int, int]],
        targets_xy: List[Tuple[int, int]],
        prev_actions: np.ndarray,  # (N,) int, 值为 0-4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建增强观测。

        Args:
            obs_list: Pogema 原生观测列表，每体 shape (3, K, K)。
            agents_xy: 每体全局坐标 (row, col)。
            targets_xy: 每体目标坐标 (row, col)。
            prev_actions: 上一步动作 (N,)，值 ∈ {0,1,2,3,4}。

        Returns:
            patches: (N, 4, K, K) float32
            scalars: (N, 8) float32
        """
        N = self.num_agents
        K = self.K

        # 记录当前位置到足迹历史
        for i in range(N):
            self._footprint_history[i].append(
                (int(agents_xy[i][0]), int(agents_xy[i][1]), self._step)
            )

        patches = np.zeros((N, 4, K, K), dtype=np.float32)
        scalars = np.zeros((N, 8), dtype=np.float32)

        for i in range(N):
            obs_i = np.asarray(obs_list[i], dtype=np.float32)
            assert obs_i.shape == (3, K, K), (
                f"INVARIANT: obs must be (3,K,K), got {obs_i.shape}"
            )

            # ---- 通道 0-2: 原生观测 ----
            patches[i, :3] = obs_i

            # ---- 通道 3: 时间衰减足迹 ----
            self._build_footprint_channel(patches[i, 3], i, agents_xy[i])

            # ---- 标量特征 ----
            scalars[i] = self._build_scalar_features(
                i, agents_xy[i], targets_xy[i], prev_actions[i]
            )

        self._step += 1
        return patches, scalars

    def _build_footprint_channel(
        self,
        channel: np.ndarray,  # (K, K) to fill
        agent_idx: int,
        agent_pos: Tuple[int, int],
    ):
        """
        构建时间衰减足迹通道。

        当前位置 = 1.0。
        历史位置按 ρ^(step - visit_step) 衰减，裁切到 [0, 0.8]。
        同一 patch 格内多次访问取最大值。

        Args:
            channel: 输出通道 (K, K)，原地修改。
            agent_idx: 智能体索引。
            agent_pos: 当前全局位置 (row, col)。
        """
        r_center, c_center = int(agent_pos[0]), int(agent_pos[1])
        half = self.obs_radius

        for r_global, c_global, visit_step in self._footprint_history[agent_idx]:
            # 转换到以 agent 为中心的 patch 坐标
            dr = r_global - r_center
            dc = c_global - c_center

            # 检查是否在 FOV 内
            if abs(dr) > half or abs(dc) > half:
                continue

            pr = half + dr  # patch row
            pc = half + dc  # patch col

            if visit_step == self._step:
                # 当前位置
                channel[pr, pc] = 1.0
            else:
                # 历史足迹：衰减
                age = self._step - visit_step
                value = self.decay_rho ** age
                # 范围 [0, 0.8]
                value = min(value, 0.8)
                # 取最大值（多次踩过同一格）
                if value > channel[pr, pc]:
                    channel[pr, pc] = value

    def _build_scalar_features(
        self,
        agent_idx: int,
        agent_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        prev_action: int,
    ) -> np.ndarray:
        """
        构建 8 维标量特征。

        Returns:
            np.ndarray shape (8,):
                [0:2]  目标相对方向（归一化 2D 向量）
                [2:7]  上一步动作 one-hot (5)
                [7]    到达标志
        """
        feats = np.zeros(8, dtype=np.float32)

        # -- 目标相对方向 (2) --
        dr = float(target_pos[0] - agent_pos[0])
        dc = float(target_pos[1] - agent_pos[1])
        dist = np.sqrt(dr * dr + dc * dc)
        if dist > 1e-8:
            feats[0] = dr / dist
            feats[1] = dc / dist
        else:
            # 已到达，方向为零向量
            feats[0] = 0.0
            feats[1] = 0.0

        # -- 上一步动作 one-hot (5) --
        action = int(prev_action)
        assert 0 <= action <= 4, f"Action must be 0-4, got {action}"
        feats[2 + action] = 1.0

        # -- 到达标志 (1) --
        feats[7] = 1.0 if dist < 0.5 else 0.0

        return feats


# ---------------------------------------------------------------------------
# 便捷函数：直接从 PogemaWrapper 构建批量观测
# ---------------------------------------------------------------------------

def build_observations(
    wrapper,  # PogemaWrapper
    obs_list: List[np.ndarray],
    prev_actions: np.ndarray,
    builder: ObservationBuilder,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 PogemaWrapper 的一次 step 结果构建增强观测。

    Args:
        wrapper: PogemaWrapper 实例。
        obs_list: wrapper.step() 或 wrapper.reset() 返回的 obs_list。
        prev_actions: 上一步动作 (N,)。
        builder: ObservationBuilder 实例。

    Returns:
        patches: (N, 4, K, K)
        scalars: (N, 8)
    """
    agents_xy = wrapper.get_agents_xy()
    targets_xy = wrapper.get_targets_xy()

    # 显式断言：本函数绝不访问全局信息（INVARIANT 2, 3）
    # global_grid / starts / goals 仅用于训练期 reward/LaCAM，
    # 绝不出现在此函数的输入或计算路径中。
    return builder.build(obs_list, agents_xy, targets_xy, prev_actions)
