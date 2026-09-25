"""
观测构建器。

在 Pogema 原生 3 通道观测之上追加：
- channel 3: 时间衰减自身足迹（范围 [0, 0.8]，当前位置 = 1.0）
- 8 维标量特征：目标相对方向(2) + 上一步动作 one-hot(5) + 到达标志(1)

输出: patch (N, 4, K, K) + scalar (N, 8)

INVARIANT 2: 输入仅为局部观测+自身历史，**不含**全局地图/距离场/LaCAM。
INVARIANT 5: 4 通道，**不得**再加与原生 ch2 重复的目标投影通道。
INVARIANT 6: 目标相对方向是合法局部特征（智能体始终知道自身 goal）。

性能策略（CLAUDE.md）：热路径禁止 per-agent/per-step Python 循环；
足迹通道与标量特征均用 NumPy 向量化实现。
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
from collections import deque

from envs.potential_field import PotentialFieldCache, UNREACHABLE_CONSTANT


class ObservationBuilder:
    """
    观测构建器：原生 3 通道 → 4 通道 patch + 8 维标量。

    用法:
        builder = ObservationBuilder(num_agents=8, obs_radius=5, decay_rho=0.9)
        builder.reset()
        for step in range(max_steps):
            patches, scalars = builder.build(obs_list, agents_xy, targets_xy, prev_actions)
            # patches:  (N, 4, K, K)  float32
            # scalars:  (N, 8)        float32
    """

    def __init__(
        self,
        num_agents: int,
        obs_radius: int = 5,
        decay_rho: float = 0.9,
        max_footprint_history: int = 200,
        use_cost2go: bool = False,
        cost2go_clip: float = 10.0,
    ):
        self.num_agents = num_agents
        self.obs_radius = obs_radius
        self.K = 2 * obs_radius + 1
        self.decay_rho = decay_rho
        self.max_footprint_history = max_footprint_history

        # cost2go 通道（T2）：FOV 内到目标的 BFS 测地距离（相对自身、居中到 [0,1]），
        # 障碍感知的导航梯度。EPH 的 heuri_map / MAPF-GPT 的 cost2go 同类。
        # 从全局静态障碍图算（对齐 baseline 惯例；INVARIANT 3 已放宽为合法局部特征）。
        self.use_cost2go = use_cost2go
        self.cost2go_clip = float(cost2go_clip)
        self.patch_channels = 5 if use_cost2go else 4
        self._c2g_cache = PotentialFieldCache()

        # 足迹历史：per-agent deque of (global_row, global_col, step)
        self._footprint_history: List[deque] = []
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
        obstacles: Optional[np.ndarray] = None,  # (H,W) 0/1，use_cost2go 时必需（算 cost2go）
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建增强观测（向量化实现）。

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
        half = self.obs_radius

        # 记录当前位置到足迹历史（顺序循环不可避免，但只有 N 次）
        for i in range(N):
            self._footprint_history[i].append(
                (int(agents_xy[i][0]), int(agents_xy[i][1]), self._step)
            )

        # ---- 通道 0-2: 原生观测 (N, 3, K, K) ----
        patches = np.empty((N, self.patch_channels, K, K), dtype=np.float32)
        for i in range(N):
            obs_i = np.asarray(obs_list[i], dtype=np.float32)
            assert obs_i.shape == (3, K, K), (
                f"INVARIANT: obs must be (3,K,K), got {obs_i.shape}"
            )
            patches[i, :3] = obs_i

        # ---- 通道 3: 时间衰减足迹（向量化计算）----
        patches[:, 3] = 0.0
        self._build_footprint_channels_vectorized(patches[:, 3], agents_xy)

        # ---- 通道 4: cost2go（FOV 内 BFS 测地距离，相对自身居中到 [0,1]）----
        if self.use_cost2go:
            patches[:, 4] = 0.5   # 默认中性（=与自身同距）
            if obstacles is not None:
                self._build_cost2go_channel(patches[:, 4], agents_xy, targets_xy, obstacles)

        # ---- 标量特征（向量化）----
        scalars = self._build_scalar_features_vectorized(agents_xy, targets_xy, prev_actions)

        self._step += 1
        return patches, scalars

    # ------------------------------------------------------------------
    # 向量化内部方法
    # ------------------------------------------------------------------

    def _build_footprint_channels_vectorized(
        self,
        channels: np.ndarray,   # (N, K, K) to fill, already zero
        agents_xy: List[Tuple[int, int]],
    ):
        """
        向量化构建 N 个足迹通道。

        对每个 agent 的足迹历史批量计算 patch 坐标与衰减值。
        """
        N = self.num_agents
        half = self.obs_radius
        step = self._step

        for i in range(N):
            hist = self._footprint_history[i]
            if not hist:
                continue

            r_center = int(agents_xy[i][0])
            c_center = int(agents_xy[i][1])

            # 提取历史为数组
            hist_arr = np.array(list(hist), dtype=np.int32)  # (H, 3): r, c, visit_step
            r_glob  = hist_arr[:, 0]
            c_glob  = hist_arr[:, 1]
            vstep   = hist_arr[:, 2]

            # patch 坐标
            dr = r_glob - r_center
            dc = c_glob - c_center

            # FOV 内过滤
            in_fov = (np.abs(dr) <= half) & (np.abs(dc) <= half)
            if not in_fov.any():
                continue

            dr = dr[in_fov]
            dc = dc[in_fov]
            vstep_f = vstep[in_fov]

            pr = half + dr  # patch row
            pc = half + dc  # patch col

            # 计算衰减值
            is_current = vstep_f == step
            ages = step - vstep_f

            # 历史足迹：decay_rho^age，截到 [0, 0.8]
            values = np.where(
                is_current,
                1.0,
                np.minimum(self.decay_rho ** ages.astype(np.float32), 0.8),
            )

            # 同一 patch 格取最大值（np.maximum.at 是向量化 scatter-max）
            np.maximum.at(channels[i], (pr, pc), values)

    def _build_cost2go_channel(
        self,
        channel: np.ndarray,          # (N, K, K) to fill（已置 0.5）
        agents_xy: List[Tuple[int, int]],
        targets_xy: List[Tuple[int, int]],
        obstacles: np.ndarray,        # (H, W) 0/1
    ):
        """填 cost2go 通道：每体 FOV 内每格到自身目标的 BFS 测地距离，
        **相对自身距离居中**、clip 到 ±cost2go_clip、映射到 [0,1]（0.5=与自身同距，
        <0.5=更近目标，>0.5=更远）。绝对偏移被消除 → 无论距目标多远，局部梯度都可见。
        从全局静态障碍图算（cache 按 (grid,goal) 复用）；INVARIANT 3 已放宽。
        """
        N, half, K = self.num_agents, self.obs_radius, self.K
        H, W = obstacles.shape
        R = self.cost2go_clip
        for i in range(N):
            r_c, c_c = int(agents_xy[i][0]), int(agents_xy[i][1])
            # 越界（如 finished 哨兵位）→ 保持中性 0.5
            if not (0 <= r_c < H and 0 <= c_c < W):
                continue
            dist = self._c2g_cache.get(obstacles, targets_xy[i])   # (H,W) int32
            d_self = dist[r_c, c_c]
            if d_self >= UNREACHABLE_CONSTANT:
                continue   # 自身不可达（罕见）→ 中性
            # FOV 窗口（padding=obs_radius 保证在界内，但仍安全裁剪）
            r0, r1 = r_c - half, r_c + half + 1
            c0, c1 = c_c - half, c_c + half + 1
            pr0, pc0 = max(0, -r0), max(0, -c0)
            rr0, rr1 = max(0, r0), min(H, r1)
            cc0, cc1 = max(0, c0), min(W, c1)
            crop = dist[rr0:rr1, cc0:cc1].astype(np.float32)
            rel = crop - float(d_self)
            # 不可达格：dist 巨大 → rel 巨大 → clip 到 +R（"很远"）
            rel = np.clip(rel, -R, R) / R          # [-1,1]
            vals = 0.5 + 0.5 * rel                  # [0,1]，居中 0.5
            ph, pw = crop.shape
            channel[i, pr0:pr0 + ph, pc0:pc0 + pw] = vals

    def _build_scalar_features_vectorized(
        self,
        agents_xy: List[Tuple[int, int]],
        targets_xy: List[Tuple[int, int]],
        prev_actions: np.ndarray,
    ) -> np.ndarray:
        """
        向量化构建 8 维标量特征 (N, 8)。

        [0:2]  目标相对方向（归一化 2D 向量）
        [2:7]  上一步动作 one-hot (5)
        [7]    到达标志
        """
        N = self.num_agents
        scalars = np.zeros((N, 8), dtype=np.float32)

        agents_arr  = np.array(agents_xy,  dtype=np.float32)   # (N, 2)
        targets_arr = np.array(targets_xy, dtype=np.float32)   # (N, 2)

        diff = targets_arr - agents_arr  # (N, 2): [dr, dc]
        dist = np.linalg.norm(diff, axis=1, keepdims=True)     # (N, 1)

        # 归一化方向
        nonzero = dist[:, 0] > 1e-8
        scalars[nonzero, 0:2] = diff[nonzero] / dist[nonzero]  # 方向向量

        # 上一步动作 one-hot
        actions = np.asarray(prev_actions, dtype=np.int32).clip(0, 4)
        scalars[np.arange(N), 2 + actions] = 1.0

        # 到达标志
        scalars[:, 7] = (dist[:, 0] < 0.5).astype(np.float32)

        return scalars


# ---------------------------------------------------------------------------
# 便捷函数：直接从 PogemaWrapper 构建批量观测
# ---------------------------------------------------------------------------

def build_observations(
    wrapper,  # PogemaWrapper
    obs_list: List[np.ndarray],
    prev_actions: np.ndarray,
    builder: "ObservationBuilder",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 PogemaWrapper 的一次 step 结果构建增强观测。

    Returns:
        patches: (N, C, K, K)  C=4 或 5（use_cost2go）
        scalars: (N, 8)
    """
    agents_xy  = wrapper.get_agents_xy()
    targets_xy = wrapper.get_targets_xy()

    # cost2go 需要全局静态障碍图（INVARIANT 3 已放宽为合法局部特征：仅取 FOV 裁剪）。
    # 非 cost2go 模式下 obstacles 不被使用，观测仍只含局部信息（INVARIANT 2）。
    obstacles = wrapper.global_grid if builder.use_cost2go else None
    return builder.build(obs_list, agents_xy, targets_xy, prev_actions, obstacles=obstacles)
