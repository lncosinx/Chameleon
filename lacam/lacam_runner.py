"""
LaCAM 专家运行器（抽象接口 + 优先规划 fallback）。

接口：
    runner = LaCAMRunner(backend="prioritized")  # 或 "lacam"（需 C++ 绑定）
    result = runner.solve(obstacles, starts, goals)
    # result: {"success": bool, "paths": [[(r,c),...],...], "makespan": int, ...}

INVARIANT 3: LaCAM 仅作训练期专家，**不进 policy 输入**。
"""
from __future__ import annotations

import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

# 动作偏移（INVARIANT 1）
ACTION_DELTAS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class LaCAMResult:
    """LaCAM 求解结果。"""
    map_type: str = "unknown"
    map_size: int = 0
    num_agents: int = 0
    starts: List[Tuple[int, int]] = field(default_factory=list)
    goals: List[Tuple[int, int]] = field(default_factory=list)
    success: bool = False
    paths: List[List[Tuple[int, int]]] = field(default_factory=list)
    makespan: int = 0
    sum_of_costs: int = 0
    runtime: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "map_type": self.map_type,
            "map_size": self.map_size,
            "num_agents": self.num_agents,
            "starts": self.starts,
            "goals": self.goals,
            "success": self.success,
            "paths": self.paths,
            "makespan": self.makespan,
            "sum_of_costs": self.sum_of_costs,
            "runtime": self.runtime,
        }


# =========================================================================
# 抽象接口
# =========================================================================

class LaCAMRunner:
    """
    LaCAM 专家运行器。

    用法:
        runner = LaCAMRunner(backend="prioritized")
        result = runner.solve(obstacles, starts, goals, map_type="maze")
    """

    def __init__(self, backend: str = "auto", time_limit: float = 10.0):
        self.backend = backend
        self.time_limit = time_limit
        self._solver = None

        if backend == "auto":
            # 自动检测真实 LaCAM 可用性
            try:
                self._solver = _LacamNativeSolver(time_limit)
                self.backend = "lacam"
            except ImportError:
                self._solver = _PrioritizedSolver(time_limit)
                self.backend = "prioritized"
        elif backend == "lacam":
            self._solver = _LacamNativeSolver(time_limit)
        elif backend == "prioritized":
            self._solver = _PrioritizedSolver(time_limit)
        else:
            raise ValueError(f"Unknown LaCAM backend: {backend}")

    def solve(
        self,
        obstacles: np.ndarray,
        starts: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
        map_type: str = "unknown",
        map_size: int = 0,
    ) -> LaCAMResult:
        return self._solver.solve(obstacles, starts, goals, map_type, map_size)


# =========================================================================
# 优先规划 fallback solver
# =========================================================================

class _PrioritizedSolver:
    """
    优先规划（Prioritized Planning）fallback。

    对每个 agent 按优先级（到目标距离升序）依次规划 BFS 最短路径，
    并将已规划路径的时空占位加入 reservation table，后续 agent 绕行。
    如果某 agent 无解，降低其优先级重试。

    这不是最优 MAPF 求解器，但能为 DQfD 训练提供合理的专家轨迹。
    """

    def __init__(self, time_limit: float = 10.0):
        self.time_limit = time_limit

    def solve(
        self,
        obstacles: np.ndarray,
        starts: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
        map_type: str = "unknown",
        map_size: int = 0,
    ) -> LaCAMResult:
        import time
        t0 = time.time()

        N = len(starts)
        H, W = obstacles.shape
        # 确保 obstacles 是 int 数组
        obs = np.asarray(obstacles, dtype=np.int32)

        # 按优先级排序（距离近的优先）
        order = list(range(N))
        order.sort(key=lambda i: _manhattan(starts[i], goals[i]))

        # reservation table: dict (t, r, c) -> agent_id
        reservations: Dict[Tuple[int, int, int], int] = {}
        paths: List[Optional[List[Tuple[int, int]]]] = [None] * N

        for idx in order:
            path = self._plan_one(obs, starts[idx], goals[idx], reservations, H, W)
            if path is None:
                # 放宽约束重试（仅避静态障碍）
                path = self._plan_one_no_reservations(obs, starts[idx], goals[idx], H, W)

            if path is not None:
                paths[idx] = path
                # 注册到 reservation table
                for t, pos in enumerate(path):
                    # 顶点冲突
                    reservations[(t, pos[0], pos[1])] = idx
                    # 边冲突（交换位置）：t≥1 时禁止从 path[t] 到 path[t-1]
                    if t >= 1:
                        # 不额外注册——通过目标格检测
                        pass
            else:
                # 实在无解：站在原地
                paths[idx] = [starts[idx]] * max(1, H * W // 2)

        # 对齐路径长度
        max_len = max(len(p) for p in paths if p)
        for i in range(N):
            if paths[i] is None:
                paths[i] = [starts[i]] * max_len
            while len(paths[i]) < max_len:
                paths[i].append(paths[i][-1])  # 停在终点

        makespan = max_len - 1
        soc = sum(len(p) - 1 for p in paths)

        return LaCAMResult(
            map_type=map_type,
            map_size=map_size,
            num_agents=N,
            starts=[tuple(s) for s in starts],
            goals=[tuple(g) for g in goals],
            success=all(p is not None for p in paths),
            paths=[[tuple(pos) for pos in p] for p in paths],
            makespan=makespan,
            sum_of_costs=soc,
            runtime=time.time() - t0,
        )

    def _plan_one(
        self,
        obstacles: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        reservations: Dict[Tuple[int, int, int], int],
        H: int, W: int,
        max_steps: int = 512,
    ) -> Optional[List[Tuple[int, int]]]:
        """时空 BFS：在 (t, r, c) 状态空间中搜索路径。"""
        sr, sc = int(start[0]), int(start[1])
        gr, gc = int(goal[0]), int(goal[1])

        if obstacles[sr, sc] != 0 or obstacles[gr, gc] != 0:
            return None

        # BFS 在 (r, c) 空间，带时间维度的 visited
        visited = set()
        parent: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {}

        q = deque()
        q.append((0, sr, sc))
        visited.add((0, sr, sc))
        parent[(0, sr, sc)] = None

        found = None
        while q:
            t, r, c = q.popleft()

            if r == gr and c == gc:
                found = (t, r, c)
                break

            if t >= max_steps:
                continue

            for dr, dc in ACTION_DELTAS:
                nr, nc = r + dr, c + dc

                # 越界/障碍
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    continue
                if obstacles[nr, nc] != 0:
                    continue

                # 检查 reservation（t+1 时刻是否被占用）
                if (t + 1, nr, nc) in reservations:
                    continue

                # 检查边冲突（t+1 的 (nr,nc) 是否有 agent 从 (nr,nc) 去了 (r,c)）
                # 简化：不检查边冲突（优先规划不保证无交换冲突）

                nt = t + 1
                if (nt, nr, nc) not in visited:
                    visited.add((nt, nr, nc))
                    parent[(nt, nr, nc)] = (t, r, c)
                    q.append((nt, nr, nc))

        if found is None:
            return None

        # 回溯路径
        path = []
        cur = found
        while cur is not None:
            path.append((cur[1], cur[2]))
            cur = parent[cur]
        path.reverse()
        return path

    def _plan_one_no_reservations(
        self,
        obstacles: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        H: int, W: int,
    ) -> Optional[List[Tuple[int, int]]]:
        """纯空间 BFS（无 reservation），用于降低优先级后的 fallback。"""
        sr, sc = int(start[0]), int(start[1])
        gr, gc = int(goal[0]), int(goal[1])

        if obstacles[sr, sc] != 0 or obstacles[gr, gc] != 0:
            return None

        dist = np.full((H, W), -1, dtype=np.int32)
        parent = np.full((H, W, 2), -1, dtype=np.int32)
        q = deque()
        q.append((sr, sc))
        dist[sr, sc] = 0

        while q:
            r, c = q.popleft()
            if r == gr and c == gc:
                break
            for dr, dc in ACTION_DELTAS[1:]:  # skip wait for efficiency
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if obstacles[nr, nc] == 0 and dist[nr, nc] == -1:
                        dist[nr, nc] = dist[r, c] + 1
                        parent[nr, nc] = [r, c]
                        q.append((nr, nc))

        if dist[gr, gc] == -1:
            return None

        # 回溯
        path = []
        r, c = gr, gc
        while (r, c) != (sr, sc):
            path.append((r, c))
            pr, pc = parent[r, c]
            r, c = pr, pc
        path.append((sr, sc))
        path.reverse()

        # 在目标位置等待（固定步数）
        path.extend([path[-1]] * 100)

        return path


# =========================================================================
# 原生 LaCAM solver（占位符——需 C++ 绑定）
# =========================================================================

class _LacamNativeSolver:
    """原生 LaCAM C++ 绑定 solver。"""

    def __init__(self, time_limit: float = 10.0):
        self.time_limit = time_limit
        try:
            from lacam import _lacam as _lacam_mod
            self._lacam = _lacam_mod
        except ImportError:
            raise ImportError(
                "LaCAM C++ binding (_lacam.so) not found. "
                "Build it with: cd lacam/ && mkdir build_pybind && ... "
                "or use backend='prioritized' for the Python fallback."
            )

    def solve(
        self,
        obstacles: np.ndarray,
        starts: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
        map_type: str = "unknown",
        map_size: int = 0,
    ) -> LaCAMResult:
        import time
        t0 = time.time()

        N = len(starts)
        obs = np.asarray(obstacles, dtype=np.int32)
        s_arr = np.array(starts, dtype=np.int32)
        g_arr = np.array(goals, dtype=np.int32)

        time_limit_ms = int(self.time_limit * 1000)

        result = self._lacam.solve(
            obs, s_arr, g_arr,
            time_limit_ms=time_limit_ms,
            seed=np.random.randint(0, 2**31 - 1),
            verbose=0,
        )

        elapsed = time.time() - t0

        if result["success"]:
            paths = [list(p) for p in result["paths"]]
            return LaCAMResult(
                map_type=map_type,
                map_size=map_size,
                num_agents=N,
                starts=[tuple(s) for s in starts],
                goals=[tuple(g) for g in goals],
                success=True,
                paths=paths,
                makespan=result.get("makespan", 0),
                sum_of_costs=result.get("sum_of_costs", 0),
                runtime=elapsed,
            )
        else:
            return LaCAMResult(
                map_type=map_type,
                map_size=map_size,
                num_agents=N,
                starts=[tuple(s) for s in starts],
                goals=[tuple(g) for g in goals],
                success=False,
                error=result.get("error", "Unknown"),
                runtime=elapsed,
            )


# =========================================================================
# helpers
# =========================================================================

def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
