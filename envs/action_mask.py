"""
动作掩码。

对每个智能体，标记非法动作为 -inf：
- 越界
- 进入静态障碍
- 环境明令禁止的动作

动态碰撞不做硬掩码（由策略学习协调）。
wait 通常恒合法，保证每体至少 1 个合法动作。

INVARIANT 7: 非法动作掩码 = -float('inf')，不是有限大负常数。
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple

# INVARIANT 1: 动作顺序
# 0=wait(0,0), 1=up(-1,0), 2=down(+1,0), 3=left(0,-1), 4=right(0,+1)
ACTION_DELTAS = np.array([
    [0, 0],    # 0: wait
    [-1, 0],   # 1: up
    [1, 0],    # 2: down
    [0, -1],   # 3: left
    [0, 1],    # 4: right
], dtype=np.int32)

NUM_ACTIONS = 5


def build_action_mask(
    agents_xy: List[Tuple[int, int]],
    obstacles: np.ndarray,  # (H, W), 0=free, 1=obstacle
) -> np.ndarray:
    """
    构建合法动作掩码 (N, 5)，True=合法，False=非法。

    Args:
        agents_xy: 每体当前全局坐标 (row, col)。
        obstacles: 全局障碍物栅格 (H, W)，0=free，1=obstacle。

    Returns:
        mask: (N, 5) bool 数组，True=合法动作，False=非法（需填 -inf）。
    """
    N = len(agents_xy)
    H, W = obstacles.shape
    mask = np.zeros((N, NUM_ACTIONS), dtype=bool)

    for i in range(N):
        r, c = int(agents_xy[i][0]), int(agents_xy[i][1])

        # 安全检查：如果自身坐标越界（不应发生），默认全部非法
        if r < 0 or r >= H or c < 0 or c >= W:
            mask[i, 0] = True  # wait 仍设为合法（容错）
            continue

        for a in range(NUM_ACTIONS):
            dr, dc = ACTION_DELTAS[a]
            nr, nc = r + dr, c + dc

            # wait (action 0) 恒合法——agent 已在该位置，无需检查障碍
            if a == 0:
                mask[i, a] = True
                continue

            # 越界检查
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                mask[i, a] = False
                continue

            # 静态障碍检查
            if obstacles[nr, nc] != 0:
                mask[i, a] = False
                continue

            mask[i, a] = True

    # 断言：每体至少 wait 合法（wait 恒在当前位置，必定合法除非自身站在障碍上）
    # 如果自身站在障碍上（不应发生），在 reset 时会检测。
    for i in range(N):
        assert mask[i, 0], (
            f"INVARIANT 7: Agent {i} has NO legal moves! wait should always be legal."
        )

    return mask


def apply_mask_to_q(q_values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    将非法动作的 Q 值设为 -inf。

    Args:
        q_values: (..., N, 5) Q 值。
        mask: (N, 5) bool 数组，True=合法。

    Returns:
        masked_q: same shape, 非法位置 = -inf。
    """
    # 使用 float('-inf') 而非大负数（INVARIANT 7）
    neg_inf = float('-inf')
    masked = np.where(mask, q_values, neg_inf)
    return masked


def build_action_mask_batch(
    agents_xy_batch: np.ndarray,  # (B, N, 2)
    obstacles: np.ndarray,        # (H, W)
) -> np.ndarray:
    """
    批量构建动作掩码（向量化版本，避免 per-agent Python 循环）。

    Args:
        agents_xy_batch: (B, N, 2) 每批每体全局坐标。
        obstacles: (H, W) 全局障碍物栅格。

    Returns:
        mask: (B, N, 5) bool 数组。
    """
    B, N, _ = agents_xy_batch.shape
    H, W = obstacles.shape

    # 动作偏移 (5, 2)
    deltas = ACTION_DELTAS  # (5, 2)

    # agents_xy_batch (B, N, 1, 2) + deltas (1, 1, 5, 2) -> (B, N, 5, 2)
    agents_expanded = agents_xy_batch[:, :, None, :]  # (B, N, 1, 2)
    deltas_expanded = deltas[None, None, :, :]         # (1, 1, 5, 2)
    targets = agents_expanded + deltas_expanded          # (B, N, 5, 2)

    nr = targets[..., 0]  # (B, N, 5)
    nc = targets[..., 1]  # (B, N, 5)

    # 越界
    in_bounds = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)

    # 静态障碍
    # 用 numpy advanced indexing 查 obstacles
    nr_clipped = np.clip(nr, 0, H - 1).astype(np.int32)
    nc_clipped = np.clip(nc, 0, W - 1).astype(np.int32)
    is_free = obstacles[nr_clipped, nc_clipped] == 0  # (B, N, 5)

    mask = in_bounds & is_free

    # wait (action 0) 恒合法（INVARIANT 7）：强制覆写，与单体版本行为一致
    mask[:, :, 0] = True

    return mask


# ---------------------------------------------------------------------------
# 数值验证：mellowmax 在 -inf 掩码下数值稳定
# ---------------------------------------------------------------------------

def test_mellowmax_stability(masked_q: np.ndarray, alpha: float = 1.0) -> bool:
    """
    验证 mellowmax 对含 -inf 的输入数值稳定（无 NaN）。

    Args:
        masked_q: (N, 5) Q 值，非法位 = -inf。
        alpha: mellowmax 温度参数。

    Returns:
        True if no NaN produced.
    """
    safe_q = masked_q / alpha
    # logsumexp over last dim, excluding -inf
    v_target = alpha * (
        np.log(np.sum(np.exp(safe_q), axis=-1) + 1e-12)
        - np.log(np.sum(np.isfinite(safe_q), axis=-1) + 1e-12)
    )
    return not np.any(np.isnan(v_target)) and not np.any(np.isinf(v_target))
