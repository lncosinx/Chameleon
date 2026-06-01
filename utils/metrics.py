"""
评价指标（§5 八类指标）。

1. 任务效果:       SR, AR, 最差体未达数
2. 路径质量:       Makespan, SoC, Inflation, 步数利用率
3. 冲突死锁:       WaitRate, NoProgressRate(曼哈顿诊断), RepeatRate, DeadlockCount
4. 学习稳定性:     L_TD, L_n, L_margin 走势（由 trainer 记录，此处提供汇总工具）
5. Q 值诊断:       Q mean/max/min/std, Expert Q advantage
6. 专家与模仿:     Acc_E, Wait_E, InterventionRate
7. 通信图:         EdgeFlipRate, MaxCC, 平均邻居数, Attention 熵
8. 系统性能:       ActorFPS, LearnerFPS（由 profiler 提供）

INVARIANT 14: NoProgressRate 等事后诊断使用曼哈顿距离，**不进 reward**。
"""
import numpy as np
from typing import List, Dict, Tuple, Optional


# ===========================================================================
# 1. 任务效果
# ===========================================================================

def compute_episode_metrics(
    agent_positions: List[List[Tuple[int, int]]],   # [step][agent] -> (r, c)
    goals: List[Tuple[int, int]],
    actions: List[List[int]],
    max_steps: int,
    lacam_makespan: float = None,
    deadlock_window: int = 32,
    deadlock_disp_threshold: float = 1.0,
) -> Dict[str, float]:
    """
    计算单 episode 的任务效果与路径质量指标。

    Args:
        agent_positions: per-step, per-agent 位置列表。
        goals:           每体目标坐标。
        actions:         per-step, per-agent 动作列表。
        max_steps:       最大步数。
        lacam_makespan:  LaCAM 参考 makespan（用于 Inflation）。
        deadlock_window: 判断死锁的回溯窗口步数。
        deadlock_disp_threshold: 窗口内总位移低于此值则判为死锁。
    """
    T = len(agent_positions)
    N = len(agent_positions[0]) if T > 0 else 0

    # ---- 到达判定 ----
    arrived      = [False] * N
    arrival_step = [T] * N
    for i in range(N):
        for t in range(T):
            r, c = agent_positions[t][i]
            gr, gc = goals[i]
            if r == gr and c == gc:
                arrived[i]      = True
                arrival_step[i] = t
                break

    sr = float(all(arrived))
    ar = float(sum(arrived)) / max(N, 1)

    # 最差体未达数
    worst_unreached = sum(1 for i in range(N) if not arrived[i])

    # ---- 路径质量 ----
    makespan  = max(arrival_step) if all(arrived) else T
    soc       = sum(arrival_step) if all(arrived) else sum(min(s, T) for s in arrival_step)
    inflation = (makespan / lacam_makespan) if (lacam_makespan and lacam_makespan > 0) else 0.0
    step_util = soc / max(N * max_steps, 1)

    # ---- WaitRate ----
    total_actions = T * N
    wait_count    = sum(1 for step_acts in actions for a in step_acts if a == 0)
    wait_rate     = wait_count / max(total_actions, 1)

    # ---- NoProgressRate（曼哈顿诊断，INVARIANT 14）----
    # 连续 2 步内曼哈顿距离未减少的比例
    no_progress_count = 0
    total_checked     = 0
    for t in range(1, T):
        for i in range(N):
            if arrived[i] and arrival_step[i] < t:
                continue
            r0, c0 = agent_positions[t - 1][i]
            r1, c1 = agent_positions[t][i]
            gr, gc = goals[i]
            d0 = abs(r0 - gr) + abs(c0 - gc)
            d1 = abs(r1 - gr) + abs(c1 - gc)
            if d1 >= d0:
                no_progress_count += 1
            total_checked += 1
    no_progress_rate = no_progress_count / max(total_checked, 1)

    # ---- RepeatRate（足迹重复率）----
    repeat_count  = 0
    total_moves   = 0
    visit_sets    = [set() for _ in range(N)]
    for t in range(T):
        for i in range(N):
            pos = agent_positions[t][i]
            if pos in visit_sets[i]:
                repeat_count += 1
            visit_sets[i].add(pos)
            total_moves += 1
    repeat_rate = repeat_count / max(total_moves, 1)

    # ---- 死锁判定 ----
    deadlock = False
    if T >= deadlock_window and not all(arrived):
        window = agent_positions[-deadlock_window:]
        total_disp = 0.0
        for t in range(1, len(window)):
            for i in range(N):
                dr = window[t][i][0] - window[t - 1][i][0]
                dc = window[t][i][1] - window[t - 1][i][1]
                total_disp += abs(dr) + abs(dc)
        deadlock = total_disp < deadlock_disp_threshold

    return {
        "SR":                float(sr),
        "AR":                float(ar),
        "WorstUnreached":    float(worst_unreached),
        "Makespan":          float(makespan),
        "SoC":               float(soc),
        "Inflation":         float(inflation),
        "StepUtilization":   float(step_util),
        "WaitRate":          float(wait_rate),
        "NoProgressRate":    float(no_progress_rate),   # 曼哈顿诊断，INVARIANT 14
        "RepeatRate":        float(repeat_rate),
        "Deadlock":          float(deadlock),
        "ArrivalSteps_mean": float(np.mean(arrival_step)),
        "ArrivalSteps_max":  float(max(arrival_step)),
    }


# ===========================================================================
# 5. Q 值诊断
# ===========================================================================

def compute_q_diagnostics(
    q_values: np.ndarray,        # (..., num_actions)
    expert_actions: np.ndarray = None,  # (...,) int
) -> Dict[str, float]:
    """
    Q 值诊断指标。

    Args:
        q_values:       Q 值，任意形状，最后一维为 actions。
        expert_actions: 专家动作（可选）。存在时计算 Expert Q advantage。

    Returns:
        dict 含 Q 统计 + （可选）Expert Q advantage。
    """
    q_flat = q_values.reshape(-1, q_values.shape[-1])

    diag = {
        "Q_mean": float(np.nanmean(q_flat)),
        "Q_max":  float(np.nanmax(q_flat)),
        "Q_min":  float(np.nanmin(q_flat)),
        "Q_std":  float(np.nanstd(q_flat)),
    }

    if expert_actions is not None:
        ea_flat = expert_actions.reshape(-1)
        B = len(ea_flat)
        q_expert = q_flat[np.arange(B), ea_flat]    # Q(s, a_E)

        # 将专家动作位设为 -inf，取非专家 Q 最大值
        q_nonexpert = q_flat.copy()
        q_nonexpert[np.arange(B), ea_flat] = -np.inf
        q_max_other = np.nanmax(q_nonexpert, axis=-1)  # max_{a ≠ a_E} Q(s, a)

        advantage = q_expert - q_max_other          # >0 ⟹ 网络认为专家更好
        diag["ExpertQ_advantage_mean"] = float(np.nanmean(advantage))
        diag["ExpertQ_advantage_neg_rate"] = float(np.mean(advantage < 0))

    return diag


# ===========================================================================
# 6. 专家与模仿
# ===========================================================================

def compute_imitation_metrics(
    policy_actions: List[int],
    expert_actions: List[int],
) -> Dict[str, float]:
    """
    计算模仿学习准确度指标。

    Args:
        policy_actions: 策略选择的动作列表。
        expert_actions: LaCAM 专家动作列表（同长度）。

    Returns:
        Acc_E: 总体匹配准确率。
        Acc_E_move: 移动动作的匹配准确率（排除 wait）。
        Acc_E_wait: wait 动作时的匹配准确率。
        Wait_E:   专家动作中 wait 的比例。
    """
    assert len(policy_actions) == len(expert_actions)
    n = len(policy_actions)
    if n == 0:
        return {"Acc_E": 0.0, "Acc_E_move": 0.0, "Acc_E_wait": 0.0, "Wait_E": 0.0}

    pa = np.array(policy_actions, dtype=np.int32)
    ea = np.array(expert_actions,  dtype=np.int32)

    match       = (pa == ea)
    expert_wait = (ea == 0)
    expert_move = ~expert_wait

    acc_e      = float(match.mean())
    acc_e_wait = float(match[expert_wait].mean()) if expert_wait.any() else 0.0
    acc_e_move = float(match[expert_move].mean()) if expert_move.any() else 0.0
    wait_e     = float(expert_wait.mean())

    return {
        "Acc_E":      acc_e,
        "Acc_E_move": acc_e_move,
        "Acc_E_wait": acc_e_wait,
        "Wait_E":     wait_e,
    }


# ===========================================================================
# 7. 通信图监控
# ===========================================================================

def compute_comm_graph_metrics(
    attn_weights: np.ndarray,         # (N, N) 注意力权重
    prev_edges: np.ndarray = None,    # (N, N) bool 上一步边集（用于 EdgeFlipRate）
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    通信图诊断指标。

    Args:
        attn_weights: (N, N) 本步注意力权重矩阵。
        prev_edges:   (N, N) bool 上一步边集（可为 None）。

    Returns:
        metrics: dict
        current_edges: (N, N) bool 本步边集（传递给下一步用于 EdgeFlipRate）
    """
    N = attn_weights.shape[0]
    current_edges = attn_weights > 0   # (N, N) bool

    # EdgeFlipRate（设计文档§5.7）：|E^t △ E^{t+1}| / |E^t ∪ E^{t+1}|
    if prev_edges is not None and prev_edges.shape == current_edges.shape:
        flips = (current_edges != prev_edges).sum()
        union = (current_edges | prev_edges).sum()
        edge_flip_rate = float(flips) / max(int(union), 1)
    else:
        edge_flip_rate = 0.0

    # 平均邻居数
    avg_neighbors = float(current_edges.sum(axis=1).mean())

    # MaxCC（最大连通分支规模，使用 BFS/Union-Find）
    max_cc = _max_connected_component(current_edges | current_edges.T)

    # Attention 熵（per-agent 熵均值）
    eps = 1e-12
    attn_norm = attn_weights / (attn_weights.sum(axis=-1, keepdims=True) + eps)
    entropy   = -(attn_norm * np.log(attn_norm + eps)).sum(axis=-1)
    attn_entropy = float(entropy.mean())

    metrics = {
        "EdgeFlipRate":  edge_flip_rate,
        "AvgNeighbors":  avg_neighbors,
        "MaxCC":         float(max_cc),
        "AttnEntropy":   attn_entropy,
    }
    return metrics, current_edges


def _max_connected_component(adj: np.ndarray) -> int:
    """BFS 计算无向图最大连通分支规模。adj: (N, N) bool 对称矩阵。"""
    N = adj.shape[0]
    visited = np.zeros(N, dtype=bool)
    max_size = 0
    for start in range(N):
        if visited[start]:
            continue
        queue   = [start]
        visited[start] = True
        size    = 0
        while queue:
            node = queue.pop()
            size += 1
            for nb in np.where(adj[node])[0]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        max_size = max(max_size, size)
    return max_size


# ===========================================================================
# 汇总工具
# ===========================================================================

def aggregate_episode_metrics(
    all_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    将多个 episode 的指标字典汇总为均值（±std）。
    """
    if not all_metrics:
        return {}
    summary = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics if key in m]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"]  = float(np.std(values))
    summary["num_episodes"] = len(all_metrics)
    return summary
