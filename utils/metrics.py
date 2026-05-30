"""
评价指标（§5 八类指标）。

- 任务效果: SR, AR, 最差体未达数
- 路径质量: Makespan, SoC, Inflation, 步数利用率
- 冲突死锁: WaitRate, NoProgressRate, DeadlockCount
- Q 值诊断: Q mean/max/min/std, Expert Q advantage
- 专家与模仿: Acc_E, Wait_E, InterventionRate
- 通信: MaxCC, 平均邻居数, Attention 熵
- 系统: ActorFPS, LearnerFPS
"""
import numpy as np
from typing import List, Dict, Tuple


def compute_episode_metrics(
    agent_positions: List[List[Tuple[int, int]]],  # per-step, per-agent positions
    goals: List[Tuple[int, int]],
    actions: List[List[int]],
    max_steps: int,
    lacam_makespan: float = None,
) -> Dict[str, float]:
    """计算单 episode 指标。"""
    T = len(agent_positions)
    N = len(agent_positions[0])

    # 到达判定
    arrived = [False] * N
    arrival_step = [T] * N
    for i in range(N):
        for t in range(T):
            pos = agent_positions[t][i]
            goal = goals[i]
            if pos == goal:
                arrived[i] = True
                arrival_step[i] = t
                break

    sr = float(all(arrived))
    ar = sum(arrived) / N

    # Makespan = 最后到达体的步数
    makespan = max(arrival_step) if all(arrived) else T
    soc = sum(arrival_step) if all(arrived) else sum(min(s, T) for s in arrival_step)

    # Inflation = len_policy / len_LaCAM
    inflation = makespan / lacam_makespan if lacam_makespan and lacam_makespan > 0 else 0.0

    # WaitRate
    wait_count = sum(1 for step_acts in actions for a in step_acts if a == 0)
    total_actions = T * N
    wait_rate = wait_count / max(total_actions, 1)

    # 死锁检测
    deadlock = False
    if T >= 32:
        last_32 = agent_positions[-32:]
        total_displacement = 0.0
        for t in range(1, len(last_32)):
            for i in range(N):
                dr = last_32[t][i][0] - last_32[t-1][i][0]
                dc = last_32[t][i][1] - last_32[t-1][i][1]
                total_displacement += abs(dr) + abs(dc)
        if total_displacement < 1.0 and not all(arrived):
            deadlock = True

    return {
        "SR": sr,
        "AR": ar,
        "Makespan": makespan,
        "SoC": soc,
        "Inflation": inflation,
        "WaitRate": wait_rate,
        "Deadlock": float(deadlock),
        "ArrivalSteps_mean": np.mean(arrival_step),
        "ArrivalSteps_max": max(arrival_step),
    }


def compute_q_diagnostics(q_values: np.ndarray) -> Dict[str, float]:
    """Q 值诊断。"""
    return {
        "Q_mean": float(np.mean(q_values)),
        "Q_max": float(np.max(q_values)),
        "Q_min": float(np.min(q_values)),
        "Q_std": float(np.std(q_values)),
    }
