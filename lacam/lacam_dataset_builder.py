"""
LaCAM 数据集构建器。

将 LaCAM 专家轨迹转换为 DQfD 训练样本：
- 成功样本 → 专家 demo buffer
- 失败样本 → hard case 池（不进入专家 margin loss）

每条样本包含：
    obs, hidden_init, action_expert, reward, next_obs, done,
    action_mask, map_type, task_id, margin_excluded
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional

from envs.observation_builder import ObservationBuilder
from envs.action_mask import build_action_mask_batch
from envs.reward_builder import RewardBuilder
from lacam.trajectory_converter import (
    path_to_actions,
    compute_observation_hash,
    mark_ambiguous_samples,
    compute_wait_ratio,
    get_wait_margin_weight,
)
from lacam.lacam_runner import LaCAMResult


def build_dataset_from_lacam(
    result: LaCAMResult,
    wrapper,  # PogemaWrapper
    obs_builder: ObservationBuilder,
    reward_builder: RewardBuilder,
    obstacles: np.ndarray,
    sequence_len: int = 32,
) -> Tuple[List[dict], List[dict]]:
    """
    从一次 LaCAM 求解结果构建训练样本。

    Args:
        result: LaCAM 求解结果（含 paths）。
        wrapper: PogemaWrapper（用于获取观测）。
        obs_builder: ObservationBuilder 实例。
        reward_builder: RewardBuilder 实例。
        obstacles: 全局障碍物栅格。
        sequence_len: 序列长度（影响 hidden_init 存储）。

    Returns:
        expert_samples: list of sample dicts（成功路径）。
        hard_cases: list of hard case dicts（失败）。
    """
    if not result.success:
        hard_case = {
            "map_type": result.map_type,
            "starts": result.starts,
            "goals": result.goals,
            "error": result.error,
            "runtime": result.runtime,
        }
        return [], [hard_case]

    # B2: _build_single_sample 未实现，此入口不可用。
    # 请使用 build_dataset_from_rollout(wrapper, obs_builder, reward_builder,
    #     obstacles, result.paths, map_type, task_id) 代替。
    raise NotImplementedError(
        "build_dataset_from_lacam() is not fully implemented (B2). "
        "Use build_dataset_from_rollout() to generate expert samples."
    )


def _build_single_sample(
    agent_idx: int,
    step: int,
    path: List[Tuple[int, int]],
    actions: List[int],
    wrapper,
    obs_builder: ObservationBuilder,
    reward_builder: RewardBuilder,
    obstacles: np.ndarray,
    map_type: str,
    task_id: str,
) -> Optional[dict]:
    """
    为单个 agent 的单个时间步构建一条训练样本。

    NOTE: 此函数**未完整实现**（B2）。
    离线重建观测需要在正确时刻 step 环境并调用 obs_builder，
    无法仅凭路径在单函数内无状态地完成。

    正确的数据生成入口是 `build_dataset_from_rollout`，
    它通过顺序执行 env.step + obs_builder.build 来重建每步观测。

    `build_dataset_from_lacam` 中对本函数的调用已被保留作为 API 占位，
    但 `build_dataset_from_lacam` 的返回值将始终为空列表，
    应使用 `build_dataset_from_rollout` 替代。
    """
    raise NotImplementedError(
        "_build_single_sample is not implemented. "
        "Use build_dataset_from_rollout() instead."
    )


def build_dataset_from_rollout(
    wrapper,  # PogemaWrapper
    obs_builder: ObservationBuilder,
    reward_builder: RewardBuilder,
    obstacles: np.ndarray,
    expert_paths: List[List[Tuple[int, int]]],
    map_type: str,
    task_id: str,
    sequence_len: int = 32,
) -> List[dict]:
    """
    通过实际 rollout 重放专家路径来构建训练样本。

    对每个时间步：
    1. 获取当前局部观测（通过 obs_builder）
    2. 从 expert_paths 中读取该步的专家动作
    3. 执行专家动作，获取 next_obs / reward / done
    4. 构建 action_mask
    5. 记录样本

    这是主要的样本生成路径（generate_lacam_data.py 使用）。
    """
    N = wrapper.num_agents
    max_steps = min(max(len(p) for p in expert_paths) - 1, wrapper.max_episode_steps)

    obs_builder.reset()
    obs_list, _ = wrapper.reset()

    prev_actions = np.zeros(N, dtype=np.int64)
    prev_dones = [False] * N
    prev_agents = wrapper.get_agents_xy()
    prev_goals = wrapper.get_targets_xy()

    # 提取专家动作
    expert_actions_per_step = []
    for t in range(max_steps):
        step_actions = []
        for i in range(N):
            if t < len(expert_paths[i]) - 1:
                dr = expert_paths[i][t + 1][0] - expert_paths[i][t][0]
                dc = expert_paths[i][t + 1][1] - expert_paths[i][t][1]
                from lacam.trajectory_converter import DELTA_TO_ACTION
                step_actions.append(DELTA_TO_ACTION.get((dr, dc), 0))
            else:
                step_actions.append(0)  # wait at goal
        expert_actions_per_step.append(step_actions)

    samples = []

    for t in range(max_steps):
        # 观测（使用 prev_actions = 上一步专家动作）
        patches, scalars = _build_obs_safe(wrapper, obs_list, prev_actions, obs_builder)

        # 掩码（向量化版本）
        agents_xy = wrapper.get_agents_xy()
        agents_arr = np.array(agents_xy, dtype=np.int32)[np.newaxis]  # (1, N, 2)
        mask = build_action_mask_batch(agents_arr, obstacles)[0]       # (N, 5) bool

        # Reward（使用上一步动作和上一步状态）
        targets_xy = wrapper.get_targets_xy()
        rewards, _ = reward_builder.compute(
            obstacles=obstacles,
            prev_agents_xy=prev_agents,
            agents_xy=agents_xy,
            targets_xy=targets_xy,
            prev_targets_xy=prev_goals,
            prev_actions=prev_actions,
            prev_dones=prev_dones,
        )

        expert_actions = expert_actions_per_step[t]

        # 执行专家动作
        obs_list, _, dones, _ = wrapper.step(expert_actions)

        for i in range(N):
            if prev_dones[i]:
                continue

            obs_hash = compute_observation_hash(patches[i], scalars[i])

            sample = {
                "obs_patch": patches[i].copy(),       # (4, K, K)
                "obs_scalar": scalars[i].copy(),       # (8,)
                "obs_hash": obs_hash,
                "_agent_idx": i,                        # 用于 next_obs 填充，稍后清理
                "hidden_init": None,                    # 由 buffer 管理
                "action_expert": expert_actions[i],
                "reward": rewards[i],
                "next_obs_patch": None,                 # 下一步填充
                "next_obs_scalar": None,
                "done": dones[i] if isinstance(dones, (list, np.ndarray)) else dones,
                "action_mask": mask[i].copy(),          # (5,) bool
                "map_type": map_type,
                "task_id": task_id,
                "margin_excluded": False,
                "margin_weight": 0.5 if expert_actions[i] == 0 else 1.0,
            }
            samples.append(sample)

        prev_actions = np.array(expert_actions, dtype=np.int64)
        prev_dones = dones if isinstance(dones, list) else [dones] * N
        prev_agents = agents_xy
        prev_goals = targets_xy

    # 填充 next_obs / next_action_mask：
    # 样本按 [a0t0, a1t0, …, aN-1t0, a0t1, …] 交错存储，
    # 必须按 agent 索引（_agent_idx）分组，避免跨体赋值错误（B3 修复）。
    for i in range(N):
        agent_indices = [idx for idx, s in enumerate(samples) if s.get("_agent_idx") == i]
        for k in range(len(agent_indices) - 1):
            cur = agent_indices[k]
            nxt = agent_indices[k + 1]
            samples[cur]["next_obs_patch"]    = samples[nxt]["obs_patch"]
            samples[cur]["next_obs_scalar"]   = samples[nxt]["obs_scalar"]
            samples[cur]["next_action_mask"]  = samples[nxt]["action_mask"]  # I2 修复

    # 清理辅助字段
    for s in samples:
        s.pop("_agent_idx", None)

    # 可解释性过滤
    samples = mark_ambiguous_samples(samples)

    return samples


def _build_obs_safe(wrapper, obs_list, prev_actions, obs_builder):
    """安全构建观测（处理可能的错误）。"""
    try:
        agents = wrapper.get_agents_xy()
        targets = wrapper.get_targets_xy()
        return obs_builder.build(obs_list, agents, targets, prev_actions)
    except Exception:
        N = wrapper.num_agents
        K = wrapper.obs_radius * 2 + 1
        dummy_patches = np.zeros((N, 4, K, K), dtype=np.float32)
        dummy_scalars = np.zeros((N, 8), dtype=np.float32)
        return dummy_patches, dummy_scalars
