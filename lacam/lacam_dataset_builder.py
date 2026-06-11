"""
LaCAM 数据集构建器。

将 LaCAM 专家轨迹转换为 DQfD 训练样本（**per-timestep × N-agent** 格式）：
- 成功样本 → 专家 demo buffer（按 episode 存储，每步含全部 N 体）
- 失败样本 → hard case 池（不进入专家 margin loss）

**为何 per-timestep：** 通信模块需要同一时刻多个共存 agent 的观测+坐标才能
计算邻居注意力并回传梯度。若按单体存储，通信候选集恒空、参数永不更新。
详见 buffers/expert_demo_buffer.py 顶部说明。

build_dataset_from_rollout 返回一个 episode 的列式数组 dict（形状 (T,N,...)），
可直接 `ExpertDemoBuffer.add_episode(ep)`。
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from envs.observation_builder import ObservationBuilder
from envs.action_mask import build_action_mask_batch
from envs.reward_builder import RewardBuilder
from lacam.trajectory_converter import (
    compute_observation_hash,
    compute_wait_ratio,
    get_wait_margin_weight,
    DELTA_TO_ACTION,
)
from lacam.lacam_runner import LaCAMResult
from buffers.expert_demo_buffer import FINISHED_AGENT_SENTINEL


def build_dataset_from_rollout(
    wrapper,  # PogemaWrapper
    obs_builder: ObservationBuilder,
    reward_builder: RewardBuilder,
    obstacles: np.ndarray,
    expert_paths: List[List[Tuple[int, int]]],
    map_type: str,
    task_id: str,
    sequence_len: int = 32,
) -> Optional[Dict[str, np.ndarray]]:
    """
    通过实际 rollout 重放专家路径，构建一个 episode 的 per-timestep 训练数据。

    对每个时间步 t：
      1. 在 step 前获取全部 N 体的局部观测（obs_builder，s_t）
      2. 从 expert_paths 读取每体该步专家动作
      3. 执行专家动作，得到 s_{t+1}、reward、done
      4. 记录 (T, N, ...) 列式数组

    返回（可直接喂 ExpertDemoBuffer.add_episode）：
        dict，键见下方 `ep`；若无有效步则返回 None。
        - positions: 已对 finished agent 置远哨兵（通信距离过滤自然排除）
        - valid:     finished agent 置 False（loss 中按 0 权重屏蔽）
    """
    N = wrapper.num_agents
    max_steps = min(max(len(p) for p in expert_paths) - 1, wrapper.max_episode_steps)
    if max_steps <= 0:
        return None

    obs_builder.reset()
    # 必须用与调用方相同的 seed 重置，保证地图与 LaCAM 解算时一致。
    obs_list, _ = wrapper.reset(seed=wrapper._seed)

    prev_actions = np.zeros(N, dtype=np.int64)
    prev_goals = wrapper.get_targets_xy()

    # 预提取每步专家动作 (T, N)
    expert_actions_per_step = np.zeros((max_steps, N), dtype=np.int64)
    for t in range(max_steps):
        for i in range(N):
            if t < len(expert_paths[i]) - 1:
                dr = expert_paths[i][t + 1][0] - expert_paths[i][t][0]
                dc = expert_paths[i][t + 1][1] - expert_paths[i][t][1]
                expert_actions_per_step[t, i] = DELTA_TO_ACTION.get((dr, dc), 0)
            else:
                expert_actions_per_step[t, i] = 0  # wait at goal

    K = obs_builder.K

    # per-step 存储（长度 max_steps；next_* 通过移位获得）
    S_patch  = np.zeros((max_steps, N, 4, K, K), dtype=np.float32)
    S_scalar = np.zeros((max_steps, N, 8),        dtype=np.float32)
    S_pos    = np.full((max_steps, N, 2), FINISHED_AGENT_SENTINEL, dtype=np.float32)
    S_action = np.zeros((max_steps, N),           dtype=np.int64)
    S_reward = np.zeros((max_steps, N),           dtype=np.float32)
    S_done   = np.zeros((max_steps, N),           dtype=bool)
    S_valid  = np.zeros((max_steps, N),           dtype=bool)
    S_mask   = np.ones((max_steps, N, 5),         dtype=bool)
    S_hash   = np.zeros((max_steps, N),           dtype=np.int64)

    # ever_done：累计完成掩码，单调递增（一旦 True 永远 True）。
    ever_done = [False] * N

    reward_builder.begin_episode()   # 归一 shaping：本 episode 起点距离缓存（首次 compute 惰性计算）
    for t in range(max_steps):
        prev_ever_done = list(ever_done)

        # ---- 观测（step 前的状态 s_t）----
        agents_xy = wrapper.get_agents_xy()
        patches, scalars = _build_obs_safe(wrapper, obs_list, prev_actions, obs_builder)
        S_patch[t]  = patches
        S_scalar[t] = scalars

        # ---- 动作掩码（step 前；wait 恒合法，保证 mellowmax 有限）----
        agents_arr = np.array(agents_xy, dtype=np.int32)[np.newaxis]
        mask = build_action_mask_batch(agents_arr, obstacles)[0]  # (N, 5) bool
        S_mask[t] = mask

        # ---- 坐标（finished agent 置远哨兵，使通信不选其为邻居）----
        pos = np.array(agents_xy, dtype=np.float32)  # (N, 2)
        for i in range(N):
            if prev_ever_done[i]:
                pos[i] = FINISHED_AGENT_SENTINEL
        S_pos[t] = pos

        # ---- valid：step 前未完成的 agent 参与 loss ----
        for i in range(N):
            S_valid[t, i] = not prev_ever_done[i]

        expert_actions = expert_actions_per_step[t]
        S_action[t] = expert_actions

        # ---- obs hash（用于歧义检测；仅对 valid 体有意义）----
        for i in range(N):
            if not prev_ever_done[i]:
                S_hash[t, i] = compute_observation_hash(patches[i], scalars[i])

        # ---- 执行专家动作，获得 s_{t+1} ----
        obs_list, _, dones, _ = wrapper.step(expert_actions.tolist())

        # ---- step 之后计算奖励（正确时序 R(s_t, a_t, s_{t+1})）----
        agents_xy_next = wrapper.get_agents_xy()
        targets_xy_next = wrapper.get_targets_xy()
        rewards, _ = reward_builder.compute(
            obstacles=obstacles,
            prev_agents_xy=agents_xy,
            agents_xy=agents_xy_next,
            targets_xy=targets_xy_next,
            prev_targets_xy=prev_goals if t > 0 else wrapper.get_targets_xy(),
            prev_actions=prev_actions,
            prev_dones=prev_ever_done,
        )
        S_reward[t] = np.asarray(rewards, dtype=np.float32)

        # ---- 更新累计完成状态 ----
        step_dones = dones if isinstance(dones, list) else [bool(dones)] * N
        for i in range(N):
            if step_dones[i]:
                ever_done[i] = True
        S_done[t] = np.array(ever_done, dtype=bool)

        prev_actions = expert_actions.copy()
        prev_goals = targets_xy_next

    # 整段无任何 valid 步（极端：所有 agent 起点即终点）→ 丢弃
    if not S_valid.any():
        return None

    # ---- 构造 next_* （移位一帧；末帧填零/哨兵）----
    next_patch  = np.zeros_like(S_patch)
    next_scalar = np.zeros_like(S_scalar)
    next_pos    = np.full_like(S_pos, FINISHED_AGENT_SENTINEL)
    next_mask   = np.ones_like(S_mask)
    if max_steps > 1:
        next_patch[:-1]  = S_patch[1:]
        next_scalar[:-1] = S_scalar[1:]
        next_pos[:-1]    = S_pos[1:]
        next_mask[:-1]   = S_mask[1:]
    # 末帧 next 保持零/哨兵/全合法（done 处 TD bootstrap 已被 done 掩码归零）

    # ---- margin_excluded：同一 obs_hash 下专家动作不唯一 → 剔除（INVARIANT 11）----
    margin_excluded = np.zeros((max_steps, N), dtype=bool)
    hash_to_actions: Dict[int, set] = defaultdict(set)
    for t in range(max_steps):
        for i in range(N):
            if S_valid[t, i]:
                hash_to_actions[int(S_hash[t, i])].add(int(S_action[t, i]))
    for t in range(max_steps):
        for i in range(N):
            if S_valid[t, i] and len(hash_to_actions[int(S_hash[t, i])]) > 1:
                margin_excluded[t, i] = True

    # ---- margin_weights：每体按 wait 比例降权（INVARIANT 11 §3.3.6）----
    margin_weights = np.ones((max_steps, N), dtype=np.float32)
    for i in range(N):
        valid_mask_i = S_valid[:, i]
        if not valid_mask_i.any():
            continue
        agent_actions = S_action[valid_mask_i, i].tolist()
        wait_ratio = compute_wait_ratio(agent_actions)
        weight = get_wait_margin_weight(wait_ratio)
        # 仅对 wait 动作降权，move 动作保持 1.0
        wait_steps = (S_action[:, i] == 0)
        margin_weights[wait_steps, i] = weight

    ep = {
        "obs_patches":       S_patch,
        "obs_scalars":       S_scalar,
        "positions":         S_pos,
        "next_obs_patches":  next_patch,
        "next_obs_scalars":  next_scalar,
        "next_positions":    next_pos,
        "actions_expert":    S_action,
        "rewards":           S_reward,
        "dones":             S_done,
        "valid":             S_valid,
        "action_masks":      S_mask,
        "next_action_masks": next_mask,
        "margin_excluded":   margin_excluded,
        "margin_weights":    margin_weights,
        "map_type":          map_type,
        "task_id":           task_id,
    }
    return ep


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
