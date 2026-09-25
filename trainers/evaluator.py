"""
评估器：关闭专家介入，评估策略真实能力。

INVARIANT: 评估时不得调用 LaCAM 或使用全局特权信号做决策。
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from envs.pogema_env_wrapper import PogemaWrapper
from envs.observation_builder import ObservationBuilder, build_observations
from envs.action_mask import build_action_mask_batch, apply_mask_to_q
from models.biintent_drqn import BiIntentGATDRQN
from utils.metrics import compute_episode_metrics, aggregate_episode_metrics


def evaluate(
    net: BiIntentGATDRQN,
    config_path: str,
    num_episodes: int = 50,
    num_agents: int = 8,
    map_size: int = 16,
    max_steps: int = 256,
    device: torch.device = None,
    epsilon: float = 0.0,  # 评估时不探索
) -> Dict[str, float]:
    """
    评估策略。

    Args:
        net: 已训练的网络。
        config_path: 环境配置。
        num_episodes: 评估 episode 数。
        num_agents: 智能体数。
        map_size: 地图尺寸。
        max_steps: 最大步数。
        device: 计算设备。
        epsilon: 探索概率（评估时建议 0）。

    Returns:
        汇总指标 dict。
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    # split='test'：评估只用 held-out 测试地图池（与训练 seed 空间不相交）。
    wrapper = PogemaWrapper(config_path, split="test")
    wrapper._raw_cfg["size"] = map_size
    wrapper._raw_cfg["num_agents"] = num_agents
    wrapper.map_size = map_size
    wrapper.num_agents = num_agents

    obs_builder = ObservationBuilder(num_agents=num_agents, obs_radius=5)

    all_metrics = []

    for ep in range(num_episodes):
        obs_builder.reset()
        # 评估第 ep 张测试地图（split='test' 映射到 held-out 池）。
        obs_list, _ = wrapper.reset(seed=ep)

        prev_actions = np.zeros(num_agents, dtype=np.int64)
        hidden = net.init_hidden(1, num_agents, device)

        # 每个 episode 开始时重置通信迟滞状态（B7 修复）
        net.reset_comm_state()

        positions_history = []
        actions_history = []

        for step in range(max_steps):
            agents_xy = wrapper.get_agents_xy()
            positions_history.append(list(agents_xy))

            # 构建观测
            patches, scalars = build_observations(
                wrapper, obs_list, prev_actions, obs_builder
            )

            # 构建动作掩码（向量化版本，避免 per-agent 循环）
            obstacles = wrapper.global_grid
            if obstacles is not None:
                agents_arr = np.array(agents_xy, dtype=np.int32)[np.newaxis]  # (1, N, 2)
                mask_np = build_action_mask_batch(agents_arr, obstacles)[0]   # (N, 5) bool
            else:
                mask_np = np.ones((num_agents, 5), dtype=bool)

            # 转 tensor
            patches_t   = torch.from_numpy(patches).unsqueeze(0).to(device)    # (1, N, 4, K, K)
            scalars_t   = torch.from_numpy(scalars).unsqueeze(0).to(device)    # (1, N, 8)
            positions_t = torch.tensor(agents_xy, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, 2)
            masks_t     = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, N, 5)

            with torch.no_grad():
                q_values, hidden, _ = net(
                    patches_t.unsqueeze(1),    # (1, 1, N, 4, K, K)
                    scalars_t.unsqueeze(1),    # (1, 1, N, 8)
                    positions_t.unsqueeze(1),  # (1, 1, N, 2)
                    hidden,
                    action_masks=masks_t,      # B9: 应用动作掩码
                )
                q_values = q_values.squeeze(0).squeeze(0)  # (N, 5)

            # ε-greedy（greedy：argmax；探索：从合法动作集中随机采样）
            if np.random.random() < epsilon:
                actions = [
                    int(np.random.choice(np.where(mask_np[i])[0]))
                    for i in range(num_agents)
                ]
            else:
                actions = q_values.argmax(dim=-1).cpu().numpy().tolist()

            actions_history.append(actions)

            # Step
            obs_list, _, dones, _ = wrapper.step(actions)
            prev_actions = np.array(actions, dtype=np.int64)

        goals = wrapper.goals or wrapper.get_targets_xy()
        metrics = compute_episode_metrics(
            positions_history, goals, actions_history, max_steps
        )
        all_metrics.append(metrics)

    return aggregate_episode_metrics(all_metrics)
