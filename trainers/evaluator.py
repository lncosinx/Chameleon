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
from models.biintent_drqn import BiIntentGATDRQN
from utils.metrics import compute_episode_metrics


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

    wrapper = PogemaWrapper(config_path)
    wrapper._raw_cfg["size"] = map_size
    wrapper._raw_cfg["num_agents"] = num_agents
    wrapper.map_size = map_size
    wrapper.num_agents = num_agents

    obs_builder = ObservationBuilder(num_agents=num_agents, obs_radius=5)

    all_metrics = []

    for ep in range(num_episodes):
        obs_builder.reset()
        obs_list, _ = wrapper.reset(seed=20000 + ep)

        prev_actions = np.zeros(num_agents, dtype=np.int64)
        hidden = net.init_hidden(1, num_agents, device)

        positions_history = []
        actions_history = []

        for step in range(max_steps):
            agents_xy = wrapper.get_agents_xy()
            positions_history.append(list(agents_xy))

            # 构建观测
            patches, scalars = build_observations(
                wrapper, obs_list, prev_actions, obs_builder
            )

            # 转 tensor
            patches_t = torch.from_numpy(patches).unsqueeze(0).to(device)  # (1, N, 4, K, K)
            scalars_t = torch.from_numpy(scalars).unsqueeze(0).to(device)  # (1, N, 8)
            positions_t = torch.tensor(agents_xy, device=device).unsqueeze(0)  # (1, N, 2)

            with torch.no_grad():
                q_values, hidden, _ = net(
                    patches_t.unsqueeze(1),    # (1, 1, N, 4, K, K)
                    scalars_t.unsqueeze(1),    # (1, 1, N, 8)
                    positions_t.unsqueeze(1),  # (1, 1, N, 2)
                    hidden,
                )
                q_values = q_values.squeeze(0).squeeze(0)  # (N, 5)

            # ε-greedy
            if np.random.random() < epsilon:
                actions = [np.random.randint(0, 5) for _ in range(num_agents)]
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

    # 汇总
    summary = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        summary[key + "_mean"] = float(np.mean(values))
        if key in ("SR", "Deadlock"):
            summary[key] = float(np.mean(values))
    summary["num_episodes"] = num_episodes

    return summary
