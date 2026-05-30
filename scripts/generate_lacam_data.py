#!/usr/bin/env python3
"""
阶段一：生成 LaCAM 专家数据。

对指定地图族批量生成专家轨迹，存入 ExpertDemoBuffer + HardCaseManager。

用法:
    python scripts/generate_lacam_data.py --config configs/env_random.yaml --num-episodes 100
"""
import sys
import os
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.pogema_env_wrapper import PogemaWrapper
from envs.observation_builder import ObservationBuilder
from envs.reward_builder import RewardBuilder
from lacam.lacam_runner import LaCAMRunner, LaCAMResult
from lacam.lacam_dataset_builder import build_dataset_from_rollout
from lacam.hard_case_manager import HardCaseManager
from buffers.expert_demo_buffer import ExpertDemoBuffer


def generate_data(
    config_path: str,
    num_episodes: int = 100,
    num_agents: int = 8,
    map_size: int = 16,
    backend: str = "prioritized",
    output_dir: str = "data",
):
    """
    对一种地图族生成专家数据。

    Args:
        config_path: env_*.yaml 配置文件路径。
        num_episodes: 生成 episode 数。
        num_agents: 每 episode 智能体数（覆盖 config）。
        map_size: 地图尺寸（覆盖 config）。
        backend: LaCAM 后端（"prioritized" 或 "lacam"）。
        output_dir: 输出目录。
    """
    os.makedirs(output_dir, exist_ok=True)

    wrapper = PogemaWrapper(config_path)
    wrapper._raw_cfg["size"] = map_size
    wrapper._raw_cfg["num_agents"] = num_agents
    wrapper.map_size = map_size
    wrapper.num_agents = num_agents

    obs_builder = ObservationBuilder(num_agents=num_agents, obs_radius=5)
    reward_builder = RewardBuilder(lambda_f=0.1)
    runner = LaCAMRunner(backend=backend)
    expert_buffer = ExpertDemoBuffer(capacity=num_episodes * 500)
    hard_case_mgr = HardCaseManager()

    map_type = wrapper.map_type
    print(f"Generating expert data for '{map_type}': "
          f"{num_episodes} episodes, {num_agents} agents, size={map_size}")

    success_count = 0
    fail_count = 0
    total_samples = 0
    t_start = time.time()

    for ep in range(num_episodes):
        seed = 10000 + ep

        # Reset 获取地图
        wrapper.reset(seed=seed)
        obstacles = wrapper.global_grid
        starts = wrapper.get_agents_xy()
        goals = wrapper.get_targets_xy()

        # LaCAM 求解
        result = runner.solve(
            obstacles=obstacles,
            starts=starts,
            goals=goals,
            map_type=map_type,
            map_size=map_size,
        )

        if not result.success:
            fail_count += 1
            hard_case_mgr.add_hard_case(
                map_type=map_type,
                starts=starts,
                goals=goals,
                reason="lacam_failure",
                obstacles=obstacles,
            )
            continue

        # 通过实际 rollout 重放专家路径，构建训练样本
        # 需要重新 reset 以正确初始化 env 状态
        wrapper.reset(seed=seed)
        samples = build_dataset_from_rollout(
            wrapper=wrapper,
            obs_builder=obs_builder,
            reward_builder=reward_builder,
            obstacles=obstacles,
            expert_paths=result.paths,
            map_type=map_type,
            task_id=f"{map_type}_ep{ep}",
        )

        if samples:
            expert_buffer.add_batch(samples)
            total_samples += len(samples)
            success_count += 1
        else:
            fail_count += 1
            hard_case_mgr.add_hard_case(
                map_type=map_type,
                starts=starts,
                goals=goals,
                reason="no_samples",
                obstacles=obstacles,
            )

        if (ep + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (ep + 1) / max(elapsed, 0.01)
            print(f"  [{ep+1}/{num_episodes}] success={success_count}, "
                  f"fail={fail_count}, samples={total_samples}, "
                  f"rate={rate:.1f} ep/s")

    elapsed = time.time() - t_start

    # 保存统计
    stats = {
        "map_type": map_type,
        "num_episodes": num_episodes,
        "num_agents": num_agents,
        "map_size": map_size,
        "success_count": success_count,
        "fail_count": fail_count,
        "total_samples": total_samples,
        "elapsed": elapsed,
        "buffer_stats": expert_buffer.get_stats(),
        "hard_case_stats": hard_case_mgr.get_stats(),
    }

    import yaml
    stats_path = os.path.join(output_dir, f"lacam_stats_{map_type}.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(stats, f)

    print(f"\nDone: {success_count} success, {fail_count} fail, "
          f"{total_samples} samples in {elapsed:.1f}s")
    print(f"Stats saved to {stats_path}")

    return expert_buffer, hard_case_mgr, stats


def main():
    parser = argparse.ArgumentParser(description="Generate LaCAM expert data")
    parser.add_argument("--config", type=str, default="configs/env_random.yaml",
                        help="Environment config YAML")
    parser.add_argument("--num-episodes", type=int, default=100,
                        help="Number of episodes to generate")
    parser.add_argument("--num-agents", type=int, default=8,
                        help="Number of agents per episode")
    parser.add_argument("--map-size", type=int, default=16,
                        help="Map size")
    parser.add_argument("--backend", type=str, default="prioritized",
                        choices=["prioritized", "lacam"],
                        help="LaCAM backend")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory")
    args = parser.parse_args()

    generate_data(
        config_path=args.config,
        num_episodes=args.num_episodes,
        num_agents=args.num_agents,
        map_size=args.map_size,
        backend=args.backend,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
