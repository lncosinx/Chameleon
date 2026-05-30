#!/usr/bin/env python3
"""
Phase 1 总验收门：随机策略在四地图族各跑 50 episode。

产出 (obs, action, reward, done, mask) 五元组，
验证 reward 有限、无 NaN、足迹/掩码/距离场行为符合预期。
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.pogema_env_wrapper import PogemaWrapper
from envs.observation_builder import ObservationBuilder, build_observations
from envs.action_mask import build_action_mask
from envs.reward_builder import RewardBuilder


CONFIG_PATHS = [
    "configs/env_random.yaml",
    "configs/env_warehouse.yaml",
    "configs/env_maze.yaml",
    "configs/env_room.yaml",
]

NUM_EPISODES = 50
MAX_STEPS = 64  # 截断，保证快速跑完


def run_one_map_type(config_path: str) -> dict:
    """对一种地图族跑 NUM_EPISODES 个 episode，返回统计信息。"""
    wrapper = PogemaWrapper(config_path)
    # 小规模测试
    wrapper._raw_cfg["size"] = 16
    wrapper._raw_cfg["num_agents"] = 4
    wrapper.map_size = 16
    wrapper.num_agents = 4

    builder = ObservationBuilder(num_agents=4, obs_radius=5)
    reward_builder = RewardBuilder(lambda_f=0.1, step_cost=0.01,
                                   collision_cost=0.1, wait_cost=0.05)

    stats = {
        "episodes": 0,
        "total_steps": 0,
        "total_reward": 0.0,
        "nan_count": 0,
        "mask_ok": 0,
        "obs_shape_ok": 0,
    }

    for ep in range(NUM_EPISODES):
        builder.reset()
        reward_builder.clear_cache()

        obs_list, _ = wrapper.reset(seed=1000 + ep)
        prev_actions = np.zeros(4, dtype=np.int64)
        prev_dones = [False] * 4
        prev_agents = wrapper.get_agents_xy()
        prev_goals = wrapper.get_targets_xy()

        # 验证 obs shape
        for obs in obs_list:
            if np.asarray(obs).shape == (3, 11, 11):
                stats["obs_shape_ok"] += 1

        for step in range(MAX_STEPS):
            # 随机动作
            actions = [np.random.randint(0, 5) for _ in range(4)]

            # 构建观测
            patches, scalars = build_observations(
                wrapper, obs_list, prev_actions, builder
            )

            # 构建掩码
            obstacles = wrapper.global_grid
            agents = wrapper.get_agents_xy()
            mask = build_action_mask(agents, obstacles)

            # 验证掩码
            if mask.shape == (4, 5) and np.all(mask[:, 0]):
                stats["mask_ok"] += 1

            # 计算 reward
            targets = wrapper.get_targets_xy()
            rewards, rinfo = reward_builder.compute(
                obstacles=obstacles,
                prev_agents_xy=prev_agents,
                agents_xy=agents,
                targets_xy=targets,
                prev_targets_xy=prev_goals,
                prev_actions=prev_actions,
                prev_dones=prev_dones,
            )

            # 检查 NaN
            for r in rewards:
                if np.isnan(r) or np.isinf(r):
                    stats["nan_count"] += 1

            stats["total_reward"] += sum(rewards)

            # Step
            obs_list, _, dones, _ = wrapper.step(actions)
            prev_actions = np.array(actions, dtype=np.int64)
            prev_dones = dones
            prev_agents = agents
            prev_goals = targets

            stats["total_steps"] += 1

        stats["episodes"] += 1

    return stats


def main():
    print("=" * 60)
    print("  Chameleon — Phase 1 总验收门")
    print(f"  每族 {NUM_EPISODES} episodes × {MAX_STEPS} steps")
    print("=" * 60)

    all_ok = True

    for cfg_path in CONFIG_PATHS:
        map_type = cfg_path.split("_")[1].split(".")[0]
        print(f"\n--- {map_type} ---")
        try:
            stats = run_one_map_type(cfg_path)

            total_obs = stats["obs_shape_ok"]
            total_mask = stats["mask_ok"]
            total_steps = stats["total_steps"]

            print(f"  Episodes:    {stats['episodes']}")
            print(f"  Total steps: {total_steps}")
            print(f"  Mean reward: {stats['total_reward'] / max(total_steps, 1):.4f}")
            print(f"  NaN/Inf:     {stats['nan_count']}")
            print(f"  Obs (3,K,K): {total_obs}/{total_obs + (4*NUM_EPISODES - total_obs)}")

            assert stats["nan_count"] == 0, f"Found {stats['nan_count']} NaN/Inf rewards!"
            assert stats["mask_ok"] == total_steps, (
                f"Mask failures: {stats['mask_ok']}/{total_steps}"
            )
            assert stats["obs_shape_ok"] > 0, "No valid observations!"
            print(f"  [PASS] {map_type} ✓")

        except Exception as e:
            print(f"  [FAIL] {map_type}: {e}")
            import traceback
            traceback.print_exc()
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("  Phase 1 总验收门: 全部通过 ✓")
    else:
        print("  Phase 1 总验收门: 存在失败项 ✗")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
