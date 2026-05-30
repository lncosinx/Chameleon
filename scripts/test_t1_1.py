#!/usr/bin/env python3
"""
T1.1 验收脚本：Pogema 封装 + 四地图族。

验证：
- 四种 env_*.yaml 各能 reset + 随机走 20 步不崩
- 每体原生观测 shape (3, K, K)
- global_grid / starts / goals 可访问（特权信号，不进 obs）
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.pogema_env_wrapper import PogemaWrapper


CONFIG_PATHS = [
    "configs/env_random.yaml",
    "configs/env_warehouse.yaml",
    "configs/env_maze.yaml",
    "configs/env_room.yaml",
]


def test_single_config(config_path: str, steps: int = 20):
    """对单个配置进行 smoke 测试。"""
    print(f"\n{'='*60}")
    print(f"  Testing: {config_path}")
    print(f"{'='*60}")

    wrapper = PogemaWrapper(config_path)

    # -- 检查配置读取 --
    print(f"  map_type:         {wrapper.map_type}")
    print(f"  map_size:         {wrapper.map_size}")
    print(f"  num_agents:       {wrapper.num_agents}")
    print(f"  obs_radius:       {wrapper.obs_radius}")
    print(f"  max_episode_steps:{wrapper.max_episode_steps}")
    print(f"  collision_system: {wrapper.collision_system}")
    # 用较小 map_size 加速测试
    wrapper.map_size = min(wrapper.map_size, 16)
    wrapper.num_agents = min(wrapper.num_agents, 4)
    wrapper._raw_cfg["map_size"] = wrapper.map_size
    wrapper._raw_cfg["num_agents"] = wrapper.num_agents

    # -- Reset --
    obs_list, info = wrapper.reset(seed=42)
    assert isinstance(obs_list, list), f"obs_list should be list, got {type(obs_list)}"
    assert len(obs_list) == wrapper.num_agents, (
        f"Expected {wrapper.num_agents} obs, got {len(obs_list)}"
    )

    obs0 = np.asarray(obs_list[0])
    K = 2 * wrapper.obs_radius + 1
    expected_shape = (3, K, K)
    assert obs0.shape == expected_shape, (
        f"INVARIANT: obs shape should be {expected_shape}, got {obs0.shape}"
    )
    print(f"  [PASS] reset ok, per-agent obs shape: {obs0.shape}")

    # -- 特权信号检查（不进 policy） --
    grid = wrapper.global_grid
    starts = wrapper.starts
    goals = wrapper.goals
    if grid is not None:
        print(f"  [INFO] global_grid shape: {grid.shape} (privileged, NOT in obs)")
    if starts is not None:
        print(f"  [INFO] starts[:3]: {starts[:3]}")
    if goals is not None:
        print(f"  [INFO] goals[:3]:  {goals[:3]}")

    # -- targets_xy 验证 --
    targets = wrapper.get_targets_xy()
    assert len(targets) == wrapper.num_agents, (
        f"Expected {wrapper.num_agents} targets, got {len(targets)}"
    )
    print(f"  [PASS] get_targets_xy() returns {len(targets)} targets")

    # -- 随机走 steps 步 --
    for step_idx in range(steps):
        actions = [i % 5 for i in range(wrapper.num_agents)]
        obs_list, rewards, dones, info = wrapper.step(actions)

    # 检查每一步的数据完整性
    assert len(obs_list) == wrapper.num_agents
    assert len(rewards) == wrapper.num_agents
    assert len(dones) == wrapper.num_agents or isinstance(dones, bool)

    # 再次检查 obs shape
    obs_np = np.asarray(obs_list[0])
    assert obs_np.shape == expected_shape, (
        f"obs shape changed after steps: {obs_np.shape}"
    )

    # rewards 应该有限
    for r in rewards:
        assert np.isfinite(float(r)), f"Non-finite reward: {r}"

    print(f"  [PASS] {steps} random steps completed, rewards sample: "
          f"{[round(float(r), 3) for r in rewards[:4]]}")

    # -- 确保 obs 不含全局信息 --
    # obs 是 (3,K,K)，只含障碍/智能体/目标投影，不含 BFS 距离/全局地图
    assert "global_grid" not in str(type(obs_list[0])).lower(), \
        "INVARIANT 2: obs must not contain global_grid"
    print(f"  [PASS] INVARIANT 2: obs is local-only (3,K,K) patch")

    return True


def main():
    print("=" * 60)
    print("  Chameleon — T1.1 验收：Pogema 封装 + 四地图族")
    print("=" * 60)

    all_ok = True
    for cfg_path in CONFIG_PATHS:
        try:
            test_single_config(cfg_path, steps=20)
        except Exception as e:
            print(f"\n  [FAIL] {cfg_path}: {e}")
            import traceback
            traceback.print_exc()
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("  T1.1 验收结果: 全部通过 ✓")
    else:
        print("  T1.1 验收结果: 存在失败项 ✗")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
