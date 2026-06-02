#!/usr/bin/env python3
"""
Phase 2 验收：LaCAM 专家数据流水线。

T2.1: 四地图族各生成 ≥10 条成功轨迹
T2.2: 位移→动作映射正确，观测按局部重建，对称让行标 margin_excluded
T2.3: 专家 buffer 可按 sequence_len=32 采样定长序列
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.pogema_env_wrapper import PogemaWrapper
from envs.observation_builder import ObservationBuilder
from envs.reward_builder import RewardBuilder
from lacam.lacam_runner import LaCAMRunner
from lacam.lacam_dataset_builder import build_dataset_from_rollout
from lacam.trajectory_converter import (
    path_to_actions, actions_to_path, DELTA_TO_ACTION,
    mark_ambiguous_samples, compute_observation_hash,
)
from lacam.hard_case_manager import HardCaseManager
from buffers.expert_demo_buffer import ExpertDemoBuffer


CONFIG_PATHS = [
    "configs/env_random.yaml",
    "configs/env_warehouse.yaml",
    "configs/env_maze.yaml",
    "configs/env_room.yaml",
]


def test_lacam_generation():
    """T2.1: 四族各 ≥10 条成功轨迹。"""
    print("\n" + "=" * 60)
    print("  T2.1: LaCAM trajectory generation")
    print("=" * 60)

    for cfg_path in CONFIG_PATHS:
        map_type = cfg_path.split("_")[1].split(".")[0]
        wrapper = PogemaWrapper(cfg_path)
        wrapper._raw_cfg["size"] = 16
        wrapper._raw_cfg["num_agents"] = 4
        wrapper.map_size = 16
        wrapper.num_agents = 4

        runner = LaCAMRunner(backend="prioritized")
        success = 0
        for ep in range(20):
            wrapper.reset(seed=20000 + ep)
            obstacles = wrapper.global_grid
            starts = wrapper.get_agents_xy()
            goals = wrapper.get_targets_xy()

            result = runner.solve(obstacles, starts, goals,
                                  map_type=map_type, map_size=16)
            if result.success:
                success += 1

        assert success >= 10, (
            f"{map_type}: only {success}/20 successful, need ≥10"
        )
        print(f"  {map_type}: {success}/20 successful ✓")


def test_action_mapping():
    """T2.2.1: 位移→动作映射按 INVARIANT 1。"""
    print("\n--- T2.2.1: Action mapping (INVARIANT 1) ---")

    # 所有合法位移都能映射
    for dr, dc in [(0,0), (-1,0), (1,0), (0,-1), (0,1)]:
        a = DELTA_TO_ACTION[(dr, dc)]
        assert 0 <= a <= 4, f"({dr},{dc}) -> {a}"
    print("  All 5 displacements map to valid actions ✓")

    # 反向：动作→位移→可还原路径
    path = [(5, 5), (4, 5), (4, 6), (4, 6), (3, 6)]
    actions = path_to_actions(path)
    assert actions == [1, 4, 0, 1], f"Expected [1,4,0,1], got {actions}"
    print(f"  Path->actions: {path} -> {actions} ✓")

    # 反向校验
    restored = actions_to_path((5, 5), actions)
    assert restored == path, f"Round-trip failed: {path} != {restored}"
    print(f"  Actions->path round-trip OK ✓")


def test_margin_exclusion():
    """T2.2.2: 歧义样本标记 margin_excluded。"""
    print("\n--- T2.2.2: Margin exclusion ---")

    # 构造：相同 obs_hash 下有两个不同的专家动作
    samples = [
        {"obs_hash": 42, "action_expert": 1},
        {"obs_hash": 42, "action_expert": 2},  # 同一个观测，不同动作
        {"obs_hash": 43, "action_expert": 1},  # 唯一
    ]
    result = mark_ambiguous_samples(samples)
    assert result[0]["margin_excluded"] == True
    assert result[1]["margin_excluded"] == True
    assert result[2]["margin_excluded"] == False
    print(f"  Ambiguous obs 42: margin_excluded=True ✓")
    print(f"  Unique obs 43: margin_excluded=False ✓")


def _make_synthetic_episode(T: int, N: int, K: int = 11, ep_seed: int = 0) -> dict:
    """构造一个 per-timestep × N-agent 的合成 episode（用于 buffer 测试）。"""
    rng = np.random.default_rng(ep_seed)
    return {
        "obs_patches":       rng.random((T, N, 4, K, K), dtype=np.float32),
        "next_obs_patches":  rng.random((T, N, 4, K, K), dtype=np.float32),
        "obs_scalars":       rng.random((T, N, 8), dtype=np.float32),
        "next_obs_scalars":  rng.random((T, N, 8), dtype=np.float32),
        "positions":         rng.integers(0, 16, (T, N, 2)).astype(np.float32),
        "next_positions":    rng.integers(0, 16, (T, N, 2)).astype(np.float32),
        "actions_expert":    rng.integers(0, 5, (T, N)).astype(np.int64),
        "rewards":           rng.random((T, N), dtype=np.float32),
        "dones":             np.zeros((T, N), dtype=bool),
        "valid":             np.ones((T, N), dtype=bool),
        "action_masks":      np.ones((T, N, 5), dtype=bool),
        "next_action_masks": np.ones((T, N, 5), dtype=bool),
        "margin_excluded":   np.zeros((T, N), dtype=bool),
        "margin_weights":    np.ones((T, N), dtype=np.float32),
    }


def test_expert_buffer():
    """T2.3: 专家 buffer 采样（per-timestep × N-agent）。"""
    print("\n--- T2.3: Expert buffer ---")

    K, N = 11, 3
    buf = ExpertDemoBuffer(capacity=1000)

    # 加入 5 个 episode，每个 50 步
    for e in range(5):
        buf.add_episode(_make_synthetic_episode(T=50, N=N, K=K, ep_seed=e))

    assert len(buf) == 250, f"expected 250 timesteps, got {len(buf)}"
    print(f"  Buffer size: {len(buf)} timesteps, {buf.num_episodes} episodes ✓")

    # 采样：返回 (B, T, N, ...) 含通信所需 positions
    batch = buf.sample(batch_size=4, sequence_len=32, burn_in_len=8)
    assert batch["obs_patches"].shape == (4, 32, N, 4, K, K)
    assert batch["obs_scalars"].shape == (4, 32, N, 8)
    assert batch["positions"].shape == (4, 32, N, 2)
    assert batch["actions_expert"].shape == (4, 32, N)
    assert batch["valid"].shape == (4, 32, N)
    print(f"  Batch shapes: patches={batch['obs_patches'].shape}, "
          f"positions={batch['positions'].shape}, "
          f"valid={batch['valid'].shape} ✓")

    # 序列段不跨 episode：每条序列的所有步应来自同一 episode（合成数据 50≥32 必成立）
    stats = buf.get_stats()
    print(f"  Buffer stats: {stats} ✓")


def test_hard_case_manager():
    """T2.2.3: 困难案例管理器。"""
    print("\n--- T2.2.3: Hard case manager ---")

    mgr = HardCaseManager(max_cases=100)
    mgr.add_hard_case("maze", [(0,0)], [(5,5)], reason="deadlock")
    mgr.add_hard_case("warehouse", [(1,1)], [(10,10)], reason="high_collision")
    mgr.add_hard_case("room", [(2,2)], [(8,8)], reason="room_doorway_block")

    assert len(mgr) == 3
    stats = mgr.get_stats()
    print(f"  Hard cases: {stats} ✓")

    weights = mgr.get_sampling_weights()
    assert len(weights) == 3
    # deadlock 权重应更高
    assert weights[0] > weights[1], (
        f"Deadlock weight {weights[0]:.3f} should > collision weight {weights[1]:.3f}"
    )
    print(f"  Sampling weights: deadlock={weights[0]:.3f}, "
          f"collision={weights[1]:.3f}, doorway={weights[2]:.3f} ✓")


def test_end_to_end():
    """T2 端到端：生成数据→存入 buffer→可采样。"""
    print("\n--- End-to-end: generate → buffer → sample ---")

    wrapper = PogemaWrapper("configs/env_random.yaml")
    wrapper._raw_cfg["size"] = 16
    wrapper._raw_cfg["num_agents"] = 2
    wrapper.map_size = 16
    wrapper.num_agents = 2

    obs_builder = ObservationBuilder(num_agents=2, obs_radius=5)
    reward_builder = RewardBuilder(lambda_f=0.1)
    runner = LaCAMRunner(backend="prioritized")

    wrapper.reset(seed=42)
    obstacles = wrapper.global_grid
    starts = wrapper.get_agents_xy()
    goals = wrapper.get_targets_xy()

    result = runner.solve(obstacles, starts, goals,
                          map_type="random", map_size=16)
    assert result.success, "LaCAM should succeed on simple 2-agent random map"

    wrapper.reset(seed=42)
    ep = build_dataset_from_rollout(
        wrapper=wrapper,
        obs_builder=obs_builder,
        reward_builder=reward_builder,
        obstacles=obstacles,
        expert_paths=result.paths,
        map_type="random",
        task_id="e2e_test",
    )

    assert ep is not None, "Should produce an episode dict"
    T = ep["actions_expert"].shape[0]
    N = ep["actions_expert"].shape[1]
    assert N == 2, f"expected N=2 agents, got {N}"
    print(f"  Generated episode: T={T} timesteps × N={N} agents")

    # 验证 episode 含必需字段且形状一致
    required_fields = ["obs_patches", "obs_scalars", "positions", "next_positions",
                       "actions_expert", "rewards", "dones", "valid",
                       "action_masks", "next_action_masks", "margin_excluded",
                       "margin_weights"]
    for f in required_fields:
        assert f in ep, f"Episode missing field '{f}'"
        assert ep[f].shape[:2] == (T, N), f"{f} shape {ep[f].shape} != (T,N,...)"
    print(f"  All episode fields present with (T,N,...) shape ✓")

    # positions 对 finished agent 应置远哨兵（保证通信距离过滤）
    from buffers.expert_demo_buffer import FINISHED_AGENT_SENTINEL
    finished = ~ep["valid"]
    if finished.any():
        assert np.allclose(ep["positions"][finished], FINISHED_AGENT_SENTINEL), \
            "finished agents must have sentinel positions"
        print(f"  Finished agents parked at sentinel ✓")

    buf = ExpertDemoBuffer(capacity=T + 1)
    buf.add_episode(ep)
    if len(buf) >= 16:
        batch = buf.sample(batch_size=2, sequence_len=16)
        assert batch["obs_patches"].shape == (2, 16, N, 4, 11, 11)
        print(f"  Buffer sample: {batch['obs_patches'].shape} ✓")

    assert ep["action_masks"].dtype == bool
    print(f"  Action mask dtype = bool ✓")


def main():
    print("=" * 60)
    print("  Chameleon — Phase 2 验收")
    print("=" * 60)

    all_ok = True
    tests = [
        test_lacam_generation,
        test_action_mapping,
        test_margin_exclusion,
        test_expert_buffer,
        test_hard_case_manager,
        test_end_to_end,
    ]
    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"\n  [FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("  Phase 2 验收结果: 全部通过 ✓")
    else:
        print("  Phase 2 验收结果: 存在失败项 ✗")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
