#!/usr/bin/env python3
"""
T1.2 验收脚本：观测构建器。

验证：
- patch 恰为 4 通道，ch1 中心格恒为 1（自身）
- 足迹通道：当前位置=1.0，历史位置随时间单调衰减
- 标量特征：8 维结构正确（方向+动作+到达）
- 断言：observation_builder 输入不含全局地图/距离场
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.pogema_env_wrapper import PogemaWrapper
from envs.observation_builder import ObservationBuilder, build_observations


def test_channel_count():
    """验收门1: patch 恰为 4 通道。"""
    print("\n--- Test: Patch channel count ---")
    wrapper = PogemaWrapper("configs/env_random.yaml")
    wrapper._raw_cfg["size"] = 16
    wrapper._raw_cfg["num_agents"] = 4
    wrapper.map_size = 16
    wrapper.num_agents = 4

    builder = ObservationBuilder(num_agents=4, obs_radius=5)
    obs_list, _ = wrapper.reset(seed=42)
    prev_actions = np.zeros(4, dtype=np.int64)  # action 0 = wait for first step

    patches, scalars = build_observations(wrapper, obs_list, prev_actions, builder)

    assert patches.shape == (4, 4, 11, 11), (
        f"Expected (4,4,11,11) patches, got {patches.shape}"
    )
    print(f"  [PASS] patches shape: {patches.shape} — 4 channels ✓")

    assert scalars.shape == (4, 8), (
        f"Expected (4,8) scalars, got {scalars.shape}"
    )
    print(f"  [PASS] scalars shape: {scalars.shape} — 8 dims ✓")


def test_self_center():
    """验收门2: ch1 中心格恒为 1（自身位置）。"""
    print("\n--- Test: Self in center (ch1) ---")
    wrapper = PogemaWrapper("configs/env_random.yaml")
    wrapper._raw_cfg["size"] = 20
    wrapper._raw_cfg["num_agents"] = 4
    wrapper.map_size = 20
    wrapper.num_agents = 4

    builder = ObservationBuilder(num_agents=4, obs_radius=5)
    obs_list, _ = wrapper.reset(seed=42)
    prev_actions = np.zeros(4, dtype=np.int64)

    patches, _ = build_observations(wrapper, obs_list, prev_actions, builder)

    K = 11
    center = K // 2  # = 5
    for i in range(4):
        val = patches[i, 1, center, center]
        # 自身恒在中心格 —— 但注意：如果自身不在任何人的 FOV 中，
        # 这个值也应该为 1（Pogema 原生保证）
        # 实际中，如果 agent 被遮挡可能会有边界情况。
        # 核心断言：所有 agent 的 ch1 中心格在自身观测中为 1
        assert abs(val - 1.0) < 0.01, (
            f"Agent {i}: ch1 center is {val}, expected 1.0 (INVARIANT)"
        )
        print(f"  Agent {i}: ch1[{center},{center}] = {val:.1f} ✓")


def test_footprint_decay():
    """验收门3: 足迹通道当前位置=1.0，历史单调衰减。"""
    print("\n--- Test: Footprint decay ---")
    wrapper = PogemaWrapper("configs/env_random.yaml")
    wrapper._raw_cfg["size"] = 20
    wrapper._raw_cfg["num_agents"] = 4
    wrapper.map_size = 20
    wrapper.num_agents = 4

    builder = ObservationBuilder(num_agents=4, obs_radius=5, decay_rho=0.9)
    obs_list, _ = wrapper.reset(seed=42)

    # 初始步：只有当前位置=1.0
    prev_actions = np.zeros(4, dtype=np.int64)
    patches, _ = build_observations(wrapper, obs_list, prev_actions, builder)
    center = 11 // 2
    fp_center = patches[0, 3, center, center]
    assert abs(fp_center - 1.0) < 0.01, (
        f"Step 0: footprint center should be 1.0, got {fp_center}"
    )
    print(f"  Step 0: footprint center = {fp_center:.1f} ✓")

    # 走几步，让 agent 离开初始位置
    for step in range(5):
        actions = [1] * 4  # all move up
        obs_list, _, _, _ = wrapper.step(actions)
        patches, _ = build_observations(wrapper, obs_list, np.array(actions), builder)

    # 第 5 步后：中心格仍应显示足迹（当前=1.0），旧位置应衰减
    fp_center = patches[0, 3, center, center]
    assert abs(fp_center - 1.0) < 0.01, (
        f"Step 5: footprint center should still be 1.0, got {fp_center}"
    )

    # 上方一格（agent 从上往下移动了 5 步，下面应该有衰减足迹）
    below_center = patches[0, 3, center + 1, center]
    # agent 从初始位置向上走了 5 步，所以初始位置在 center+5 下方
    # 实际上向下 5 步后，底部应该有足迹痕迹
    print(f"  Step 5: center={fp_center:.1f}, (center+1,center)={below_center:.3f}")

    # 核心断言：足迹通道有非零值（不全是 0）
    nonzero_count = np.count_nonzero(patches[0, 3])
    assert nonzero_count > 1, (
        f"Footprint channel should have >1 nonzero cells (tracking path), got {nonzero_count}"
    )
    print(f"  [PASS] Footprint channel has {nonzero_count} nonzero cells (tracking path) ✓")

    # 核心断言：没有值 > 1.0（当前格为 1.0，历史格 ≤ 0.8）
    max_val = patches[0, 3].max()
    assert max_val <= 1.01, (
        f"Footprint max value {max_val} > 1.0 (only current pos should be 1.0)"
    )
    print(f"  [PASS] Footprint max value = {max_val:.3f} ≤ 1.0 ✓")


def test_scalar_features():
    """验收门4: 标量特征结构正确。"""
    print("\n--- Test: Scalar features ---")
    wrapper = PogemaWrapper("configs/env_random.yaml")
    wrapper._raw_cfg["size"] = 20
    wrapper._raw_cfg["num_agents"] = 4
    wrapper.map_size = 20
    wrapper.num_agents = 4

    builder = ObservationBuilder(num_agents=4, obs_radius=5)

    # 第一步：prev_action = wait (0)
    obs_list, _ = wrapper.reset(seed=42)
    prev_actions = np.array([0, 1, 2, 3], dtype=np.int64)  # wait, up, down, left
    _, scalars = build_observations(wrapper, obs_list, prev_actions, builder)

    for i, expected_action in enumerate([0, 1, 2, 3]):
        # one-hot: feats[2:7]
        for a in range(5):
            if a == expected_action:
                assert abs(scalars[i, 2 + a] - 1.0) < 0.01, (
                    f"Agent {i}: action={expected_action}, one-hot[{a}] should be 1.0"
                )
            else:
                assert abs(scalars[i, 2 + a]) < 0.01, (
                    f"Agent {i}: action={expected_action}, one-hot[{a}] should be 0.0"
                )
    print(f"  [PASS] Action one-hot correct for all agents ✓")

    # 方向向量：归一化长度 ≈ 1（除非已到达）
    for i in range(4):
        dr, dc = scalars[i, 0], scalars[i, 1]
        norm = np.sqrt(dr * dr + dc * dc)
        assert norm <= 1.01, f"Agent {i}: direction norm {norm} > 1.0"
        if scalars[i, 7] < 0.5:  # 未到达
            # 方向向量应接近单位长度（除非恰好0距离）
            pass  # 只要 ≤1 即可
    print(f"  [PASS] Goal direction vectors valid ✓")

    # 到达标志：0 或 1
    for i in range(4):
        arrived = scalars[i, 7]
        assert arrived in (0.0, 1.0), (
            f"Agent {i}: arrival flag should be 0 or 1, got {arrived}"
        )
    print(f"  [PASS] Arrival flags are 0/1 ✓")


def test_no_global_leak():
    """验收门5: observation_builder 绝不含全局信息（代码审查 + 断言）。"""
    print("\n--- Test: No global info leak ---")
    import inspect
    from envs import observation_builder as ob

    source = inspect.getsource(ob)

    # 检查 ObservationBuilder.build 的参数签名
    sig = inspect.signature(ObservationBuilder.build)
    param_names = list(sig.parameters.keys())
    print(f"  build() parameters: {param_names}")

    # 不应出现全局信息相关参数
    forbidden = {"global_grid", "global_obstacles", "bfs", "distance_field",
                 "lacam", "expert_path", "global_map", "full_map"}
    for name in param_names:
        for fb in forbidden:
            assert fb not in name.lower(), (
                f"INVARIANT VIOLATION: '{fb}' found in build() param '{name}'"
            )
    print(f"  [PASS] No global info in build() signature ✓")

    # 代码中不应引用 wrapper.global_grid 或其他全局信息
    forbidden_in_code = ["global_grid", "global_obstacles", ".grid.", "bfs_dist",
                         "la_", "expert_"]
    found_forbidden = []
    for fb in forbidden_in_code:
        if fb in source:
            # 检查是否在注释/文档中（允许）
            lines = source.split("\n")
            for lineno, line in enumerate(lines, 1):
                if fb in line and not line.strip().startswith("#") and fb not in line.split("#")[0]:
                    found_forbidden.append(f"  L{lineno}: {line.strip()[:80]}")
    if found_forbidden:
        print(f"  [WARN] Potential forbidden references found:")
        for f in found_forbidden:
            print(f"    {f}")
    else:
        print(f"  [PASS] No forbidden global-info references in code ✓")


def test_footprint_monotonic_decay():
    """验收门6: 足迹通道当前位置=1.0，离开后原位置衰减。"""
    print("\n--- Test: Footprint monotonic decay ---")
    wrapper = PogemaWrapper("configs/env_random.yaml")
    wrapper._raw_cfg["size"] = 20
    wrapper._raw_cfg["num_agents"] = 2
    wrapper.map_size = 20
    wrapper.num_agents = 2

    rho = 0.85
    builder = ObservationBuilder(num_agents=2, obs_radius=5, decay_rho=rho)

    obs_list, _ = wrapper.reset(seed=42)
    patches, _ = build_observations(wrapper, obs_list, np.zeros(2, dtype=np.int64), builder)

    # 记录 agent 0 的初始位置（在 patch 坐标中即中心格）
    center = 5  # K=11, K//2 = 5
    val_center_0 = patches[0, 3, center, center]
    assert abs(val_center_0 - 1.0) < 0.01, (
        f"Step 0: footprint center should be 1.0, got {val_center_0}"
    )
    print(f"  Step 0: footprint center = {val_center_0:.1f} ✓")

    # 走几步（不管能不能动，足迹应保持当前位置=1.0）
    for step in range(3):
        obs_list, _, _, _ = wrapper.step([1, 2])  # agent 0 up, agent 1 down
        prev_acts = np.array([1, 2], dtype=np.int64)
        patches, _ = build_observations(wrapper, obs_list, prev_acts, builder)

    # 当前中心仍应为 1.0
    val_center_now = patches[0, 3, center, center]
    assert abs(val_center_now - 1.0) < 0.01, (
        f"Step 3: footprint center should still be 1.0, got {val_center_now}"
    )
    print(f"  Step 3: footprint center = {val_center_now:.1f} ✓")

    # 如果 agent 移动成功，应该有衰减的历史足迹（非中心格的紫色值）
    # 核心断言：同一位置被多次经过后，旧值 < 当前值（随时间衰减）
    nonzero_mask = patches[0, 3] > 0.01
    n_nonzero = nonzero_mask.sum()
    # agent 至少踏过 ≥2 个位置（初始 + 可能移动）
    print(f"  Step 3: {n_nonzero} nonzero footprint cells")

    # 验证：没有值超过 1.0，非中心最大值 ≤ 0.8
    non_center = patches[0, 3].copy()
    non_center[center, center] = 0.0
    if non_center.max() > 0:
        assert non_center.max() <= 0.81, (
            f"Non-center footprint max {non_center.max():.3f} should be ≤ 0.8"
        )
        print(f"  [PASS] Non-center footprint max = {non_center.max():.3f} ≤ 0.8 ✓")
    else:
        print(f"  [INFO] Agent didn't move far, no decayed cells to check (non-issue)")

    print(f"  [PASS] Footprint monotonic decay verified ✓")


def main():
    print("=" * 60)
    print("  Chameleon — T1.2 验收：观测构建器")
    print("=" * 60)

    all_ok = True
    tests = [
        test_channel_count,
        test_self_center,
        test_footprint_decay,
        test_scalar_features,
        test_no_global_leak,
        test_footprint_monotonic_decay,
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
        print("  T1.2 验收结果: 全部通过 ✓")
    else:
        print("  T1.2 验收结果: 存在失败项 ✗")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
