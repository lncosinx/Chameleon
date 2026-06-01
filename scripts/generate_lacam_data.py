#!/usr/bin/env python3
"""
阶段一：生成 LaCAM 专家数据并保存到磁盘。

对指定地图族批量生成专家轨迹，保存为压缩 npz 文件供阶段二加载。

用法:
    python scripts/generate_lacam_data.py --config configs/env_random.yaml \\
        --num-episodes 500 --num-agents 96 --map-size 64 --output-dir data
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


# ---------------------------------------------------------------------------
# 磁盘 I/O
# ---------------------------------------------------------------------------

def save_buffer_to_disk(buffer: ExpertDemoBuffer, path: str):
    """
    将 ExpertDemoBuffer 中的样本保存为压缩 npz 文件。

    存储格式（节省空间）：
    - obs_patches:      uint8  (N, 4, K, K)
    - next_obs_patches: uint8  (N, 4, K, K)
    - obs_scalars:      float16 (N, 8)
    - next_obs_scalars: float16 (N, 8)
    - actions:          int8   (N,)
    - rewards:          float16 (N,)
    - dones:            bool   (N,)
    - action_masks:     bool   (N, 5)
    - next_action_masks:bool   (N, 5)
    - margin_excluded:  bool   (N,)
    - margin_weights:   float16 (N,)
    """
    if len(buffer) == 0:
        return

    samples = list(buffer._buffer)
    N = len(samples)
    K = samples[0].get("obs_patch", np.zeros((4, 11, 11))).shape[-1]

    zero_patch  = np.zeros((4, K, K), dtype=np.float32)
    zero_scalar = np.zeros(8, dtype=np.float32)
    ones_mask   = np.ones(5, dtype=bool)

    def _get_scalar(key, default, dtype):
        return np.array([s.get(key, default) for s in samples], dtype=dtype)

    def _get_patch(key):
        """提取 patch 字段，None 用零填充，输出 uint8 (N,4,K,K)。"""
        arr = np.empty((N, 4, K, K), dtype=np.float32)
        for i, s in enumerate(samples):
            v = s.get(key)
            arr[i] = v if v is not None else zero_patch
        return (arr * 255).clip(0, 255).astype(np.uint8)

    def _get_arr(key, default_fn):
        """提取 1D/2D 数组字段，None 用 default_fn() 填充。"""
        rows = []
        for s in samples:
            v = s.get(key)
            rows.append(v if v is not None else default_fn())
        return np.array(rows)

    arrays = dict(
        obs_patches       = _get_patch("obs_patch"),
        next_obs_patches  = _get_patch("next_obs_patch"),
        obs_scalars       = _get_arr("obs_scalar",       lambda: zero_scalar).astype(np.float16),
        next_obs_scalars  = _get_arr("next_obs_scalar",  lambda: zero_scalar).astype(np.float16),
        actions           = _get_scalar("action_expert", 0,     np.int8),
        rewards           = _get_scalar("reward",        0.0,   np.float16),
        dones             = _get_scalar("done",          False,  bool),
        action_masks      = _get_arr("action_mask",      lambda: ones_mask).astype(bool),
        next_action_masks = _get_arr("next_action_mask", lambda: ones_mask).astype(bool),
        margin_excluded   = _get_scalar("margin_excluded", False, bool),
        margin_weights    = _get_scalar("margin_weight",  1.0,   np.float16),
    )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved {N:,} samples → {path}  ({size_mb:.1f} MB)")


def load_buffer_from_disk(path: str, buffer: ExpertDemoBuffer):
    """
    从 npz 文件加载样本到 ExpertDemoBuffer。
    """
    if not os.path.exists(path):
        return 0

    data = np.load(path, allow_pickle=False)
    N = len(data["actions"])

    # uint8 patch → float32 [0,1]
    obs_p  = data["obs_patches"].astype(np.float32)  / 255.0
    nobs_p = data["next_obs_patches"].astype(np.float32) / 255.0

    for i in range(N):
        s = {
            "obs_patch":        obs_p[i],
            "obs_scalar":       data["obs_scalars"][i].astype(np.float32),
            "action_expert":    int(data["actions"][i]),
            "reward":           float(data["rewards"][i]),
            "next_obs_patch":   nobs_p[i],
            "next_obs_scalar":  data["next_obs_scalars"][i].astype(np.float32),
            "done":             bool(data["dones"][i]),
            "action_mask":      data["action_masks"][i],
            "next_action_mask": data["next_action_masks"][i],
            "margin_excluded":  bool(data["margin_excluded"][i]),
            "margin_weight":    float(data["margin_weights"][i]),
        }
        buffer.add(s)

    print(f"  Loaded {N:,} samples ← {path}")
    return N


def generate_data(
    config_path: str,
    num_episodes: int = 100,
    num_agents: int = 8,
    map_size: int = 16,
    backend: str = "prioritized",
    output_dir: str = "data",
    max_buffer_samples: int = 2_000_000,
    save_to_disk: bool = True,
):
    """
    对一种地图族生成专家数据并保存到磁盘。

    Args:
        config_path:        env_*.yaml 配置文件路径。
        num_episodes:       生成 episode 数。
        num_agents:         每 episode 智能体数（覆盖 config）。
        map_size:           地图尺寸（覆盖 config）。
        backend:            LaCAM 后端（"prioritized" 或 "lacam"）。
        output_dir:         输出目录（保存 npz + stats yaml）。
        max_buffer_samples: buffer 容量上限（默认 2M 样本）。
        save_to_disk:       是否保存 npz 到磁盘（供 train_meta.py 加载）。
    """
    os.makedirs(output_dir, exist_ok=True)

    wrapper = PogemaWrapper(config_path)
    wrapper.map_size = map_size
    wrapper.num_agents = num_agents

    obs_builder = ObservationBuilder(num_agents=num_agents, obs_radius=5)
    reward_builder = RewardBuilder(lambda_f=0.1)
    runner = LaCAMRunner(backend=backend)
    # buffer 容量用 max_buffer_samples，不再用 num_episodes×500（后者在目标规模下丢弃 97% 数据）
    expert_buffer = ExpertDemoBuffer(capacity=max_buffer_samples)
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

    # 保存专家样本到磁盘（供 train_meta.py 加载）
    npz_path = os.path.join(output_dir, f"expert_{map_type}.npz")
    if save_to_disk:
        save_buffer_to_disk(expert_buffer, npz_path)

    # 保存统计
    stats = {
        "map_type": map_type,
        "num_episodes": num_episodes,
        "num_agents": num_agents,
        "map_size": map_size,
        "success_count": success_count,
        "fail_count": fail_count,
        "total_samples": total_samples,
        "buffer_size": len(expert_buffer),
        "elapsed": elapsed,
        "npz_path": npz_path if save_to_disk else None,
        "buffer_stats": expert_buffer.get_stats(),
        "hard_case_stats": hard_case_mgr.get_stats(),
    }

    import yaml
    stats_path = os.path.join(output_dir, f"lacam_stats_{map_type}.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(stats, f)

    print(f"\nDone: {success_count} success, {fail_count} fail, "
          f"{len(expert_buffer):,} samples in buffer, {elapsed:.1f}s")
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
                        help="Output directory（保存 expert_<map>.npz 和 stats yaml）")
    parser.add_argument("--max-buffer-samples", type=int, default=2_000_000,
                        help="Buffer 最大样本数（默认 2M）")
    args = parser.parse_args()

    generate_data(
        config_path=args.config,
        num_episodes=args.num_episodes,
        num_agents=args.num_agents,
        map_size=args.map_size,
        backend=args.backend,
        output_dir=args.output_dir,
        max_buffer_samples=args.max_buffer_samples,
        save_to_disk=True,
    )


if __name__ == "__main__":
    main()
