#!/usr/bin/env python3
"""
阶段一：生成 LaCAM 专家数据并保存到磁盘。

支持多进程并行（--num-workers N），各 worker 独立运行 LaCAM + rollout，
结果写入临时 npz 后由主进程合并，避免跨进程大数组传输开销。
"""
import sys
import os
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# 磁盘 I/O
# ---------------------------------------------------------------------------

# per-timestep × N-agent 列式存储的全部键（不含 episode_ids）
_STORAGE_KEYS = [
    "obs_patches", "next_obs_patches", "obs_scalars", "next_obs_scalars",
    "positions", "next_positions", "actions_expert", "rewards",
    "dones", "valid", "action_masks", "next_action_masks",
    "margin_excluded", "margin_weights",
]


def _episode_to_storage(ep: dict, ep_id: int) -> dict:
    """将 build_dataset_from_rollout 的 episode dict 转为压缩存储数组 + episode_ids。"""
    T = ep["actions_expert"].shape[0]
    return {
        "obs_patches":       (np.asarray(ep["obs_patches"]) * 255).clip(0, 255).astype(np.uint8),
        "next_obs_patches":  (np.asarray(ep["next_obs_patches"]) * 255).clip(0, 255).astype(np.uint8),
        "obs_scalars":       np.asarray(ep["obs_scalars"]).astype(np.float16),
        "next_obs_scalars":  np.asarray(ep["next_obs_scalars"]).astype(np.float16),
        "positions":         np.asarray(ep["positions"]).astype(np.float16),
        "next_positions":    np.asarray(ep["next_positions"]).astype(np.float16),
        "actions_expert":    np.asarray(ep["actions_expert"]).astype(np.int8),
        "rewards":           np.asarray(ep["rewards"]).astype(np.float16),
        "dones":             np.asarray(ep["dones"]).astype(bool),
        "valid":             np.asarray(ep["valid"]).astype(bool),
        "action_masks":      np.asarray(ep["action_masks"]).astype(bool),
        "next_action_masks": np.asarray(ep["next_action_masks"]).astype(bool),
        "margin_excluded":   np.asarray(ep["margin_excluded"]).astype(bool),
        "margin_weights":    np.asarray(ep["margin_weights"]).astype(np.float16),
        "episode_ids":       np.full(T, ep_id, dtype=np.int32),
    }


def save_buffer_to_disk(buffer, path: str):
    """将 ExpertDemoBuffer（per-timestep 列式存储）保存为压缩 npz。"""
    if not getattr(buffer, "_arrays_inited", False) or len(buffer) == 0:
        return
    n = buffer._size
    arrays = {
        "obs_patches":       buffer._obs_patches[:n],
        "next_obs_patches":  buffer._next_obs_patches[:n],
        "obs_scalars":       buffer._obs_scalars[:n].astype(np.float16),
        "next_obs_scalars":  buffer._next_obs_scalars[:n].astype(np.float16),
        "positions":         buffer._positions[:n].astype(np.float16),
        "next_positions":    buffer._next_positions[:n].astype(np.float16),
        "actions_expert":    buffer._actions_expert[:n].astype(np.int8),
        "rewards":           buffer._rewards[:n].astype(np.float16),
        "dones":             buffer._dones[:n],
        "valid":             buffer._valid[:n],
        "action_masks":      buffer._action_masks[:n],
        "next_action_masks": buffer._next_action_masks[:n],
        "margin_excluded":   buffer._margin_excluded[:n],
        "margin_weights":    buffer._margin_weights[:n].astype(np.float16),
        "episode_ids":       buffer._episode_ids[:n].astype(np.int32),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **arrays)
    print(f"  Saved {n:,} timesteps → {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


def load_buffer_from_disk(path: str, buffer):
    """
    从 npz 批量加载 per-timestep 数据到 ExpertDemoBuffer（按 episode 切分）。

    检测旧版（per-agent，无 positions/episode_ids）格式并明确报错，要求重新生成。
    """
    if not os.path.exists(path):
        return 0

    data = np.load(path, allow_pickle=False)

    # 旧格式拒绝：缺少通信所需的 positions / episode 边界 → 必须重新生成
    if "positions" not in data.files or "episode_ids" not in data.files:
        raise ValueError(
            f"{path} is in the OLD per-agent format (no 'positions'/'episode_ids'). "
            f"It cannot feed the communication-aware per-timestep buffer. "
            f"Regenerate with: python scripts/generate_lacam_data.py "
            f"--config configs/env_<map>.yaml --num-workers 47"
        )

    arrays = {
        "obs_patches":       data["obs_patches"],
        "next_obs_patches":  data["next_obs_patches"],
        "obs_scalars":       data["obs_scalars"].astype(np.float32),
        "next_obs_scalars":  data["next_obs_scalars"].astype(np.float32),
        "positions":         data["positions"].astype(np.float32),
        "next_positions":    data["next_positions"].astype(np.float32),
        "actions_expert":    data["actions_expert"].astype(np.int64),
        "rewards":           data["rewards"].astype(np.float32),
        "dones":             data["dones"].astype(bool),
        "valid":             data["valid"].astype(bool),
        "action_masks":      data["action_masks"].astype(bool),
        "next_action_masks": data["next_action_masks"].astype(bool),
        "margin_excluded":   data["margin_excluded"].astype(bool),
        "margin_weights":    data["margin_weights"].astype(np.float32),
        "episode_ids":       data["episode_ids"].astype(np.int64),
    }
    n_steps = len(arrays["episode_ids"])
    buffer.load_from_arrays(arrays)
    print(f"  Loaded {n_steps:,} timesteps ← {path}")
    return n_steps


# ---------------------------------------------------------------------------
# 专家数据质量指标（判断专家是否够好，prioritized vs 真 LaCAM 差异很大）
# ---------------------------------------------------------------------------

_QUALITY_KEYS = [
    "makespan", "soc", "wait_ratio", "margin_excluded_ratio",
    "mean_reward", "vertex_collision_rate",
]


def _episode_quality(ep: dict, result) -> dict:
    """
    从一个 episode（per-timestep dict）+ LaCAM 结果计算质量指标。

    - wait_ratio:            valid agent-步中 wait(0) 占比（过高 → 过度等待）
    - vertex_collision_rate: 执行轨迹中顶点冲突的超额占用 / valid agent-步
                             （真 LaCAM 应 ≈0；prioritized fallback 通常 >0）
    - makespan/soc:          来自求解器（路径质量）
    - margin_excluded_ratio: 歧义样本占比
    - mean_reward:           valid agent-步平均 PBRS reward
    """
    valid = np.asarray(ep["valid"]).astype(bool)        # (T, N)
    acts  = np.asarray(ep["actions_expert"])             # (T, N)
    pos   = np.asarray(ep["positions"])                  # (T, N, 2)
    T, N  = acts.shape
    nvalid = int(valid.sum())

    wait = float((acts[valid] == 0).mean()) if nvalid else 0.0
    mexr = float(np.asarray(ep["margin_excluded"])[valid].mean()) if nvalid else 0.0
    mrew = float(np.asarray(ep["rewards"])[valid].mean()) if nvalid else 0.0

    # 顶点冲突：每步在 valid 体中统计同格超额占用（finished 体已被 valid 掩掉）
    coll = 0
    for t in range(T):
        vt = valid[t]
        if int(vt.sum()) < 2:
            continue
        p = pos[t][vt]
        _, cnt = np.unique(p, axis=0, return_counts=True)
        coll += int((cnt[cnt > 1] - 1).sum())
    coll_rate = coll / max(nvalid, 1)

    return {
        "makespan":              float(getattr(result, "makespan", 0)),
        "soc":                   float(getattr(result, "sum_of_costs", 0)),
        "wait_ratio":            wait,
        "margin_excluded_ratio": mexr,
        "mean_reward":           mrew,
        "vertex_collision_rate": coll_rate,
    }


def _new_quality_accumulator() -> dict:
    """质量指标累加器（按 episode 求和 + 计数）。"""
    acc = {k: 0.0 for k in _QUALITY_KEYS}
    acc["_n"] = 0
    return acc


def _accumulate_quality(acc: dict, q: dict):
    for k in _QUALITY_KEYS:
        acc[k] += q.get(k, 0.0)
    acc["_n"] += 1


def _finalize_quality(acc: dict) -> dict:
    """把累加器转为均值（写入 stats）。"""
    n = max(int(acc.get("_n", 0)), 1)
    out = {f"{k}_mean": round(acc[k] / n, 4) for k in _QUALITY_KEYS}
    out["episodes_measured"] = int(acc.get("_n", 0))
    return out


def _merge_quality_accumulators(accs: list) -> dict:
    """合并多个 worker 的质量累加器。"""
    total = _new_quality_accumulator()
    for a in accs:
        if not a:
            continue
        for k in _QUALITY_KEYS:
            total[k] += a.get(k, 0.0)
        total["_n"] += int(a.get("_n", 0))
    return total


# ---------------------------------------------------------------------------
# 多进程 Worker（模块级，spawn-safe）
# ---------------------------------------------------------------------------

def _worker_generate(args: tuple):
    """
    多进程 worker：处理一批 episode seeds，保存结果到临时 npz。

    模块级函数保证 spawn 启动方式下可 pickle。
    不创建 CUDA 上下文（env/LaCAM 均为 CPU-only）。
    """
    (config_path, num_agents, map_size, backend, seeds, worker_id, tmp_dir,
     use_cost2go, cost2go_clip) = args

    # spawn 下需确保 sys.path 含项目根目录
    _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)

    import yaml
    from envs.pogema_env_wrapper import PogemaWrapper
    from envs.observation_builder import ObservationBuilder
    from envs.reward_builder import RewardBuilder
    from lacam.lacam_runner import LaCAMRunner
    from lacam.lacam_dataset_builder import build_dataset_from_rollout

    wrapper = PogemaWrapper(config_path)
    wrapper.map_size = map_size
    wrapper.num_agents = num_agents
    obs_builder = ObservationBuilder(num_agents=num_agents, obs_radius=5,
                                     use_cost2go=use_cost2go, cost2go_clip=cost2go_clip)
    with open(config_path) as f:
        env_cfg = yaml.safe_load(f)
    reward_builder = RewardBuilder.from_config(env_cfg.get("reward", {}))
    runner = LaCAMRunner(backend=backend)
    map_type = wrapper.map_type

    # 按 episode 分批积累，每 episode 立刻转存储数组释放 Python 对象
    chunk_arrays = []   # list of per-episode storage dicts（含 episode_ids）
    success_count = 0
    fail_count = 0
    quality = _new_quality_accumulator()

    for seed in seeds:
        wrapper.reset(seed=seed)
        obstacles = wrapper.global_grid
        starts    = wrapper.get_agents_xy()
        goals     = wrapper.get_targets_xy()

        result = runner.solve(obstacles=obstacles, starts=starts, goals=goals,
                              map_type=map_type, map_size=map_size)

        if not result.success:
            fail_count += 1
            continue

        ep = build_dataset_from_rollout(
            wrapper=wrapper,
            obs_builder=obs_builder,
            reward_builder=reward_builder,
            obstacles=obstacles,
            expert_paths=result.paths,
            map_type=map_type,
            task_id=f"{map_type}_w{worker_id}_ep{seed}",
        )

        if ep is None:
            fail_count += 1
            continue

        _accumulate_quality(quality, _episode_quality(ep, result))
        # episode_id 在本 worker 内唯一（success_count 递增）
        chunk_arrays.append(_episode_to_storage(ep, ep_id=success_count))
        success_count += 1

    total_steps = sum(v["episode_ids"].shape[0] for v in chunk_arrays)
    if total_steps == 0:
        return None, success_count, fail_count, 0, quality

    # 合并本 worker 所有 episode 的数组（episode_ids 已在本 worker 内唯一）
    all_keys = _STORAGE_KEYS + ["episode_ids"]
    merged = {
        k: np.concatenate([d[k] for d in chunk_arrays], axis=0)
        for k in all_keys
    }

    tmp_path = os.path.join(tmp_dir, f".tmp_worker_{worker_id:04d}.npz")
    np.savez(tmp_path, **merged)
    print(f"  [worker {worker_id:2d}] ok={success_count} fail={fail_count} "
          f"timesteps={total_steps}", flush=True)
    return tmp_path, success_count, fail_count, total_steps, quality


def _merge_tmp_npz(tmp_paths: list, output_path: str, max_samples: int):
    """
    合并多个 worker 的临时 npz，写入最终 npz，删除临时文件。
    各 worker 的 episode_ids 加全局偏移保证唯一；按 max_samples（时间步）截断。
    """
    keys = _STORAGE_KEYS + ["episode_ids"]
    parts = {k: [] for k in keys}

    ep_offset = 0
    for p in tmp_paths:
        if p is None or not os.path.exists(p):
            continue
        data = np.load(p, allow_pickle=False)
        if "episode_ids" not in data.files:
            os.remove(p)
            continue
        ids = data["episode_ids"].astype(np.int64) + ep_offset
        ep_offset = int(ids.max()) + 1 if len(ids) else ep_offset
        for k in _STORAGE_KEYS:
            if k in data.files:
                parts[k].append(data[k])
        parts["episode_ids"].append(ids)
        os.remove(p)

    merged = {}
    for k in keys:
        if parts[k]:
            merged[k] = np.concatenate(parts[k], axis=0)

    # 按时间步截断（保留最后 max_samples 步；episode_ids 仍标记边界，
    # 被截断的首个 episode 只是行数变少，采样的端点同 episode 检查可正确处理）
    N = len(merged.get("episode_ids", []))
    if N > max_samples:
        for k in merged:
            merged[k] = merged[k][-max_samples:]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # 不压缩：zlib 对这些数值数组要单核跑 ~13min/族 才省 7× 磁盘，而磁盘管够。
    # 改 np.savez 后合并从 ~13min 降到 ~1min，且训练启动时加载也免去解压（快得多）。
    np.savez(output_path, **merged)
    final_n = len(merged.get("episode_ids", []))
    print(f"  Saved {final_n:,} timesteps → {output_path}  "
          f"({os.path.getsize(output_path)/1e6:.1f} MB)")
    return final_n


# ---------------------------------------------------------------------------
# 主生成函数
# ---------------------------------------------------------------------------

def generate_data(
    config_path: str,
    num_episodes: int = 100,
    num_agents: int = 8,
    map_size: int = 16,
    backend: str = "prioritized",
    output_dir: str = "data",
    max_buffer_samples: int = 2_000_000,
    save_to_disk: bool = True,
    num_workers: int = 1,
    use_cost2go: bool = False,
    cost2go_clip: float = 10.0,
):
    """
    对一种地图族生成专家数据，支持多进程并行。

    Args:
        num_workers: 并行进程数。1=单进程，-1=auto(cpu_count-1)，N=N 进程。
    """
    import yaml
    from envs.pogema_env_wrapper import PogemaWrapper
    from buffers.expert_demo_buffer import ExpertDemoBuffer
    from lacam.hard_case_manager import HardCaseManager

    os.makedirs(output_dir, exist_ok=True)

    with open(config_path) as f:
        env_cfg = yaml.safe_load(f)
    map_type = env_cfg.get("map_type", "random")

    # 确定有效 worker 数
    if num_workers == -1:
        num_workers = max(1, (os.cpu_count() or 4) - 1)
    num_workers = min(num_workers, num_episodes)

    print(f"Generating expert data for '{map_type}': "
          f"{num_episodes} episodes, {num_agents} agents, size={map_size}, "
          f"workers={num_workers}")

    all_seeds = [10000 + ep for ep in range(num_episodes)]
    t_start = time.time()

    if num_workers <= 1:
        # ---- 单进程路径（向后兼容）----
        from envs.observation_builder import ObservationBuilder
        from envs.reward_builder import RewardBuilder
        from lacam.lacam_runner import LaCAMRunner
        from lacam.lacam_dataset_builder import build_dataset_from_rollout

        wrapper = PogemaWrapper(config_path)
        wrapper.map_size = map_size
        wrapper.num_agents = num_agents
        obs_builder    = ObservationBuilder(num_agents=num_agents, obs_radius=5,
                                            use_cost2go=use_cost2go, cost2go_clip=cost2go_clip)
        reward_builder = RewardBuilder.from_config(env_cfg.get("reward", {}))
        runner         = LaCAMRunner(backend=backend)
        # 容量按「时间步」计（每步含 N 体）；单进程按实际 episode 数上界分配，
        # 避免用 max_buffer_samples（可能很大）预分配巨型数组。
        cap_steps = min(max_buffer_samples,
                        num_episodes * wrapper.max_episode_steps)
        expert_buffer  = ExpertDemoBuffer(capacity=max(cap_steps, 1),
                                          num_agents=num_agents)
        hard_case_mgr  = HardCaseManager()

        success_count = fail_count = total_samples = 0
        quality = _new_quality_accumulator()

        for ep, seed in enumerate(all_seeds):
            wrapper.reset(seed=seed)
            obstacles = wrapper.global_grid
            starts    = wrapper.get_agents_xy()
            goals     = wrapper.get_targets_xy()

            result = runner.solve(obstacles=obstacles, starts=starts, goals=goals,
                                  map_type=map_type, map_size=map_size)
            if not result.success:
                fail_count += 1
                hard_case_mgr.add_hard_case(map_type=map_type, starts=starts,
                                             goals=goals, reason="lacam_failure",
                                             obstacles=obstacles)
                continue

            ep_data = build_dataset_from_rollout(
                wrapper=wrapper, obs_builder=obs_builder,
                reward_builder=reward_builder, obstacles=obstacles,
                expert_paths=result.paths, map_type=map_type,
                task_id=f"{map_type}_ep{ep}",
            )
            if ep_data is not None:
                _accumulate_quality(quality, _episode_quality(ep_data, result))
                expert_buffer.add_episode(ep_data)
                total_samples += int(ep_data["actions_expert"].shape[0])  # 时间步数
                success_count += 1
            else:
                fail_count += 1
                hard_case_mgr.add_hard_case(map_type=map_type, starts=starts,
                                             goals=goals, reason="no_samples",
                                             obstacles=obstacles)

            if (ep + 1) % 10 == 0:
                elapsed = time.time() - t_start
                rate = (ep + 1) / max(elapsed, 0.01)
                print(f"  [{ep+1}/{num_episodes}] ok={success_count} fail={fail_count} "
                      f"timesteps={total_samples} rate={rate:.1f} ep/s")

        stats = dict(map_type=map_type, num_episodes=num_episodes,
                     num_agents=num_agents, map_size=map_size, backend=backend,
                     success_count=success_count, fail_count=fail_count,
                     total_samples=total_samples, buffer_size=len(expert_buffer),
                     elapsed=time.time()-t_start,
                     npz_path=os.path.join(output_dir, f"expert_{map_type}.npz") if save_to_disk else None,
                     quality=_finalize_quality(quality),
                     buffer_stats=expert_buffer.get_stats(),
                     hard_case_stats=hard_case_mgr.get_stats())

    else:
        # ---- 多进程路径 ----
        import multiprocessing as mp

        # 将 seeds 均匀分配给 workers
        seed_chunks = [all_seeds[i::num_workers] for i in range(num_workers)]
        tmp_dir = output_dir  # 临时文件与输出在同一目录

        worker_args = [
            (config_path, num_agents, map_size, backend,
             seed_chunks[i], i, tmp_dir, use_cost2go, cost2go_clip)
            for i in range(num_workers)
        ]

        # spawn 避免 fork 带来的 CUDA 上下文 / 文件描述符泄漏
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(_worker_generate, worker_args)

        # 汇总统计
        tmp_paths = [r[0] for r in results]
        success_count = sum(r[1] for r in results)
        fail_count    = sum(r[2] for r in results)
        total_samples = sum(r[3] for r in results)
        quality = _merge_quality_accumulators([r[4] for r in results if len(r) > 4])

        # 合并临时文件到最终 npz
        npz_path = os.path.join(output_dir, f"expert_{map_type}.npz")
        if save_to_disk:
            final_n = _merge_tmp_npz(tmp_paths, npz_path, max_buffer_samples)
        else:
            for p in tmp_paths:
                if p and os.path.exists(p):
                    os.remove(p)
            final_n = 0

        # 多进程路径不用 ExpertDemoBuffer（直接写 npz）
        # 返回空 buffer 保持接口兼容
        expert_buffer = ExpertDemoBuffer(capacity=1)
        hard_case_mgr = HardCaseManager()

        stats = dict(map_type=map_type, num_episodes=num_episodes,
                     num_agents=num_agents, map_size=map_size, backend=backend,
                     num_workers=num_workers,
                     success_count=success_count, fail_count=fail_count,
                     total_samples=total_samples, buffer_size=final_n,
                     elapsed=time.time()-t_start,
                     npz_path=npz_path if save_to_disk else None,
                     quality=_finalize_quality(quality),
                     buffer_stats={}, hard_case_stats={})

    elapsed = time.time() - t_start
    stats["elapsed"] = elapsed

    # 单进程路径保存 npz
    if num_workers <= 1 and save_to_disk:
        npz_path = os.path.join(output_dir, f"expert_{map_type}.npz")
        save_buffer_to_disk(expert_buffer, npz_path)

    # 保存统计
    import yaml
    stats_path = os.path.join(output_dir, f"lacam_stats_{map_type}.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(stats, f)

    rate = success_count / max(elapsed, 0.01)
    print(f"\nDone: {success_count} success, {fail_count} fail, "
          f"{stats.get('buffer_size', total_samples):,} samples, "
          f"{elapsed:.1f}s ({rate:.2f} ep/s)")
    print(f"Stats saved to {stats_path}")

    return expert_buffer, hard_case_mgr, stats


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate LaCAM expert data")
    parser.add_argument("--config",       type=str, default="configs/env_random.yaml")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-agents",   type=int, default=8)
    parser.add_argument("--map-size",     type=int, default=16)
    parser.add_argument("--backend",      type=str, default="auto",
                        choices=["auto", "lacam", "prioritized"],
                        help="auto=有 _lacam.so 则用真 LaCAM，否则回退 prioritized；"
                             "lacam=强制真 LaCAM；prioritized=纯 Python fallback（质量差，"
                             "有碰撞/SoC 翻倍，仅无绑定时应急用）")
    parser.add_argument("--output-dir",   type=str, default="data")
    parser.add_argument("--max-buffer-samples", type=int, default=2_000_000)
    parser.add_argument("--num-workers",  type=int, default=-1,
                        help="并行进程数：-1=auto(cpu_count-1), 1=单进程, N=N 进程")
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
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
