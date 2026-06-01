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

def save_buffer_to_disk(buffer, path: str):
    """将 ExpertDemoBuffer（列式存储）保存为压缩 npz。"""
    if not getattr(buffer, "_arrays_inited", False) or len(buffer) == 0:
        return
    N = buffer._size
    arrays = {
        "obs_patches":       buffer._obs_patches[:N],
        "next_obs_patches":  buffer._next_obs_patches[:N],
        "obs_scalars":       buffer._obs_scalars[:N].astype(np.float16),
        "next_obs_scalars":  buffer._next_obs_scalars[:N].astype(np.float16),
        "actions":           buffer._actions_expert[:N].astype(np.int8),
        "rewards":           buffer._rewards[:N].astype(np.float16),
        "dones":             buffer._dones[:N],
        "action_masks":      buffer._action_masks[:N],
        "next_action_masks": buffer._next_action_masks[:N],
        "margin_excluded":   buffer._margin_excluded[:N],
        "margin_weights":    buffer._margin_weights[:N].astype(np.float16),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"  Saved {N:,} samples → {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


def load_buffer_from_disk(path: str, buffer):
    """
    从 npz 文件批量加载样本到 ExpertDemoBuffer。

    np.load 的 NpzFile 每次 __getitem__ 都会重新解压整个数组，
    必须先一次性读入所有字段，再做 O(1) 次 numpy 批量写入。
    """
    if not os.path.exists(path):
        return 0

    data = np.load(path, allow_pickle=False)
    N = len(data["actions"])

    obs_p   = data["obs_patches"]
    nobs_p  = data["next_obs_patches"]
    sc      = data["obs_scalars"].astype(np.float32)
    nsc     = data["next_obs_scalars"].astype(np.float32)
    acts    = data["actions"].astype(np.int64)
    rews    = data["rewards"].astype(np.float32)
    dns     = data["dones"].astype(bool)
    amask   = data["action_masks"].astype(bool)
    namask  = data["next_action_masks"].astype(bool)
    mexcl   = data["margin_excluded"].astype(bool)
    mwt     = data["margin_weights"].astype(np.float32)

    C, K = obs_p.shape[1], obs_p.shape[3]
    if not buffer._arrays_inited:
        buffer._init_arrays(C, K)

    n    = min(N, buffer.capacity)
    head = buffer._head
    cap  = buffer.capacity

    def _write(dst_arr, src_arr):
        if head + n <= cap:
            dst_arr[head:head + n] = src_arr[:n]
        else:
            first = cap - head
            dst_arr[head:cap]  = src_arr[:first]
            dst_arr[0:n-first] = src_arr[first:n]

    _write(buffer._obs_patches,       obs_p)
    _write(buffer._next_obs_patches,  nobs_p)
    _write(buffer._obs_scalars,       sc)
    _write(buffer._next_obs_scalars,  nsc)
    _write(buffer._actions_expert,    acts)
    _write(buffer._rewards,           rews)
    _write(buffer._dones,             dns)
    _write(buffer._action_masks,      amask)
    _write(buffer._next_action_masks, namask)
    _write(buffer._margin_excluded,   mexcl)
    _write(buffer._margin_weights,    mwt)

    buffer._head = (head + n) % cap
    buffer._size = min(buffer._size + n, cap)

    print(f"  Loaded {N:,} samples ← {path}")
    return N


# ---------------------------------------------------------------------------
# 样本字典 → numpy 数组（供 worker 使用，避免保留大量 Python 对象）
# ---------------------------------------------------------------------------

def _samples_to_numpy(samples: list) -> dict | None:
    """将 build_dataset_from_rollout 返回的样本列表转换为列式 numpy 数组。"""
    if not samples:
        return None

    p0 = samples[0].get("obs_patch")
    C, K = (p0.shape[0], p0.shape[-1]) if p0 is not None else (4, 11)
    N = len(samples)
    z3 = np.zeros((C, K, K), dtype=np.float32)
    z8 = np.zeros(8, dtype=np.float32)
    o5 = np.ones(5, dtype=bool)

    # 预分配
    obs_p  = np.zeros((N, C, K, K), dtype=np.uint8)
    nobs_p = np.zeros((N, C, K, K), dtype=np.uint8)
    sc     = np.zeros((N, 8),       dtype=np.float16)
    nsc    = np.zeros((N, 8),       dtype=np.float16)
    acts   = np.zeros(N,            dtype=np.int8)
    rews   = np.zeros(N,            dtype=np.float16)
    dns    = np.zeros(N,            dtype=bool)
    amask  = np.ones((N, 5),        dtype=bool)
    namask = np.ones((N, 5),        dtype=bool)
    mexcl  = np.zeros(N,            dtype=bool)
    mwt    = np.ones(N,             dtype=np.float16)

    for i, s in enumerate(samples):
        p = s.get("obs_patch", z3)
        obs_p[i] = (np.asarray(p) * 255).clip(0, 255).astype(np.uint8)
        np_val = s.get("next_obs_patch", z3)
        nobs_p[i] = (np.asarray(np_val) * 255).clip(0, 255).astype(np.uint8)
        v = s.get("obs_scalar", z8)
        sc[i] = np.asarray(v, dtype=np.float16)
        v = s.get("next_obs_scalar", z8)
        nsc[i] = np.asarray(v, dtype=np.float16)
        acts[i]  = int(s.get("action_expert", 0))
        rews[i]  = float(s.get("reward", 0.0))
        dns[i]   = bool(s.get("done", False))
        v = s.get("action_mask", o5)
        if v is not None:
            amask[i] = np.asarray(v, dtype=bool)
        v = s.get("next_action_mask", o5)
        if v is not None:
            namask[i] = np.asarray(v, dtype=bool)
        mexcl[i] = bool(s.get("margin_excluded", False))
        mwt[i]   = float(s.get("margin_weight", 1.0))

    return {
        "obs_patches": obs_p, "next_obs_patches": nobs_p,
        "obs_scalars": sc,    "next_obs_scalars": nsc,
        "actions": acts,      "rewards": rews,
        "dones": dns,         "action_masks": amask,
        "next_action_masks": namask,
        "margin_excluded": mexcl, "margin_weights": mwt,
    }


# ---------------------------------------------------------------------------
# 多进程 Worker（模块级，spawn-safe）
# ---------------------------------------------------------------------------

def _worker_generate(args: tuple):
    """
    多进程 worker：处理一批 episode seeds，保存结果到临时 npz。

    模块级函数保证 spawn 启动方式下可 pickle。
    不创建 CUDA 上下文（env/LaCAM 均为 CPU-only）。
    """
    config_path, num_agents, map_size, backend, seeds, worker_id, tmp_dir = args

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
    obs_builder = ObservationBuilder(num_agents=num_agents, obs_radius=5)
    with open(config_path) as f:
        env_cfg = yaml.safe_load(f)
    reward_builder = RewardBuilder.from_config(env_cfg.get("reward", {}))
    runner = LaCAMRunner(backend=backend)
    map_type = wrapper.map_type

    # 按 episode 分批积累，每 episode 立刻转 numpy 释放 Python 对象
    chunk_arrays = []   # list of per-episode numpy dicts
    success_count = 0
    fail_count = 0

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

        samples = build_dataset_from_rollout(
            wrapper=wrapper,
            obs_builder=obs_builder,
            reward_builder=reward_builder,
            obstacles=obstacles,
            expert_paths=result.paths,
            map_type=map_type,
            task_id=f"{map_type}_w{worker_id}_ep{seed}",
        )

        if not samples:
            fail_count += 1
            continue

        ep_arr = _samples_to_numpy(samples)
        if ep_arr is not None:
            chunk_arrays.append(ep_arr)
            success_count += 1

    total_samples = sum(v["actions"].shape[0] for v in chunk_arrays)
    if total_samples == 0:
        return None, success_count, fail_count, 0

    # 合并本 worker 所有 episode 的数组
    merged = {
        k: np.concatenate([d[k] for d in chunk_arrays], axis=0)
        for k in chunk_arrays[0]
    }

    tmp_path = os.path.join(tmp_dir, f".tmp_worker_{worker_id:04d}.npz")
    np.savez_compressed(tmp_path, **merged)
    print(f"  [worker {worker_id:2d}] ok={success_count} fail={fail_count} "
          f"samples={total_samples}", flush=True)
    return tmp_path, success_count, fail_count, total_samples


def _merge_tmp_npz(tmp_paths: list, output_path: str, max_samples: int):
    """合并多个 worker 的临时 npz 文件，写入最终 npz，并删除临时文件。"""
    keys = [
        "obs_patches", "next_obs_patches", "obs_scalars", "next_obs_scalars",
        "actions", "rewards", "dones", "action_masks", "next_action_masks",
        "margin_excluded", "margin_weights",
    ]
    parts = {k: [] for k in keys}

    for p in tmp_paths:
        if p is None or not os.path.exists(p):
            continue
        data = np.load(p, allow_pickle=False)
        for k in keys:
            if k in data:
                parts[k].append(data[k])
        os.remove(p)

    merged = {}
    for k in keys:
        if parts[k]:
            merged[k] = np.concatenate(parts[k], axis=0)

    N = len(merged.get("actions", []))
    if N > max_samples:
        for k in merged:
            merged[k] = merged[k][-max_samples:]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(output_path, **merged)
    final_n = len(merged.get("actions", []))
    print(f"  Saved {final_n:,} samples → {output_path}  "
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
        obs_builder    = ObservationBuilder(num_agents=num_agents, obs_radius=5)
        reward_builder = RewardBuilder.from_config(env_cfg.get("reward", {}))
        runner         = LaCAMRunner(backend=backend)
        expert_buffer  = ExpertDemoBuffer(capacity=max_buffer_samples)
        hard_case_mgr  = HardCaseManager()

        success_count = fail_count = total_samples = 0

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

            samples = build_dataset_from_rollout(
                wrapper=wrapper, obs_builder=obs_builder,
                reward_builder=reward_builder, obstacles=obstacles,
                expert_paths=result.paths, map_type=map_type,
                task_id=f"{map_type}_ep{ep}",
            )
            if samples:
                expert_buffer.add_batch(samples)
                total_samples += len(samples)
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
                      f"samples={total_samples} rate={rate:.1f} ep/s")

        stats = dict(map_type=map_type, num_episodes=num_episodes,
                     num_agents=num_agents, map_size=map_size,
                     success_count=success_count, fail_count=fail_count,
                     total_samples=total_samples, buffer_size=len(expert_buffer),
                     elapsed=time.time()-t_start,
                     npz_path=os.path.join(output_dir, f"expert_{map_type}.npz") if save_to_disk else None,
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
             seed_chunks[i], i, tmp_dir)
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
                     num_agents=num_agents, map_size=map_size,
                     num_workers=num_workers,
                     success_count=success_count, fail_count=fail_count,
                     total_samples=total_samples, buffer_size=final_n,
                     elapsed=time.time()-t_start,
                     npz_path=npz_path if save_to_disk else None,
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
    parser.add_argument("--backend",      type=str, default="prioritized",
                        choices=["prioritized", "lacam"])
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
