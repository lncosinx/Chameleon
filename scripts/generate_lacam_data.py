#!/usr/bin/env python3
"""
阶段一：生成 LaCAM 专家数据并保存到磁盘。

支持多进程并行（--num-workers N），各 worker 独立运行 LaCAM + rollout，
结果写入临时 npz 后由主进程合并，避免跨进程大数组传输开销。
"""
import sys
import os
import shutil
import argparse
import time
import numpy as np
from numpy.lib.format import open_memmap

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

# next_* 字段（next_obs_patches 等）只有已废弃的 DQfD/TD 损失要用；v3+ 的纯 IL 主线
# 从不读取，而 next_obs_patches 单项就占每族约一半磁盘（15k-ep 规模 39GB）。默认**不生成**，
# 需要时用 generate_expert_data.py --store-next 打开（同时 buffers 侧见 attach_disk_dir）。
STORE_NEXT = os.environ.get("CHAMELEON_STORE_NEXT", "0") == "1"


def storage_keys():
    return [k for k in _STORAGE_KEYS if STORE_NEXT or not k.startswith("next_")]


# 单 worker 攒够这么多 episode 就落一次盘（而不是攒完分给它的全部 episode 才写）。
# 2026-08-10 事故：15000-ep 那次每 worker 攒 ~1875 episode 全程不落盘，8 worker
# 一起把 62GB 内存限额打穿、连 sshd 都被 OOM 牵连。见 memory feedback_nohup_long_running。
_FLUSH_EVERY_EPISODES = 300


def _episode_to_storage(ep: dict, ep_id: int) -> dict:
    """将 build_dataset_from_rollout 的 episode dict 转为压缩存储数组 + episode_ids。
    STORE_NEXT=False（默认）时不构建 next_* 字段（省一半磁盘与写入时间）。"""
    T = ep["actions_expert"].shape[0]
    out = {
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
    if not STORE_NEXT:
        for k in [k for k in out if k.startswith("next_")]:
            del out[k]
    return out


def expert_data_exists(base_path: str) -> bool:
    """新格式（目录）或旧格式（单个 .npz）任一存在即算已生成。"""
    return os.path.exists(base_path + ".npz") or \
        os.path.exists(os.path.join(base_path, "episode_ids.npy"))


def save_buffer_to_disk(buffer, base_path: str):
    """
    将 ExpertDemoBuffer（per-timestep 列式存储）保存为**目录**，每个 key 一个
    独立 .npy（而非单个 .npz）——numpy 的 mmap_mode 只对独立 .npy 生效，对 zip
    内的 npz 成员不生效（训练时想 mmap 加载、不把整族数据读进内存，落盘格式
    就不能是 npz）。见 buffers/expert_demo_buffer.py 的 mmap_dir 与
    load_buffer_from_disk。
    """
    if not getattr(buffer, "_arrays_inited", False) or len(buffer) == 0:
        return
    n = buffer._size
    arrays = {
        "obs_patches":       buffer._obs_patches[:n],
        "next_obs_patches":  buffer._next_obs_patches[:n],
        "obs_scalars":       buffer._obs_scalars[:n],
        "next_obs_scalars":  buffer._next_obs_scalars[:n],
        "positions":         buffer._positions[:n],
        "next_positions":    buffer._next_positions[:n],
        "actions_expert":    buffer._actions_expert[:n],
        "rewards":           buffer._rewards[:n],
        "dones":             buffer._dones[:n],
        "valid":             buffer._valid[:n],
        "action_masks":      buffer._action_masks[:n],
        "next_action_masks": buffer._next_action_masks[:n],
        "margin_excluded":   buffer._margin_excluded[:n],
        "margin_weights":    buffer._margin_weights[:n],
        "episode_ids":       buffer._episode_ids[:n],
    }
    arrays = {k: v for k, v in arrays.items() if v is not None
              and (STORE_NEXT or not k.startswith("next_"))}
    os.makedirs(base_path, exist_ok=True)
    total_bytes = 0
    for k, v in arrays.items():
        p = os.path.join(base_path, f"{k}.npy")
        np.save(p, v)
        total_bytes += os.path.getsize(p)
    print(f"  Saved {n:,} timesteps → {base_path}/  ({total_bytes/1e6:.1f} MB)")


def load_buffer_from_disk(base_path: str, buffer):
    """
    批量加载 per-timestep 数据到 ExpertDemoBuffer（按 episode 切分）。

    自动探测两种落盘格式：
      - 新格式（目录，`{base_path}/{key}.npy`）：每个 key 用 mmap_mode='r' 打开，
        真正的磁盘内存映射——之后 add_episode 按 episode 切片搬运，峰值内存只
        有"一个 episode"那么大，与整族数据量无关。2026-08-10 之后生成的数据
        用这个格式。
      - 旧格式（单个 `{base_path}.npz`）：整族一次性读进内存（npz 的 zip 成员
        不支持真 mmap，加载即读满，实测验证过）。兼容 2026-08-09 之前生成的
        数据，量小（几十 GB 级）时安全。

    检测旧版（per-agent，无 positions/episode_ids）格式并明确报错，要求重新生成。
    """
    keys = storage_keys() + ["episode_ids"]
    npz_path = base_path + ".npz"
    dir_ready = os.path.isdir(base_path) and os.path.exists(os.path.join(base_path, "episode_ids.npy"))

    if dir_ready and not getattr(buffer, "_arrays_inited", False):
        # 零拷贝挂载：大字段直接只读 mmap 源目录，不重新分配/搬运（见
        # ExpertDemoBuffer.attach_disk_dir 的事故背景注释）。只对全新空 buffer
        # 生效——这是 train_il_meta.py 的标准用法（整族一次性只读加载）。
        n_steps = buffer.attach_disk_dir(base_path)
        print(f"  Attached {n_steps:,} timesteps ← {base_path}/ (zero-copy mmap)")
        return n_steps

    if dir_ready:
        arrays = {k: np.load(os.path.join(base_path, f"{k}.npy"), mmap_mode="r")
                  for k in keys if os.path.exists(os.path.join(base_path, f"{k}.npy"))}
        src_desc = f"{base_path}/ (mmap)"
    elif os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=False)
        if "positions" not in data.files or "episode_ids" not in data.files:
            raise ValueError(
                f"{npz_path} is in the OLD per-agent format (no 'positions'/'episode_ids'). "
                f"It cannot feed the communication-aware per-timestep buffer. "
                f"Regenerate with: python scripts/generate_lacam_data.py "
                f"--config configs/env_<map>.yaml --num-workers 47"
            )
        arrays = {k: data[k] for k in keys if k in data.files}
        src_desc = npz_path
    else:
        return 0

    if "positions" not in arrays or "episode_ids" not in arrays:
        raise ValueError(f"{base_path}: missing 'positions'/'episode_ids', regenerate the data.")

    n_steps = len(arrays["episode_ids"])
    buffer.load_from_arrays(arrays)
    print(f"  Loaded {n_steps:,} timesteps ← {src_desc}")
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

    # 按 episode 分批积累，每 _FLUSH_EVERY_EPISODES 个立刻落盘一次并释放，
    # 而不是攒完分给这个 worker 的全部 episode 才写——否则单 worker 内存随
    # num_episodes 无界增长（15000-ep 的事故根因，见上方常量注释）。
    chunk_arrays = []   # list of per-episode storage dicts（含 episode_ids），仅本次 flush 区间
    tmp_paths = []
    success_count = 0
    fail_count = 0
    total_steps = 0
    part_idx = 0
    quality = _new_quality_accumulator()

    def _flush():
        nonlocal chunk_arrays, part_idx, total_steps
        if not chunk_arrays:
            return
        steps = sum(v["episode_ids"].shape[0] for v in chunk_arrays)
        all_keys = storage_keys() + ["episode_ids"]
        merged = {
            k: np.concatenate([d[k] for d in chunk_arrays], axis=0)
            for k in all_keys
        }
        tmp_path = os.path.join(tmp_dir, f".tmp_worker_{worker_id:04d}_{part_idx:04d}.npz")
        np.savez(tmp_path, **merged)
        tmp_paths.append(tmp_path)
        total_steps += steps
        part_idx += 1
        chunk_arrays = []

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
        # episode_id 在本 worker 内唯一（success_count 递增，跨 flush 不重置）
        chunk_arrays.append(_episode_to_storage(ep, ep_id=success_count))
        success_count += 1

        if len(chunk_arrays) >= _FLUSH_EVERY_EPISODES:
            _flush()

    _flush()  # 收尾：落盘不足一整批的残余 episode

    if total_steps == 0:
        return [], success_count, fail_count, 0, quality

    print(f"  [worker {worker_id:2d}] ok={success_count} fail={fail_count} "
          f"timesteps={total_steps} parts={len(tmp_paths)}", flush=True)
    return tmp_paths, success_count, fail_count, total_steps, quality


def _merge_tmp_parts(worker_tmp_paths: list, output_dir: str, max_samples: int) -> int:
    """
    合并多个 worker（各自可能多个 part，因增量 flush）的临时 npz，流式写入最终
    目录（每个 key 一个独立 .npy）。

    Args:
        worker_tmp_paths: list of list[str]，外层按 worker 分组、内层为该 worker
            按时间顺序 flush 出的 part 路径（`_worker_generate` 的返回值直接拼起来）。
            **必须按 worker 分组**而不是拍平成一条列表——同一 worker 内 episode_id
            连续递增、跨 part 不重置，若拍平后不按 worker 累加偏移量会把同一 worker
            后续 part 的 episode_id 重复加偏移（2026-08-10 联调时抓到过这个 bug）。

    2026-08-10 改动：原实现把所有 part 读进主进程内存拼成一个大数组再
    `np.savez` 一次性写出——15000-ep 规模下这一步本身就要在主进程里再攒一份
    ~90GB，跟 `_worker_generate` 的无界累积是同一次 OOM 事故的另一半根因。
    现在：
      1. 输出改为**目录 + 每个 key 一个独立 .npy**（而不是单个 .npz）——numpy 的
         `mmap_mode` 只对独立 .npy 生效，对 zip 内的 npz 成员**不生效**（会整
         个读进内存，实测验证过），所以想要训练时真正 mmap 加载，落盘格式就
         不能是 npz。
      2. 用 `open_memmap` 建目标 .npy（磁盘映射），每个 worker part（大小已被
         `_FLUSH_EVERY_EPISODES` 卡住，读进内存很小）读入后立刻拷进目标 mmap
         对应偏移量再释放，峰值内存 ≈ 一个 part 的大小，与总数据量无关。
    每个 worker 的 episode_ids 整体加一个全局偏移保证跨 worker 唯一（偏移只在
    这个 worker 的最后一个 part 处理完后才推进一次）；按 max_samples（时间步）
    截断（保留最后 max_samples 步，语义与旧实现一致）。
    """
    keys = storage_keys() + ["episode_ids"]
    # (path, is_last_part_of_this_worker) 展平，保留 worker 边界信息
    flat = []
    for paths in worker_tmp_paths:
        valid = [p for p in paths if p and os.path.exists(p)]
        for i, p in enumerate(valid):
            flat.append((p, i == len(valid) - 1))
    if not flat:
        return 0

    # ---- Pass 1：只读 episode_ids 长度算总步数（单个 part 本就小，直接读没关系）----
    file_steps = []
    for p, _ in flat:
        with np.load(p, allow_pickle=False) as data:
            file_steps.append(int(data["episode_ids"].shape[0]) if "episode_ids" in data.files else 0)
    total_steps = sum(file_steps)
    if total_steps == 0:
        for p, _ in flat:
            os.remove(p)
        return 0

    skip = max(0, total_steps - max_samples)   # 截断：丢弃最早的 skip 步
    kept = total_steps - skip

    # 用第一个非空 part 确定每个 key 的 dtype / 单步形状
    shapes, dtypes = {}, {}
    for p, _ in flat:
        with np.load(p, allow_pickle=False) as data:
            if int(data["episode_ids"].shape[0]) == 0:
                continue
            for k in keys:
                if k in data.files:
                    shapes[k] = data[k].shape[1:]
                    dtypes[k] = data[k].dtype
        if shapes:
            break

    os.makedirs(output_dir, exist_ok=True)
    out_arrays = {
        k: open_memmap(os.path.join(output_dir, f"{k}.npy"), mode="w+",
                        dtype=dtypes[k], shape=(kept,) + shapes[k])
        for k in keys if k in shapes
    }

    # ---- Pass 2：按 part 顺序流式拷进目标 mmap，越过被截断的 skip 区间 ----
    global_pos = 0    # 截断前的绝对步数游标
    write_cursor = 0  # 目标 mmap 里的写游标
    ep_offset = 0
    for (p, is_last_of_worker), n_i in zip(flat, file_steps):
        if n_i == 0:
            os.remove(p)
            continue
        file_start, file_end = global_pos, global_pos + n_i
        global_pos = file_end
        with np.load(p, allow_pickle=False) as data:
            ids_raw = data["episode_ids"]
            lo = max(file_start, skip)
            if lo < file_end:
                s, e = lo - file_start, file_end - file_start
                n_write = e - s
                out_arrays["episode_ids"][write_cursor:write_cursor + n_write] = \
                    ids_raw[s:e].astype(np.int64) + ep_offset
                for k in storage_keys():
                    if k in out_arrays:
                        out_arrays[k][write_cursor:write_cursor + n_write] = data[k][s:e]
                write_cursor += n_write
            if is_last_of_worker:
                # 该 worker 的 episode_id 在其全部 part 里连续递增、不重置，
                # 用最后一个 part 的最大值才是这个 worker 贡献的 episode 总数。
                ep_offset += int(ids_raw[-1]) + 1
        os.remove(p)

    for arr in out_arrays.values():
        arr.flush()

    total_bytes = sum(os.path.getsize(os.path.join(output_dir, f"{k}.npy")) for k in out_arrays)
    print(f"  Saved {kept:,} timesteps → {output_dir}/  ({total_bytes/1e6:.1f} MB)")
    return kept


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
                     data_path=os.path.join(output_dir, f"expert_{map_type}") if save_to_disk else None,
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

        # 汇总统计（每个 worker 现在可能因增量 flush 返回多个 part 文件；
        # worker_tmp_paths 保持按 worker 分组——episode_id 偏移量的合并逻辑依赖这个分组）
        worker_tmp_paths = [r[0] for r in results]
        tmp_paths = [p for paths in worker_tmp_paths for p in paths]
        success_count = sum(r[1] for r in results)
        fail_count    = sum(r[2] for r in results)
        total_samples = sum(r[3] for r in results)
        quality = _merge_quality_accumulators([r[4] for r in results if len(r) > 4])

        # 合并临时文件到最终目录（每 key 一个独立 .npy，支持训练时 mmap 加载）
        data_path = os.path.join(output_dir, f"expert_{map_type}")
        if save_to_disk:
            final_n = _merge_tmp_parts(worker_tmp_paths, data_path, max_buffer_samples)
        else:
            for p in tmp_paths:
                if p and os.path.exists(p):
                    os.remove(p)
            final_n = 0

        # 多进程路径不用 ExpertDemoBuffer（直接写 npy 目录）
        # 返回空 buffer 保持接口兼容
        expert_buffer = ExpertDemoBuffer(capacity=1)
        hard_case_mgr = HardCaseManager()

        stats = dict(map_type=map_type, num_episodes=num_episodes,
                     num_agents=num_agents, map_size=map_size, backend=backend,
                     num_workers=num_workers,
                     success_count=success_count, fail_count=fail_count,
                     total_samples=total_samples, buffer_size=final_n,
                     elapsed=time.time()-t_start,
                     data_path=data_path if save_to_disk else None,
                     quality=_finalize_quality(quality),
                     buffer_stats={}, hard_case_stats={})

    elapsed = time.time() - t_start
    stats["elapsed"] = elapsed

    # 单进程路径保存
    if num_workers <= 1 and save_to_disk:
        data_path = os.path.join(output_dir, f"expert_{map_type}")
        save_buffer_to_disk(expert_buffer, data_path)

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
