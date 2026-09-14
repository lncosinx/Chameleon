#!/usr/bin/env python3
"""
DAgger 式在线纠错数据生成（M3，SCHEDULE.md §3）。

**为什么需要**：纯 BC（离线 IL）只见过专家轨迹上的状态。策略一旦偏离（高密度下必然偏离），
就进入训练分布外的状态，没有任何纠正信号 —— 实测表现为高密度 Deadlock 0.6~0.85、
ISR 尚可（多数体到达）但 CSR 塌到 0（少数体互锁）。这正是 MAPF-GPT-DDG 相对 MAPF-GPT
的增益来源：**把策略自己走出来的失败状态交给专家重新求解，再学回来**。

**两阶段**（因为 LaCAM 的 `_lacam.so` 是 py3.10 编译，而策略推断在 py3.12 的 base 环境，
两者不能同进程；见 CLAUDE.md 的致命坑）：

  phase=collect  （base 环境，用 GPU）
      用当前策略在训练分布上 rollout，检测"卡死状态"（某体连续多步测地距离无改善），
      把该时刻的 (地图, 全体位置, 全体目标) 快照存盘。

  phase=label    （lacam310 环境，纯 CPU，多进程）
      对每个快照，用 LaCAM **从该状态重新求解**，replay 成标准专家数据格式落盘。
      产出的数据 = "策略会走到的困难状态 → 专家的正确解法"。

用法：
  python scripts/generate_dagger_data.py --phase collect --ckpt ckpt_il/xxx.pt \
      --episodes 2000 --out /root/autodl-tmp/dagger/snapshots.npz
  conda activate lacam310
  python scripts/generate_dagger_data.py --phase label \
      --snapshots /root/autodl-tmp/dagger/snapshots.npz \
      --output-dir /root/autodl-tmp/chameleon_data/dagger_r1 --num-workers 20
"""
from __future__ import annotations
import sys, os, argparse, json, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# phase = collect（策略 rollout，找卡死状态）
# ===========================================================================
def phase_collect(args):
    """批量 rollout：一次开 E 个同 agent 数的 env，合成一次 (E,1,N,...) 前向。
    单 episode 逐个跑在本机（25 核被数据生成占满时）实测慢到不可用，故与 eval_bench 一样批量化。"""
    import torch, yaml, tempfile as _tf
    from envs.pogema_env_wrapper import PogemaWrapper
    from envs.observation_builder import ObservationBuilder, build_observations
    from envs.action_mask import build_action_mask_batch
    from scripts.train_il_meta import build_net
    from utils.seed import set_seed

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nc = yaml.safe_load(open(args.network_config))
    for kv in args.set_net:
        k, _, v = kv.partition("="); nc[k.strip()] = yaml.safe_load(v)
        print(f"[set-net] {k.strip()} = {nc[k.strip()]}")
    use_c2g = int(nc.get("patch_channels", 4)) >= 5
    c2g_clip = float(nc.get("cost2go_clip", 10.0))
    net = build_net(nc)
    ck = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["net"].items()})
    net.to(device).eval()
    print(f"[collect] {args.ckpt} (iter={ck.get('meta_iter','?')}) | device={device} | "
          f"batch={args.batch_envs}", flush=True)

    cfgs = [c.strip() for c in args.env_configs.split(",") if c.strip()]
    agent_counts = [int(a) for a in args.agent_counts.split(",")]
    snaps = {"maps": [], "positions": [], "goals": [], "num_agents": []}
    rng = np.random.default_rng(args.seed)
    t0, done_eps = time.time(), 0

    while done_eps < args.episodes:
        cfg_path = cfgs[done_eps // max(1, args.batch_envs) % len(cfgs)]
        na = int(agent_counts[rng.integers(0, len(agent_counts))])
        E = min(args.batch_envs, args.episodes - done_eps)
        base_cfg = yaml.safe_load(open(cfg_path))
        base_cfg.update(num_agents=na, max_episode_steps=args.max_steps, on_target="nothing")
        with _tf.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(base_cfg, f); tmp = f.name

        envs, obs_l, builders, grids, goals, maps_s, dfields = [], [], [], [], [], [], []
        for _ in range(E):
            w = PogemaWrapper(tmp, split="train")     # 训练 seed 段，绝不碰评估地图
            o, _ = w.reset(seed=int(rng.integers(0, 900_000)))
            ob = ObservationBuilder(num_agents=na, obs_radius=5, use_cost2go=use_c2g,
                                    cost2go_clip=c2g_clip)
            ob.reset()
            g = np.array(w.get_targets_xy(), dtype=np.int32)
            envs.append(w); obs_l.append(o); builders.append(ob)
            grids.append(w.global_grid); goals.append(g)
            maps_s.append(_grid_to_str(w.global_grid, pad=5))
            dfields.append([ob._c2g_cache.get(w.global_grid, tuple(g[i])) for i in range(na)])
        os.unlink(tmp)

        best_d = np.full((E, na), np.inf, dtype=np.float32)
        no_improve = np.zeros((E, na), dtype=np.int32)
        taken = np.zeros(E, dtype=np.int32)
        prev = np.zeros((E, na), dtype=np.int64)
        net.reset_comm_state()

        with torch.no_grad():
            for t in range(args.max_steps):
                C = 5 if use_c2g else 4
                patches = np.empty((E, na, C, 11, 11), dtype=np.float32)
                scalars = np.empty((E, na, 8), dtype=np.float32)
                masks = np.ones((E, na, 5), dtype=bool)
                pos = np.zeros((E, na, 2), dtype=np.float32)
                for e, w in enumerate(envs):
                    axy = np.array(w.get_agents_xy(), dtype=np.int32)
                    patches[e], scalars[e] = build_observations(w, obs_l[e], prev[e], builders[e])
                    pos[e] = axy
                    masks[e] = build_action_mask_batch(axy[np.newaxis], grids[e])[0]
                    H, W = grids[e].shape
                    ok = (axy[:, 0] >= 0) & (axy[:, 0] < H) & (axy[:, 1] >= 0) & (axy[:, 1] < W)
                    d = np.full(na, np.inf, dtype=np.float32)
                    idx = np.nonzero(ok)[0]
                    for i in idx:
                        d[i] = dfields[e][i][axy[i, 0], axy[i, 1]]
                    at_goal = d <= 0.0
                    improved = d < best_d[e]
                    best_d[e] = np.where(at_goal, 0.0, np.minimum(best_d[e], d))
                    no_improve[e] = np.where(at_goal | improved, 0, no_improve[e] + 1)

                    stuck = (no_improve[e] >= args.stuck_k) & (best_d[e] > 0)
                    if (taken[e] < args.snaps_per_episode and t >= args.min_step
                            and int(stuck.sum()) >= args.stuck_min):
                        snaps["maps"].append(maps_s[e])
                        snaps["positions"].append(axy.copy())
                        snaps["goals"].append(goals[e].copy())
                        snaps["num_agents"].append(na)
                        taken[e] += 1
                        no_improve[e] = 0     # 同一僵局不重复取样

                pt = torch.from_numpy(patches).unsqueeze(1).to(device)
                st = torch.from_numpy(scalars).unsqueeze(1).to(device)
                po = torch.from_numpy(pos).unsqueeze(1).to(device)
                mt = torch.from_numpy(masks).unsqueeze(1).to(device)
                logits, _ = net(pt, st, po, action_masks=mt, reset_comm=False, stateful=True)
                probs = torch.softmax(logits[:, 0], dim=-1)
                acts = torch.multinomial(probs.reshape(-1, 5), 1).reshape(E, na).cpu().numpy()
                for e, w in enumerate(envs):
                    obs_l[e], _, _, _ = w.step(acts[e].tolist())
                prev = acts.astype(np.int64)

        done_eps += E
        print(f"  [collect] {done_eps}/{args.episodes} eps  snapshots={len(snaps['maps'])}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out,
                        maps=np.array(snaps["maps"], dtype=object),
                        positions=np.array(snaps["positions"], dtype=object),
                        goals=np.array(snaps["goals"], dtype=object),
                        num_agents=np.array(snaps["num_agents"], dtype=np.int32))
    print(f"[collect] {len(snaps['maps'])} 个卡死状态快照 → {args.out}  "
          f"({time.time()-t0:.0f}s)", flush=True)


def _grid_to_str(grid: np.ndarray, pad: int = 0) -> str:
    """把 global_grid 转成地图字符串。

    ★ 坐标系：`wrapper.global_grid` 是 **pogema 已经 pad 了 obs_radius 圈之后**的栅格，
    而 `get_agents_xy()` 返回的也是这个 padded 系。若直接把 padded 栅格当地图串存下来，
    重建环境时 pogema 会**再 pad 一圈**，于是注入的坐标整体偏移 obs_radius —— 起终点落到
    障碍上（pogema 只打一句 Warning 然后偷偷挪到别的空格），数据被静默污染。
    故这里剥掉 pad 圈：剥完之后 "padded 坐标" 正好等于 "新环境的 get_agents_xy 坐标"，
    与 PogemaWrapper.fixed_agents_xy 的约定（收 padded 系）自洽。"""
    g = grid[pad:grid.shape[0] - pad, pad:grid.shape[1] - pad] if pad > 0 else grid
    return "\n".join("".join("#" if v else "." for v in row) for row in g)


# ===========================================================================
# phase = label（LaCAM 从快照状态重解，落成标准专家数据）
# ===========================================================================
def _label_worker(a: tuple):
    (payload_path, idxs, worker_id, tmp_dir, use_cost2go, cost2go_clip, max_steps) = a
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import yaml, tempfile as _tf
    from envs.pogema_env_wrapper import PogemaWrapper
    from envs.observation_builder import ObservationBuilder
    from envs.reward_builder import RewardBuilder
    from lacam.lacam_runner import LaCAMRunner
    from lacam.lacam_dataset_builder import build_dataset_from_rollout
    from scripts.generate_lacam_data import _episode_to_storage, storage_keys

    d = np.load(payload_path, allow_pickle=True)
    maps, positions, goals = d["maps"], d["positions"], d["goals"]
    runner = LaCAMRunner(backend="auto")
    if runner.backend != "lacam":
        raise RuntimeError(f"LaCAM 后端回退成 {runner.backend}！必须在 lacam310 环境跑 "
                           "（见 CLAUDE.md 致命坑），否则专家数据被污染。")

    chunk, tmp_paths, part, ok, fail, total = [], [], 0, 0, 0, 0

    def flush():
        nonlocal chunk, part, total
        if not chunk:
            return
        keys = storage_keys() + ["episode_ids"]
        merged = {k: np.concatenate([c[k] for c in chunk], axis=0) for k in keys}
        p = os.path.join(tmp_dir, f".tmp_worker_{worker_id:04d}_{part:04d}.npz")
        np.savez(p, **merged)
        tmp_paths.append(p); total += merged["episode_ids"].shape[0]
        part += 1; chunk = []

    for j in idxs:
        map_str = str(maps[j])
        pos = np.asarray(positions[j], dtype=np.int32)
        goal = np.asarray(goals[j], dtype=np.int32)
        na = pos.shape[0]
        rows = map_str.splitlines()
        cfg = dict(map_type="random", fixed_map=map_str, fixed_map_name=f"dagger{j}",
                   map_size=max(len(rows), len(rows[0])), num_agents=na, obs_radius=5,
                   max_episode_steps=max_steps, collision_system="soft", on_target="nothing",
                   split="all",
                   fixed_agents_xy=pos.tolist(), fixed_targets_xy=goal.tolist())
        with _tf.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f); cp = f.name
        try:
            w = PogemaWrapper(cp)
        finally:
            os.unlink(cp)
        w.reset(seed=0)
        got_a = np.array(w.get_agents_xy(), dtype=np.int32)
        got_g = np.array(w.get_targets_xy(), dtype=np.int32)
        if not (np.array_equal(got_a, pos) and np.array_equal(got_g, goal)):
            # pogema 遇到"起终点落在障碍上"只打 Warning 然后偷偷换格子 → 必须显式拦截，
            # 否则 DAgger 数据里的状态根本不是策略真实卡住的那个状态。
            fail += 1
            continue
        obstacles = w.global_grid
        res = runner.solve(obstacles=obstacles, starts=w.get_agents_xy(),
                           goals=w.get_targets_xy(), map_type="random",
                           map_size=cfg["map_size"])
        if not res.success:
            fail += 1; continue
        ob = ObservationBuilder(num_agents=na, obs_radius=5, use_cost2go=use_cost2go,
                                cost2go_clip=cost2go_clip)
        rb = RewardBuilder.from_config({})   # 纯 IL 不用 reward，取默认值即可
        ep = build_dataset_from_rollout(wrapper=w, obs_builder=ob, reward_builder=rb,
                                        obstacles=obstacles, expert_paths=res.paths,
                                        map_type="random", task_id=f"dagger_w{worker_id}_{j}")
        if ep is None:
            fail += 1; continue
        chunk.append(_episode_to_storage(ep, ep_id=ok))
        ok += 1
        if len(chunk) >= 200:
            flush()
    flush()
    print(f"  [worker {worker_id:>2}] ok={ok} fail={fail} timesteps={total} parts={len(tmp_paths)}",
          flush=True)
    return tmp_paths


def phase_label(args):
    import multiprocessing as mp
    from scripts.generate_lacam_data import _merge_tmp_parts

    d = np.load(args.snapshots, allow_pickle=True)
    n = len(d["maps"])
    n = min(n, args.max_snapshots) if args.max_snapshots > 0 else n
    print(f"[label] {n} 个快照 → LaCAM 重解，workers={args.num_workers}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, ".tmp"); os.makedirs(tmp_dir, exist_ok=True)

    # 按 agent 数分组：同一输出目录的数据集必须 N 一致（buffer 是 (T,N,...) 列式存储）
    nas = np.asarray(d["num_agents"])[:n]
    for na in sorted(set(nas.tolist())):
        idxs = [int(i) for i in np.nonzero(nas == na)[0]]
        out_dir = os.path.join(args.output_dir, f"expert_dagger{na}")
        if os.path.exists(os.path.join(out_dir, "episode_ids.npy")):
            print(f"  [skip-existing] {out_dir}"); continue
        nw = min(args.num_workers, max(1, len(idxs)))
        shards = [idxs[i::nw] for i in range(nw)]
        t0 = time.time()
        with mp.get_context("spawn").Pool(nw) as pool:
            res = pool.map(_label_worker, [
                (args.snapshots, sh, wid, tmp_dir, args.cost2go, args.cost2go_clip,
                 args.max_steps) for wid, sh in enumerate(shards)])
        total = _merge_tmp_parts(res, out_dir, args.max_samples)
        print(f"  agents={na}: {total:,} timesteps → {out_dir}  ({time.time()-t0:.0f}s)",
              flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=["collect", "label"])
    # collect
    p.add_argument("--ckpt", default=None)
    p.add_argument("--network-config", default="configs/network_gru_c2g.yaml")
    p.add_argument("--set-net", action="append", default=[], metavar="K=V",
                   help="覆盖 network config 键。★ 必须与 --ckpt 的架构一致（如 "
                        "intent_mode=action 会多出 pre_head、comm_in 192→197）。"
                        "凡是'另起进程加载某个 ckpt'的地方都要带上它——这是第 4 次踩。")
    p.add_argument("--env-configs", default="configs/env_random_small.yaml,configs/env_maze_small.yaml")
    p.add_argument("--agent-counts", default="32,48,64")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=128)
    p.add_argument("--stuck-k", type=int, default=8, help="连续多少步无测地进展算卡住")
    p.add_argument("--stuck-min", type=int, default=2, help="至少几个体同时卡住才拍快照")
    p.add_argument("--min-step", type=int, default=8, help="episode 前多少步不拍（还没成形）")
    p.add_argument("--snaps-per-episode", type=int, default=2)
    p.add_argument("--batch-envs", type=int, default=24, help="一次并行 rollout 的 episode 数")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="/root/autodl-tmp/dagger/snapshots.npz")
    # label
    p.add_argument("--snapshots", default="/root/autodl-tmp/dagger/snapshots.npz")
    p.add_argument("--output-dir", default="/root/autodl-tmp/chameleon_data/dagger_r1")
    p.add_argument("--num-workers", type=int, default=20)
    p.add_argument("--max-snapshots", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=20_000_000)
    p.add_argument("--cost2go", action="store_true", default=True)
    p.add_argument("--cost2go-clip", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    (phase_collect if args.phase == "collect" else phase_label)(args)


if __name__ == "__main__":
    main()
