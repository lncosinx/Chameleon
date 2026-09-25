#!/usr/bin/env python3
"""
把 marl-gpt 的 POGEMA 评测地图 + POGEMA 确定性放置的起终点，导出成 EPH 测试集格式
(list of (map[H,W]{0,1}, starts[N,2], goals[N,2]))，供 EPH 在**同一套实例**上评测。

实例与 scripts/eval_marlgpt_benchmark.py 完全一致：同 20 张抽样地图/族、同 agent 数网格、
同 seed（默认 0），因此 EPH 与 comm2 跑的是逐张相同的 (map, starts, goals)。

坐标处理：POGEMA 的 global_grid 会按 obs_radius 外扩填边；已实测 global_grid[r:r+H, r:r+W]
恰等于原始 ascii 地图（r=obs_radius）。故裁回内部真图、位置整体减 r，得到忠实的原图实例。

用法（在 Chameleon 根目录、marlgpt 环境下）:
  python scripts/convert_pogema_to_eph_instances.py --out /path/to/eph_instances
"""
from __future__ import annotations
import sys, os, argparse, pickle, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import yaml
from envs.pogema_env_wrapper import PogemaWrapper

MARLGPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "marl-gpt", "eval_configs")
FAMILY_DIR = {"random": "01-random", "mazes": "02-mazes"}
OBS_RADIUS = 5


def sample_map_names(maps: dict, num_maps: int) -> list:
    names = sorted(maps.keys())
    if num_maps <= 0 or num_maps >= len(names):
        return names
    idx = sorted(set(np.linspace(0, len(names) - 1, num_maps).round().astype(int).tolist()))
    return [names[i] for i in idx]


def make_instance(map_str, name, num_agents, seed):
    rows = map_str.splitlines()
    H, W = len(rows), len(rows[0])
    ascii_grid = np.array([[1 if c == '#' else 0 for c in r] for r in rows], dtype=np.int64)
    cfg = dict(map_type="random", fixed_map=map_str, fixed_map_name=name, map_size=max(H, W),
               num_agents=num_agents, obs_radius=OBS_RADIUS, max_episode_steps=256,
               collision_system="soft", on_target="finish", split="test")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f); p = f.name
    try:
        w = PogemaWrapper(p)
    finally:
        os.unlink(p)
    w.reset(seed=seed)
    starts = np.array(w.get_agents_xy(), dtype=np.int64) - OBS_RADIUS
    goals = np.array(w.get_targets_xy(), dtype=np.int64) - OBS_RADIUS
    # 完整性校验：起终点必须落在真图空地、且在界内
    assert starts.shape == (num_agents, 2) and goals.shape == (num_agents, 2)
    for p_ in np.vstack([starts, goals]):
        assert 0 <= p_[0] < H and 0 <= p_[1] < W, f"pos {p_} out of {H}x{W}"
        assert ascii_grid[p_[0], p_[1]] == 0, f"pos {p_} on obstacle"
    # 起点互不重叠、终点互不重叠（POGEMA 保证，但再确认）
    assert np.unique(starts, axis=0).shape[0] == num_agents
    assert np.unique(goals, axis=0).shape[0] == num_agents
    return (ascii_grid, starts, goals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="random,mazes")
    ap.add_argument("--num-maps", type=int, default=20)
    ap.add_argument("--agent-counts", default="8,16,24,32,48,64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/data/lncosinx/docker_tmp/claude-1011/-home-lncosinx-Chameleon/"
                                     "3d007f56-a6c3-4c19-82a3-a2e88badc3e3/scratchpad/eph_instances")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    agent_counts = [int(x) for x in args.agent_counts.split(",")]

    for fam in families:
        d = os.path.join(MARLGPT, FAMILY_DIR[fam])
        maps = yaml.safe_load(open(os.path.join(d, "maps.yaml")))
        names = sample_map_names(maps, args.num_maps)
        for na in agent_counts:
            instances = []
            for nm in names:
                instances.append(make_instance(maps[nm], nm, na, args.seed))
            out_path = os.path.join(args.out, f"{fam}_{na}agents.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(instances, f)
            szs = [inst[0].shape for inst in instances]
            print(f"[{fam}] {na} agents: {len(instances)} instances -> {out_path} "
                  f"(map sizes {min(szs)}..{max(szs)})")
    print("done.")


if __name__ == "__main__":
    main()
