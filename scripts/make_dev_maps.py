#!/usr/bin/env python3
"""
生成**独立 dev 地图集**，用于调推断期超参（温度、卡住阈值等）。

为什么必须有这个：benchmark 的 validation-* 是**报告用的测试集**。在它上面挑温度、挑
escape_k，挑完再报同一批地图的数字，就是在拟合测试集——审稿人有理由不认。这里用与训练
**同一套在线生成器**（20×20 random / 21×21 maze）另造一批地图，seed 段与训练/评测都不重叠，
调参只在这上面做，最终只报 validation-* 的数字。

输出格式与 marl-gpt/eval_configs/*/maps.yaml 完全一致（name -> 地图字符串），
可直接被 scripts/eval_bench.py 的 --maps-file 读取。

用法：
  python scripts/make_dev_maps.py --family random --num 64 --out /root/autodl-tmp/dev_maps/random.yaml
"""
from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

from envs.pogema_env_wrapper import _generate_maze_map, _generate_warehouse_map

# dev seed 段：与 envs/dataset_split.py 的 train/test 段都不重叠（那两段用的是
# resolve_seed 的哈希映射，这里直接取一段远离 0 的大 seed，且只在本脚本内使用）。
DEV_SEED_BASE = 7_000_000


def gen_random_map(size: int, density: float, seed: int) -> str:
    import numpy as np
    rng = np.random.default_rng(seed)
    grid = (rng.random((size, size)) < density)
    rows = ["".join("#" if c else "." for c in row) for row in grid]
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["random", "mazes", "warehouse"])
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg_path = {"random": "configs/env_random_small.yaml",
                "mazes": "configs/env_maze_small.yaml",
                "warehouse": "configs/env_warehouse_bench.yaml"}[args.family]
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), cfg_path)))

    maps = {}
    for i in range(args.num):
        sd = DEV_SEED_BASE + i
        if args.family == "random":
            m = gen_random_map(int(cfg.get("map_size", 20)),
                               float(cfg.get("obstacle_density", 0.2)), sd)
        elif args.family == "mazes":
            m = _generate_maze_map(cfg, seed=sd)
        else:
            m = _generate_warehouse_map(cfg, seed=sd)
        maps[f"dev-{args.family}-seed-{i:03d}"] = m

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    yaml.safe_dump(maps, open(args.out, "w"))
    rows = list(maps.values())[0].splitlines()
    occ = sum(r.count("#") for r in rows) / (len(rows) * len(rows[0]))
    print(f"{len(maps)} 张 dev 地图 → {args.out}  "
          f"(首张 {len(rows)}x{len(rows[0])}，障碍率 {occ:.2f})")


if __name__ == "__main__":
    main()
