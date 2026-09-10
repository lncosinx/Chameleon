#!/usr/bin/env python3
"""
从残留的 `.tmp_worker_XXXX_YYYY.npz` 分片抢救一次中断的数据生成。

用途：生成进程被杀/机器重启后，worker 已经落盘的分片仍然完好（每个分片是一次完整的
`np.savez`），只是没走到最后的合并步骤。重跑一遍要几小时，直接合并只要几分钟。

安全性：
  - 逐个分片试打开并读 `episode_ids`，**损坏/截断的分片自动跳过**（进程被杀时正在写的
    那一个通常是坏的），不会污染输出；
  - 分片按 worker 分组、组内按 part 序号排序后交给 `_merge_tmp_parts`，保持它对
    episode_id 偏移的假设（同一 worker 的 episode_id 跨 part 连续递增，偏移只在
    worker 的最后一个 part 之后推进）。

用法：
  python scripts/recover_tmp_parts.py --tmp-dir /root/autodl-tmp/chameleon_data/c2g_a64 \
      --output-dir /root/autodl-tmp/chameleon_data/c2g_a64/expert_maze [--delete-parts]
"""
from __future__ import annotations
import sys, os, argparse, glob, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from scripts.generate_lacam_data import _merge_tmp_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=10_000_000)
    ap.add_argument("--delete-parts", action="store_true", help="合并成功后删除分片")
    args = ap.parse_args()

    parts = sorted(glob.glob(os.path.join(args.tmp_dir, ".tmp_worker_*.npz")))
    if not parts:
        print("没有找到分片"); return
    print(f"发现 {len(parts)} 个分片，逐个校验…", flush=True)

    by_worker = collections.defaultdict(list)
    bad = []
    total_steps = 0
    for p in parts:
        m = re.search(r"\.tmp_worker_(\d+)_(\d+)\.npz$", p)
        w, k = int(m.group(1)), int(m.group(2))
        try:
            with np.load(p) as d:
                n = len(d["episode_ids"])
                _ = d["obs_patches"].shape        # 触发实际读取，截断文件会在这里炸
            by_worker[w].append((k, p))
            total_steps += n
        except Exception as e:  # noqa: BLE001
            bad.append((p, type(e).__name__))
    for p, e in bad:
        print(f"  [跳过损坏] {os.path.basename(p)} ({e})")
    print(f"可用分片 {sum(len(v) for v in by_worker.values())} 个 / {len(by_worker)} 个 worker，"
          f"合计 {total_steps:,} 时间步", flush=True)

    grouped = [[p for _, p in sorted(v)] for _, v in sorted(by_worker.items())]
    n = _merge_tmp_parts(grouped, args.output_dir, args.max_samples)
    print(f"合并完成：{n:,} 时间步 → {args.output_dir}")

    if args.delete_parts:
        # _merge_tmp_parts 成功后会自行删除它处理过的分片，这里只清理它没碰过的
        # （损坏分片）并容忍已不存在的文件。
        removed = 0
        for p in [q for g in grouped for q in g] + [q for q, _ in bad]:
            try:
                os.remove(p); removed += 1
            except FileNotFoundError:
                pass
        print(f"清理残留分片 {removed} 个（其余已由合并步骤删除）")


if __name__ == "__main__":
    main()
