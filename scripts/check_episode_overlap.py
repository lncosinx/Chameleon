#!/usr/bin/env python3
"""专家数据批次间 episode 重复检查。

用每个 episode 首帧的全体观测（地图 + 全体起点 + 目标 + cost2go 都编码在里面）做哈希作为
episode 身份，报告各批次两两重合的 episode 数。

背景（2026-09-17）：generate_lacam_data 的 seed 曾写死为 10000+ep，"gen2" 扩充批与原批
逐 episode 比对 random 重复 66%、maze 重复 89~98%，"数据 2.3×" 实际只多 1.31× 不重复 episode。
追加任何新批次前后都应跑一次本脚本。

用法：
  python scripts/check_episode_overlap.py DIR_A DIR_B [DIR_C ...]
  （DIR 为 expert_{fam}/ 目录，可以是符号链接）
"""
from __future__ import annotations
import sys, os, hashlib, itertools
import numpy as np


def episode_hashes(path: str) -> set:
    eid = np.asarray(np.load(os.path.join(path, "episode_ids.npy"), mmap_mode="r"))
    obs = np.load(os.path.join(path, "obs_patches.npy"), mmap_mode="r")
    starts = np.flatnonzero(np.r_[True, eid[1:] != eid[:-1]])
    return {hashlib.blake2b(np.ascontiguousarray(obs[s]).tobytes(), digest_size=12).digest()
            for s in starts}


def main():
    dirs = sys.argv[1:]
    if len(dirs) < 2:
        print(__doc__); return 2
    hs = {}
    for d in dirs:
        hs[d] = episode_hashes(d)
        print(f"  {len(hs[d]):>8,} ep  {d}")
    worst = 0.0
    for a, b in itertools.combinations(dirs, 2):
        ov = len(hs[a] & hs[b]); frac = ov / max(1, min(len(hs[a]), len(hs[b])))
        worst = max(worst, frac)
        print(f"  重合 {ov:>8,} ep（占较小批 {frac:6.1%}）  {os.path.basename(a.rstrip('/'))} ↔ {b}")
    total = len(set().union(*hs.values()))
    print(f"  不重复 episode 合计 {total:,}")
    return 1 if worst > 0.01 else 0


if __name__ == "__main__":
    sys.exit(main())
