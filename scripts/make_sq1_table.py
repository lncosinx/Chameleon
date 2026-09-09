#!/usr/bin/env python3
"""
SQ1 主表生成：把我们 eval_bench.py 的结果与 pogema-benchmark leaderboard 的官方数字
合成一张同协议对照表（CSR，可选 ISR/SoC）。

引用合法性（三条都已核对，见 SCHEDULE.md §4）：
  1. 地图逐字节相同：pogema-benchmark/raw_data_MAPF/{01-random,02-mazes}/maps.yaml
     与 marl-gpt/eval_configs 下同名文件 128/128 完全一致；
  2. 协议相同：on_target=nothing、max_episode_steps=128、collision_system=soft、
     agents ∈ {8,16,24,32,48,64}、seed=0；
  3. 实例相同：eval_bench.py 用 split="all"（不重映射 seed），起终点布局与 leaderboard 一致。

用法：
  python scripts/make_sq1_table.py --ours /root/autodl-tmp/eval_v5/v5_sample.json \
      --label MADCL --metric CSR
"""
from __future__ import annotations
import sys, os, json, glob, argparse, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LB = os.path.join(ROOT, "pogema-benchmark", "raw_data_MAPF")
FAM_DIR = {"random": "01-random", "mazes": "02-mazes",
           "warehouse": "03-warehouse", "movingai": "04-movingai"}
# 分类：规划器是天花板（不作对手），去中心化 PO 学习方法才是真对手
CEILING = {"LaCAM", "CBS", "RHCR"}
AGENTS = [8, 16, 24, 32, 48, 64]


def leaderboard(fam: str, metric: str) -> dict:
    out = {}
    d = os.path.join(LB, FAM_DIR[fam])
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        alg = os.path.basename(f)[:-5]
        agg = collections.defaultdict(list)
        for r in json.load(open(f)):
            agg[r["env_grid_search"]["num_agents"]].append(r["metrics"][metric])
        out[alg] = {k: sum(v) / len(v) for k, v in agg.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", action="append", default=[], metavar="PATH[:LABEL]",
                    help="eval_bench.py 的输出 json，可多次；可用 path:LABEL 指定显示名")
    ap.add_argument("--metric", default="CSR", choices=["CSR", "ISR", "SoC", "makespan"])
    ap.add_argument("--families", default="random,mazes")
    ap.add_argument("--out", default=None, help="额外写出 markdown 文件")
    args = ap.parse_args()

    lines = []
    for fam in [f.strip() for f in args.families.split(",") if f.strip()]:
        lb = leaderboard(fam, args.metric)
        ours = {}
        for spec in args.ours:
            path, _, label = spec.partition(":")
            if not os.path.exists(path):
                print(f"[warn] 缺 {path}，跳过"); continue
            j = json.load(open(path))
            label = label or j.get("tag") or os.path.basename(path)[:-5]
            res = j["results"].get(fam, {})
            ours[label] = {int(k): v[args.metric] for k, v in res.items()}

        lines.append(f"\n### {fam} — {args.metric}（POGEMA benchmark 协议：nothing / 128 步 / "
                     f"128 图 / 同实例）\n")
        hdr = "| 方法 | 类别 | " + " | ".join(str(a) for a in AGENTS) + " |"
        lines.append(hdr)
        lines.append("|" + "---|" * (len(AGENTS) + 2))

        def row(name, vals, kind):
            cells = []
            for a in AGENTS:
                v = vals.get(a)
                cells.append("—" if v is None else f"{v:.3f}")
            return f"| {name} | {kind} | " + " | ".join(cells) + " |"

        for name in sorted(lb):
            if name in CEILING:
                lines.append(row(name, lb[name], "天花板(集中式满信息)"))
        for label, vals in ours.items():
            lines.append(row(f"**{label}（本文）**", vals, "去中心化 PO"))
        for name in sorted(lb):
            if name not in CEILING:
                lines.append(row(name, lb[name], "去中心化 PO"))

        # 判决：对每个"真对手"，我们是否每档都 ≥
        if ours:
            best_label = list(ours)[-1]
            mine = ours[best_label]
            lines.append("")
            for name in sorted(lb):
                if name in CEILING:
                    continue
                wins = sum(1 for a in AGENTS
                           if a in mine and a in lb[name] and mine[a] >= lb[name][a] - 1e-9)
                tot = sum(1 for a in AGENTS if a in mine and a in lb[name])
                mark = "✅ 全档胜" if wins == tot else f"❌ {wins}/{tot} 档胜"
                lines.append(f"- vs **{name}**: {mark}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w").write(text + "\n")
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
