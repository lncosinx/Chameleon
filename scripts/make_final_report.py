#!/usr/bin/env python3
"""
把四个 SQ 的所有结果汇总成一份论文可用的 markdown 报告。

输入 = 各脚本落盘的 json（eval_bench.py / run_mapfgpt_bench.py / run_sq2_adaptation.py /
run_sq4_continual.py）+ pogema-benchmark leaderboard 原始数据。输出 = 一份报告，含：
  - SQ1 主表（三类方法分开：天花板规划器 / 去中心化 PO 真对手 / foundation model）
  - SQ1 判决行（对每个真对手是否全档 ≥）
  - SQ3 通信消融表 + 通信量-EdgeFlip 列
  - SQ2 适应曲线表 / SQ4 持续学习表（有结果才出）

设计原则：**缺的结果就写"未完成"，绝不留空表或用占位数字**——这份报告是要直接进论文的，
任何一个看不出来源的数字都是隐患。

用法：
  python scripts/make_final_report.py --out /root/autodl-tmp/report/RESULTS.md
"""
from __future__ import annotations
import sys, os, json, glob, argparse, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
LB = os.path.join(ROOT, "pogema-benchmark", "raw_data_MAPF")
EV = "/root/autodl-tmp/eval_v5"
FAM_DIR = {"random": "01-random", "mazes": "02-mazes",
           "warehouse": "03-warehouse", "movingai": "04-movingai"}
CEILING = {"LaCAM", "CBS", "RHCR"}
AGENTS_BY_FAM = {"random": [8, 16, 24, 32, 48, 64], "mazes": [8, 16, 24, 32, 48, 64],
                 "warehouse": [32, 64, 96, 128, 160, 192], "movingai": [64, 128, 192, 256]}


def leaderboard(fam, metric="CSR"):
    out = {}
    for f in sorted(glob.glob(os.path.join(LB, FAM_DIR[fam], "*.json"))):
        alg = os.path.basename(f)[:-5]
        agg = collections.defaultdict(list)
        for r in json.load(open(f)):
            agg[r["env_grid_search"]["num_agents"]].append(r["metrics"][metric])
        out[alg] = {k: sum(v) / len(v) for k, v in agg.items()}
    return out


def load(path, fam, metric="CSR"):
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    res = j.get("results", {}).get(fam)
    if not res:
        return None
    return {int(k): v.get(metric) for k, v in res.items() if v.get(metric) is not None}


def table(fam, rows, agents, metric):
    out = [f"\n**{fam} — {metric}**\n",
           "| 方法 | 类别 | " + " | ".join(str(a) for a in agents) + " |",
           "|" + "---|" * (len(agents) + 2)]
    for name, kind, vals in rows:
        cells = ["—" if vals.get(a) is None else f"{vals[a]:.3f}" for a in agents]
        out.append(f"| {name} | {kind} | " + " | ".join(cells) + " |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=None, help="主模型结果 json；不给则自动挑最新的 v5final/v5main")
    ap.add_argument("--label", default="MADCL")
    ap.add_argument("--metric", default="CSR")
    ap.add_argument("--out", default="/root/autodl-tmp/report/RESULTS.md")
    args = ap.parse_args()

    ours_path = args.ours
    if ours_path is None:
        for cand in ["v5final_15000.json", "v5final_5000.json", "v5main_40000.json",
                     "v5main_20k.json", "v5main_10k_early.json"]:
            if os.path.exists(os.path.join(EV, cand)):
                ours_path = os.path.join(EV, cand); break
    L = ["# MADCL 实验结果汇总", "",
         f"主模型结果文件：`{ours_path}`", "",
         "协议：POGEMA，`on_target=nothing`，`max_episode_steps=128`，validation-* 全 128 图，",
         "`collision_system=soft`，seed 与 leaderboard 逐实例一致。CSR/ISR 口径已与 pogema",
         "自带 `info['metrics']` 逐 episode 对拍一致。", ""]

    # ---------------- SQ1 ----------------
    L += ["## SQ1 整体性能", ""]
    for fam in ["random", "mazes"]:
        agents = AGENTS_BY_FAM[fam]
        lb = leaderboard(fam, args.metric)
        rows = [(n, "天花板(集中式满信息)", lb[n]) for n in sorted(lb) if n in CEILING]
        mine = load(ours_path, fam, args.metric) if ours_path else None
        if mine:
            rows.append((f"**{args.label}（本文）**", "去中心化 PO", mine))
        rows += [(n, "去中心化 PO", lb[n]) for n in sorted(lb) if n not in CEILING]
        for w in ["MAPF-GPT-2M", "MAPF-GPT-DDG-2M", "MAPF-GPT-85M"]:
            v = load(os.path.join(EV, f"gpt_{w}.json"), fam, args.metric)
            if v:
                rows.append((w, "foundation model(本地复现)", v))
        L += table(fam, rows, agents, args.metric)

        if mine:
            L += ["", "**判决（对每个真对手是否全档 ≥）**：", ""]
            for name, kind, vals in rows:
                if kind.startswith("天花板") or name.startswith("**"):
                    continue
                cmp_ = [(a, mine.get(a), vals.get(a)) for a in agents
                        if mine.get(a) is not None and vals.get(a) is not None]
                if not cmp_:
                    continue
                win = sum(1 for _, m, o in cmp_ if m >= o - 1e-9)
                mark = "✅ 全档胜" if win == len(cmp_) else f"❌ {win}/{len(cmp_)} 档胜"
                L.append(f"- vs **{name}**（{kind}）：{mark}")
        L.append("")

    # ---------------- 缺席基线的交代 ----------------
    L += ["", "### 路线图点名但无法同协议对比的基线", "",
          "| 方法 | 状态 | 说明 |", "|---|---|---|",
          "| PRIMAL (ICRA'19) | ⚠️ 无 POGEMA 协议数字 | 不在 pogema-benchmark leaderboard 内；"
          "原论文用自己的环境/协议（含不同的碰撞与到达语义），直接搬数字不可比。|",
          "| PRIMAL2 (RA-L'21) | ⚠️ 同上 | 同上。|",
          "| ALPHA (2025) | ⚠️ 无公开权重/同协议数字 | 需原作者权重才能同环境复现。|",
          "| CBS | ⚠️ 未跑 | 集中式最优规划器，属天花板类；LaCAM 已代表该类（random/mazes 全档 1.000）。|",
          "",
          "**可用的替代论证**：SCRIMP 原论文报告其在同一族任务上全面优于 PRIMAL/PRIMAL2；"
          "本文与 SCRIMP **同协议、同地图、同实例**直接对比，故对 SCRIMP 的结论可传递到 "
          "PRIMAL/PRIMAL2。若审稿要求显式行，需把 PRIMAL2 移植进 POGEMA（独立工作量，未做）。",
          "**不要**把这些方法原论文里的数字直接填进主表——协议不同，那是错误对比。", ""]

    # ---------------- SQ3 ----------------
    L += ["", "## SQ3 通信机制消融（头条）", ""]
    arms = [("madcl", "本文：entmax 软稀疏 + 迟滞"), ("nocomm", "无通信"),
            ("softmax", "dense softmax（无稀疏）"), ("nohyst", "无迟滞"),
            ("hardtopk", "hard top-k 做选择"), ("globalcomm", "全局通信（无 FOV 限制）"),
            ("commnet", "CommNet'16"), ("ic3net", "IC3Net'19"), ("tarmac", "TarMAC'19"),
            ("dial", "DIAL'16"), ("atoc", "ATOC'18"), ("dgn", "DGN'20"),
            ("it_feat", "消息=观测特征"), ("it_act", "消息=显式动作意图")]
    found = False
    for fam in ["random", "mazes"]:
        rows = []
        for tag, desc in arms:
            v = load(os.path.join(EV, f"sq3_{tag}.json"), fam, args.metric) \
                or load(os.path.join(EV, f"{tag}.json"), fam, args.metric)
            if v:
                rows.append((desc, "消融臂", v)); found = True
        if rows:
            L += table(fam, rows, [32, 48, 64], args.metric)
    if not found:
        L += ["_SQ3 消融尚未完成（训练队列中）。_", ""]

    # 通信量 / 稳定性
    comm_rows = []
    for tag, desc in arms:
        p = os.path.join(EV, f"sq3_{tag}.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p)).get("results", {}).get("mazes", {})
        if "64" in j:
            r = j["64"]
            comm_rows.append((desc, r.get("avg_neighbors"), r.get("edge_flip"), r.get(args.metric)))
    if comm_rows:
        L += ["", "**通信量 / 稳定性（mazes@64）**", "",
              "| 臂 | 平均通信邻居数 | EdgeFlipRate | CSR |", "|---|---|---|---|"]
        for d, n, f_, c in comm_rows:
            L.append(f"| {d} | {n:.2f} | {f_:.3f} | {c:.3f} |")

    # ---------------- SQ2 / SQ4 ----------------
    for sq, path, title in [("SQ2", "/root/autodl-tmp/eval_v5/sq2_adaptation.json",
                             "## SQ2 元学习的快速适应"),
                            ("SQ4", "/root/autodl-tmp/eval_v5/sq4_continual.json",
                             "## SQ4 持续分布漂移")]:
        L += ["", title, ""]
        if os.path.exists(path):
            j = json.load(open(path))
            L += ["```json", json.dumps(j, indent=2, ensure_ascii=False)[:4000], "```"]
        else:
            L += [f"_{sq} 尚未完成。_"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:60]))
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
