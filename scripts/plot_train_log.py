#!/usr/bin/env python3
"""解析 train_full_0611_compile.log 训练日志并作图，用于汇报。

用法:
    python scripts/plot_train_log.py [logfile]
默认解析 logs/train_full_0611_compile.log，输出 PNG 到 logs/figs/。
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/train_full_0611_compile.log")
OUTDIR = LOG.parent / "figs"
OUTDIR.mkdir(exist_ok=True)

# 形如:
# iter  18300/20000: loss=0.2357  Qmax=4.64  Qmean=3.12  r=0.13/5.00
#   [td=0.03 n=0.03 mgn=0.33 l2=0.03]  map=... er=0.50  sp=512  3.45s/it
#   tasks=4  meta_lr=0.00883
ITER_RE = re.compile(
    r"iter\s+(\d+)/(\d+):\s+loss=([-\d.]+)\s+Qmax=([-\d.]+)\s+Qmean=([-\d.]+)\s+"
    r"r=([-\d.]+)/([-\d.]+)\s+\[td=([-\d.]+)\s+n=([-\d.]+)\s+mgn=([-\d.]+)\s+l2=([-\d.]+)\]"
    r".*?er=([-\d.]+)\s+sp=(\d+)\s+([\d.]+)s/it.*?meta_lr=([-\d.]+)"
)
PHASE_RE = re.compile(r"\[phase\]\s+(.+?)\s+@ iter\s+(\d+)")

rows = []
phases = []
for line in LOG.read_text().splitlines():
    m = ITER_RE.search(line)
    if m:
        g = m.groups()
        rows.append(
            dict(
                it=int(g[0]),
                loss=float(g[2]),
                qmax=float(g[3]),
                qmean=float(g[4]),
                r=float(g[5]),
                td=float(g[7]),
                n=float(g[8]),
                mgn=float(g[9]),
                l2=float(g[10]),
                er=float(g[11]),
                sp=int(g[12]),
                spit=float(g[13]),
                mlr=float(g[14]),
            )
        )
        continue
    p = PHASE_RE.search(line)
    if p:
        phases.append((p.group(1), int(p.group(2))))

assert rows, "未解析到任何 iter 行"
it = np.array([r["it"] for r in rows])
total = int(re.search(r"/(\d+):", [l for l in LOG.read_text().splitlines() if "iter" in l][1]).group(1)) if False else 20000


def col(k):
    return np.array([r[k] for r in rows], float)


def mark_phases(ax):
    for name, x in phases:
        ax.axvline(x, color="gray", ls="--", lw=0.8, alpha=0.7)
        ax.text(x, ax.get_ylim()[1], name.split()[0], rotation=90,
                va="top", ha="right", fontsize=7, color="gray")


# ---- 图1: 总览 3x3，覆盖日志全部指标 ----
fig, axes = plt.subplots(3, 3, figsize=(16, 11))

ax = axes[0, 0]
ax.plot(it, col("loss"), color="C0")
ax.set_title("Total Loss"); ax.set_xlabel("iter"); ax.set_ylabel("loss")
mark_phases(ax)

ax = axes[0, 1]
ax.plot(it, col("qmax"), label="Qmax", color="C1")
ax.plot(it, col("qmean"), label="Qmean", color="C2")
ax.set_title("Q values"); ax.set_xlabel("iter"); ax.legend()
mark_phases(ax)

ax = axes[0, 2]
for k, c in [("td", "C3"), ("n", "C4"), ("mgn", "C5"), ("l2", "C6")]:
    ax.plot(it, col(k), label=k, color=c)
ax.set_title("Loss components"); ax.set_xlabel("iter"); ax.legend()
mark_phases(ax)

ax = axes[1, 0]
ax.plot(it, col("er"), color="C7")
ax.set_title("Expert ratio (er)"); ax.set_xlabel("iter"); ax.set_ylabel("er")
mark_phases(ax)

ax = axes[1, 1]
ax.plot(it, col("spit"), color="C8")
ax.set_title("Throughput (s/iter)"); ax.set_xlabel("iter"); ax.set_ylabel("s/it")
mark_phases(ax)

ax = axes[1, 2]
ax.plot(it, col("mlr"), color="C9")
ax.set_title("Meta LR"); ax.set_xlabel("iter"); ax.set_ylabel("meta_lr")
mark_phases(ax)

ax = axes[2, 0]
ax.plot(it, col("r"), color="C0")
ax.set_title("Reward (r)"); ax.set_xlabel("iter"); ax.set_ylabel("reward")
mark_phases(ax)

ax = axes[2, 1]
ax.plot(it, col("sp"), color="C1")
ax.set_title("SelfPlay buffer (sp)"); ax.set_xlabel("iter"); ax.set_ylabel("segs")
mark_phases(ax)

axes[2, 2].axis("off")  # 占位

fig.suptitle(f"Training overview — {LOG.name}", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
f1 = OUTDIR / "train_overview.png"
fig.savefig(f1, dpi=130)

# ---- 图2: Loss 分量堆叠（占比演化）----
fig2, ax = plt.subplots(figsize=(12, 5))
comps = ["td", "n", "mgn", "l2"]
ys = [col(k) for k in comps]
ax.stackplot(it, *ys, labels=comps, alpha=0.85)
ax.set_title("Loss composition (stacked)"); ax.set_xlabel("iter"); ax.set_ylabel("loss")
ax.legend(loc="upper right")
mark_phases(ax)
fig2.tight_layout()
f2 = OUTDIR / "train_loss_stack.png"
fig2.savefig(f2, dpi=130)

# ---- 控制台摘要 ----
print(f"解析 {len(rows)} 个采样点, 范围 iter {it.min()}–{it.max()}")
print("阶段切换:", phases)
print(f"  loss: {col('loss')[0]:.3f} -> {col('loss')[-1]:.3f}")
print(f"  Qmean: {col('qmean')[0]:.2f} -> {col('qmean')[-1]:.2f} | Qmax 末值 {col('qmax')[-1]:.2f}")
print(f"  expert_ratio: {col('er')[0]:.2f} -> {col('er')[-1]:.2f}")
print(f"  平均吞吐 BC段 {col('spit')[it<1000].mean():.2f}s/it | DQfD段 {col('spit')[it>=1000].mean():.2f}s/it")
tot_h = (col('spit').mean() * 20000) / 3600
print(f"  估算总训练耗时 ≈ {tot_h:.1f} h (按平均 s/it × 20000)")
print(f"图已保存:\n  {f1}\n  {f2}")
