"""绘制 comm_sweep 训练曲线对比（comm_rounds = 0/1/2/3）。

解析 logs/comm_sweep/comm{0,1,2,3}.log 中的 iter 行，作图对比通信轮数对训练的影响。
每行格式示例：
  iter   2020/20000: loss=0.5677  Qmax=1.06  Qmean=0.00  r=0.10/5.00  \
      [td=0.08 n=0.17 mgn=0.35 l2=0.03]  map=...  er=0.90  tasks=4  meta_lr=0.04546
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = Path("logs/comm_sweep")
OUT = LOG_DIR / "comm_sweep_curves.png"
BC_DQFD_BOUNDARY = 2000  # BC → DQfD 相变点
ROUNDS = (0, 1, 2, 3)    # 4 个模型：通信轮数 0/1/2/3
EMA_WEIGHT = 0.9         # TensorBoard 式指数滑动平均权重（越大越平滑）

ITER_RE = re.compile(
    r"iter\s+(\d+)/\d+:\s+loss=([\d.]+)\s+Qmax=([\d.-]+)\s+Qmean=([\d.-]+)\s+"
    r"r=([\d.-]+)/[\d.]+\s+\[td=([\d.-]+)\s+n=([\d.-]+)\s+mgn=([\d.-]+)\s+l2=([\d.-]+)\]"
    r".*?er=([\d.]+)"
)


def ema(values, weight=EMA_WEIGHT):
    """TensorBoard 式指数滑动平均（含 debias 校正，避免起点被拉向 0）。"""
    smoothed = []
    last = 0.0
    debias = 0.0
    for v in values:
        last = weight * last + (1 - weight) * v
        debias = weight * debias + (1 - weight)
        smoothed.append(last / debias if debias > 0 else v)
    return smoothed


def parse(path):
    cols = {k: [] for k in
            ("iter", "loss", "Qmax", "Qmean", "r", "td", "n", "mgn", "l2", "er")}
    for line in path.read_text().splitlines():
        m = ITER_RE.search(line)
        if not m:
            continue
        vals = m.groups()
        for k, v in zip(cols, vals):
            cols[k].append(float(v))
    cols["iter"] = [int(x) for x in cols["iter"]]
    return cols


def main():
    runs = {}
    for r in ROUNDS:
        p = LOG_DIR / f"comm{r}.log"
        runs[r] = parse(p)
        print(f"comm{r}: {len(runs[r]['iter'])} 个数据点, "
              f"iter {runs[r]['iter'][0]}..{runs[r]['iter'][-1]}")

    colors = {0: "#888888", 1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}
    labels = {0: "comm_rounds=0 (no-comm baseline)",
              1: "comm_rounds=1",
              2: "comm_rounds=2",
              3: "comm_rounds=3"}

    panels = [
        ("loss", "Total loss", "log"),
        ("mgn", "Margin loss (BC main term)", "linear"),
        ("td", "TD loss (DQfD)", "linear"),
        ("n", "n-step loss (DQfD)", "linear"),
        ("Qmax", "Qmax", "linear"),
        ("Qmean", "Qmean", "linear"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()
    for ax, (key, title, yscale) in zip(axes, panels):
        for r in ROUNDS:
            x = runs[r]["iter"]
            # 生数据薄背景 + EMA 平滑前景
            ax.plot(x, runs[r][key], color=colors[r], lw=0.6, alpha=0.18)
            ax.plot(x, ema(runs[r][key]),
                    color=colors[r], lw=1.6, alpha=0.95, label=labels[r])
        ax.axvline(BC_DQFD_BOUNDARY, color="green", ls="--", lw=1, alpha=0.6)
        ax.text(BC_DQFD_BOUNDARY, ax.get_ylim()[1], " BC->DQfD",
                color="green", fontsize=8, va="top")
        ax.set_title(title)
        ax.set_xlabel("meta-iter")
        ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("comm_sweep training curves (comm_rounds = 0 / 1 / 2 / 3)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, dpi=130)
    print(f"已保存: {OUT}")


if __name__ == "__main__":
    main()
