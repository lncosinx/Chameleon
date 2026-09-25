#!/usr/bin/env python3
"""
SQ2：元学习是否带来跨场景快速适应？（对应路线图 Q2 / SCHEDULE.md §4-SQ2）

**设计要点**：v4 已实测 Reptile 放进预训练会拖累峰值 CSR（joint 完胜），所以本文不把
Reptile 放预训练，只放**适应阶段**。本实验回答的是："同样的适应预算下，Reptile 学到的
初始化是不是更容易被少量新场景数据掰过来。"

  - held-out 族（默认 warehouse）**完全不进**预训练；
  - 两个初始化：Reptile-init vs joint-init（预训练预算、数据、超参完全相同，唯一变量=外层是否 Reptile）；
  - 适应：在 held-out 族的少量专家数据上跑 k ∈ {0,10,50,200} 步 SGD；
  - 每个 k 都在 held-out 族的评测集上测 CSR，画适应曲线。

用法：
  python scripts/run_sq2_adaptation.py --init-reptile ckpt_il/v5reptile_iter_40000.pt \
      --init-joint ckpt_il/v5d4_iter_40000.pt --family warehouse \
      --data-dir /root/autodl-tmp/chameleon_data/c2g_wh --steps 0,10,50,200
"""
from __future__ import annotations
import sys, os, argparse, json, copy, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, yaml

from buffers.expert_demo_buffer import ExpertDemoBuffer
from scripts.generate_lacam_data import load_buffer_from_disk
from scripts.train_il_meta import build_net
from trainers.train_il_reptile import ILReptileTrainer
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint


def adapt(init_ckpt, buf, cfg, nc, steps, device, seed=0, out=None):
    """从 init_ckpt 出发在 held-out 族数据上跑 `steps` 步 SGD，返回适应后的 ckpt 路径。"""
    set_seed(seed)
    net = build_net(nc)
    ck = torch.load(init_ckpt, map_location="cpu")
    net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["net"].items()})
    tr = ILReptileTrainer(net, {**cfg, "reptile": False}, device=device)
    rng = np.random.default_rng(seed)
    for i in range(steps):
        try:
            batch = buf.sample(batch_size=cfg.get("batch_size", 32),
                               sequence_len=cfg.get("sequence_len", 32),
                               burn_in_len=0, rng=rng, need_next=False)
        except (ValueError, IndexError):
            continue
        tr._inner_step(batch)
    save_checkpoint(tr.state_dict(), out, "meta")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-reptile", required=True)
    ap.add_argument("--init-joint", required=True)
    ap.add_argument("--family", default="warehouse", help="held-out 族（评测台家族名）")
    ap.add_argument("--data-family", default=None,
                    help="适应数据的 expert_{name} 名；**可逗号分隔多个**（默认同 --family）。"
                         "★ 2026-09-23：原先只给 wh96 单一体数，而评测覆盖 32~192 六档，"
                         "适应把策略拉向 96 体行为、高密度档因此退化——那是适应集与评测分布"
                         "不匹配，不能据此说'快速适应无效'。分层适应集用 wh96,wh192。")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", default="configs/train_il_v5.yaml")
    ap.add_argument("--network-config", default="configs/network_gru_c2g.yaml")
    ap.add_argument("--steps", default="0,10,50,200")
    ap.add_argument("--agent-counts", default="16,32,48,64")
    ap.add_argument("--num-maps", type=int, default=128)
    ap.add_argument("--no-auto-protocol", dest="auto_protocol", action="store_false",
                    default=True, help="不套用各族官方协议（调试用；出表必须用官方协议）")
    ap.add_argument("--ckpt-dir", default="/root/autodl-tmp/sq2")
    ap.add_argument("--out", default="/root/autodl-tmp/eval_v5/sq2_adaptation.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    nc = yaml.safe_load(open(args.network_config))
    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    fam_data = args.data_family or args.family
    fams_data = [f.strip() for f in fam_data.split(",") if f.strip()]
    base = os.path.join(args.data_dir, f"expert_{fams_data[0]}")
    n = int(np.load(os.path.join(base, "episode_ids.npy"), mmap_mode="r").shape[0])
    buf = ExpertDemoBuffer(capacity=n)
    load_buffer_from_disk(base, buf)
    if len(fams_data) > 1:
        # 分层适应集：每次 sample 随机选一个体数档，使适应分布覆盖评测的高低密度两端。
        class _MixBuffer:
            def __init__(self, bufs): self.bufs = bufs
            def __len__(self): return sum(len(b) for b in self.bufs)
            def sample(self, **kw):
                rng = kw.get("rng")
                i = int(rng.integers(0, len(self.bufs))) if rng is not None else 0
                return self.bufs[i].sample(**kw)
        extra = []
        for f in fams_data[1:]:
            b2 = os.path.join(args.data_dir, f"expert_{f}")
            n2 = int(np.load(os.path.join(b2, "episode_ids.npy"), mmap_mode="r").shape[0])
            bb = ExpertDemoBuffer(capacity=n2); load_buffer_from_disk(b2, bb); extra.append(bb)
            print(f"[SQ2] 追加适应数据 {len(bb):,} steps ← {b2}", flush=True)
        buf = _MixBuffer([buf] + extra)
    print(f"[SQ2] 适应数据 {len(buf):,} steps ← {base}", flush=True)

    results = {}
    for arm, init in [("reptile-init", args.init_reptile), ("joint-init", args.init_joint)]:
        results[arm] = {}
        for k in [int(x) for x in args.steps.split(",")]:
            ck = os.path.join(args.ckpt_dir, f"{arm}_k{k}.pt")
            t0 = time.time()
            adapt(init, buf, cfg, nc, k, device, seed=0, out=ck)
            ev = os.path.join(args.ckpt_dir, f"eval_{arm}_k{k}.json")
            # ★ --auto-protocol：各族的官方协议并不相同（warehouse 是 **1 张图 × 128 个 seed**，
            # 不是 128 张图；agent 档位也是 32..192 而非 8..64）。不传这个开关会退化成
            # "1 张图 × 1 个 seed"，n=1 的 CSR 只是噪声，且与 leaderboard 不可比。
            cmd = [sys.executable, "scripts/eval_bench.py", "--ckpt", ck,
                   "--network-config", args.network_config, "--families", args.family,
                   "--batch-envs", "16", "--action-mode", "sample", "--out", ev]
            cmd += ["--auto-protocol"] if args.auto_protocol else [
                "--agent-counts", args.agent_counts, "--num-maps", str(args.num_maps)]
            subprocess.run(cmd, check=True)
            r = json.load(open(ev))["results"][args.family]
            results[arm][k] = {a: v["CSR"] for a, v in r.items()}
            print(f"[SQ2] {arm} k={k}: " +
                  " ".join(f"{a}:{v['CSR']:.3f}" for a, v in r.items()) +
                  f"  ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
