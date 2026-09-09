#!/usr/bin/env python3
"""
SQ4：持续分布漂移下框架是否保持性能？（路线图 Q4 / SCHEDULE.md §4-SQ4）
对应创新：**Replay + Meta Update**。

任务序列（每段固定训练预算，段末在**全部已见族**上评测）：
    random48 → maze48 → random64 → maze64          （密度与结构双重漂移）

四个臂（唯一变量 = 是否 replay / 是否 meta update）：
    madcl     : replay ✓ + meta update ✓   （本文完整方法）
    replay    : replay ✓ + meta update ✗
    naive     : replay ✗ + meta update ✗   （纯顺序微调，测灾难性遗忘）
    joint     : 同时训全族                  （multi-task 上界参照）

指标（标准持续学习口径）：
    final CSR  = 全部任务学完后在各族的平均 CSR
    forgetting = 平均 max_t(CSR_t) − CSR_final  （越低越好）
    BWT        = 后向迁移 = 平均 (CSR_final − CSR_学完该任务时)

replay 实现：保留一个**跨族**的经验回放（按族均匀采样已见数据），每步以 replay_ratio
的概率从旧族采 batch。meta update 实现：每段结束做一次 Reptile 式外层平均
θ ← θ_old + ε(θ_new − θ_old)，抑制单段过拟合导致的漂移。
"""
from __future__ import annotations
import sys, os, argparse, json, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, yaml

from buffers.expert_demo_buffer import ExpertDemoBuffer
from scripts.generate_lacam_data import load_buffer_from_disk
from scripts.train_il_meta import build_net
from trainers.train_il_reptile import ILReptileTrainer
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint

FAM_EVAL = {"random48": "random", "random64": "random",
            "maze48": "mazes", "maze64": "mazes"}
FAM_AGENTS = {"random48": "48", "random64": "64", "maze48": "48", "maze64": "64"}


def load_buffers(data_dir, fams):
    out = {}
    for f in fams:
        base = os.path.join(data_dir, f"expert_{f}")
        if not os.path.exists(os.path.join(base, "episode_ids.npy")):
            print(f"  [skip] {base} 不存在"); continue
        n = int(np.load(os.path.join(base, "episode_ids.npy"), mmap_mode="r").shape[0])
        b = ExpertDemoBuffer(capacity=n); load_buffer_from_disk(base, b); out[f] = b
    return out


def train_segment(tr, cfg, buffers, cur_fam, seen, steps, replay_ratio, rng):
    """在 cur_fam 上训 steps 步；replay_ratio>0 时按该比例从**已见旧族**采 batch。"""
    for i in range(steps):
        fam = cur_fam
        if replay_ratio > 0 and seen and rng.random() < replay_ratio:
            fam = seen[int(rng.integers(0, len(seen)))]
        try:
            batch = buffers[fam].sample(batch_size=cfg.get("batch_size", 32),
                                        sequence_len=cfg.get("sequence_len", 32),
                                        burn_in_len=0, rng=rng, need_next=False)
        except (ValueError, IndexError):
            continue
        tr._inner_step(batch)


def evaluate(ckpt, fams, nc_path, num_maps, tag_dir):
    """在给定族集合上评测，返回 {fam: CSR}。"""
    out = {}
    for f in fams:
        ev = os.path.join(tag_dir, f"eval_{os.path.basename(ckpt)[:-3]}_{f}.json")
        subprocess.run([sys.executable, "scripts/eval_bench.py", "--ckpt", ckpt,
                        "--network-config", nc_path, "--families", FAM_EVAL[f],
                        "--agent-counts", FAM_AGENTS[f], "--num-maps", str(num_maps),
                        "--batch-envs", "16", "--action-mode", "sample", "--out", ev],
                       check=True, stdout=subprocess.DEVNULL)
        r = json.load(open(ev))["results"][FAM_EVAL[f]]
        out[f] = float(list(r.values())[0]["CSR"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/chameleon_data/c2g_mix")
    ap.add_argument("--sequence", default="random48,maze48,random64,maze64")
    ap.add_argument("--arms", default="madcl,replay,naive,joint")
    ap.add_argument("--init", default="ckpt_il/il_smallwarm100k_iter_100000.pt")
    ap.add_argument("--steps-per-segment", type=int, default=2000)
    ap.add_argument("--replay-ratio", type=float, default=0.5)
    ap.add_argument("--meta-lr", type=float, default=0.5, help="段末 Reptile 式外层步长")
    ap.add_argument("--config", default="configs/train_il_v5.yaml")
    ap.add_argument("--network-config", default="configs/network_gru_c2g.yaml")
    ap.add_argument("--num-maps", type=int, default=64)
    ap.add_argument("--ckpt-dir", default="/root/autodl-tmp/sq4")
    ap.add_argument("--out", default="/root/autodl-tmp/eval_v5/sq4_continual.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)); nc = yaml.safe_load(open(args.network_config))
    seq = [f.strip() for f in args.sequence.split(",")]
    os.makedirs(args.ckpt_dir, exist_ok=True)
    buffers = load_buffers(args.data_dir, seq)
    seq = [f for f in seq if f in buffers]
    print(f"[SQ4] 任务序列: {' → '.join(seq)}", flush=True)
    device = torch.device(args.device)
    results = {}

    for arm in [a.strip() for a in args.arms.split(",")]:
        set_seed(0)
        net = build_net(nc)
        ck0 = torch.load(args.init, map_location="cpu")
        net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck0["net"].items()})
        tr = ILReptileTrainer(net, {**cfg, "reptile": False}, device=device)
        rng = np.random.default_rng(0)
        use_replay = arm in ("madcl", "replay")
        use_meta = arm == "madcl"
        curve, seen = [], []
        t0 = time.time()

        if arm == "joint":      # 上界臂：全族混训，总预算与顺序臂相同
            total = args.steps_per_segment * len(seq)
            for i in range(total):
                fam = seq[int(rng.integers(0, len(seq)))]
                try:
                    b = buffers[fam].sample(batch_size=cfg.get("batch_size", 32),
                                            sequence_len=cfg.get("sequence_len", 32),
                                            burn_in_len=0, rng=rng, need_next=False)
                except (ValueError, IndexError):
                    continue
                tr._inner_step(b)
            ck = os.path.join(args.ckpt_dir, f"{arm}_final.pt")
            save_checkpoint(tr.state_dict(), ck, "meta")
            final = evaluate(ck, seq, args.network_config, args.num_maps, args.ckpt_dir)
            results[arm] = {"final": final, "curve": [], "forgetting": 0.0, "bwt": 0.0}
            print(f"[SQ4] {arm}: final={final} ({time.time()-t0:.0f}s)", flush=True)
            continue

        for si, fam in enumerate(seq):
            theta_before = {k: v.detach().cpu().clone() for k, v in tr.net.state_dict().items()}
            train_segment(tr, cfg, buffers, fam, seen, args.steps_per_segment,
                          args.replay_ratio if use_replay else 0.0, rng)
            if use_meta:   # 段末 Reptile 式外层平均：θ ← θ_before + ε(θ_after − θ_before)
                with torch.no_grad():
                    for k, v in tr.net.state_dict().items():
                        tb = theta_before[k].to(v.device)
                        if v.dtype.is_floating_point:
                            v.copy_(tb + args.meta_lr * (v - tb))
            seen.append(fam)
            ck = os.path.join(args.ckpt_dir, f"{arm}_seg{si}.pt")
            save_checkpoint(tr.state_dict(), ck, "meta")
            scores = evaluate(ck, seen, args.network_config, args.num_maps, args.ckpt_dir)
            curve.append({"segment": si, "trained": fam, "scores": scores})
            print(f"[SQ4] {arm} seg{si}({fam}): {scores}", flush=True)

        final = curve[-1]["scores"]
        # forgetting = 平均 (历史最好 − 最终)；BWT = 平均 (最终 − 刚学完该任务时)
        best = {f: max(c["scores"].get(f, 0.0) for c in curve if f in c["scores"]) for f in seq}
        just = {c["trained"]: c["scores"][c["trained"]] for c in curve}
        forgetting = float(np.mean([best[f] - final.get(f, 0.0) for f in seq]))
        bwt = float(np.mean([final.get(f, 0.0) - just[f] for f in seq if f in just]))
        results[arm] = {"final": final, "curve": curve,
                        "forgetting": forgetting, "bwt": bwt,
                        "avg_final": float(np.mean(list(final.values())))}
        print(f"[SQ4] {arm}: avg_final={results[arm]['avg_final']:.3f} "
              f"forgetting={forgetting:.3f} BWT={bwt:.3f} ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
