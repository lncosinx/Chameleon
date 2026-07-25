#!/usr/bin/env python3
"""通信层数消融 · 分族评估：对 comm_rounds=0/1/2 的 checkpoint 在 when-to-switch
每个地图族 (random/mazes/sc1/street/wc3) 的 held-out test split 上分别评估，
打印 "族 × 通信轮数" 透视表（AR / SR / Deadlock 等）。

与 scripts/eval_comm_sweep.py 的区别：后者只在混合全池上给一个聚合数，本脚本**细分每族**。

默认 --device cpu（训练占用 GPU 时用），--threads 限制 CPU 线程避免抢占 env-worker。

用法:
  python scripts/eval_comm_family.py --rounds 0,1,2 --iter 20000 \
      --num-agents 64 --episodes 50 --max-steps 256 --device cpu --threads 16
"""
import sys, os, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from utils.seed import set_seed
from scripts.train_meta import _build_network
from trainers.evaluator import evaluate

# 评估器汇总后键名带 _mean/_std 后缀（aggregate_episode_metrics）
KEYS = ["SR_mean", "AR_mean", "Deadlock_mean", "NoProgressRate_mean",
        "WaitRate_mean", "Makespan_mean", "SoC_mean", "ArrivalSteps_mean_mean"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", default="0,1,2", help="逗号分隔的 comm_rounds 列表")
    p.add_argument("--iter", type=int, default=20000)
    p.add_argument("--ckpt-tpl", default="checkpoints/comm{r}_iter_{it}.pt")
    p.add_argument("--families", default="random,mazes,sc1,street,wc3")
    p.add_argument("--config-tpl", default="configs/env_{fam}.yaml")
    p.add_argument("--network-config", default="configs/network_biintent_drqn.yaml")
    p.add_argument("--num-agents", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=16, help="torch CPU 线程上限（别抢 env-worker）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="logs/eval/comm_family_eval.json")
    args = p.parse_args()

    set_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    rounds = [int(x) for x in args.rounds.split(",") if x.strip() != ""]
    families = [f.strip() for f in args.families.split(",") if f.strip() != ""]
    net_cfg = yaml.safe_load(open(args.network_config))

    # results[r][fam] = metrics dict
    results = {r: {} for r in rounds}
    for r in rounds:
        ckpt_path = args.ckpt_tpl.format(r=r, it=args.iter)
        if not os.path.exists(ckpt_path):
            print(f"[skip] comm_rounds={r}: {ckpt_path} 不存在", flush=True); continue
        nc = dict(net_cfg); nc["comm_rounds"] = r
        net = _build_network(nc, use_torch_compile=False)
        ckpt = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ckpt["net"])
        net.to(device)
        for fam in families:
            cfg = args.config_tpl.format(fam=fam)
            if not os.path.exists(cfg):
                print(f"[skip] family={fam}: {cfg} 不存在", flush=True); continue
            # 每族重置种子 → 各 comm 轮在同一族用同一组测试场景，可比
            set_seed(args.seed)
            print(f"\n===== comm_rounds={r} | family={fam} ({cfg}) | "
                  f"agents={args.num_agents} steps={args.max_steps} eps={args.episodes} test-split =====",
                  flush=True)
            m = evaluate(net=net, config_path=cfg, num_episodes=args.episodes,
                         num_agents=args.num_agents, map_size=64, max_steps=args.max_steps,
                         device=device, epsilon=0.0)
            results[r][fam] = m
            print(f"  SR={m.get('SR_mean',0):.3f}  AR={m.get('AR_mean',0):.3f}  "
                  f"Deadlock={m.get('Deadlock_mean',0):.3f}  "
                  f"NoProg={m.get('NoProgressRate_mean',0):.3f}  "
                  f"WaitRate={m.get('WaitRate_mean',0):.3f}  "
                  f"ArrSteps={m.get('ArrivalSteps_mean_mean', float('nan')):.1f}", flush=True)

    # ---- 透视表：每个指标一张 (comm_rounds × family) 表 ----
    def pivot(metric_key, title):
        print(f"\n========== {title} | 族 × 通信轮数 (test split) ==========")
        hdr = "comm | " + " | ".join(f"{f:>8}" for f in families) + " | " + f"{'MEAN':>8}"
        print(hdr); print("-" * len(hdr))
        for r in rounds:
            if not results.get(r): continue
            vals = [results[r].get(f, {}).get(metric_key, float('nan')) for f in families]
            valid = [v for v in vals if v == v]
            mean = sum(valid) / len(valid) if valid else float('nan')
            print(f"{r:>4} | " + " | ".join(f"{v:>8.3f}" for v in vals) + " | " + f"{mean:>8.3f}")

    pivot("AR_mean", "Arrival Rate (到达率, 越高越好)")
    pivot("SR_mean", "Success Rate (全员到达率, 越高越好)")
    pivot("Deadlock_mean", "Deadlock (死锁率, 越低越好)")
    pivot("WaitRate_mean", "WaitRate (等待率)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({str(r): results[r] for r in results}, f, indent=2)
    print(f"\n结果已存 {args.output}", flush=True)


if __name__ == "__main__":
    main()
