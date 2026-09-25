#!/usr/bin/env python3
"""
通信层数消融评估：对 comm_rounds=0/1/2/3 的 checkpoint 在 held-out test split 上跑评估，
打印 SR 等指标对比表。地图来自 when-to-switch 全族 test split（envs/map_registry.py，
PogemaWrapper(split='test')）。

用法:
  python scripts/eval_comm_sweep.py --rounds 0,1,2 --iter 20000 --num-agents 64 --episodes 50
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
    p.add_argument("--iter", type=int, default=20000, help="评估哪个里程碑 ckpt（comm{r}_iter_{N}.pt）")
    p.add_argument("--ckpt-tpl", default="checkpoints/comm{r}_iter_{it}.pt")
    p.add_argument("--config", default="configs/env_eval_all.yaml")
    p.add_argument("--network-config", default="configs/network_biintent_drqn.yaml")
    p.add_argument("--num-agents", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="logs/eval/comm_sweep_eval.json")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rounds = [int(x) for x in args.rounds.split(",") if x.strip() != ""]
    net_cfg = yaml.safe_load(open(args.network_config))

    results = {}
    for r in rounds:
        ckpt_path = args.ckpt_tpl.format(r=r, it=args.iter)
        if not os.path.exists(ckpt_path):
            print(f"[skip] comm_rounds={r}: {ckpt_path} 不存在"); continue
        nc = dict(net_cfg); nc["comm_rounds"] = r
        net = _build_network(nc, use_torch_compile=False)
        ckpt = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ckpt["net"])
        net.to(device)
        print(f"\n===== 评估 comm_rounds={r}  ({ckpt_path}) | agents={args.num_agents} "
              f"steps={args.max_steps} eps={args.episodes} test-split =====", flush=True)
        m = evaluate(net=net, config_path=args.config, num_episodes=args.episodes,
                     num_agents=args.num_agents, map_size=64, max_steps=args.max_steps,
                     device=device, epsilon=0.0)
        results[r] = m
        print(f"  SR={m.get('SR_mean',0):.3f}±{m.get('SR_std',0):.3f}  "
              + "  ".join(f"{k.replace('_mean',''):}={m[k]:.3f}" for k in KEYS[1:] if k in m), flush=True)

    # 对比表
    print("\n========== 通信层数消融对比 (test split) ==========")
    cols = [k.replace("_mean", "") for k in KEYS]
    hdr = "comm_rounds | " + " | ".join(f"{c:>12}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for r in rounds:
        if r not in results: continue
        m = results[r]
        print(f"{r:>11} | " + " | ".join(f"{m.get(k, float('nan')):>12.4f}" for k in KEYS))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({str(r): results[r] for r in results}, f, indent=2)
    print(f"\n结果已存 {args.output}")


if __name__ == "__main__":
    main()
