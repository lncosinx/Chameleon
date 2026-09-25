#!/usr/bin/env python3
"""
在外部提供的实例集 (scripts/convert_pogema_to_eph_instances.py 产出) 上评测 EPH。

- 纯学习策略（use_stepinfer=False, 无 A* 引导, 无 AEP）——对我们纯通信网络的公平对照。
- 每族 max_steps 对齐 marl-gpt/comm2 评测协议：random=256, mazes=128。
- 报告 SR(全员到达) / AR(到达率) / 平均步数 / 平均通信事件数(comm_mask 之和)。

用法（在 EPH/ 目录、marlgpt 环境下）:
  CONFIG=eph-quick python eval_instances.py --instances <dir> --ckpt 65000
"""
import os, sys, argparse, pickle, glob, time
import numpy as np
import torch
from hydra.utils import instantiate

from src.config import config
from src.environment import Environment

FAMILY_MAXSTEPS = {"random": 256, "mazes": 128}


def build_network(ckpt_name, device):
    Network = instantiate({"_target_": config.model_target, "_partial_": True})
    net = Network()
    net.eval()
    net.to(device)
    sd = torch.load(os.path.join(config.save_path, f"{ckpt_name}.pth"), map_location=device)
    net.load_state_dict(sd)
    net.eval()
    return net


@torch.inference_mode()
def run_one(net, inst, max_steps, device):
    map_, starts, goals = inst
    env = Environment()
    env.load(np.asarray(map_), np.asarray(starts), np.asarray(goals))
    # 纯策略：关闭所有搜索外挂
    env.astar_type = None
    env.active_agent_radius = None
    env.use_aep = False
    env.aep_astar_type = None

    obs, last_act, pos = env.observe()
    net.reset()
    comm_total = 0
    done = False
    step = 0
    while not done and env.steps < max_steps:
        actions, q_val, _, _, comm_mask = net.step(
            torch.as_tensor(obs.astype(np.float32)).to(device),
            torch.as_tensor(last_act.astype(np.float32)).to(device),
            torch.as_tensor(pos.astype(int)).to(device),
        )
        (obs, last_act, pos), _, done, info = env.step(actions)
        comm_total += int(np.sum(comm_mask))
        step += 1

    arrived = int(np.sum(np.all(env.agents_pos == env.goals_pos, axis=1)))
    n = env.num_agents
    success = int(arrived == n)
    return success, arrived / n, step, comm_total, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="/data/lncosinx/docker_tmp/claude-1011/"
                    "-home-lncosinx-Chameleon/3d007f56-a6c3-4c19-82a3-a2e88badc3e3/scratchpad/eph_instances")
    ap.add_argument("--ckpt", default="65000")
    ap.add_argument("--families", default="random,mazes")
    ap.add_argument("--agent-counts", default="8,16,24,32,48,64")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default="/data/lncosinx/docker_tmp/claude-1011/"
                    "-home-lncosinx-Chameleon/3d007f56-a6c3-4c19-82a3-a2e88badc3e3/scratchpad/eph_samemap_results.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    net = build_network(args.ckpt, device)
    print(f"EPH pure-policy | ckpt={args.ckpt} | device={device} | "
          f"params={sum(p.numel() for p in net.parameters())/1e6:.3f}M", flush=True)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    agent_counts = [int(x) for x in args.agent_counts.split(",")]
    results = {}
    for fam in families:
        max_steps = FAMILY_MAXSTEPS.get(fam, 256)
        print(f"\n===== EPH family={fam} max_steps={max_steps} =====", flush=True)
        results[fam] = {}
        for na in agent_counts:
            path = os.path.join(args.instances, f"{fam}_{na}agents.pkl")
            if not os.path.exists(path):
                print(f"  [skip] {path} missing"); continue
            insts = pickle.load(open(path, "rb"))
            t0 = time.time()
            rows = [run_one(net, inst, max_steps, device) for inst in insts]
            succ, ar, steps, comm, ns = zip(*rows)
            agg = dict(SR=float(np.mean(succ)), AR=float(np.mean(ar)),
                       steps=float(np.mean(steps)), comm=float(np.mean(comm)),
                       n=len(insts))
            results[fam][na] = agg
            print(f"  agents={na:>3}: SR={agg['SR']:.3f}  AR={agg['AR']:.3f}  "
                  f"steps={agg['steps']:.1f}  comm={agg['comm']:.1f}  "
                  f"(n={agg['n']}, {time.time()-t0:.1f}s)", flush=True)

    import json
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.output}")


if __name__ == "__main__":
    main()
