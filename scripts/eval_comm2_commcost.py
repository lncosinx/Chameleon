#!/usr/bin/env python3
"""
在与 EPH 相同的 marl-gpt 地图/起终点/agent 网格上评测 comm2，并记录**通信开销**指标，
用于"协调质量 vs 通信量" Pareto 对比。

指标：
  SR / AR          —— 应与 scripts/eval_marlgpt_benchmark.py(comm2) 一致(同实例交叉校验)
  comm_degree      —— 每 agent 每轮平均邻居数(avg_neighbors，与 episode 长度无关，架构级)
  edge_flip        —— EdgeFlipRate(通信图时间稳定性，越低越好，本文卖点)
  total_messages   —— 全 episode 有向消息总数 = Σ_step avg_neighbors·N·comm_rounds
                      (与 EPH 的 comm 事件总数同口径，用于 Pareto 横轴)

max_steps 对齐 marl-gpt/EPH：random=256, mazes=128。seed=0，与实例生成一致。

用法(Chameleon 根、marlgpt 环境):
  python scripts/eval_comm2_commcost.py --ckpt checkpoints/comm2_iter_20000.pt --comm-rounds 2
"""
import sys, os, argparse, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import yaml

from utils.seed import set_seed
from utils.metrics import compute_episode_metrics
from envs.pogema_env_wrapper import PogemaWrapper
from envs.observation_builder import ObservationBuilder, build_observations
from envs.action_mask import build_action_mask_batch
from scripts.train_meta import _build_network

MARLGPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "marl-gpt", "eval_configs")
FAMILY_DIR = {"random": "01-random", "mazes": "02-mazes"}
FAMILY_MAXSTEPS = {"random": 256, "mazes": 128}


def sample_map_names(maps, num_maps):
    names = sorted(maps.keys())
    if num_maps <= 0 or num_maps >= len(names):
        return names
    idx = sorted(set(np.linspace(0, len(names) - 1, num_maps).round().astype(int).tolist()))
    return [names[i] for i in idx]


@torch.no_grad()
def run_episode(net, comm_rounds, map_str, name, num_agents, max_steps, device, seed):
    rows = map_str.splitlines(); H, W = len(rows), len(rows[0])
    cfg = dict(map_type="random", fixed_map=map_str, fixed_map_name=name, map_size=max(H, W),
               num_agents=num_agents, obs_radius=5, max_episode_steps=max_steps,
               collision_system="soft", on_target="finish", split="test")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f); p = f.name
    try:
        w = PogemaWrapper(p)
    finally:
        os.unlink(p)

    ob = ObservationBuilder(num_agents=num_agents, obs_radius=5); ob.reset()
    obs_list, _ = w.reset(seed=seed)
    prev = np.zeros(num_agents, dtype=np.int64)
    hidden = net.init_hidden(1, num_agents, device)
    net.reset_comm_state()

    pos_hist, act_hist = [], []
    deg_sum, act_deg_sum, flip_sum, msg_total, nsteps = 0.0, 0.0, 0.0, 0.0, 0
    for _ in range(max_steps):
        axy = w.get_agents_xy(); pos_hist.append(list(axy))
        patches, scalars = build_observations(w, obs_list, prev, ob)
        obstacles = w.global_grid
        if obstacles is not None:
            aarr = np.array(axy, dtype=np.int32)[np.newaxis]
            mask_np = build_action_mask_batch(aarr, obstacles)[0]
        else:
            mask_np = np.ones((num_agents, 5), dtype=bool)
        pt = torch.from_numpy(patches).unsqueeze(0).to(device)
        st = torch.from_numpy(scalars).unsqueeze(0).to(device)
        post = torch.tensor(axy, dtype=torch.float32, device=device).unsqueeze(0)
        mt = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
        q, hidden, info = net(pt.unsqueeze(1), st.unsqueeze(1), post.unsqueeze(1), hidden, action_masks=mt)
        q = q.squeeze(0).squeeze(0)
        actions = q.argmax(dim=-1).cpu().numpy().tolist()
        act_hist.append(actions)
        deg = float(info.get("avg_neighbors", 0.0))            # topk 前候选(高估)
        act_deg = float(info.get("avg_active_neighbors", 0.0)) # topk 后真实度数
        deg_sum += deg
        act_deg_sum += act_deg
        flip_sum += float(info.get("edge_flip_rate", 0.0))
        msg_total += act_deg * num_agents * comm_rounds        # 真实消息量口径
        nsteps += 1
        obs_list, _, _, _ = w.step(actions)
        prev = np.array(actions, dtype=np.int64)

    goals = w.goals or w.get_targets_xy()
    m = compute_episode_metrics(pos_hist, goals, act_hist, max_steps)
    return dict(SR=m["SR"], AR=m["AR"],
                comm_degree_precap=deg_sum / max(nsteps, 1),
                comm_degree=act_deg_sum / max(nsteps, 1),   # 真实(topk 后)度数
                edge_flip=flip_sum / max(nsteps, 1),
                total_messages=msg_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/comm2_iter_20000.pt")
    ap.add_argument("--network-config", default="configs/network_biintent_drqn.yaml")
    ap.add_argument("--comm-rounds", type=int, default=2)
    ap.add_argument("--families", default="random,mazes")
    ap.add_argument("--agent-counts", default="8,16,24,32,48,64")
    ap.add_argument("--num-maps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default="/data/lncosinx/docker_tmp/claude-1011/"
                    "-home-lncosinx-Chameleon/3d007f56-a6c3-4c19-82a3-a2e88badc3e3/scratchpad/comm2_commcost.json")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net_cfg = dict(yaml.safe_load(open(args.network_config))); net_cfg["comm_rounds"] = args.comm_rounds
    net = _build_network(net_cfg, use_torch_compile=False)
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt["net"]); net.to(device); net.eval()
    print(f"comm2 | ckpt={args.ckpt} comm_rounds={args.comm_rounds} "
          f"params={sum(p.numel() for p in net.parameters())/1e6:.3f}M", flush=True)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    acs = [int(x) for x in args.agent_counts.split(",")]
    results = {}
    for fam in families:
        maps = yaml.safe_load(open(os.path.join(MARLGPT, FAMILY_DIR[fam], "maps.yaml")))
        names = sample_map_names(maps, args.num_maps)
        ms = FAMILY_MAXSTEPS.get(fam, 256)
        print(f"\n===== comm2 family={fam} max_steps={ms} maps={len(names)} =====", flush=True)
        results[fam] = {}
        for na in acs:
            rows = [run_episode(net, args.comm_rounds, maps[nm], nm, na, ms, device, args.seed) for nm in names]
            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            agg["n"] = len(rows)
            results[fam][na] = agg
            print(f"  agents={na:>3}: SR={agg['SR']:.3f}  AR={agg['AR']:.3f}  "
                  f"degree={agg['comm_degree']:.2f}(cap6, precap={agg['comm_degree_precap']:.1f})  "
                  f"edgeflip={agg['edge_flip']:.3f}  msgs={agg['total_messages']:.0f}  (n={agg['n']})", flush=True)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.output}")


if __name__ == "__main__":
    main()
