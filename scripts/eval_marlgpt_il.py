#!/usr/bin/env python3
"""
在 marl-gpt 的固定测试地图（validation-*）上评测 v3 IL 模型（MAPFCommNet，5ch cost2go，
full-flatten 编码器），产出与 scripts/eval_marlgpt_benchmark.py（comm2）、EPH、MAPF-GPT
**同图同协议**可比的 SR/AR 表。

与 eval_marlgpt_benchmark.py 的唯一区别：网络换成 MAPFCommNet（build_net）、rollout 用
逐步 stateful 前向、观测启用 cost2go（obstacles 从 wrapper.global_grid 现算，与 baseline 惯例一致）。

用法（marlgpt 环境，Chameleon 根）：
  python scripts/eval_marlgpt_il.py --ckpt ckpt_il/il_joint_iter_20000.pt \
      --network-config configs/network_gru_c2g.yaml --families random,mazes \
      --num-maps-per-family 20 --episodes-per-combo 1
"""
from __future__ import annotations
import sys, os, argparse, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, yaml

from utils.seed import set_seed
from utils.metrics import compute_episode_metrics, aggregate_episode_metrics
from envs.pogema_env_wrapper import PogemaWrapper
from envs.observation_builder import ObservationBuilder, build_observations
from envs.action_mask import build_action_mask_batch
from scripts.train_il_meta import build_net

MARLGPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "marl-gpt", "eval_configs")
FAMILY_DIR = {"random": "01-random", "mazes": "02-mazes"}
FAMILY_MAXSTEPS = {"random": 256, "mazes": 128}


def _sample_map_names(maps, num_maps):
    names = sorted(maps.keys())
    if num_maps <= 0 or num_maps >= len(names):
        return names
    idx = sorted(set(np.linspace(0, len(names) - 1, num_maps).round().astype(int).tolist()))
    return [names[i] for i in idx]


@torch.no_grad()
def run_episode(net, use_cost2go, cost2go_clip, map_str, name, num_agents, max_steps, device, seed,
                on_target="finish"):
    rows = map_str.splitlines(); H, W = len(rows), len(rows[0])
    cfg = dict(map_type="random", fixed_map=map_str, fixed_map_name=name, map_size=max(H, W),
               num_agents=num_agents, obs_radius=5, max_episode_steps=max_steps,
               collision_system="soft", on_target=on_target, split="test")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f); p = f.name
    try:
        w = PogemaWrapper(p)
    finally:
        os.unlink(p)

    ob = ObservationBuilder(num_agents=num_agents, obs_radius=5,
                            use_cost2go=use_cost2go, cost2go_clip=cost2go_clip)
    ob.reset()
    obs_list, _ = w.reset(seed=seed)
    prev = np.zeros(num_agents, dtype=np.int64)
    net.reset_comm_state()

    pos_hist, act_hist = [], []
    for _ in range(max_steps):
        axy = w.get_agents_xy(); pos_hist.append(list(axy))
        patches, scalars = build_observations(w, obs_list, prev, ob)  # cost2go 内部取 w.global_grid
        obstacles = w.global_grid
        if obstacles is not None:
            aarr = np.array(axy, dtype=np.int32)[np.newaxis]
            mask_np = build_action_mask_batch(aarr, obstacles)[0]
        else:
            mask_np = np.ones((num_agents, 5), dtype=bool)
        pt = torch.from_numpy(patches).unsqueeze(0).unsqueeze(0).to(device)
        st = torch.from_numpy(scalars).unsqueeze(0).unsqueeze(0).to(device)
        post = torch.tensor(axy, dtype=torch.float32, device=device)[None, None]
        mt = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
        logits, _ = net(pt, st, post, action_masks=mt, reset_comm=False, stateful=True)
        actions = logits.squeeze(0).squeeze(0).argmax(dim=-1).cpu().numpy().tolist()
        act_hist.append(actions)
        obs_list, _, _, _ = w.step(actions)
        prev = np.array(actions, dtype=np.int64)

    goals = w.goals or w.get_targets_xy()
    m = compute_episode_metrics(pos_hist, goals, act_hist, max_steps)
    # leaderboard 口径（on_target=nothing）：CSR/ISR 按**终局**位置算（到达者不消失、需守住目标）。
    final = np.array(w.get_agents_xy()); g = np.array(goals)
    at_goal = np.all(final == g, axis=1)
    m["ISR"] = float(at_goal.mean())       # = pogema ISR（个体成功率，终局）
    m["CSR"] = float(at_goal.all())        # = pogema CSR（全员成功，终局）
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--network-config", default="configs/network_gru_c2g.yaml")
    p.add_argument("--families", default="random,mazes")
    p.add_argument("--agent-counts", default="8,16,24,32,48,64")
    p.add_argument("--num-maps-per-family", type=int, default=20)
    p.add_argument("--episodes-per-combo", type=int, default=1)
    p.add_argument("--on-target", default="finish", choices=["finish", "nothing"],
                   help="nothing=leaderboard 协议（到达者不消失、变静态障碍）")
    p.add_argument("--max-steps", type=int, default=0,
                   help=">0 覆盖各族 horizon（leaderboard MAPF 两族都用 128）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="/data/lncosinx/docker_tmp/claude-1011/-home-lncosinx-Chameleon/"
                                       "3d007f56-a6c3-4c19-82a3-a2e88badc3e3/scratchpad/marlgpt_il.json")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nc = yaml.safe_load(open(args.network_config))
    use_c2g = int(nc.get("patch_channels", 4)) >= 5
    net = build_net(nc)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["net"].items()}
    net.load_state_dict(state); net.to(device).eval()
    print(f"Loaded {args.ckpt} | MAPFCommNet {sum(p.numel() for p in net.parameters())/1e6:.2f}M | "
          f"cost2go={use_c2g} (meta_iter={ckpt.get('meta_iter','?')})", flush=True)

    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    acs = [int(x) for x in args.agent_counts.split(",")]
    results = {}
    for fam in fams:
        maps = yaml.safe_load(open(os.path.join(MARLGPT, FAMILY_DIR[fam], "maps.yaml")))
        names = _sample_map_names(maps, args.num_maps_per_family)
        ms = args.max_steps if args.max_steps > 0 else FAMILY_MAXSTEPS.get(fam, 256)
        print(f"\n===== il family={fam} maps={len(names)}/{len(maps)} max_steps={ms} =====", flush=True)
        results[fam] = {}
        for na in acs:
            rows = []
            for nm in names:
                for ep in range(args.episodes_per_combo):
                    rows.append(run_episode(net, use_c2g, float(nc.get("cost2go_clip", 10.0)),
                                            maps[nm], nm, na, ms, device, args.seed + ep,
                                            on_target=args.on_target))
            agg = aggregate_episode_metrics(rows)
            results[fam][na] = agg
            print(f"  agents={na:>3}: CSR={agg.get('CSR_mean',0):.3f}  ISR={agg.get('ISR_mean',0):.3f}  "
                  f"(SR_arrive={agg.get('SR_mean',0):.3f} AR_arrive={agg.get('AR_mean',0):.3f})  "
                  f"Deadlock={agg.get('Deadlock_mean',0):.3f}  (n={agg.get('num_episodes',0)})", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(results, open(args.output, "w"), indent=2)
    print(f"\nsaved -> {args.output}")


if __name__ == "__main__":
    main()
