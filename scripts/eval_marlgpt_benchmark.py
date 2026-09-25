#!/usr/bin/env python3
"""
在 marl-gpt 自带的固定测试地图（marl-gpt/eval_configs/{01-random,02-mazes,...}/maps.yaml）
上评估本仓库的通信策略 checkpoint（如 comm2 = 2 层通信）。

与 marl-gpt 官方评测协议的差异（已与用户确认）：
  - on_target 用 'finish'（一次性 MAPF，到达即停，SR 指标），而非 marl-gpt 用的
    'restart'（持续换目标、测吞吐量）——本仓库模型训练时就是 finish 语义，数值不直接
    对标 marl-gpt 论文汇报值，但能反映模型在其测试地图分布上的真实一次性 MAPF 能力。
  - 默认只轻量抽样每族部分地图（--num-maps-per-family），而非跑满整张 grid_search。

用法:
  python scripts/eval_marlgpt_benchmark.py --families random,mazes \
      --checkpoint checkpoints/comm2_iter_20000.pt --comm-rounds 2 \
      --num-maps-per-family 20 --episodes-per-combo 1
"""
from __future__ import annotations

import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from utils.seed import set_seed
from utils.metrics import compute_episode_metrics, aggregate_episode_metrics
from envs.pogema_env_wrapper import PogemaWrapper
from envs.observation_builder import ObservationBuilder, build_observations
from envs.action_mask import build_action_mask_batch
from scripts.train_meta import _build_network

MARLGPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "marl-gpt", "eval_configs")

# family 目录名 -> (family yaml, maps yaml)
_FAMILY_DIRS = {
    "random": "01-random",
    "mazes": "02-mazes",
    "warehouse": "03-warehouse",
    "movingai": "04-movingai",
}


def _load_family(family: str):
    d = os.path.join(MARLGPT_DIR, _FAMILY_DIRS[family])
    cfg = yaml.safe_load(open(os.path.join(d, f"{_FAMILY_DIRS[family]}.yaml")))
    maps = yaml.safe_load(open(os.path.join(d, "maps.yaml")))
    env_cfg = cfg["environment"]
    max_steps = env_cfg.get("max_episode_steps", 256)
    collision_system = env_cfg.get("collision_system", "soft")
    na = env_cfg.get("num_agents", 8)
    if isinstance(na, dict) and "grid_search" in na:
        agent_counts = list(na["grid_search"])
    else:
        agent_counts = [int(na)]
    return maps, max_steps, collision_system, agent_counts


def _sample_map_names(maps: dict, num_maps: int) -> list:
    names = sorted(maps.keys())
    if num_maps <= 0 or num_maps >= len(names):
        return names
    idx = np.linspace(0, len(names) - 1, num_maps).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [names[i] for i in idx]


def run_episode(net, map_str, map_name, num_agents, max_steps, collision_system,
                on_target, obs_radius, device, seed):
    cfg = {
        "map_type": "random",  # 占位，fixed_map 会覆盖地图来源
        "fixed_map": map_str,
        "fixed_map_name": map_name,
        "map_size": max(len(map_str.splitlines()), len(map_str.splitlines()[0])),
        "num_agents": num_agents,
        "obs_radius": obs_radius,
        "max_episode_steps": max_steps,
        "collision_system": collision_system,
        "on_target": on_target,
        "split": "test",
    }
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp_path = f.name
    try:
        wrapper = PogemaWrapper(tmp_path)
    finally:
        os.unlink(tmp_path)

    obs_builder = ObservationBuilder(num_agents=num_agents, obs_radius=obs_radius)
    obs_builder.reset()
    obs_list, _ = wrapper.reset(seed=seed)

    prev_actions = np.zeros(num_agents, dtype=np.int64)
    hidden = net.init_hidden(1, num_agents, device)
    net.reset_comm_state()

    positions_history, actions_history = [], []

    for _ in range(max_steps):
        agents_xy = wrapper.get_agents_xy()
        positions_history.append(list(agents_xy))

        patches, scalars = build_observations(wrapper, obs_list, prev_actions, obs_builder)

        obstacles = wrapper.global_grid
        if obstacles is not None:
            agents_arr = np.array(agents_xy, dtype=np.int32)[np.newaxis]
            mask_np = build_action_mask_batch(agents_arr, obstacles)[0]
        else:
            mask_np = np.ones((num_agents, 5), dtype=bool)

        patches_t   = torch.from_numpy(patches).unsqueeze(0).to(device)
        scalars_t   = torch.from_numpy(scalars).unsqueeze(0).to(device)
        positions_t = torch.tensor(agents_xy, dtype=torch.float32, device=device).unsqueeze(0)
        masks_t     = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values, hidden, _ = net(
                patches_t.unsqueeze(1), scalars_t.unsqueeze(1), positions_t.unsqueeze(1),
                hidden, action_masks=masks_t,
            )
            q_values = q_values.squeeze(0).squeeze(0)

        actions = q_values.argmax(dim=-1).cpu().numpy().tolist()
        actions_history.append(actions)

        obs_list, _, _, _ = wrapper.step(actions)
        prev_actions = np.array(actions, dtype=np.int64)

    goals = wrapper.goals or wrapper.get_targets_xy()
    return compute_episode_metrics(positions_history, goals, actions_history, max_steps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--families", default="random,mazes", help="逗号分隔: random,mazes,warehouse,movingai")
    p.add_argument("--checkpoint", default="checkpoints/comm2_iter_20000.pt")
    p.add_argument("--network-config", default="configs/network_biintent_drqn.yaml")
    p.add_argument("--comm-rounds", type=int, default=2)
    p.add_argument("--num-maps-per-family", type=int, default=20,
                    help="每族抽样地图数（<=0 表示用全部地图）")
    p.add_argument("--agent-counts", default="", help="逗号分隔覆盖 agent 数网格；留空则用各族自带 grid_search")
    p.add_argument("--episodes-per-combo", type=int, default=1)
    p.add_argument("--on-target", default="finish", choices=["finish", "restart"])
    p.add_argument("--obs-radius", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="/data/lncosinx/docker_tmp/claude-1011/-home-lncosinx-Chameleon/"
                                        "3d007f56-a6c3-4c19-82a3-a2e88badc3e3/scratchpad/marlgpt_benchmark.json",
                    help="logs/eval/ 目录当前为 root 所有、本用户不可写，默认改存 scratchpad")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    net_cfg = yaml.safe_load(open(args.network_config))
    net_cfg = dict(net_cfg)
    net_cfg["comm_rounds"] = args.comm_rounds
    net = _build_network(net_cfg, use_torch_compile=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt["net"])
    net.to(device)
    net.eval()
    print(f"Loaded {args.checkpoint} (train_step={ckpt.get('train_step', 'N/A')}, comm_rounds={args.comm_rounds})")

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    override_agent_counts = ([int(x) for x in args.agent_counts.split(",") if x.strip()]
                              if args.agent_counts else None)

    all_results = {}
    for family in families:
        maps, max_steps, collision_system, agent_counts = _load_family(family)
        if override_agent_counts:
            agent_counts = override_agent_counts
        map_names = _sample_map_names(maps, args.num_maps_per_family)
        print(f"\n===== family={family}  maps={len(map_names)}/{len(maps)}  "
              f"agent_counts={agent_counts}  max_steps={max_steps}  on_target={args.on_target} =====")

        per_agent_count = {n: [] for n in agent_counts}
        family_metrics = []
        for num_agents in agent_counts:
            for map_name in map_names:
                map_str = maps[map_name]
                for ep in range(args.episodes_per_combo):
                    seed = args.seed + ep
                    m = run_episode(net, map_str, map_name, num_agents, max_steps,
                                     collision_system, args.on_target, args.obs_radius,
                                     device, seed)
                    per_agent_count[num_agents].append(m)
                    family_metrics.append(m)
            agg = aggregate_episode_metrics(per_agent_count[num_agents])
            print(f"  agents={num_agents:>4}: SR={agg.get('SR_mean', float('nan')):.3f}  "
                  f"AR={agg.get('AR_mean', float('nan')):.3f}  "
                  f"WaitRate={agg.get('WaitRate_mean', float('nan')):.3f}  "
                  f"Deadlock={agg.get('Deadlock_mean', float('nan')):.3f}  "
                  f"(n={agg.get('num_episodes', 0)})")

        all_results[family] = {
            "overall": aggregate_episode_metrics(family_metrics),
            "by_agent_count": {str(n): aggregate_episode_metrics(v) for n, v in per_agent_count.items()},
            "map_names": map_names,
        }
        print(f"  [{family} overall] SR={all_results[family]['overall'].get('SR_mean', float('nan')):.3f}"
              f" over {all_results[family]['overall'].get('num_episodes', 0)} episodes")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已存 {args.output}")


if __name__ == "__main__":
    main()
