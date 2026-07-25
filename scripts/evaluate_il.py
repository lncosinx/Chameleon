#!/usr/bin/env python3
"""v3 评估：MAPFCommNet（CNN+通信+Mamba）闭环 rollout，关专家、greedy argmax。

与旧 evaluate_policy.py 的区别：新 net 无 GRU hidden、出 logits；**Mamba 时间核逐步 stateful 推断**
（episode 起点 reset_comm_state，之后每步 stateful=True 跨步携带状态）。指标同旧（SR/AR/...）。
"""
import sys, os, argparse, yaml, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

from envs.pogema_env_wrapper import PogemaWrapper
from envs.observation_builder import ObservationBuilder, build_observations
from envs.action_mask import build_action_mask_batch
from utils.metrics import compute_episode_metrics, aggregate_episode_metrics
from utils.seed import set_seed
from scripts.train_il_meta import build_net


@torch.no_grad()
def evaluate_il(net, config_path, num_episodes, num_agents, map_size, max_steps, device,
                use_cost2go=False, cost2go_clip=10.0):
    net.to(device).eval()
    wrapper = PogemaWrapper(config_path, split="test")  # held-out 测试地图池
    wrapper._raw_cfg["size"] = map_size; wrapper._raw_cfg["num_agents"] = num_agents
    wrapper.map_size = map_size; wrapper.num_agents = num_agents
    obs_builder = ObservationBuilder(num_agents=num_agents, obs_radius=5,
                                     use_cost2go=use_cost2go, cost2go_clip=cost2go_clip)

    all_metrics = []
    for ep in range(num_episodes):
        obs_builder.reset()
        obs_list, _ = wrapper.reset(seed=ep)   # 第 ep 张测试地图
        prev_actions = np.zeros(num_agents, dtype=np.int64)
        net.reset_comm_state()                       # episode 起点重置通信迟滞 + Mamba 状态
        positions_history, actions_history = [], []

        for step in range(max_steps):
            agents_xy = wrapper.get_agents_xy()
            positions_history.append(list(agents_xy))
            patches, scalars = build_observations(wrapper, obs_list, prev_actions, obs_builder)

            obstacles = wrapper.global_grid
            if obstacles is not None:
                agents_arr = np.array(agents_xy, dtype=np.int32)[np.newaxis]
                mask_np = build_action_mask_batch(agents_arr, obstacles)[0]
            else:
                mask_np = np.ones((num_agents, 5), dtype=bool)

            patches_t = torch.from_numpy(patches).unsqueeze(0).unsqueeze(0).to(device)   # (1,1,N,C,K,K)
            scalars_t = torch.from_numpy(scalars).unsqueeze(0).unsqueeze(0).to(device)   # (1,1,N,8)
            positions_t = torch.tensor(agents_xy, dtype=torch.float32, device=device)[None, None]  # (1,1,N,2)
            masks_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)     # (1,1,N,5)

            logits, _ = net(patches_t, scalars_t, positions_t, action_masks=masks_t,
                            reset_comm=False, stateful=True)
            actions = logits.squeeze(0).squeeze(0).argmax(dim=-1).cpu().numpy().tolist()

            actions_history.append(actions)
            obs_list, _, dones, _ = wrapper.step(actions)
            prev_actions = np.array(actions, dtype=np.int64)

        goals = wrapper.goals or wrapper.get_targets_xy()
        all_metrics.append(compute_episode_metrics(positions_history, goals, actions_history, max_steps))
    return aggregate_episode_metrics(all_metrics)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default="configs/env_random.yaml")
    p.add_argument("--network-config", default="configs/network_gru.yaml")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--num-agents", type=int, default=64)
    p.add_argument("--map-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    nc = yaml.safe_load(open(args.network_config))
    net = build_net(nc)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["net"].items()}
    net.load_state_dict(state)
    print(f"Loaded {args.ckpt} (meta_iter={ckpt.get('meta_iter','N/A')})")

    use_c2g = int(nc.get("patch_channels", 4)) >= 5
    m = evaluate_il(net, args.config, args.episodes, args.num_agents, args.map_size,
                    args.max_steps, device,
                    use_cost2go=use_c2g, cost2go_clip=float(nc.get("cost2go_clip", 10.0)))
    print("\n=== Eval (IL, MAPFCommNet) ===")
    for k in ["SR_mean", "AR_mean", "Deadlock_mean", "Makespan_mean", "WaitRate_mean"]:
        if k in m: print(f"  {k}: {m[k]:.4f}")
    if args.output:
        json.dump(m, open(args.output, "w"), indent=2)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
