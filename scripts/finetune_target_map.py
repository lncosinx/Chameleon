#!/usr/bin/env python3
"""
阶段三：目标地图近 on-policy 微调。

用法:
    python scripts/finetune_target_map.py --config configs/finetune_online.yaml \
        --env configs/env_maze.yaml --init checkpoints/meta_init.pt
"""
import sys, os, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from utils.seed import set_seed
from models.biintent_drqn import BiIntentGATDRQN
from trainers.finetune_online import FinetuneOnlineTrainer
from envs.pogema_env_wrapper import PogemaWrapper
from envs.observation_builder import ObservationBuilder, build_observations
from envs.reward_builder import RewardBuilder
from lacam.lacam_runner import LaCAMRunner
from lacam.lacam_dataset_builder import build_dataset_from_rollout
from utils.checkpoint import save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/finetune_online.yaml")
    parser.add_argument("--network-config", type=str, default="configs/network_biintent_drqn.yaml")
    parser.add_argument("--env", type=str, default="configs/env_maze.yaml")
    parser.add_argument("--init", type=str, required=True, help="Meta checkpoint path")
    parser.add_argument("--num-segments", type=int, default=500)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="checkpoints/finetuned.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.network_config) as f:
        net_cfg = yaml.safe_load(f)

    # 网络
    net = BiIntentGATDRQN(
        patch_channels=net_cfg.get("patch_channels", 4),
        cnn_channels=net_cfg.get("cnn_channels", [32, 64, 64]),
        scalar_feat_dim=net_cfg.get("scalar_feat_dim", 8),
        intent_dim=net_cfg.get("intent_dim", 64),
        comm_dim=net_cfg.get("comm_dim", 64),
        comm_radius=net_cfg.get("comm_radius", 4),
        comm_topk=net_cfg.get("comm_topk", 6),
        gru_hidden_dim=net_cfg.get("gru_hidden_dim", 256),
    )

    trainer = FinetuneOnlineTrainer(net, cfg, device=device)
    trainer.load_meta_checkpoint(args.init)

    # 环境
    wrapper = PogemaWrapper(args.env)
    wrapper._raw_cfg.update(size=args.map_size, num_agents=args.num_agents)
    wrapper.map_size = args.map_size
    wrapper.num_agents = args.num_agents

    obs_builder = ObservationBuilder(num_agents=args.num_agents, obs_radius=5)
    reward_builder = RewardBuilder(lambda_f=0.1)
    runner = LaCAMRunner(backend="auto")

    print(f"Finetuning on {wrapper.map_type} | {args.num_segments} segments | "
          f"agents={args.num_agents} | device={device}")

    for seg in range(args.num_segments):
        # 同步段采样
        wrapper.reset(seed=args.seed * 1000 + seg)
        obstacles = wrapper.global_grid

        # 以当前概率调用 LaCAM 专家
        if np.random.random() < trainer.expert_prob():
            starts = wrapper.get_agents_xy()
            goals = wrapper.get_targets_xy()
            result = runner.solve(obstacles, starts, goals,
                                  map_type=wrapper.map_type, map_size=args.map_size)
            if result.success:
                wrapper.reset(seed=args.seed * 1000 + seg)
                samples = build_dataset_from_rollout(
                    wrapper, obs_builder, reward_builder, obstacles,
                    result.paths, wrapper.map_type, f"ft_{seg}",
                )
            else:
                samples = []
        else:
            # 纯自游走采样（用当前策略 rollout）
            samples = _rollout_with_policy(
                wrapper, obs_builder, reward_builder, net, device,
                trainer.rollout_segment_len,
            )

        if samples:
            trainer.add_rollout_samples(samples, trainer.train_step)
            metrics = trainer.train_on_segment()

            if (seg + 1) % 20 == 0:
                print(f"  seg {seg+1}/{args.num_segments}: "
                      f"loss={metrics.get('loss_total', 0):.4f}, "
                      f"expert_p={metrics.get('expert_prob', 0):.2f}, "
                      f"margin_lambda={metrics.get('lambda_margin', 0):.3f}")

    save_checkpoint(trainer.state_dict(), args.output, "finetuned")
    print(f"Checkpoint saved to {args.output}")


def _rollout_with_policy(wrapper, obs_builder, reward_builder, net, device, segment_len):
    """用当前策略 rollout 一段。"""
    N = wrapper.num_agents
    samples = []
    obs_list, _ = wrapper.reset()
    prev_actions = np.zeros(N, dtype=np.int64)
    hidden = net.init_hidden(1, N, device)

    for step in range(segment_len):
        patches, scalars = build_observations(wrapper, obs_list, prev_actions, obs_builder)
        patches_t = torch.from_numpy(patches).unsqueeze(0).unsqueeze(1).to(device)
        scalars_t = torch.from_numpy(scalars).unsqueeze(0).unsqueeze(1).to(device)
        positions_t = torch.zeros(1, 1, N, 2, device=device)

        with torch.no_grad():
            q_values, hidden, _ = net(patches_t, scalars_t, positions_t, hidden)

        actions = q_values.squeeze().argmax(dim=-1).cpu().numpy()
        obs_list, rewards, dones, _ = wrapper.step(actions.tolist())

        for i in range(N):
            samples.append({
                "obs_patch": patches[i].copy(),
                "obs_scalar": scalars[i].copy(),
                "action_expert": int(actions[i]),
                "reward": float(rewards[i]),
                "done": dones[i],
                "action_mask": np.ones(5, dtype=bool),
                "map_type": wrapper.map_type,
                "task_id": f"rollout",
                "margin_excluded": True,  # 自游走样本不进 margin loss
                "margin_weight": 0.0,
            })

        prev_actions = actions

    return samples


if __name__ == "__main__":
    main()
