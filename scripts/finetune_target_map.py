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
from envs.action_mask import build_action_mask_batch
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
        d_patch=net_cfg.get("d_patch", 128),
        scalar_feat_dim=net_cfg.get("scalar_feat_dim", 8),
        d_scalar=net_cfg.get("d_scalar", 32),
        intent_dim=net_cfg.get("intent_dim", 64),
        comm_dim=net_cfg.get("comm_dim", 64),
        comm_radius=net_cfg.get("comm_radius", 4),
        comm_topk_cap=net_cfg.get("comm_topk_cap", 6),
        num_attention_heads=net_cfg.get("num_attention_heads", 4),
        entmax_alpha=net_cfg.get("entmax_alpha", 1.5),
        tau_in=net_cfg.get("tau_in", 0.05),
        tau_out=net_cfg.get("tau_out", 0.02),
        stickiness_bonus=net_cfg.get("stickiness_bonus", 0.1),
        ema_decay=net_cfg.get("ema_decay", 0.0),
        gru_hidden_dim=net_cfg.get("gru_hidden_dim", 256),
        dueling_hidden_dim=net_cfg.get("dueling_hidden_dim", 256),
    )

    trainer = FinetuneOnlineTrainer(net, cfg, device=device)
    trainer.load_meta_checkpoint(args.init)

    # 环境
    wrapper = PogemaWrapper(args.env)
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

    # episode 开始时重置通信迟滞状态
    net.reset_comm_state()

    prev_agents = wrapper.get_agents_xy()
    prev_goals  = wrapper.get_targets_xy()
    obstacles   = wrapper.global_grid

    for step in range(segment_len):
        patches, scalars = build_observations(wrapper, obs_list, prev_actions, obs_builder)
        agents_xy = wrapper.get_agents_xy()

        # 向量化动作掩码（避免 per-agent 循环）
        if obstacles is not None:
            agents_arr = np.array(agents_xy, dtype=np.int32)[np.newaxis]  # (1, N, 2)
            mask_np = build_action_mask_batch(agents_arr, obstacles)[0]   # (N, 5) bool
        else:
            mask_np = np.ones((N, 5), dtype=bool)

        patches_t  = torch.from_numpy(patches).unsqueeze(0).unsqueeze(1).to(device)
        scalars_t  = torch.from_numpy(scalars).unsqueeze(0).unsqueeze(1).to(device)
        positions_t = torch.tensor(agents_xy, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        masks_t    = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,N,5)

        with torch.no_grad():
            q_values, hidden, _ = net(
                patches_t, scalars_t, positions_t, hidden,
                action_masks=masks_t,
            )

        # q_values: (1, 1, N, 5) → (N, 5)；-inf 掩码已由网络施加
        actions = q_values.squeeze(0).squeeze(0).argmax(dim=-1).cpu().numpy()
        obs_list, rewards, dones, _ = wrapper.step(actions.tolist())

        # 计算 PBRS reward（用于近 on-policy 训练）
        curr_agents = wrapper.get_agents_xy()
        curr_goals  = wrapper.get_targets_xy()
        if obstacles is not None:
            shaped_rewards, _ = reward_builder.compute(
                obstacles=obstacles,
                prev_agents_xy=agents_xy,
                agents_xy=curr_agents,
                targets_xy=curr_goals,
                prev_targets_xy=prev_goals,
                prev_actions=prev_actions,
            )
        else:
            shaped_rewards = [float(r) for r in rewards]

        for i in range(N):
            samples.append({
                "obs_patch": patches[i].copy(),
                "obs_scalar": scalars[i].copy(),
                "action_expert": int(actions[i]),
                "reward": shaped_rewards[i],
                "done": dones[i] if isinstance(dones, list) else bool(dones),
                "action_mask": mask_np[i].copy(),   # B10: 真实掩码
                "map_type": wrapper.map_type,
                "task_id": "rollout",
                "margin_excluded": True,  # 自游走样本不进 margin loss
                "margin_weight": 0.0,
            })

        prev_actions = actions
        prev_agents  = curr_agents
        prev_goals   = curr_goals

    return samples


if __name__ == "__main__":
    main()
