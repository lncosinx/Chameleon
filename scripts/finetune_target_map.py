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
        d_patch=net_cfg.get("d_patch", 256),  # 与 train_meta 默认一致，防 ckpt 维度不匹配
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
        use_torch_compile=bool(cfg.get("use_torch_compile", False)),
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
        segment = None
        if np.random.random() < trainer.expert_prob():
            starts = wrapper.get_agents_xy()
            goals = wrapper.get_targets_xy()
            result = runner.solve(obstacles, starts, goals,
                                  map_type=wrapper.map_type, map_size=args.map_size)
            if result.success:
                wrapper.reset(seed=args.seed * 1000 + seg)
                segment = build_dataset_from_rollout(
                    wrapper, obs_builder, reward_builder, obstacles,
                    result.paths, wrapper.map_type, f"ft_{seg}",
                )
        else:
            # 纯自游走采样（用当前策略 rollout，per-timestep × N-agent segment）
            segment = _rollout_with_policy(
                wrapper, obs_builder, reward_builder, net, device,
                trainer.rollout_segment_len,
            )

        if segment is not None:
            trainer.add_rollout_samples(segment, trainer.train_step)
            metrics = trainer.train_on_segment()

            if (seg + 1) % 20 == 0:
                print(f"  seg {seg+1}/{args.num_segments}: "
                      f"loss={metrics.get('loss_total', 0):.4f}, "
                      f"expert_p={metrics.get('expert_prob', 0):.2f}, "
                      f"margin_lambda={metrics.get('lambda_margin', 0):.3f}")

    save_checkpoint(trainer.state_dict(), args.output, "finetuned")
    print(f"Checkpoint saved to {args.output}")


def _rollout_with_policy(wrapper, obs_builder, reward_builder, net, device, segment_len):
    """
    用当前策略 rollout 一段，返回 per-timestep × N-agent 的 segment dict
    （与 build_dataset_from_rollout / ExpertDemoBuffer 同构，供通信训练）。

    自游走样本：actions=策略动作，margin_excluded=True、margin_weight=0（不进 margin）。
    finished agent：坐标置远哨兵 + valid=False（通信排除 + loss 零权屏蔽）。
    """
    from buffers.expert_demo_buffer import FINISHED_AGENT_SENTINEL

    N = wrapper.num_agents
    K = wrapper.obs_radius * 2 + 1

    obs_builder.reset()
    obs_list, _ = wrapper.reset()
    obstacles = wrapper.global_grid
    prev_actions = np.zeros(N, dtype=np.int64)
    hidden = net.init_hidden(1, N, device)
    net.reset_comm_state()   # episode 起始重置通信迟滞

    T = segment_len
    S_patch  = np.zeros((T, N, 4, K, K), dtype=np.float32)
    S_scalar = np.zeros((T, N, 8),        dtype=np.float32)
    S_pos    = np.full((T, N, 2), FINISHED_AGENT_SENTINEL, dtype=np.float32)
    S_action = np.zeros((T, N),           dtype=np.int64)
    S_reward = np.zeros((T, N),           dtype=np.float32)
    S_done   = np.zeros((T, N),           dtype=bool)
    S_valid  = np.zeros((T, N),           dtype=bool)
    S_mask   = np.ones((T, N, 5),         dtype=bool)

    ever_done = [False] * N
    prev_goals = wrapper.get_targets_xy()

    for t in range(T):
        prev_ever_done = list(ever_done)
        patches, scalars = build_observations(wrapper, obs_list, prev_actions, obs_builder)
        agents_xy = wrapper.get_agents_xy()

        if obstacles is not None:
            agents_arr = np.array(agents_xy, dtype=np.int32)[np.newaxis]
            mask_np = build_action_mask_batch(agents_arr, obstacles)[0]   # (N, 5) bool
        else:
            mask_np = np.ones((N, 5), dtype=bool)

        # 坐标：finished agent 置远哨兵（通信排除），与 train 一致
        pos = np.array(agents_xy, dtype=np.float32)
        for i in range(N):
            if prev_ever_done[i]:
                pos[i] = FINISHED_AGENT_SENTINEL

        S_patch[t]  = patches
        S_scalar[t] = scalars
        S_pos[t]    = pos
        S_mask[t]   = mask_np
        for i in range(N):
            S_valid[t, i] = not prev_ever_done[i]

        # 策略前向（推理时施加掩码选动作；传真实多体坐标，通信生效）
        patches_t   = torch.from_numpy(patches).unsqueeze(0).unsqueeze(1).to(device)
        scalars_t   = torch.from_numpy(scalars).unsqueeze(0).unsqueeze(1).to(device)
        positions_t = torch.from_numpy(pos).unsqueeze(0).unsqueeze(1).to(device)
        masks_t     = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values, hidden, _ = net(patches_t, scalars_t, positions_t, hidden,
                                      action_masks=masks_t)
        actions = q_values.squeeze(0).squeeze(0).argmax(dim=-1).cpu().numpy()
        S_action[t] = actions

        obs_list, rewards, dones, _ = wrapper.step(actions.tolist())

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
                prev_dones=prev_ever_done,
            )
            S_reward[t] = np.asarray(shaped_rewards, dtype=np.float32)
        else:
            S_reward[t] = np.asarray([float(r) for r in rewards], dtype=np.float32)

        step_dones = dones if isinstance(dones, list) else [bool(dones)] * N
        for i in range(N):
            if step_dones[i]:
                ever_done[i] = True
        S_done[t] = np.array(ever_done, dtype=bool)

        prev_actions = actions
        prev_goals   = curr_goals

    if not S_valid.any():
        return None

    # next_* 移位一帧（末帧零/哨兵/全合法）
    next_patch  = np.zeros_like(S_patch)
    next_scalar = np.zeros_like(S_scalar)
    next_pos    = np.full_like(S_pos, FINISHED_AGENT_SENTINEL)
    next_mask   = np.ones_like(S_mask)
    if T > 1:
        next_patch[:-1]  = S_patch[1:]
        next_scalar[:-1] = S_scalar[1:]
        next_pos[:-1]    = S_pos[1:]
        next_mask[:-1]   = S_mask[1:]

    return {
        "obs_patches":       S_patch,
        "obs_scalars":       S_scalar,
        "positions":         S_pos,
        "next_obs_patches":  next_patch,
        "next_obs_scalars":  next_scalar,
        "next_positions":    next_pos,
        "actions_expert":    S_action,
        "rewards":           S_reward,
        "dones":             S_done,
        "valid":             S_valid,
        "action_masks":      S_mask,
        "next_action_masks": next_mask,
        "margin_excluded":   np.ones((T, N), dtype=bool),    # 自游走不进 margin
        "margin_weights":    np.zeros((T, N), dtype=np.float32),
    }


if __name__ == "__main__":
    main()
