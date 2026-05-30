#!/usr/bin/env python3
"""阶段二：DQfD + Reptile 元预训练（单 GPU）。"""
import sys, os, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.biintent_drqn import BiIntentGATDRQN
from trainers.train_reptile_dqfd import ReptileDQfDTrainer
from buffers.expert_demo_buffer import ExpertDemoBuffer
from buffers.selfplay_buffer import SelfPlayBuffer
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint
from scripts.generate_lacam_data import generate_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_reptile_dqfd.yaml")
    parser.add_argument("--network-config", type=str, default="configs/network_biintent_drqn.yaml")
    parser.add_argument("--meta-iters", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.network_config) as f:
        net_cfg = yaml.safe_load(f)

    # 生成专家数据（小规模）
    print("Generating expert data...")
    for map_type in ["random", "warehouse", "maze", "room"]:
        generate_data(
            config_path=f"configs/env_{map_type}.yaml",
            num_episodes=20, num_agents=4, map_size=16,
            output_dir="data",
        )

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

    # 简化训练：用 dummy buffer 验证流程
    trainer = ReptileDQfDTrainer(net, cfg, device=device)

    # 创建 dummy buffers
    from envs.pogema_env_wrapper import PogemaWrapper
    from envs.observation_builder import ObservationBuilder
    from lacam.lacam_dataset_builder import build_dataset_from_rollout
    from lacam.lacam_runner import LaCAMRunner
    from envs.reward_builder import RewardBuilder

    expert_buf = ExpertDemoBuffer(capacity=5000)
    obs_builder = ObservationBuilder(num_agents=4, obs_radius=5)
    reward_builder = RewardBuilder(lambda_f=0.1)
    runner = LaCAMRunner(backend="prioritized")

    # 生成一些专家数据
    wrapper = PogemaWrapper("configs/env_random.yaml")
    wrapper._raw_cfg.update(size=16, num_agents=4)
    wrapper.map_size = 16
    wrapper.num_agents = 4

    for ep in range(30):
        wrapper.reset(seed=50000 + ep)
        obstacles = wrapper.global_grid
        starts = wrapper.get_agents_xy()
        goals = wrapper.get_targets_xy()
        result = runner.solve(obstacles, starts, goals,
                              map_type="random", map_size=16)
        if result.success:
            wrapper.reset(seed=50000 + ep)
            samples = build_dataset_from_rollout(
                wrapper, obs_builder, reward_builder, obstacles,
                result.paths, "random", f"meta_{ep}"
            )
            expert_buf.add_batch(samples)

    print(f"Expert buffer: {len(expert_buf)} samples")

    # 训练循环
    print(f"\nStarting training: {args.meta_iters} meta-iterations")
    for i in range(args.meta_iters):
        metrics = trainer.meta_step(expert_buf, None, None)

        if (i + 1) % 20 == 0:
            print(f"  iter {i+1}/{args.meta_iters}: "
                  f"loss={metrics['inner_loss_mean']:.4f}, "
                  f"map={metrics['map_type']}, "
                  f"expert_ratio={metrics['expert_ratio']:.2f}")

    # 保存 checkpoint
    save_checkpoint(trainer.state_dict(), "checkpoints/meta_init.pt", "meta")
    print("Checkpoint saved to checkpoints/meta_init.pt")


if __name__ == "__main__":
    main()
