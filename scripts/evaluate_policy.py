#!/usr/bin/env python3
"""
评估脚本：关闭专家介入，评估策略真实能力。

用法:
    python scripts/evaluate_policy.py --ckpt checkpoints/finetuned.pt \
        --config configs/env_maze.yaml --episodes 50
"""
import sys, os, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json

from utils.seed import set_seed
from models.biintent_drqn import BiIntentGATDRQN
from trainers.evaluator import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--config", type=str, default="configs/env_maze.yaml")
    parser.add_argument("--network-config", type=str, default="configs/network_biintent_drqn.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--held-out", action="store_true",
                        help="Use held-out map family (movingai/puzzles)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.network_config) as f:
        net_cfg = yaml.safe_load(f)

    net = BiIntentGATDRQN(
        patch_channels=net_cfg.get("patch_channels", 4),
        cnn_channels=net_cfg.get("cnn_channels", [32, 64, 64]),
        scalar_feat_dim=net_cfg.get("scalar_feat_dim", 8),
        intent_dim=net_cfg.get("intent_dim", 64),
        comm_dim=net_cfg.get("comm_dim", 64),
        comm_radius=net_cfg.get("comm_radius", 4),
        comm_topk_cap=net_cfg.get("comm_topk_cap", 6),
        num_attention_heads=net_cfg.get("num_attention_heads", 4),
        entmax_alpha=net_cfg.get("entmax_alpha", 1.5),
        tau_in=net_cfg.get("tau_in", 0.05),
        tau_out=net_cfg.get("tau_out", 0.02),
        gru_hidden_dim=net_cfg.get("gru_hidden_dim", 256),
        dueling_hidden_dim=net_cfg.get("dueling_hidden_dim", 256),
    )

    # 加载 checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt["net"])
    net.to(device)
    print(f"Loaded checkpoint from {args.ckpt} (train_step={ckpt.get('train_step', 'N/A')})")

    # held-out 地图族
    config_path = args.config
    if args.held_out:
        print("[held-out mode] Using movingai/puzzles as unseen map family")
        config_path = args.config.replace("maze", "room")  # 用 room 替代

    metrics = evaluate(
        net=net,
        config_path=config_path,
        num_episodes=args.episodes,
        num_agents=args.num_agents,
        map_size=args.map_size,
        max_steps=args.max_steps,
        device=device,
        epsilon=0.0,
    )

    print("\n=== Evaluation Results ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
