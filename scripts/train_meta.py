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
from scripts.generate_lacam_data import generate_data, load_buffer_from_disk


def _load_or_generate(data_dir: str, map_type: str,
                      num_agents: int, map_size: int) -> ExpertDemoBuffer:
    """
    优先从 data_dir/expert_<map_type>.npz 加载预生成数据；
    文件不存在时退为小规模在线生成（开发/调试用）。
    """
    npz = os.path.join(data_dir, f"expert_{map_type}.npz")
    buf = ExpertDemoBuffer(capacity=2_000_000)

    if os.path.exists(npz):
        n = load_buffer_from_disk(npz, buf)
        if n > 0:
            return buf
        print(f"  [warn] {npz} 存在但为空，回退到在线生成")

    # fallback：小规模生成（仅供调试）
    print(f"  [fallback] 未找到 {npz}，在线生成 20 episodes (4 agents, 16x16) ...")
    gen_buf, _, _ = generate_data(
        config_path=f"configs/env_{map_type}.yaml",
        num_episodes=20, num_agents=4, map_size=16,
        output_dir=data_dir, save_to_disk=False,
    )
    for s in list(gen_buf._buffer):
        buf.add(s)
    return buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         type=str, default="configs/train_reptile_dqfd.yaml")
    parser.add_argument("--network-config", type=str, default="configs/network_biintent_drqn.yaml")
    parser.add_argument("--meta-iters",     type=int, default=200)
    parser.add_argument("--device",         type=str, default="cuda:0")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--data-dir",       type=str, default="data",
                        help="阶段一 generate_lacam_data.py 的输出目录（含 expert_*.npz）")
    parser.add_argument("--log-interval",   type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.network_config) as f:
        net_cfg = yaml.safe_load(f)

    # ---- 加载专家数据（四种地图族合并到一个 buffer）----
    print(f"\nLoading expert data from '{args.data_dir}' ...")
    expert_buf = ExpertDemoBuffer(capacity=4_000_000)  # 四种地图族合计上限
    total_loaded = 0
    for map_type in ["random", "warehouse", "maze", "room"]:
        per_buf = _load_or_generate(args.data_dir, map_type,
                                    num_agents=cfg.get("num_agents", 96),
                                    map_size=64)
        for s in list(per_buf._buffer):
            expert_buf.add(s)
        total_loaded += len(per_buf)
    print(f"Expert buffer: {len(expert_buf):,} samples (from {total_loaded:,} loaded)")

    # ---- 网络 ----
    net = BiIntentGATDRQN(
        patch_channels   =net_cfg.get("patch_channels",    4),
        cnn_channels     =net_cfg.get("cnn_channels",      [32, 64, 64]),
        d_patch          =net_cfg.get("d_patch",            256),
        scalar_feat_dim  =net_cfg.get("scalar_feat_dim",   8),
        d_scalar         =net_cfg.get("d_scalar",          32),
        intent_dim       =net_cfg.get("intent_dim",        64),
        comm_dim         =net_cfg.get("comm_dim",          64),
        comm_radius      =net_cfg.get("comm_radius",       4),
        comm_topk_cap    =net_cfg.get("comm_topk_cap",     6),
        num_attention_heads=net_cfg.get("num_attention_heads", 4),
        entmax_alpha     =net_cfg.get("entmax_alpha",      1.5),
        tau_in           =net_cfg.get("tau_in",            0.05),
        tau_out          =net_cfg.get("tau_out",           0.02),
        stickiness_bonus =net_cfg.get("stickiness_bonus",  0.1),
        ema_decay        =net_cfg.get("ema_decay",         0.0),
        gru_hidden_dim   =net_cfg.get("gru_hidden_dim",    256),
        dueling_hidden_dim=net_cfg.get("dueling_hidden_dim", 256),
    )
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Network: {n_params:,} parameters ({n_params/1e6:.2f}M)")

    trainer    = ReptileDQfDTrainer(net, cfg, device=device)
    sp_buf     = SelfPlayBuffer(capacity=500_000)

    # ---- 训练循环 ----
    print(f"\nStarting training: {args.meta_iters} meta-iterations")
    for i in range(args.meta_iters):
        metrics = trainer.meta_step(expert_buf, sp_buf, env_factory=None)

        if (i + 1) % args.log_interval == 0:
            print(f"  iter {i+1:>6}/{args.meta_iters}: "
                  f"loss={metrics['inner_loss_mean']:.4f}  "
                  f"map={metrics['map_type']:<10}  "
                  f"er={metrics['expert_ratio']:.2f}")

    # ---- 保存 checkpoint ----
    os.makedirs("checkpoints", exist_ok=True)
    save_checkpoint(trainer.state_dict(), "checkpoints/meta_init.pt", "meta")
    print("Checkpoint saved to checkpoints/meta_init.pt")


if __name__ == "__main__":
    main()
