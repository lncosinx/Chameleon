#!/usr/bin/env python3
"""
阶段二：DQfD + Reptile 元预训练。

自动选择训练器：
- 2 GPU 可用且 --gpus 指定两卡 → DualGPUReptileTrainer（task-parallel，§4.4）
- 否则                          → ReptileDQfDTrainer（单 GPU）
"""
import sys, os, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from models.biintent_drqn import BiIntentGATDRQN
from trainers.train_reptile_dqfd import ReptileDQfDTrainer, DualGPUReptileTrainer
from buffers.expert_demo_buffer import ExpertDemoBuffer
from buffers.selfplay_buffer import SelfPlayBuffer
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint
from scripts.generate_lacam_data import generate_data, load_buffer_from_disk


def _peek_npz_timesteps(npz_path: str) -> int:
    """读取 npz 的 episode_ids 长度（时间步数），用于精确分配 buffer 容量。"""
    data = np.load(npz_path, allow_pickle=False)
    if "episode_ids" not in data.files:
        return -1   # 旧格式标记
    return int(len(data["episode_ids"]))


def _build_network(net_cfg: dict) -> BiIntentGATDRQN:
    return BiIntentGATDRQN(
        patch_channels    = net_cfg.get("patch_channels",     4),
        cnn_channels      = net_cfg.get("cnn_channels",       [32, 64, 64]),
        d_patch           = net_cfg.get("d_patch",             256),
        scalar_feat_dim   = net_cfg.get("scalar_feat_dim",    8),
        d_scalar          = net_cfg.get("d_scalar",           32),
        intent_dim        = net_cfg.get("intent_dim",         64),
        comm_dim          = net_cfg.get("comm_dim",           64),
        comm_radius       = net_cfg.get("comm_radius",        4),
        comm_topk_cap     = net_cfg.get("comm_topk_cap",      6),
        num_attention_heads=net_cfg.get("num_attention_heads", 4),
        entmax_alpha      = net_cfg.get("entmax_alpha",       1.5),
        tau_in            = net_cfg.get("tau_in",             0.05),
        tau_out           = net_cfg.get("tau_out",            0.02),
        stickiness_bonus  = net_cfg.get("stickiness_bonus",   0.1),
        ema_decay         = net_cfg.get("ema_decay",          0.0),
        gru_hidden_dim    = net_cfg.get("gru_hidden_dim",     256),
        dueling_hidden_dim= net_cfg.get("dueling_hidden_dim", 256),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         type=str, default="configs/train_reptile_dqfd.yaml")
    parser.add_argument("--network-config", type=str, default="configs/network_biintent_drqn.yaml")
    parser.add_argument("--meta-iters",     type=int, default=200)
    parser.add_argument("--gpus",           type=str, default="auto",
                        help="GPU 设备列表，逗号分隔 (e.g. '0,1') 或 'auto'（自动检测）")
    parser.add_argument("--data-dir",       type=str, default="data",
                        help="阶段一输出目录（含 expert_*.npz）")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--log-interval",   type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config)         as f: cfg     = yaml.safe_load(f)
    with open(args.network_config) as f: net_cfg = yaml.safe_load(f)

    # ---- 确定使用哪些 GPU ----
    n_cuda = torch.cuda.device_count()
    if args.gpus == "auto":
        cfg_gpus = cfg.get("parallel", {}).get("task_parallel_gpus", [0, 1])
        requested = [torch.device(f"cuda:{g}") for g in cfg_gpus]
    else:
        requested = [torch.device(f"cuda:{g.strip()}") for g in args.gpus.split(",")]

    available = [d for d in requested if d.index < n_cuda] if n_cuda > 0 else []

    use_dual = len(available) >= 2
    if use_dual:
        devices   = available[:2]
        device_str = f"GPU {devices[0].index} + GPU {devices[1].index} (task-parallel)"
    elif len(available) == 1:
        devices   = available[:1]
        device_str = f"GPU {devices[0].index} (single)"
    else:
        devices   = [torch.device("cpu")]
        device_str = "CPU"

    print(f"Device: {device_str}")

    # ---- 加载专家数据（per-timestep × N-agent 格式）----
    # 单一混合 buffer，四种地图随机混采。容量按时间步计，先 peek 各 npz 长度精确分配。
    # 注意：所有地图必须用相同 num_agents 生成（buffer 要求 N 一致）。
    from scripts.generate_lacam_data import load_buffer_from_disk
    map_types = ["random", "warehouse", "maze", "room"]

    print(f"\nLoading expert data from '{args.data_dir}' ...")
    present = []
    total_steps = 0
    for map_type in map_types:
        npz = os.path.join(args.data_dir, f"expert_{map_type}.npz")
        if not os.path.exists(npz):
            print(f"  [skip] {npz} not found — train on remaining maps. "
                  f"Generate it with: python scripts/generate_lacam_data.py "
                  f"--config configs/env_{map_type}.yaml --num-workers 47")
            continue
        n = _peek_npz_timesteps(npz)
        if n < 0:
            raise ValueError(
                f"{npz} is OLD per-agent format (no positions/episode_ids). "
                f"Communication-aware training requires regenerated data. "
                f"Run: python scripts/generate_lacam_data.py "
                f"--config configs/env_{map_type}.yaml --num-workers 47"
            )
        present.append(npz)
        total_steps += n

    if not present:
        raise FileNotFoundError(
            f"No expert data found in '{args.data_dir}'. Generate it first "
            f"(see scripts/generate_lacam_data.py)."
        )

    # 容量上限（时间步），可由 config 收紧以控显存/内存；默认装下全部
    cap = min(total_steps, cfg.get("expert_buffer_capacity_steps", total_steps))
    expert_buf = ExpertDemoBuffer(capacity=max(cap, 1))
    for npz in present:
        load_buffer_from_disk(npz, expert_buf)

    print(f"Expert buffer: {len(expert_buf):,} timesteps × {expert_buf.num_agents} agents "
          f"({expert_buf.num_episodes} episodes)")

    sp_buf = SelfPlayBuffer(capacity=500_000)

    # ---- 构建训练器 ----
    net = _build_network(net_cfg)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Network: {n_params:,} parameters ({n_params/1e6:.2f}M)")

    if use_dual:
        trainer = DualGPUReptileTrainer(net, cfg, devices=[str(d) for d in devices])
        print(f"Trainer: DualGPUReptileTrainer  (task-parallel, {len(devices)} GPUs)")
    else:
        trainer = ReptileDQfDTrainer(net, cfg, device=devices[0])
        print(f"Trainer: ReptileDQfDTrainer     (single device: {devices[0]})")

    # ---- 训练循环 ----
    print(f"\nStarting training: {args.meta_iters} meta-iterations\n")
    for i in range(args.meta_iters):
        metrics = trainer.meta_step(expert_buf, sp_buf, env_factory=None)

        if (i + 1) % args.log_interval == 0:
            # 兼容单/双 GPU 返回的 map_type vs map_types
            mt = metrics.get("map_types", [metrics.get("map_type", "?")])
            mt_str = "+".join(mt) if isinstance(mt, list) else str(mt)
            n_tasks = metrics.get("num_tasks", metrics.get("num_gpus", 1))
            meta_lr = metrics.get("meta_lr", cfg.get("meta_lr_start", 0.0))
            print(f"  iter {i+1:>6}/{args.meta_iters}: "
                  f"loss={metrics['inner_loss_mean']:.4f}  "
                  f"map={mt_str:<20}  "
                  f"er={metrics['expert_ratio']:.2f}  "
                  f"tasks={n_tasks}  meta_lr={meta_lr:.5f}")

    # ---- 保存 checkpoint ----
    os.makedirs("checkpoints", exist_ok=True)
    save_checkpoint(trainer.state_dict(), "checkpoints/meta_init.pt", "meta")
    print("\nCheckpoint saved to checkpoints/meta_init.pt")


if __name__ == "__main__":
    main()
