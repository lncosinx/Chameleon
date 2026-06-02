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


def _copy_columnar_buffer(src: ExpertDemoBuffer, dst: ExpertDemoBuffer, n: int):
    """将 src 的前 n 条数据批量写入 dst（列式 buffer 直接数组拷贝）。"""
    if n == 0 or not src._arrays_inited:
        return
    if not dst._arrays_inited:
        dst._init_arrays(src.obs_channels, src.patch_size)
    n = min(n, src._size, dst.capacity)
    head = dst._head
    cap = dst.capacity
    if head + n <= cap:
        dst._obs_patches[head:head+n]       = src._obs_patches[:n]
        dst._next_obs_patches[head:head+n]  = src._next_obs_patches[:n]
        dst._obs_scalars[head:head+n]       = src._obs_scalars[:n]
        dst._next_obs_scalars[head:head+n]  = src._next_obs_scalars[:n]
        dst._actions_expert[head:head+n]    = src._actions_expert[:n]
        dst._rewards[head:head+n]           = src._rewards[:n]
        dst._dones[head:head+n]             = src._dones[:n]
        dst._action_masks[head:head+n]      = src._action_masks[:n]
        dst._next_action_masks[head:head+n] = src._next_action_masks[:n]
        dst._margin_excluded[head:head+n]   = src._margin_excluded[:n]
        dst._margin_weights[head:head+n]    = src._margin_weights[:n]
    else:
        first = cap - head
        second = n - first
        for sa, da in [
            ("_obs_patches", "_obs_patches"), ("_next_obs_patches", "_next_obs_patches"),
            ("_obs_scalars", "_obs_scalars"), ("_next_obs_scalars", "_next_obs_scalars"),
            ("_actions_expert", "_actions_expert"), ("_rewards", "_rewards"),
            ("_dones", "_dones"), ("_action_masks", "_action_masks"),
            ("_next_action_masks", "_next_action_masks"),
            ("_margin_excluded", "_margin_excluded"), ("_margin_weights", "_margin_weights"),
        ]:
            getattr(dst, da)[head:cap]   = getattr(src, sa)[:first]
            getattr(dst, da)[0:second]   = getattr(src, sa)[first:n]
    dst._head = (head + n) % cap
    dst._size = min(dst._size + n, cap)


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

    # ---- 加载专家数据 ----
    # 每种地图族各取 1M 条（共 4M），保证四种地图平衡分布。
    # 使用新的列式 buffer 直接写入，不再经过 per-map 中间 buffer。
    map_types = ["random", "warehouse", "maze", "room"]
    total_cap = cfg.get("expert_buffer_capacity", 4_000_000)
    per_map_cap = total_cap // len(map_types)   # 每种地图各 1M
    expert_buf = ExpertDemoBuffer(capacity=total_cap)

    print(f"\nLoading expert data from '{args.data_dir}' ...")
    for map_type in map_types:
        npz = os.path.join(args.data_dir, f"expert_{map_type}.npz")
        if os.path.exists(npz):
            # 直接批量加载到 expert_buf，限制每种地图最多 per_map_cap 条
            tmp = ExpertDemoBuffer(capacity=per_map_cap)
            from scripts.generate_lacam_data import load_buffer_from_disk
            load_buffer_from_disk(npz, tmp)
            # 将 tmp 的列式数据批量拷贝到 expert_buf
            n = len(tmp)
            if n > 0:
                _copy_columnar_buffer(tmp, expert_buf, n)
        else:
            print(f"  [fallback] {npz} not found, generating inline ...")
            import yaml as _yaml
            cfg_env = f"configs/env_{map_type}.yaml"
            gen_buf, _, _ = generate_data(
                config_path=cfg_env,
                num_episodes=20, num_agents=4, map_size=16,
                output_dir=args.data_dir, save_to_disk=False,
            )
            n = len(gen_buf)
            if n > 0:
                _copy_columnar_buffer(gen_buf, expert_buf, n)

    print(f"Expert buffer: {len(expert_buf):,} samples")

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
