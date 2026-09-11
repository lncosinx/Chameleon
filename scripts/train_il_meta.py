#!/usr/bin/env python3
"""v3 入口：离线 IL + Reptile 元预训练（Transformer 骨干）。取代 train_meta.py（DQfD）。

每个地图族一个独立 buffer（Reptile 任务=族）。held_out_family 留出不进元训练，用于后续
few-shot 适应评估（Reptile 的存在理由）。
"""
import sys, os, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

from models.mapf_comm_net import MAPFCommNet
from buffers.expert_demo_buffer import ExpertDemoBuffer
from scripts.generate_lacam_data import load_buffer_from_disk, expert_data_exists
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint
from trainers.train_il_reptile import ILReptileTrainer, DualGPUILReptileTrainer


def build_net(nc: dict) -> MAPFCommNet:
    return MAPFCommNet(
        patch_channels=nc.get("patch_channels", 4), scalar_feat_dim=nc.get("scalar_feat_dim", 8),
        obs_radius=nc.get("obs_radius", 5), num_actions=nc.get("num_actions", 5),
        d_model=nc.get("d_model", 192), cnn_channels=nc.get("cnn_channels", 64),
        d_scalar=nc.get("d_scalar", 32), n_res_blocks=nc.get("n_res_blocks", 1),
        comm_rounds=nc.get("comm_rounds", 1), intent_dim=nc.get("intent_dim", 128),
        comm_dim=nc.get("comm_dim", 128), comm_radius=nc.get("comm_radius", 4),
        comm_topk_cap=nc.get("comm_topk_cap", 6), entmax_alpha=nc.get("entmax_alpha", 1.5),
        tau_in=nc.get("tau_in", 0.05), tau_out=nc.get("tau_out", 0.02),
        stickiness_bonus=nc.get("stickiness_bonus", 0.1), ema_decay=nc.get("ema_decay", 0.0),
        num_attention_heads=nc.get("num_attention_heads", 4),
        temporal_backend=nc.get("temporal_backend", "mamba"),
        temporal_layers=nc.get("temporal_layers", 2), d_state=nc.get("d_state", 16),
        d_conv=nc.get("d_conv", 4), mamba_expand=nc.get("mamba_expand", 2),
        head_hidden=nc.get("head_hidden", 128),
        spatial_pool=nc.get("spatial_pool", 0),   # 0=全展平（保留空间布局）
        comm_type=nc.get("comm_type", "madcl"),   # SQ3：只换通信模块的消融轴
        intent_mode=nc.get("intent_mode", "feature"),  # SQ3：消息内容（特征 vs 显式动作意图）
        hard_topk_select=nc.get("hard_topk_select", False),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_il_reptile.yaml")
    p.add_argument("--network-config", default="configs/network_gru.yaml")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--meta-iters", type=int, default=20000)
    p.add_argument("--gpus", default="0", help="逗号分隔，如 '0,1' 启用双卡 task-并行")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--ckpt-interval", type=int, default=1000)
    p.add_argument("--ckpt-milestones", type=str, default="",
                   help="逗号分隔 meta-iter 里程碑，另存 checkpoints/il_iter_{N}.pt（SR-vs-iter 用）")
    p.add_argument("--batch-size", type=int, default=None, help="覆盖 config batch_size")
    p.add_argument("--seq-len", type=int, default=None, help="覆盖 config sequence_len")
    p.add_argument("--d4-augment", dest="d4_augment", action="store_true", default=None,
                   help="开启 D4 对称数据增强（M1，等效 8× 数据）")
    p.add_argument("--no-d4-augment", dest="d4_augment", action="store_false",
                   help="关闭 D4 增强（消融对照臂）")
    p.add_argument("--comm-rounds", type=int, default=None, help="覆盖 network comm_rounds（通信消融轴）")
    p.add_argument("--set-net", action="append", default=[], metavar="K=V",
                   help="覆盖 network config 键（消融用，可多次），如 --set-net tau_in=0 --set-net entmax_alpha=1.0")
    p.add_argument("--tag", default="il", help="ckpt 前缀，区分不同 run（里程碑存 checkpoints/{tag}_iter_N.pt）")
    p.add_argument("--init", default=None, help="热启动 ckpt 路径（训练前 load_state_dict，strict=False）")
    p.add_argument("--resume", default=None,
                   help="从 ckpt 断点续训：载入权重**并从它的 meta_iter 继续计数**（里程碑对齐）。"
                        "与 --init 的区别：--init 总是从 0 开始计数。长训练被 OOM/重启打断时用这个。")
    p.add_argument("--reinit-comm", action="store_true",
                   help="载入 --init 后**重新随机初始化通信模块**。SQ3 必须开：否则本文的"
                        "消融臂（参数名相同）会继承已训练好的通信权重，而外部通信基线"
                        "（参数名不同）只能随机起步 —— 那不是唯一变量对照，是给自己开小灶。")
    p.add_argument("--out", default=None)
    p.add_argument("--buffer-mmap-dir", default=None,
                   help="给定则各族 ExpertDemoBuffer 用磁盘内存映射（mmap）存储，不整族占用 "
                        "常驻 RAM——数据量超出可用内存时用（见 CLAUDE.md 服务器迁移记忆：本机 "
                        "内存硬限额 62GB）。默认 None=常驻内存（小数据集更快）。会在此目录下按 "
                        "族建子目录，内容为该次训练的临时磁盘缓存，可随时删除。")
    args = p.parse_args()
    set_seed(args.seed)
    milestones = {int(x) for x in args.ckpt_milestones.split(",") if x.strip()}
    if args.out is None:
        args.out = f"checkpoints/{args.tag}_latest.pt"

    cfg = yaml.safe_load(open(args.config))
    if args.batch_size is not None: cfg["batch_size"] = args.batch_size
    if args.seq_len is not None:    cfg["sequence_len"] = args.seq_len
    if args.d4_augment is not None: cfg["d4_augment"] = args.d4_augment
    nc = yaml.safe_load(open(args.network_config))
    if args.comm_rounds is not None: nc["comm_rounds"] = args.comm_rounds
    for kv in args.set_net:
        k, _, v = kv.partition("="); nc[k.strip()] = yaml.safe_load(v)
        print(f"[set-net] {k.strip()} = {nc[k.strip()]}")
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_cuda = torch.cuda.device_count()
    avail = [g for g in gpu_ids if int(g) < n_cuda] if n_cuda > 0 else []
    use_dual = len(avail) >= 2
    devices = [f"cuda:{g}" for g in avail[:2]] if use_dual else \
              ([f"cuda:{avail[0]}"] if avail else ["cpu"])
    device = torch.device(devices[0])

    held_out = cfg.get("held_out_family")
    fams = [f for f in cfg.get("families", ["random", "warehouse", "maze", "room"]) if f != held_out]

    print(f"Device: {device} | meta-train families: {fams}"
          + (f" | held-out: {held_out}" if held_out else ""))
    def _peek_steps(base_path: str) -> int:
        """新格式（目录）只读 episode_ids.npy 的 shape；旧格式（.npz）读 episode_ids 键。"""
        ep_npy = os.path.join(base_path, "episode_ids.npy")
        if os.path.exists(ep_npy):
            return int(np.load(ep_npy, mmap_mode="r").shape[0])
        npz_path = base_path + ".npz"
        if os.path.exists(npz_path):
            d = np.load(npz_path, allow_pickle=False)
            return int(len(d["episode_ids"])) if "episode_ids" in d.files else -1
        return -1

    family_buffers = {}
    for fam in fams:
        base_path = os.path.join(args.data_dir, f"expert_{fam}")
        if not expert_data_exists(base_path):
            print(f"  [skip] {base_path}(.npz) not found"); continue
        cap = _peek_steps(base_path)
        if cap < 0:
            raise ValueError(f"{base_path} 为旧格式（无 episode_ids），需重新生成")
        mmap_dir = os.path.join(args.buffer_mmap_dir, fam) if args.buffer_mmap_dir else None
        buf = ExpertDemoBuffer(capacity=max(cap, 1), mmap_dir=mmap_dir)   # 按数据实际步数分配容量
        load_buffer_from_disk(base_path, buf)
        family_buffers[fam] = buf
        print(f"  {fam}: {len(buf):,} steps × {buf.num_agents} agents"
              + (f" (mmap: {mmap_dir})" if mmap_dir else ""))
    if not family_buffers:
        raise FileNotFoundError(f"No expert_* data under {args.data_dir}")

    net = build_net(nc)
    n_params = sum(p.numel() for p in net.parameters())
    # 热启动：从已训 ckpt 初始化(如 il_nothing)，把预训练的导航/守位能力带过来，
    # 只在新数据上微调"拥堵协调"。strict=False 容忍 comm_rounds 等结构差异。
    if args.init:
        ck = torch.load(args.init, map_location="cpu")
        state = {k.replace("_orig_mod.", ""): v for k, v in ck["net"].items()}
        if args.reinit_comm:
            # 同时丢掉 comm_in / comm_out（消息的入口/出口投影）。只丢 communication.* 是不够的：
            # 比较不同"消息内容"（intent_mode）时，feature 臂会带着训练好的 comm_in 出发、
            # action 臂因为输入维度变了只能随机初始化 —— 那就不是唯一变量了。两臂都从
            # 全新的通信通路出发，差异才只剩"消息里装什么"。
            state = {k: v for k, v in state.items()
                     if not k.startswith(("communication.", "comm_in.", "comm_out.", "pre_head."))}
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f"[init] warm-start from {args.init} "
              f"(missing={len(missing)} unexpected={len(unexpected)})"
              + ("  [通信模块已重新随机初始化]" if args.reinit_comm else ""))
    print(f"Net: MAPFCommNet {n_params/1e6:.2f}M params | temporal={nc.get('temporal_backend','mamba')} "
          f"| {'DualGPU '+str(devices) if use_dual else 'single '+devices[0]}"
          + (f" | init={args.init}" if args.init else ""))

    trainer = (DualGPUILReptileTrainer(net, cfg, devices=devices) if use_dual
               else ILReptileTrainer(net, cfg, device=device))
    os.makedirs("checkpoints", exist_ok=True)
    import time as _time
    start_iter = 0
    if args.resume:
        rck = torch.load(args.resume, map_location="cpu")
        rstate = {k.replace("_orig_mod.", ""): v for k, v in rck["net"].items()}
        net.load_state_dict(rstate)
        start_iter = int(rck.get("meta_iter", 0))
        trainer.meta_iter = start_iter
        if hasattr(trainer, "_meta_state"):
            trainer._meta_state = {k: v.cpu() for k, v in rstate.items()}
        print(f"[resume] 从 {args.resume} 续训，已完成 {start_iter} iter")

    print(f"\nStart: {start_iter} → {args.meta_iters}\n")
    _last = _time.perf_counter()
    for i in range(start_iter, args.meta_iters):
        m = trainer.meta_step(family_buffers, seed=args.seed + i)
        if (i + 1) % args.log_interval == 0:
            _now = _time.perf_counter(); _spi = (_now - _last) / args.log_interval; _last = _now
            _mode = m.get("mode", "reptile")
            _lrlabel = "inner_lr" if _mode == "joint" else "meta_lr"
            print(f"  iter {i+1:>6}/{args.meta_iters}: loss={m['loss']:.4f} acc={m['acc']:.3f} "
                  f"wait_pred={m.get('wait_pred',0):.3f} edge_flip={m['edge_flip']:.3f} "
                  f"{_spi:.2f}s/it [{_mode}] tasks={'+'.join(m['tasks'])} {_lrlabel}={m['meta_lr']:.4f}", flush=True)
        if args.ckpt_interval > 0 and (i + 1) % args.ckpt_interval == 0:
            save_checkpoint(trainer.state_dict(), args.out, "meta")
        if (i + 1) in milestones:
            # 里程碑 ckpt 与 --out 同目录（checkpoints/ 常为 root 所有不可写）
            mpath = os.path.join(os.path.dirname(args.out) or ".", f"{args.tag}_iter_{i+1}.pt")
            save_checkpoint(trainer.state_dict(), mpath, "meta")
            print(f"  [ckpt] {mpath}", flush=True)
    save_checkpoint(trainer.state_dict(), args.out, "meta")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
