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


def _build_network(net_cfg: dict, use_torch_compile: bool = False) -> BiIntentGATDRQN:
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
        use_torch_compile = use_torch_compile,
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
    parser.add_argument("--ckpt-interval",  type=int, default=1000,
                        help="每 N 个 meta-iter 原子写一次 checkpoints/meta_latest.pt（断点续跑用）")
    parser.add_argument("--resume",         type=str, default="none",
                        help="'none'=从头训；'auto'=若有 checkpoints/meta_latest.pt 则续；或指定 ckpt 路径")
    parser.add_argument("--ckpt-milestones", type=str, default="",
                        help="逗号分隔的 meta-iter 里程碑，在这些点另存 checkpoints/meta_iter_{N}.pt（SR-vs-iters 曲线用）")
    parser.add_argument("--no-selfplay", action="store_true",
                        help="强制关闭自游走（隔离测 DQfD 核心；覆盖 config 的 selfplay.enable）")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                        help="覆盖 config 顶层键（可多次，值经 yaml 解析），用于实验不污染 config。"
                             "如 --set lambda_td=0 --set inner_lr=0.003")
    parser.add_argument("--bc-iters", type=int, default=0,
                        help="两阶段：前 N 个 meta-iter 纯 BC（强制 lambda_td=lambda_nstep=0，"
                             "只学专家 margin 排序），之后恢复全 λ 做 DQfD 价值 grounding。0=关闭。")
    args = parser.parse_args()
    milestones = {int(x) for x in args.ckpt_milestones.split(",") if x.strip()}

    set_seed(args.seed)

    with open(args.config)         as f: cfg     = yaml.safe_load(f)
    with open(args.network_config) as f: net_cfg = yaml.safe_load(f)
    if args.no_selfplay:
        cfg.setdefault("selfplay", {})["enable"] = False
    # BC 步数注入 cfg → trainer 用它把 expert_ratio 衰减时钟偏移 bc_iters
    # （BC 期 er 冻结在 start，self-play 从 iter=bc_iters 起才温和衰减；见 scheduler.expert_ratio）。
    cfg["bc_iters"] = args.bc_iters
    for kv in args.set:
        k, _, v = kv.partition("=")
        val = yaml.safe_load(v)
        keys = k.strip().split(".")          # 支持嵌套键，如 selfplay.families
        d = cfg
        for kk in keys[:-1]:
            d = d.setdefault(kk, {})
        d[keys[-1]] = val
        print(f"[override] {k.strip()} = {val}")

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

    # 自游走段是 (T,N,...) 大对象（每段 ~十几 MB），容量按「段数」计，
    # 勿沿用单步样本的 500k（会占 TB 级内存）。
    sp_buf = SelfPlayBuffer(
        capacity=cfg.get("selfplay", {}).get("buffer_capacity_segments", 512)
    )

    # ---- 构建训练器 ----
    _use_compile = bool(cfg.get("parallel", {}).get("use_torch_compile", False))
    net = _build_network(net_cfg, use_torch_compile=_use_compile)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Network: {n_params:,} parameters ({n_params/1e6:.2f}M)")

    if use_dual:
        trainer = DualGPUReptileTrainer(net, cfg, devices=[str(d) for d in devices])
        print(f"Trainer: DualGPUReptileTrainer  (task-parallel, {len(devices)} GPUs)")
    else:
        trainer = ReptileDQfDTrainer(net, cfg, device=devices[0])
        print(f"Trainer: ReptileDQfDTrainer     (single device: {devices[0]})")

    # ---- 自游走 producer（阶段二 off-policy 自游走，填充 sp_buf；enable=False 时全 no-op）----
    from trainers.selfplay_producer import SelfPlayProducer
    sp_producer = SelfPlayProducer(net, cfg, sp_buf, device=devices[0])
    if sp_producer.enabled:
        print(f"SelfPlay: ON (async overlap) | families={sp_producer.families} | "
              f"{sp_producer.n_env}env/family × {sp_producer.num_agents}agents "
              f"@size{sp_producer.map_size} | sync every {sp_producer.K} meta-iters "
              f"| buffer cap={sp_buf._buffer.maxlen} segs")
    else:
        print("SelfPlay: OFF (selfplay.enable=false)")
    # 后台 rollout 线程：与 inner loop 重叠（rollout 是 CPU/延迟 bound，藏进训练 GPU 空泡）。
    sp_producer.start()

    # ---- 断点续跑 ----
    os.makedirs("checkpoints", exist_ok=True)
    latest_path = "checkpoints/meta_latest.pt"
    if args.resume and args.resume != "none":
        resume_path = latest_path if args.resume == "auto" else args.resume
        if os.path.exists(resume_path):
            from utils.checkpoint import load_checkpoint
            trainer.load_state_dict(load_checkpoint(resume_path))
            print(f"Resumed from {resume_path} at meta_iter={trainer.meta_iter}")
        elif args.resume == "auto":
            print(f"No {latest_path}; starting fresh.")
        else:
            raise FileNotFoundError(f"--resume {resume_path} not found")

    def _atomic_save(state, path):
        """先写 .tmp 再 os.replace，避免写一半被杀导致 ckpt 损坏。"""
        save_checkpoint(state, path + ".tmp", "meta")
        os.replace(path + ".tmp", path)

    # ---- 两阶段（--bc-iters）：前 N iter 纯 BC（lambda_td=lambda_nstep=0），之后恢复全 λ ----
    # 设在 trainer（DualGPU 则各 sub_trainer）的 lambda_td/lambda_nstep 上，_inner_step 即时生效。
    _cfg_lam_td = cfg.get("lambda_td", 1.0)
    _cfg_lam_n  = cfg.get("lambda_nstep", 1.0)
    def _set_td_lambdas(lam_td, lam_n):
        for t in (getattr(trainer, "sub_trainers", None) or [trainer]):
            t.lambda_td = lam_td
            t.lambda_nstep = lam_n

    # ---- 训练循环（按 trainer.meta_iter 驱动，支持续跑）----
    start_iter = trainer.meta_iter
    print(f"\nStarting training: {start_iter} → {args.meta_iters} meta-iterations"
          + (f"  (BC pretrain 前 {args.bc_iters} iter)" if args.bc_iters > 0 else "") + "\n")
    _prev_bc = None
    import time as _time
    _last_log_t = _time.perf_counter()
    for i in range(start_iter, args.meta_iters):
        in_bc = i < args.bc_iters
        if in_bc != _prev_bc:
            _set_td_lambdas(0.0, 0.0) if in_bc else _set_td_lambdas(_cfg_lam_td, _cfg_lam_n)
            print(f"  [phase] {'BC (margin only)' if in_bc else 'DQfD (full λ, +TD)'} @ iter {i}", flush=True)
            _prev_bc = in_bc
        metrics = trainer.meta_step(expert_buf, sp_buf, env_factory=None)

        # 自游走（异步重叠）：每 K 个 meta-iter 非阻塞地把**当前** θ_meta 快照推给后台线程，
        # 由它在训练 GPU 空泡里 rollout 填 sp_buf（不再阻塞主循环）。BC 期 TD 关、自游走
        # TD-only loss 贡献为 0，跳过 sync 省算力（sp_buf 空时消费端自然 no-op）。
        if sp_producer.enabled and not in_bc and (i + 1) % sp_producer.K == 0:
            theta_meta = trainer._meta_state if hasattr(trainer, "_meta_state") else trainer.net.state_dict()
            sp_producer.sync_weights(theta_meta)

        if (i + 1) % args.log_interval == 0:
            # 兼容单/双 GPU 返回的 map_type vs map_types
            mt = metrics.get("map_types", [metrics.get("map_type", "?")])
            mt_str = "+".join(mt) if isinstance(mt, list) else str(mt)
            n_tasks = metrics.get("num_tasks", metrics.get("num_gpus", 1))
            meta_lr = metrics.get("meta_lr", cfg.get("meta_lr_start", 0.0))
            _now = _time.perf_counter()
            _spi = (_now - _last_log_t) / args.log_interval   # 墙钟 s/iter（含 rollout 重叠后净值）
            _last_log_t = _now
            # loss 分量（加权贡献，应约等于 loss_total）。BC 期 td/n 的有效 λ=0，显示按有效值。
            _lt = 0.0 if in_bc else cfg.get("lambda_td", 1.0)
            _ln = 0.0 if in_bc else cfg.get("lambda_nstep", 1.0)
            w_td  = _lt * metrics.get("loss_td", 0.0)
            w_n   = _ln * metrics.get("loss_nstep", 0.0)
            w_mgn = cfg.get("lambda_margin", 0.5) * metrics.get("loss_margin", 0.0)
            w_l2  = cfg.get("lambda_l2", 1e-4)    * metrics.get("l2_reg", 0.0)
            print(f"  iter {i+1:>6}/{args.meta_iters}: "
                  f"loss={metrics['inner_loss_mean']:.4f}  "
                  f"Qmax={metrics.get('q_max', 0.0):.2f}  "
                  f"Qmean={metrics.get('q_mean', 0.0):.2f}  "
                  f"r={metrics.get('r_mean', 0.0):.2f}/{metrics.get('r_max', 0.0):.2f}  "
                  f"[td={w_td:.2f} n={w_n:.2f} mgn={w_mgn:.2f} l2={w_l2:.2f}]  "
                  f"map={mt_str:<20}  "
                  f"er={metrics['expert_ratio']:.2f}  "
                  f"sp={len(sp_buf)}  {_spi:.2f}s/it  "
                  f"tasks={n_tasks}  meta_lr={meta_lr:.5f}", flush=True)

        # 周期性原子存档（断点续跑）
        if args.ckpt_interval > 0 and (i + 1) % args.ckpt_interval == 0:
            _atomic_save(trainer.state_dict(), latest_path)
            print(f"  [ckpt] meta_latest.pt @ iter {i+1}", flush=True)

        # 里程碑编号存档（SR-vs-iters 曲线评估用，不被覆盖）
        if (i + 1) in milestones:
            mpath = f"checkpoints/meta_iter_{i+1}.pt"
            save_checkpoint(trainer.state_dict(), mpath, "meta")
            print(f"  [ckpt] {mpath} @ iter {i+1}", flush=True)

    # ---- 保存最终 checkpoint ----
    save_checkpoint(trainer.state_dict(), "checkpoints/meta_init.pt", "meta")
    _atomic_save(trainer.state_dict(), latest_path)
    print("\nCheckpoint saved to checkpoints/meta_init.pt")

    # 清理自游走 worker 进程（spawn 的 CPU 池）
    sp_producer.close()


if __name__ == "__main__":
    main()
