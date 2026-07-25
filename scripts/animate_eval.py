#!/usr/bin/env python3
"""用 checkpoint 在 warehouse 地图跑一遍贪心评估，并用 Pogema AnimationMonitor 录制 SVG 动画。

用法:
    python scripts/animate_eval.py \
        --ckpt checkpoints/meta_iter_20000.pt \
        --config configs/env_warehouse.yaml \
        --num-agents 16 --map-size 32 --max-steps 256 \
        --out renders/warehouse_iter20000.svg

输出一个可在浏览器里播放的 SVG 动画，直观看到各机器人在仓库地图中的移动轨迹。
"""
import sys, os, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pogema import AnimationMonitor, AnimationConfig

from utils.seed import set_seed
from models.biintent_drqn import BiIntentGATDRQN
from envs.pogema_env_wrapper import PogemaWrapper
from envs.observation_builder import ObservationBuilder, build_observations
from envs.action_mask import build_action_mask_batch


def build_net(net_cfg, device):
    net = BiIntentGATDRQN(
        patch_channels=net_cfg.get("patch_channels", 4),
        cnn_channels=net_cfg.get("cnn_channels", [32, 64, 64]),
        d_patch=net_cfg.get("d_patch", 256),
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
        comm_rounds=net_cfg.get("comm_rounds", 1),  # 必须与 ckpt 一致，否则 comm 层数不匹配
    )
    return net.to(device)


def load_state(net, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["net"]
    # torch.compile 保存的权重带 `_orig_mod.` 前缀，去掉后才能装进未编译网络
    sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    print(f"Loaded {ckpt_path}  (meta_iter={ckpt.get('meta_iter', 'N/A')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/meta_iter_20000.pt")
    ap.add_argument("--config", default="configs/env_warehouse.yaml")
    ap.add_argument("--network-config", default="configs/network_biintent_drqn.yaml")
    ap.add_argument("--num-agents", type=int, default=16)
    ap.add_argument("--map-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--comm-rounds", type=int, default=None,
                    help="通信层数，必须与训练时一致（comm0/comm2 与默认 comm1 的 comm 层数不同）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="renders/warehouse_iter20000.svg")
    ap.add_argument("--no-comm", action="store_true",
                    help="消融：将通信特征置零（保留张量形状），看通信对协调的贡献")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.network_config) as f:
        net_cfg = yaml.safe_load(f)
    if args.comm_rounds is not None:
        net_cfg["comm_rounds"] = args.comm_rounds
    net = build_net(net_cfg, device)
    load_state(net, args.ckpt, device)
    net.eval()

    # 消融：把通信特征整段置零（不改形状），等价于"FOV 内不交换任何邻居意图"
    if args.no_comm:
        _orig_comm = net.communication.forward_sequence
        def _zeroed_comm(intent, positions, *a, **k):
            comm_feat, info = _orig_comm(intent, positions, *a, **k)
            return torch.zeros_like(comm_feat), info
        net.communication.forward_sequence = _zeroed_comm
        print("[ablation] 通信已关闭 (comm_feat ≡ 0)")

    # 构建 warehouse 环境，并让底层 pogema env 套上 AnimationMonitor
    wrapper = PogemaWrapper(args.config, split="test")  # 与 evaluator 同用 held-out 测试地图
    wrapper._raw_cfg["size"] = args.map_size
    wrapper._raw_cfg["num_agents"] = args.num_agents
    wrapper.map_size = args.map_size
    wrapper.num_agents = args.num_agents

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    anim_cfg = AnimationConfig(
        directory=os.path.dirname(args.out) or ".",
        show_agents=True,
        show_grid_lines=True,
        save_every_idx_episode=None,  # 关闭自动保存，结束后手动保存
    )

    # 让 _create_env 产出的 pogema env 被 AnimationMonitor 包裹（reset 前的初始帧也会被记录）
    orig_create = wrapper._create_env
    wrapper._create_env = lambda: AnimationMonitor(orig_create(), anim_cfg)

    num_agents = args.num_agents
    obs_builder = ObservationBuilder(num_agents=num_agents, obs_radius=5)
    obs_builder.reset()
    obs_list, _ = wrapper.reset(seed=args.seed)   # 第 args.seed 张测试地图

    prev_actions = np.zeros(num_agents, dtype=np.int64)
    hidden = net.init_hidden(1, num_agents, device)
    net.reset_comm_state()

    reached_steps = [None] * num_agents  # 记录每体首次到达 goal 的步
    goals0 = wrapper.get_targets_xy()

    for step in range(args.max_steps):
        agents_xy = wrapper.get_agents_xy()
        cur_targets = wrapper.get_targets_xy()
        for i in range(num_agents):
            if reached_steps[i] is None and tuple(agents_xy[i]) == tuple(cur_targets[i]):
                reached_steps[i] = step

        patches, scalars = build_observations(wrapper, obs_list, prev_actions, obs_builder)

        obstacles = wrapper.global_grid
        if obstacles is not None:
            agents_arr = np.array(agents_xy, dtype=np.int32)[np.newaxis]
            mask_np = build_action_mask_batch(agents_arr, obstacles)[0]
        else:
            mask_np = np.ones((num_agents, 5), dtype=bool)

        patches_t = torch.from_numpy(patches).unsqueeze(0).to(device)
        scalars_t = torch.from_numpy(scalars).unsqueeze(0).to(device)
        positions_t = torch.tensor(agents_xy, dtype=torch.float32, device=device).unsqueeze(0)
        masks_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values, hidden, _ = net(
                patches_t.unsqueeze(1),
                scalars_t.unsqueeze(1),
                positions_t.unsqueeze(1),
                hidden,
                action_masks=masks_t,
            )
            q_values = q_values.squeeze(0).squeeze(0)

        actions = q_values.argmax(dim=-1).cpu().numpy().tolist()
        obs_list, _, dones, _ = wrapper.step(actions)
        prev_actions = np.array(actions, dtype=np.int64)

        if all(dones):
            print(f"All agents done at step {step + 1}")
            break

    # 保存动画
    wrapper._env.save_animation(args.out)

    n_reached = sum(r is not None for r in reached_steps)
    fam_label = wrapper._raw_cfg.get("map_name", os.path.basename(args.config))
    print(f"\n=== Animation saved: {args.out} ===")
    print(f"  map={fam_label}  comm_rounds={net_cfg.get('comm_rounds', 1)}  "
          f"agents={num_agents}  map_size={args.map_size}  max_steps={args.max_steps}")
    print(f"  到达目标的智能体: {n_reached}/{num_agents}  (success_rate={n_reached/num_agents:.2%})")
    if n_reached:
        avg = np.mean([r for r in reached_steps if r is not None])
        print(f"  平均到达步数: {avg:.1f}")


if __name__ == "__main__":
    main()
