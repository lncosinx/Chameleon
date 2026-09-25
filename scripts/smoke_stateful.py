#!/usr/bin/env python3
"""校验：Mamba 时间核 + 通信迟滞的 stateful 逐步推断 == 整段 forward（同序列同输入）。
不等价则 rollout 评估会静默错误。比较整段 vs 逐步的 logits 最大差 + argmax 一致率。"""
import sys, os, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.mapf_comm_net import MAPFCommNet

cfg = yaml.safe_load(open(os.environ.get("NETCFG", "configs/network_gru.yaml")))
if len(sys.argv) > 1:
    cfg["comm_rounds"] = int(sys.argv[1])   # 测多轮通信的 stateful 等价性
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
print(f"comm_rounds={cfg['comm_rounds']}")

net = MAPFCommNet(
    patch_channels=cfg["patch_channels"], scalar_feat_dim=cfg["scalar_feat_dim"],
    obs_radius=cfg["obs_radius"], num_actions=cfg["num_actions"], d_model=cfg["d_model"],
    cnn_channels=cfg["cnn_channels"], d_scalar=cfg["d_scalar"], n_res_blocks=cfg["n_res_blocks"],
    comm_rounds=cfg["comm_rounds"], intent_dim=cfg["intent_dim"], comm_dim=cfg["comm_dim"],
    comm_radius=cfg["comm_radius"], comm_topk_cap=cfg["comm_topk_cap"], entmax_alpha=cfg["entmax_alpha"],
    tau_in=cfg["tau_in"], tau_out=cfg["tau_out"], stickiness_bonus=cfg["stickiness_bonus"],
    ema_decay=cfg["ema_decay"], num_attention_heads=cfg["num_attention_heads"],
    temporal_backend=cfg["temporal_backend"], temporal_layers=cfg["temporal_layers"],
    d_state=cfg["d_state"], d_conv=cfg["d_conv"], mamba_expand=cfg["mamba_expand"],
    head_hidden=cfg["head_hidden"],
).to(dev).eval()

B, T, N, K = 1, 20, 8, 2 * cfg["obs_radius"] + 1
patches = torch.rand(B, T, N, cfg["patch_channels"], K, K, device=dev)
scalars = torch.rand(B, T, N, cfg["scalar_feat_dim"], device=dev)
positions = torch.randint(0, 24, (B, T, N, 2), device=dev).float()
masks = torch.ones(B, T, N, 5, dtype=torch.bool, device=dev)

with torch.no_grad():
    full, _ = net(patches, scalars, positions, action_masks=masks, reset_comm=True, stateful=False)
    # 逐步
    net.reset_comm_state()
    steps = []
    for t in range(T):
        lt, _ = net(patches[:, t:t+1], scalars[:, t:t+1], positions[:, t:t+1],
                    action_masks=masks[:, t:t+1], reset_comm=False, stateful=True)
        steps.append(lt)
    step = torch.cat(steps, dim=1)   # (B,T,N,5)

diff = (full - step).abs().max().item()
agree = (full.argmax(-1) == step.argmax(-1)).float().mean().item()
print(f"max|Δlogit|={diff:.3e}  argmax_agree={agree*100:.1f}%")
assert agree > 0.999, f"argmax 不一致 ({agree:.3f}) → stateful 推断有 bug"
assert diff < 1e-2, f"logit 差过大 ({diff:.2e})"
print("STATEFUL OK：逐步推断与整段 forward 等价")
