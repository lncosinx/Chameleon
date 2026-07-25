#!/usr/bin/env python3
"""v3 网络核心 smoke：构建 TransformerCommNet，跑前向 + IL 交叉熵 + 反传，校验 shape/无 nan。"""
import sys, os, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.transformer_comm_net import TransformerCommNet
from losses.il_losses import il_cross_entropy_loss

cfg = yaml.safe_load(open("configs/network_transformer.yaml"))
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = TransformerCommNet(
    patch_channels=cfg["patch_channels"], scalar_feat_dim=cfg["scalar_feat_dim"],
    obs_radius=cfg["obs_radius"], num_actions=cfg["num_actions"],
    d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
    d_ff=cfg["d_ff"], dropout=cfg["dropout"], norm_first=cfg["norm_first"],
    comm_rounds=cfg["comm_rounds"], intent_dim=cfg["intent_dim"], comm_dim=cfg["comm_dim"],
    comm_radius=cfg["comm_radius"], comm_topk_cap=cfg["comm_topk_cap"],
    entmax_alpha=cfg["entmax_alpha"], tau_in=cfg["tau_in"], tau_out=cfg["tau_out"],
    stickiness_bonus=cfg["stickiness_bonus"], ema_decay=cfg["ema_decay"],
    num_attention_heads=cfg["num_attention_heads"], head_hidden=cfg["head_hidden"],
).to(dev)
n_params = sum(p.numel() for p in net.parameters())
print(f"params: {n_params:,} ({n_params/1e6:.2f}M)  | n_tokens={net.tokenizer.n_tokens}")

B, T, N, K = 2, 4, 8, 2 * cfg["obs_radius"] + 1
patches = torch.rand(B, T, N, cfg["patch_channels"], K, K, device=dev)
scalars = torch.rand(B, T, N, cfg["scalar_feat_dim"], device=dev)
positions = torch.randint(0, 32, (B, T, N, 2), device=dev).float()
masks = torch.ones(B, T, N, 5, dtype=torch.bool, device=dev)
masks[..., 1] = False                      # 制造一个非法动作
expert = torch.randint(0, 5, (B, T, N), device=dev)
expert[expert == 1] = 0                     # 专家动作恒合法
valid = torch.ones(B, T, N, dtype=torch.bool, device=dev)

logits, info = net(patches, scalars, positions, action_masks=masks, reset_comm=True)
assert logits.shape == (B, T, N, 5), logits.shape
assert torch.isinf(logits[..., 1]).all(), "非法动作应为 -inf"
loss, linfo = il_cross_entropy_loss(logits, expert, valid=valid)
loss.backward()
gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)

assert torch.isfinite(loss), loss
print(f"forward OK | logits {tuple(logits.shape)} | loss={linfo['loss_il']:.4f} "
      f"acc={linfo['acc_expert']:.3f} | grad_norm={gnorm:.3f} | "
      f"edge_flip={info['edge_flip_rate']:.3f} avg_nbr={info['avg_neighbors']:.2f}")
print("SMOKE OK")
