#!/usr/bin/env python3
"""拟合 / 泛化探针：专家动作准确率（贪心 argmax）在「训练批」vs「真·留出批」vs「外分布 warehouse」上的差。

2026-09-17 重写：旧探针把 mix12 的 *b 批当"同分布留出集"，但 *b 批 66~98% 与原批是同一批 episode
（seed 写死），留出集被训练数据污染，得出的"gap≈0.009 ⇒ 欠拟合"作废。这里的留出集来自
chameleon_data/c2g_heldout/（seed 段与所有训练批次不相交，永不参与训练）。

用法：python scripts/probe_generalization.py [--batches 60]
"""
from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, yaml
from buffers.expert_demo_buffer import ExpertDemoBuffer
from scripts.generate_lacam_data import load_buffer_from_disk
from scripts.train_il_meta import build_net
from trainers.train_il_reptile import ILReptileTrainer

D = "/root/autodl-tmp/chameleon_data"
H = f"{D}/c2g_heldout"
# (族, 训练批（mix6 与 mix12 都含）, 真留出批)
PAIRS = [("random32", f"{D}/c2g_dense32/expert_random",   f"{H}/r32/expert_random"),
         ("random64", f"{D}/c2g_dense64/expert_random",   f"{H}/r64b/expert_random"),
         ("maze32",   f"{D}/c2g_dense32m/expert_maze",    f"{H}/m32/expert_maze"),
         ("maze48",   f"{D}/c2g_small_15k/expert_maze",   f"{H}/m48/expert_maze"),
         ("maze64",   f"{D}/c2g_a64/expert_maze",         f"{H}/m64/expert_maze")]
OOD = [("wh96", f"{D}/c2g_whmix/expert_wh96"), ("wh192", f"{D}/c2g_whmix/expert_wh192")]
MODELS = [("it_act 2.16M (mix6)",     "ckpt_il/it_act_iter_8000.pt",    "configs/network_gru_c2g.yaml", {"intent_mode": "action"}),
          ("小网 2.16M @8k (mix12)",  "ckpt_il/capsmall_iter_8000.pt",  "configs/network_gru_c2g.yaml", {"intent_mode": "action"}),
          ("大网 7.45M @14k (mix12)", "ckpt_il/capbig_iter_14000.pt",   "configs/network_big.yaml", {}),
          ("大网 7.45M @20k (mix12)", "ckpt_il/capbig_iter_20000.pt",   "configs/network_big.yaml", {})]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--batches", type=int, default=60)
    args = ap.parse_args()
    bufs = {}
    for p in [x for _, a, b in PAIRS for x in (a, b)] + [p for _, p in OOD]:
        n = int(np.load(p + "/episode_ids.npy", mmap_mode="r").shape[0])
        b = ExpertDemoBuffer(capacity=n); load_buffer_from_disk(p, b); bufs[p] = b
    cfg = yaml.safe_load(open("configs/train_il_cap_lr1e4.yaml"))

    def acc(tr, path):
        rng = np.random.default_rng(13); cor = tot = 0
        with torch.no_grad():
            for _ in range(args.batches):
                g = tr._to_gpu(bufs[path].sample(batch_size=8, sequence_len=32, burn_in_len=0,
                                                 rng=rng, need_next=False))
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    lg, _ = tr.net(g["patches"], g["scalars"], g["positions"],
                                   action_masks=g["masks"], reset_comm=True)
                v = g["valid"].bool()
                cor += (lg.argmax(-1)[v] == g["actions"][v]).sum().item(); tot += int(v.sum())
        return cor / tot

    print(f"{'模型':26s}" + "".join(f"{f:>18s}" for f, _, _ in PAIRS) + f"{'gap均':>8}{'OOD均':>8}")
    print(f"{'':26s}" + "".join(f"{'训练/留出/gap':>18s}" for _ in PAIRS))
    for lbl, ck, ncfg, over in MODELS:
        nc = yaml.safe_load(open(ncfg)); nc.update(over)
        net = build_net(nc); sd = torch.load(ck, map_location="cpu")["net"]
        net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
        tr = ILReptileTrainer(net, {**cfg, "reptile": False}, device=torch.device("cuda:0"))
        tr.net.eval(); tr.d4_augment = False
        cells, gaps = [], []
        for fam, trn, ho in PAIRS:
            a, h = acc(tr, trn), acc(tr, ho); gaps.append(a - h)
            cells.append(f"{a:.3f}/{h:.3f}/{a-h:+.3f}")
        ood = np.mean([acc(tr, p) for _, p in OOD])
        print(f"{lbl:26s}" + "".join(f"{c:>18s}" for c in cells) + f"{np.mean(gaps):>+8.3f}{ood:>8.3f}", flush=True)
        del tr, net; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
