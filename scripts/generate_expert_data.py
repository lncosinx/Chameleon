#!/usr/bin/env python3
"""
为 when-to-switch 原生 5 族（random/mazes/sc1/street/wc3）批量生成 LaCAM 专家数据。

每族只用其 **train split**（envs/map_registry.py 内 8:2 划分，wrapper 默认 split='train'），
输出 data/expert_{族}.npz。Reptile 任务=族，训练时各族一个独立 buffer。

用法：
  python scripts/generate_expert_data.py                       # 5 族, 96 agents, 3000 ep, 40 workers
  python scripts/generate_expert_data.py --families random,mazes --num-episodes 500
  python scripts/generate_expert_data.py --num-agents 8 --num-episodes 20 --num-workers 4   # 冒烟
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_lacam_data import generate_data

FAMILIES = ["random", "mazes", "sc1", "street", "wc3"]


def main():
    p = argparse.ArgumentParser(description="批量生成 5 族 LaCAM 专家数据（train split）")
    p.add_argument("--families", type=str, default=",".join(FAMILIES),
                   help="逗号分隔的族列表，对应 configs/env_{族}.yaml")
    p.add_argument("--num-episodes", type=int, default=3000, help="每族 episode 数")
    p.add_argument("--num-agents", type=int, default=96, help="智能体数（所有族必须一致）")
    p.add_argument("--map-size", type=int, default=64, help="仅元数据；实际尺寸由 registry 地图决定")
    p.add_argument("--num-workers", type=int, default=40, help="每族并行进程数（-1=auto）")
    p.add_argument("--backend", type=str, default="auto", help="LaCAM 后端：auto/native/prioritized")
    p.add_argument("--output-dir", type=str, default="data")
    p.add_argument("--config-tpl", type=str, default="configs/env_{fam}.yaml")
    p.add_argument("--skip-existing", action="store_true",
                   help="若 data/expert_{族}.npz 已存在则跳过该族")
    p.add_argument("--cost2go", action="store_true",
                   help="观测追加 cost2go 通道（patch 4→5 通道，T2）；输出到 --output-dir")
    p.add_argument("--cost2go-clip", type=float, default=10.0, help="cost2go 相对距离 clip 半径")
    args = p.parse_args()

    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    print(f"[generate_expert_data] families={fams} | num_agents={args.num_agents} "
          f"| episodes/族={args.num_episodes} | workers={args.num_workers} | backend={args.backend}")
    t0 = time.time()
    summary = {}
    for fam in fams:
        cfg = args.config_tpl.format(fam=fam)
        if not os.path.exists(cfg):
            print(f"  [skip] 配置不存在: {cfg}"); continue
        out_npz = os.path.join(args.output_dir, f"expert_{fam}.npz")
        if args.skip_existing and os.path.exists(out_npz):
            print(f"  [skip-existing] {out_npz} 已存在"); continue
        print(f"\n===== 生成族 '{fam}'  (config={cfg}) =====", flush=True)
        ft = time.time()
        generate_data(
            config_path=cfg,
            num_episodes=args.num_episodes,
            num_agents=args.num_agents,
            map_size=args.map_size,
            backend=args.backend,
            output_dir=args.output_dir,
            save_to_disk=True,
            num_workers=args.num_workers,
            use_cost2go=args.cost2go,
            cost2go_clip=args.cost2go_clip,
        )
        summary[fam] = time.time() - ft
        print(f"===== 族 '{fam}' 完成，用时 {summary[fam]/60:.1f} min =====", flush=True)

    print(f"\n[done] 全部用时 {(time.time()-t0)/60:.1f} min")
    for fam, sec in summary.items():
        print(f"  {fam}: {sec/60:.1f} min  -> data/expert_{fam}.npz")


if __name__ == "__main__":
    main()
