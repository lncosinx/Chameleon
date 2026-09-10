#!/usr/bin/env python3
"""
为 when-to-switch 原生 5 族（random/mazes/sc1/street/wc3）批量生成 LaCAM 专家数据。

每族只用其 **train split**（envs/map_registry.py 内 8:2 划分，wrapper 默认 split='train'），
输出 data/expert_{族}/（目录，每个字段一个独立 .npy，支持训练时 mmap 加载，不必整族占内存；
见 scripts/generate_lacam_data.py 的 save_buffer_to_disk/load_buffer_from_disk）。
Reptile 任务=族，训练时各族一个独立 buffer。

用法：
  python scripts/generate_expert_data.py                       # 5 族, 96 agents, 3000 ep, 40 workers
  python scripts/generate_expert_data.py --families random,mazes --num-episodes 500
  python scripts/generate_expert_data.py --num-agents 8 --num-episodes 20 --num-workers 4   # 冒烟
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_lacam_data import generate_data, expert_data_exists

FAMILIES = ["random", "mazes", "sc1", "street", "wc3"]


def main():
    p = argparse.ArgumentParser(description="批量生成 5 族 LaCAM 专家数据（train split）")
    p.add_argument("--families", type=str, default=",".join(FAMILIES),
                   help="逗号分隔的族列表，对应 configs/env_{族}.yaml")
    p.add_argument("--num-episodes", type=int, default=3000, help="每族 episode 数")
    p.add_argument("--num-agents", type=int, default=96, help="智能体数（所有族必须一致）")
    p.add_argument("--map-size", type=int, default=None,
                   help="覆盖配置里的 map_size。**不给就用配置值**（默认行为）。"
                        "★2026-09-10 修：原来默认 64 会**静默覆盖** env_*_small.yaml 的 "
                        "map_size:20 —— 于是'小图高密度'配置从未生效，random 专家数据一直是 "
                        "64x64 大图(64 体密度 2%)，而评测是 20x21(密度 18%)，9 倍失配。"
                        "help 原文写'仅元数据'是错的：generate_lacam_data.py 里 "
                        "`wrapper.map_size = map_size` 真的会改地图生成尺寸。")
    p.add_argument("--num-workers", type=int, default=40, help="每族并行进程数（-1=auto）")
    p.add_argument("--backend", type=str, default="auto", help="LaCAM 后端：auto/native/prioritized")
    p.add_argument("--output-dir", type=str, default="data")
    p.add_argument("--config-tpl", type=str, default="configs/env_{fam}.yaml")
    p.add_argument("--skip-existing", action="store_true",
                   help="若 data/expert_{族}.npz 已存在则跳过该族")
    p.add_argument("--store-next", action="store_true",
                   help="同时存 next_obs/next_* 字段（只有已废弃的 DQfD/TD 损失需要；"
                        "默认不存，省约一半磁盘）")
    p.add_argument("--cost2go", action="store_true",
                   help="观测追加 cost2go 通道（patch 4→5 通道，T2）；输出到 --output-dir")
    p.add_argument("--cost2go-clip", type=float, default=10.0, help="cost2go 相对距离 clip 半径")
    args = p.parse_args()

    # spawn 出来的 worker 会重新 import 模块，靠环境变量把开关传下去（模块级赋值传不到子进程）
    os.environ["CHAMELEON_STORE_NEXT"] = "1" if args.store_next else "0"
    import scripts.generate_lacam_data as _gld
    _gld.STORE_NEXT = bool(args.store_next)
    print(f"[storage] next_* 字段: {'存' if args.store_next else '不存（纯 IL 用不到，省一半磁盘）'}")

    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    print(f"[generate_expert_data] families={fams} | num_agents={args.num_agents} "
          f"| episodes/族={args.num_episodes} | workers={args.num_workers} | backend={args.backend}")
    t0 = time.time()
    summary = {}
    for fam in fams:
        cfg = args.config_tpl.format(fam=fam)
        if not os.path.exists(cfg):
            print(f"  [skip] 配置不存在: {cfg}"); continue
        out_base = os.path.join(args.output_dir, f"expert_{fam}")
        if args.skip_existing and expert_data_exists(out_base):
            print(f"  [skip-existing] {out_base}(.npz) 已存在"); continue
        print(f"\n===== 生成族 '{fam}'  (config={cfg}) =====", flush=True)
        ft = time.time()
        generate_data(
            config_path=cfg,
            num_episodes=args.num_episodes,
            num_agents=args.num_agents,
            map_size=args.map_size,   # None = 用配置里的值
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
        print(f"  {fam}: {sec/60:.1f} min  -> data/expert_{fam}/")


if __name__ == "__main__":
    main()
