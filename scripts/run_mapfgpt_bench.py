#!/usr/bin/env python3
"""
在 POGEMA benchmark 协议下跑 MAPF-GPT / MAPF-GPT-DDG（SQ1 的 foundation-model 对照点）。

**必须用 marlgpt 环境**（pogema==1.4.0 / numpy 1.26 / cppimport 编译的 C++ 观测生成器）：
    /root/autodl-tmp/envs/marlgpt/bin/python3.10 scripts/run_mapfgpt_bench.py ...

为什么不直接用官方 benchmark.py：它一次跑全部 5 个 folder、强依赖 wandb、且把结果写进它自己的
目录结构。这里只跑我们主表要的族与 agent 档，输出格式与 `scripts/eval_bench.py` 完全一致
（同一份 make_sq1_table.py 就能合表）。协议参数直接取自 pogema-benchmark 的官方 yaml，
不手抄，避免抄错。

用法：
  python scripts/run_mapfgpt_bench.py --weights MAPF-GPT-2M.pt --families random,mazes \
      --num-maps 128 --out /root/autodl-tmp/eval_v5/mapfgpt2m.json
"""
from __future__ import annotations
import sys, os, argparse, json, time

MG = "/root/Chameleon/MAPF-GPT"
sys.path.insert(0, MG)
os.chdir(MG)   # cppimport 需要相对路径找到 gpt/observation_generator.cpp

import numpy as np, yaml
import cppimport.import_hook  # noqa: F401  （触发 C++ 观测生成器编译/加载）
from pogema import GridConfig, pogema_v0
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig

FAM_DIR = {"random": "01-random", "mazes": "02-mazes",
           "warehouse": "03-warehouse", "movingai": "04-movingai"}


def load_protocol(fam: str):
    """从 pogema-benchmark 官方 yaml 读该族的 agents / map 列表 / seed / 步数。"""
    p = os.path.join("/root/Chameleon/pogema-benchmark/raw_data_MAPF",
                     FAM_DIR[fam], f"{FAM_DIR[fam]}.yaml")
    cfg = yaml.safe_load(open(p))["environment"]
    agents = cfg["num_agents"]["grid_search"]
    mp = cfg["map_name"]
    maps = mp["grid_search"] if isinstance(mp, dict) else [mp]
    sd = cfg.get("seed", 0)
    seeds = sd["grid_search"] if isinstance(sd, dict) else [sd]
    return agents, maps, seeds, int(cfg["max_episode_steps"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="MAPF-GPT-2M.pt")
    ap.add_argument("--families", default="random,mazes")
    ap.add_argument("--num-maps", type=int, default=128)
    ap.add_argument("--num-seeds", type=int, default=0, help=">0 时截断 seed 数（warehouse 用）")
    ap.add_argument("--agent-counts", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    algo = MAPFGPTInference(MAPFGPTInferenceConfig(
        path_to_weights=f"hf_weights/{args.weights}", device=args.device))
    label = args.tag or args.weights.replace(".pt", "")
    print(f"[MAPF-GPT] weights={args.weights}", flush=True)

    results = {}
    for fam in [f.strip() for f in args.families.split(",") if f.strip()]:
        agents, map_names, seeds, steps = load_protocol(fam)
        all_maps = yaml.safe_load(open(os.path.join("eval_configs", FAM_DIR[fam], "maps.yaml")))
        if args.num_maps > 0:
            map_names = map_names[:args.num_maps]
        if args.num_seeds > 0:
            seeds = seeds[:args.num_seeds]
        if args.agent_counts:
            agents = [int(a) for a in args.agent_counts.split(",")]
        print(f"\n===== {fam}: {len(map_names)} maps × {len(seeds)} seed(s) × {steps} steps "
              f"| agents={agents} =====", flush=True)
        results[fam] = {}
        for na in agents:
            t0, rows = time.time(), []
            for mn in map_names:
                for sd in seeds:
                    gc = GridConfig(map=all_maps[mn], map_name=mn, num_agents=na,
                                    obs_radius=5, max_episode_steps=steps, seed=sd,
                                    on_target="nothing", collision_system="soft",
                                    observation_type="MAPF")
                    env = pogema_v0(grid_config=gc)
                    obs, info = env.reset(seed=sd)
                    algo.reset_states()
                    for _ in range(steps):
                        obs, _, term, trunc, info = env.step(algo.act(obs))
                        if all(term) or all(trunc):
                            break
                    # pogema 1.4 的指标在 episode 结束时挂在 per-agent info 的 'metrics' 里
                    # （没有 env.get_metrics()）。CSR/ISR/SoC/makespan 与 leaderboard 同源同口径。
                    m = (info[0].get("metrics") or {}) if isinstance(info, list) else {}
                    rows.append((float(m.get("CSR", 0.0)), float(m.get("ISR", 0.0)),
                                 float(m.get("SoC", 0.0)), float(m.get("makespan", 0.0))))
            arr = np.array(rows)
            agg = dict(CSR=float(arr[:, 0].mean()), ISR=float(arr[:, 1].mean()),
                       SoC=float(arr[:, 2].mean()), makespan=float(arr[:, 3].mean()),
                       n=len(rows))
            agg["CSR_ci95"] = float(1.96 * np.sqrt(max(agg["CSR"] * (1 - agg["CSR"]), 1e-9) / len(rows)))
            results[fam][na] = agg
            print(f"  agents={na:>3}: CSR={agg['CSR']:.3f}±{agg['CSR_ci95']:.3f}  "
                  f"ISR={agg['ISR']:.3f}  SoC={agg['SoC']:.1f}  "
                  f"(n={agg['n']}, {time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"ckpt": args.weights, "tag": label, "results": results}, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
