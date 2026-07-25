#!/usr/bin/env python3
"""
用官方 marl-gpt 推理管线（gpt/inference.py + pogema_toolbox）跑一个轻量子集评测。

用法（须在 marl-gpt/ 目录下、且用装了 loguru/dask/cppimport/pybind11 的环境跑）:
  python run_light_bench.py --family 01-random --num-maps 20 \
      --agent-counts 8,16,24,32,48,64 --weights out/ckpt.pt --output-dir eval_out_light
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml
import cppimport.import_hook  # noqa: F401  触发 C++ observation_generator JIT 编译

from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.create_env import Environment
from gpt.inference import MARLGPTInference, InferenceConfig
from envs.pogema_env.create_env import create_eval_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="01-random")
    p.add_argument("--num-maps", type=int, default=20)
    p.add_argument("--agent-counts", default="")
    p.add_argument("--on-target", default="", help="留空则用族自带 on_target；否则覆盖 (finish/restart/nothing)")
    p.add_argument("--max-steps", type=int, default=0, help=">0 覆盖 max_episode_steps（leaderboard=128）")
    p.add_argument("--weights", default="out/ckpt.pt")
    p.add_argument("--parallel-backend", default="sequential")
    p.add_argument("--output-dir", default="eval_out_light")
    args = p.parse_args()

    ToolboxRegistry.register_env("pogema", create_eval_env, Environment)
    ToolboxRegistry.register_algorithm("MARL-GPT", MARLGPTInference, InferenceConfig)

    cfg_path = os.path.join("eval_configs", args.family, f"{args.family}.yaml")
    maps_path = os.path.join("eval_configs", args.family, "maps.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    maps = yaml.safe_load(open(maps_path))
    ToolboxRegistry.register_maps(maps)

    names = sorted(maps.keys())
    if 0 < args.num_maps < len(names):
        idx = sorted(set(np.linspace(0, len(names) - 1, args.num_maps).round().astype(int).tolist()))
        sampled = [names[i] for i in idx]
    else:
        sampled = names
    cfg["environment"]["map_name"]["grid_search"] = sampled

    if args.agent_counts:
        cfg["environment"]["num_agents"]["grid_search"] = [int(x) for x in args.agent_counts.split(",")]
    if args.on_target:
        cfg["environment"]["on_target"] = args.on_target
    if args.max_steps > 0:
        cfg["environment"]["max_episode_steps"] = args.max_steps

    cfg["algorithms"]["MARL-GPT"]["parallel_backend"] = args.parallel_backend
    cfg["algorithms"]["MARL-GPT"]["path_to_weights"] = args.weights

    print(f"[run_light_bench] family={args.family} maps={len(sampled)}/{len(names)} "
          f"agent_counts={cfg['environment']['num_agents']['grid_search']} "
          f"on_target={cfg['environment'].get('on_target')} weights={args.weights}")

    eval_dir = os.path.join(args.output_dir, args.family)
    os.makedirs(eval_dir, exist_ok=True)
    results = evaluation(cfg, eval_dir=eval_dir)

    with open(os.path.join(eval_dir, "raw_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    sr_vals = [r.get("metrics", {}).get("ISR") for r in results if "ISR" in r.get("metrics", {})]
    csr_vals = [r.get("metrics", {}).get("CSR") for r in results if "CSR" in r.get("metrics", {})]
    print(f"[run_light_bench] {len(results)} rows saved to {eval_dir}")
    if sr_vals:
        print(f"  ISR mean={np.mean(sr_vals):.3f}")
    if csr_vals:
        print(f"  CSR mean={np.mean(csr_vals):.3f}")


if __name__ == "__main__":
    main()
