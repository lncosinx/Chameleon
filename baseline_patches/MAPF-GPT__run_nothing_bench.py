#!/usr/bin/env python3
"""专用 MAPF-GPT(AAAI'25)在 nothing 协议(128步, validation 图)下评测 random+mazes。
配置本就是 nothing/128/MAPF——直接用其 eval_configs，权重从 HF 自动下载。
跑 MAPF-GPT-2M 与 MAPF-GPT-DDG-2M(尺度对齐 il_joint 2.16M）。

用法(MAPF-GPT/ 目录, marlgpt 环境):
  python run_nothing_bench.py --families 01-random,02-mazes --num-maps 0
"""
import sys, os, argparse, yaml, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pogema_toolbox.create_env import Environment
from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry
from create_env import create_eval_env
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="01-random,02-mazes")
    ap.add_argument("--num-maps", type=int, default=0, help="0=全部图;>0=均匀抽样")
    ap.add_argument("--backend", default="sequential")
    ap.add_argument("--output-dir", default="eval_out_nothing")
    args = ap.parse_args()

    ToolboxRegistry.register_env("Environment", create_eval_env, Environment)
    ToolboxRegistry.register_algorithm("MAPF-GPT", MAPFGPTInference, MAPFGPTInferenceConfig)

    for fam in args.families.split(","):
        fam = fam.strip()
        maps = yaml.safe_load(open(f"eval_configs/{fam}/maps.yaml"))
        ToolboxRegistry.register_maps(maps)
        cfg = yaml.safe_load(open(f"eval_configs/{fam}/{fam}.yaml"))
        names = sorted(maps)
        if 0 < args.num_maps < len(names):
            idx = sorted(set(np.linspace(0, len(names)-1, args.num_maps).round().astype(int).tolist()))
            cfg["environment"]["map_name"]["grid_search"] = [names[i] for i in idx]
        for a in cfg["algorithms"].values():
            a["parallel_backend"] = args.backend
            # 触发 HF 自动下载：inference.py 仅当路径含 'hf_weights' 才下载
            if "path_to_weights" in a and "hf_weights" not in a["path_to_weights"]:
                a["path_to_weights"] = a["path_to_weights"].replace("weights/", "hf_weights/")
        print(f"[{fam}] on_target={cfg['environment'].get('on_target')} "
              f"max_steps={cfg['environment'].get('max_episode_steps')} "
              f"algos={list(cfg['algorithms'])} maps={len(cfg['environment']['map_name']['grid_search'])}", flush=True)
        for algo, acfg in cfg["algorithms"].items():
            ToolboxRegistry.create_algorithm(acfg["name"], **acfg)  # 触发权重下载
        eval_dir = os.path.join(args.output_dir, fam)
        os.makedirs(eval_dir, exist_ok=True)
        evaluation(cfg, eval_dir=eval_dir)


if __name__ == "__main__":
    main()
