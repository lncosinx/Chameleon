"""
性能 profiler：诊断 env / forward / batch / 拷贝 / backward / optimizer 各段耗时。

剖析策略（CLAUDE.md §4.3）：
- GPU 利用率上不去 → launch 开销/Python 循环（做批量化/编译/重叠）
- GPU 满载但仍慢 → 网络太重（降级 GAT：每 N 步通信 → mean-pool）
- profiler 明确指向某 CPU 观测构建热点 + numba 压不下去 → 才轮到 Rust
"""
from __future__ import annotations

import time
import torch
import numpy as np
from typing import Dict, List, Callable
from collections import defaultdict
from contextlib import contextmanager


class Profiler:
    """简单的分段计时器。"""

    def __init__(self):
        self._times: Dict[str, List[float]] = defaultdict(list)
        self._current: Dict[str, float] = {}

    @contextmanager
    def segment(self, name: str):
        """上下文管理器：记录一段代码的耗时。"""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._times[name].append(elapsed)

    def start(self, name: str):
        """手动开始计时。"""
        self._current[name] = time.perf_counter()

    def stop(self, name: str):
        """手动停止计时。"""
        if name in self._current:
            elapsed = time.perf_counter() - self._current.pop(name)
            self._times[name].append(elapsed)

    def report(self, sort_by: str = "total") -> Dict[str, float]:
        """生成耗时报告。"""
        report = {}
        for name, times in self._times.items():
            if times:
                report[name] = {
                    "count": len(times),
                    "total_s": sum(times),
                    "mean_ms": np.mean(times) * 1000,
                    "max_ms": np.max(times) * 1000,
                    "pct": 0.0,  # filled below
                }

        total = sum(r["total_s"] for r in report.values())
        if total > 0:
            for r in report.values():
                r["pct"] = r["total_s"] / total * 100

        # 排序
        if sort_by == "total":
            report = dict(sorted(report.items(), key=lambda x: -x[1]["total_s"]))
        elif sort_by == "mean":
            report = dict(sorted(report.items(), key=lambda x: -x[1]["mean_ms"]))

        return report

    def print_report(self):
        """打印耗时报告。"""
        report = self.report()
        print(f"{'Segment':<30} {'Count':>6} {'Total(s)':>10} {'Mean(ms)':>10} {'%':>6}")
        print("-" * 62)
        for name, stats in report.items():
            print(f"{name:<30} {stats['count']:>6} {stats['total_s']:>10.3f} "
                  f"{stats['mean_ms']:>10.2f} {stats['pct']:>5.1f}%")

    def reset(self):
        """重置所有计时。"""
        self._times.clear()
        self._current.clear()


def profile_env_step(env, num_steps: int = 100) -> Dict:
    """
    对环境步进做快速 profiling。

    Returns:
        dict with env_step_ms, obs_build_ms, reward_ms, total_ms
    """
    prof = Profiler()

    for _ in range(num_steps):
        with prof.segment("env_step"):
            actions = [np.random.randint(0, 5) for _ in range(env.num_agents)]
            obs, _, _, _ = env.step(actions)

    return prof.report()


def profile_forward(net, sample_batch: Dict, num_iters: int = 50) -> Dict:
    """对网络前向做 profiling。"""
    prof = Profiler()
    device = next(net.parameters()).device

    # warmup
    for _ in range(5):
        _ = net(**{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in sample_batch.items()})

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(num_iters):
        with prof.segment("forward"):
            _ = net(**{k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in sample_batch.items()})
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    return prof.report()


def estimate_throughput(
    env_step_ms: float, forward_ms: float, num_envs: int, num_agents: int
) -> float:
    """
    估算吞吐量（env-steps/s）。

    Args:
        env_step_ms: 单个 env step 耗时（ms）。
        forward_ms: 单次前向耗时（ms）。
        num_envs: 并行 env 数。
        num_agents: 每 env agent 数。

    Returns:
        env_steps_per_second
    """
    total_ms_per_step = env_step_ms + forward_ms
    steps_per_second = 1000.0 / max(total_ms_per_step, 0.01)
    return steps_per_second * num_envs
