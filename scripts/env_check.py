#!/usr/bin/env python3
"""
T0.1 环境自检脚本
验证 pogema 动作顺序（INVARIANT 1）、GPU 可见性、CPU 核数、关键依赖版本。
"""
import sys
import os

# 将仓库根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPECTED_MOVES = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]


def check_pogema_moves():
    """INVARIANT 1: 实测 pogema 动作顺序，wait 必须为 action 0。"""
    from pogema import GridConfig
    gc = GridConfig()
    actual = [list(m) for m in gc.MOVES]

    print(f"[CHECK] Pogema MOVES: {actual}")
    print(f"        Expected   : {EXPECTED_MOVES}")

    if actual != EXPECTED_MOVES:
        print("\n!!!  FATAL: Pogema MOVES 与预期不符！后续所有动作语义依赖此顺序。")
        print("!!!  请检查 pogema 版本并确认 GridConfig().MOVES。")
        sys.exit(1)

    print("[PASS]  INVARIANT 1: 动作顺序正确，wait=action 0 ✓")
    return True


def check_gpu():
    """检查 GPU 数量 & CUDA 可用性。记录差异：设计文档假设 2×RTX 5090。"""
    import torch
    count = torch.cuda.device_count()
    cuda_ok = torch.cuda.is_available()

    print(f"\n[CHECK] CUDA available: {cuda_ok}")
    print(f"        GPU count     : {count}")
    for i in range(count):
        print(f"        GPU {i}: {torch.cuda.get_device_name(i)}")

    if not cuda_ok:
        print("[WARN]  CUDA 不可用，将回退 CPU。")
    elif count < 2:
        print(f"[WARN]  仅有 {count} 块 GPU，目标硬件为 2。双 GPU task-并行需 ≥2 卡。")
        print("        代码仍支持双 GPU 模式，但当前只能在单卡上运行。")
    else:
        print("[PASS]  GPU: ≥2 块可用 ✓")
    return cuda_ok, count


def check_cpu():
    """检查 CPU 核数。"""
    n = os.cpu_count()
    print(f"\n[CHECK] CPU 核心数: {n}")
    if n is None or n < 4:
        print("[WARN]  CPU 核数异常或过少（<4）。")
    else:
        print(f"[INFO]  可用核数 {n}（目标硬件为 47 核）。多进程采样可正常运行。")
    return n


def check_deps():
    """检查关键依赖版本。"""
    print("\n[CHECK] 关键依赖版本:")
    deps = {}

    import pogema
    deps['pogema'] = getattr(pogema, '__version__', 'unknown')
    print(f"        pogema   : {deps['pogema']}")

    import torch
    deps['torch'] = torch.__version__
    print(f"        torch    : {deps['torch']}")

    import numpy
    deps['numpy'] = numpy.__version__
    print(f"        numpy    : {deps['numpy']}")

    try:
        import numba
        deps['numba'] = numba.__version__
        print(f"        numba    : {deps['numba']}")
    except ImportError:
        deps['numba'] = None
        print("        numba    : NOT INSTALLED")

    import yaml
    deps['pyyaml'] = getattr(yaml, '__version__', 'unknown')
    print(f"        pyyaml   : {deps['pyyaml']}")

    try:
        import pogema_toolbox
        deps['pogema_toolbox'] = getattr(pogema_toolbox, '__version__', 'installed')
        print(f"        pogema-toolbox: {deps['pogema_toolbox']}")
    except ImportError:
        deps['pogema_toolbox'] = None
        print("        pogema-toolbox: NOT INSTALLED")

    print("[PASS]  所有关键依赖已安装 ✓")
    return deps


def _unpack_reset(reset_result):
    """兼容 Pogema 1.x 和 2.x (Gymnasium 风格) 的 reset 返回值。"""
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result  # (obs_list, info)
    return reset_result, {}


def _unpack_step(step_result, num_agents):
    """兼容 Pogema 1.x 和 2.x 的 step 返回值。"""
    if isinstance(step_result, tuple):
        # Gymnasium: (obs, reward, terminated, truncated, info) or (obs, reward, done, info)
        if len(step_result) == 5:
            obs, reward, term, trunc, info = step_result
            done = [t or trunc[i] if isinstance(trunc, (list, tuple)) else t or trunc
                    for i, t in enumerate(term)]
            return obs, reward, done, info
        elif len(step_result) == 4:
            obs, reward, done, info = step_result
            return obs, reward, done, info
    # Fallback: old-style single return
    return step_result, [0] * num_agents, [False] * num_agents, {}


def check_pogema_env():
    """快速 smoke: 创建一个最小 random 环境，走几步确保不崩。"""
    from pogema import GridConfig, pogema_v0
    print("\n[CHECK] Pogema 环境 smoke:")
    gc = GridConfig(size=16, num_agents=4, obs_radius=5,
                    max_episode_steps=128, density=0.3)
    env = pogema_v0(gc)
    obs_list, info = _unpack_reset(env.reset())
    print(f"        reset ok, obs per agent shape: {obs_list[0].shape} (expect (3,11,11))")

    for step in range(10):
        actions = [i % 5 for i in range(gc.num_agents)]  # dummy actions
        obs_list, reward, done, info = _unpack_step(env.step(actions), gc.num_agents)
    print(f"        10 steps ok, rewards sample: {[round(float(r), 3) for r in reward[:4]]}")
    print("[PASS]  Pogema 环境 smoke ✓")


def main():
    print("=" * 60)
    print("  Chameleon — T0.1 环境自检")
    print("=" * 60)

    all_ok = True

    # 1. 动作顺序（硬约束）
    try:
        check_pogema_moves()
    except Exception as e:
        print(f"[FAIL] 动作顺序检查异常: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False

    # 2. GPU
    try:
        check_gpu()
    except Exception as e:
        print(f"[FAIL] GPU 检查异常: {e}")
        all_ok = False

    # 3. CPU
    try:
        check_cpu()
    except Exception as e:
        print(f"[FAIL] CPU 检查异常: {e}")
        all_ok = False

    # 4. 依赖
    try:
        check_deps()
    except Exception as e:
        print(f"[FAIL] 依赖检查异常: {e}")
        all_ok = False

    # 5. env smoke
    try:
        check_pogema_env()
    except Exception as e:
        print(f"[FAIL] 环境 smoke 异常: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("  T0.1 验收结果: 全部通过 ✓")
    else:
        print("  T0.1 验收结果: 存在失败项 ✗")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
