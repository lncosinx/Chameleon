"""
训练/测试地图集划分（对齐 when-to-switch 的确定性划分逻辑）。

背景
----
本项目地图是**按 seed 程序化生成**（random 用 pogema 原生 generator；maze/room/
warehouse 经 pogema-toolbox 由 seed 生成结构 + 由 seed 放置起终点）。原实现里训练与
评估各自用一段**互相重叠的低位 seed**（专家数据 `10000+ep`、selfplay `base+i`、评估
`20000+ep`），导致评估地图可能就是训练见过的地图 —— 测不出泛化，SR 被高估。

when-to-switch 的做法（见 `when-to-switch/pomapf_env/custom_maps.py`）是：在一个**固定
地图池**上，用一条确定性谓词（名称/尺寸正则 `_test_regexp`，如 `mazes-s0..s39=test,
s40..s49=train`）把实例划成**互不相交**的 train/test 两份。

本模块复刻同一逻辑：以 seed 作为"地图身份"，把整数 seed 空间确定性地切成两段不相交区间：

    train seed ∈ [0, TEST_SEED_BASE)                     —— 训练（约等于无限多张）
    test  seed ∈ [TEST_SEED_BASE, TEST_SEED_BASE + TEST_POOL_SIZE)   —— held-out 评估池

任何训练用的 raw 值经 `resolve_seed(raw, "train")` 折叠到 train 段（恒 < TEST_SEED_BASE），
任何评估索引经 `resolve_seed(idx, "test")` 映射到 test 段，两段整数集合保证不相交，
因此训练永远不会见到评估用的地图实例。

注意
----
- random/maze/room：seed 决定结构 + 起终点，train/test 是**完全不同的地图**。
- warehouse：当前 generator 结构固定（不吃 seed），但起终点仍由 env seed 随机放置，
  故 train/test 仍是**不相交的任务实例**（同结构、不同起终点配置）。
"""
from __future__ import annotations

from typing import Optional

# 评估池起点：放在高位，远离训练常用低位 seed。
TEST_SEED_BASE: int = 1_000_000
# held-out 评估地图数量（when-to-switch procedural 测试池量级，足够稳定估计 SR）。
TEST_POOL_SIZE: int = 256

_INT32_MAX = 2 ** 31 - 1


def resolve_seed(raw_seed: int, split: Optional[str] = "train") -> int:
    """把任意 raw 整数确定性地映射到指定 split 的 seed 空间。

    Args:
        raw_seed: 调用方传入的原始 seed（可来自任意公式，允许 >TEST_SEED_BASE 或负）。
        split:    'train' / 'test'(='eval') / 'all'(不重映射, 旧行为)。

    Returns:
        映射后的 seed。train 与 test 的返回值集合互不相交。
    """
    raw = int(raw_seed) % _INT32_MAX  # 防溢出/负数，保持确定性
    if split in ("test", "eval"):
        return TEST_SEED_BASE + raw % TEST_POOL_SIZE
    if split in ("train", None):
        return raw % TEST_SEED_BASE  # 恒 < TEST_SEED_BASE，与 test 块不相交
    if split == "all":
        return raw
    raise ValueError(f"unknown split: {split!r} (expected 'train'/'test'/'all')")


def test_seed(index: int) -> int:
    """第 index 个评估地图的 seed（确定性、可复现）。"""
    return resolve_seed(index, "test")


def train_seed(raw_seed: int) -> int:
    """把 raw 值折叠进 train 空间（保证落在 test 区间之外）。"""
    return resolve_seed(raw_seed, "train")


def is_test_seed(seed: int) -> bool:
    """该 seed 是否落在 held-out 评估池内。"""
    return TEST_SEED_BASE <= int(seed) < TEST_SEED_BASE + TEST_POOL_SIZE
