"""
直接复用 when-to-switch 的固定地图池（231 张），按 8:2 划分 train/test。

设计动机
--------
原先地图按 seed 在线程序化生成，导致两个问题：(1) 训练/评估 seed 段重叠，测不出
泛化；(2) maze/room/warehouse 生成尺寸 ≠ 配置 map_size。本模块改为**直接加载**
`when-to-switch/pomapf_env/maps.yaml` 这一固定地图 registry（共 231 张）。

划分逻辑（对齐论文"全部地图 8:2 划分训练/测试"）：
  **默认全部 231 张图一起用，不按 random/maze/room/warehouse 分类**（配置 `map_name: "all"`）。
  地图按名称排序后**均匀跨步**取约 1/5 作测试、其余作训练（默认 TEST_RATIO=0.2，stride=5）。
  跨步而非取字母尾巴 → 测试集有代表性。训练与评估因此用**互不相交的真实地图结构**。
  → 全局约 185 训练 / 46 测试。

也支持把 `map_name` 设成正则前缀只用某族（如 'random-' / 'mazes-' / 'sc1-' / 'street-'），
此时在该族内按同样 8:2 划分。所有地图均为标准 '.'(空)/'#'(障碍)。

注：when-to-switch 代码里的 `_test_regexp` 是个测试集占 ~81% 的特殊 held-out 方案
（它训练主要靠在线 random generator），与论文 8:2 相反，故这里不采用。
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import yaml

# when-to-switch 地图 registry 路径（可用环境变量覆盖）。
_DEFAULT_YAML = (
    Path(__file__).resolve().parent.parent / "when-to-switch" / "pomapf_env" / "maps.yaml"
)
MAPS_YAML = Path(os.environ.get("CHAMELEON_MAPS_YAML", _DEFAULT_YAML))

# 测试集占比（论文 8:2 → 0.2）。族内按排序均匀跨步取 1/stride 作测试。
TEST_RATIO = 0.2
# 匹配"全部地图"的 pattern 取值（默认所有配置都用全池，不分类别）。
_ALL = ("", "all", "*", ".*")


@lru_cache(maxsize=1)
def _registry() -> dict:
    if not MAPS_YAML.exists():
        raise FileNotFoundError(
            f"地图 registry 不存在: {MAPS_YAML}\n"
            "请确认 when-to-switch/ 在仓库根目录，或设环境变量 CHAMELEON_MAPS_YAML 指向 maps.yaml。"
        )
    return yaml.safe_load(MAPS_YAML.read_text())


def _matches(pattern: str, name: str) -> bool:
    if pattern in _ALL:
        return True
    return re.match(pattern, name) is not None


@lru_cache(maxsize=None)
def get_pool(pattern: str, split: str = "train") -> Tuple[Tuple[str, str], ...]:
    """返回匹配 `pattern` 且属于 `split` 的地图列表 [(name, map_str), ...]。

    pattern: 正则前缀；`''/'all'/'*'/'.*'` 表示**全部地图**（默认用法，不分类别）。
    split:   'train' / 'test'(='eval') / 'all'。
    划分：匹配的地图按名称排序后均匀跨步，i % stride == 0 → 测试，其余 → 训练
         （stride=round(1/TEST_RATIO)=5 → 约 8:2，且测试集在族内有代表性）。
    """
    reg = _registry()
    names = sorted(n for n in reg if _matches(pattern, n))
    if not names:
        raise KeyError(f"无地图匹配 pattern={pattern!r}（检查 map_name 与 registry）。")
    stride = max(2, round(1.0 / TEST_RATIO))
    train, test = [], []
    for i, name in enumerate(names):
        (test if i % stride == 0 else train).append((name, reg[name]))
    if split == "train":
        pool = train
    elif split in ("test", "eval"):
        pool = test
    elif split == "all":
        pool = [(n, reg[n]) for n in names]
    else:
        raise ValueError(f"unknown split: {split!r}")
    if not pool:
        raise KeyError(f"split={split!r} 池为空 (pattern={pattern!r})。")
    return tuple(pool)


def pick_map(pattern: str, split: str, seed: int) -> Tuple[str, str]:
    """按 seed 从 (pattern, split) 池中确定性地取一张地图 (name, map_str)。

    用 Knuth 乘法哈希打散：池按名称排序，直接 seed%n 会让连续 seed（如一个 rollout
    batch 的 base+i）扎堆在同族区段；乘法哈希让相邻 seed 跨族均匀分布。仍完全确定性。
    """
    pool = get_pool(pattern, split)
    idx = (int(seed) * 2654435761) % len(pool)
    return pool[idx]
