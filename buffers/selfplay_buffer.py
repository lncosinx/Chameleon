"""自游走回放缓冲（阶段二 off-policy 自游走数据）。"""
import numpy as np
from collections import deque
from typing import Dict, List, Optional


class SelfPlayBuffer:
    """循环队列式自游走缓冲。"""

    def __init__(self, capacity: int = 100_000):
        self._buffer = deque(maxlen=capacity)

    def add(self, sample: dict):
        self._buffer.append(sample)

    def add_batch(self, samples: List[dict]):
        for s in samples:
            self._buffer.append(s)

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> List[dict]:
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.integers(0, len(self._buffer), size=min(n, len(self._buffer)))
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self):
        self._buffer.clear()
