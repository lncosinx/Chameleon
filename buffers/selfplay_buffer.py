"""自游走回放缓冲（阶段二 off-policy 自游走数据）。

线程安全：阶段二异步 self-play producer 在后台线程 add，trainer 在主/预取线程
sample（见 trainers/selfplay_producer.py 重叠设计）。deque 的单次 append 虽因 GIL
原子，但 sample 内 len()+多次按索引取值跨并发 append-evict 会读到偏移/重复元素，
故用 Lock 包住所有 deque 操作。锁仅护 deque 本身；sample 返回的 dict 引用在锁外使用
安全（即使随后被 evict，Python 引用计数保活）。
"""
import numpy as np
import threading
from collections import deque
from typing import List, Optional


class SelfPlayBuffer:
    """循环队列式自游走缓冲（线程安全）。"""

    def __init__(self, capacity: int = 100_000):
        self._buffer = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def add(self, sample: dict):
        with self._lock:
            self._buffer.append(sample)

    def add_batch(self, samples: List[dict]):
        with self._lock:
            for s in samples:
                self._buffer.append(s)

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> List[dict]:
        if rng is None:
            rng = np.random.default_rng()
        with self._lock:
            size = len(self._buffer)
            if size == 0:
                return []
            indices = rng.integers(0, size, size=min(n, size))
            return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self):
        with self._lock:
            self._buffer.clear()
