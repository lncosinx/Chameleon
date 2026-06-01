"""
专家演示回放缓冲（列式预分配，uint8 patches）。

存储格式：所有字段预分配为连续 numpy 数组（columnar storage），
patches 用 uint8 存储（采样时才归一化为 float32），消除 per-sample
Python 对象开销。

内存估算（capacity=4M）：
  obs_patches    uint8  (4M, 4, 11, 11)  → 1.94 GB
  next_obs_patch uint8  (4M, 4, 11, 11)  → 1.94 GB
  obs_scalars   float32 (4M, 8)          → 0.13 GB
  next_obs_scalars float32 (4M, 8)       → 0.13 GB
  其余小字段                              → ~0.10 GB
  合计 ≈ 4.2 GB（vs 旧 Python dict 方案的 50-60 GB）
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


# 如果加载时 K 尚未确定，先用占位容量 0，第一次 add 时自动 _init_arrays。
_UNINIT_K = -1


class ExpertDemoBuffer:
    """
    专家演示回放缓冲（循环队列，列式预分配）。

    用法:
        buf = ExpertDemoBuffer(capacity=4_000_000)
        buf.add(sample)   # sample dict 字段见 _init_arrays
        batch = buf.sample(batch_size=32, sequence_len=32, burn_in_len=8)
    """

    def __init__(self, capacity: int = 500_000, obs_channels: int = 4, patch_size: int = 11):
        self.capacity = capacity
        self.obs_channels = obs_channels
        self.patch_size = patch_size

        self._size = 0      # 当前有效样本数
        self._head = 0      # 下一次写入位置（循环）
        self._arrays_inited = False

        self._rng = np.random.default_rng()

    def _init_arrays(self, obs_channels: int, K: int):
        """第一次 add 时（或手动调用）初始化所有列式数组。"""
        C = obs_channels
        cap = self.capacity
        self.obs_channels = C
        self.patch_size = K

        self._obs_patches       = np.zeros((cap, C, K, K), dtype=np.uint8)
        self._next_obs_patches  = np.zeros((cap, C, K, K), dtype=np.uint8)
        self._obs_scalars       = np.zeros((cap, 8),        dtype=np.float32)
        self._next_obs_scalars  = np.zeros((cap, 8),        dtype=np.float32)
        self._actions_expert    = np.zeros(cap,             dtype=np.int64)
        self._rewards           = np.zeros(cap,             dtype=np.float32)
        self._dones             = np.zeros(cap,             dtype=bool)
        self._action_masks      = np.ones((cap, 5),         dtype=bool)
        self._next_action_masks = np.ones((cap, 5),         dtype=bool)
        self._margin_excluded   = np.zeros(cap,             dtype=bool)
        self._margin_weights    = np.ones(cap,              dtype=np.float32)

        self._arrays_inited = True

    def _ensure_init(self, sample: dict):
        if not self._arrays_inited:
            p = sample.get("obs_patch")
            if p is not None:
                arr = np.asarray(p)
                C = arr.shape[0] if arr.ndim == 3 else self.obs_channels
                K = arr.shape[-1] if arr.ndim >= 2 else self.patch_size
            else:
                C, K = self.obs_channels, self.patch_size
            self._init_arrays(C, K)

    def _patch_to_uint8(self, p) -> np.ndarray:
        """将 patch 转为 uint8（输入可以是 float [0,1] 或 uint8）。"""
        if p is None:
            return np.zeros((self.obs_channels, self.patch_size, self.patch_size), dtype=np.uint8)
        arr = np.asarray(p)
        if arr.dtype == np.uint8:
            return arr
        # float [0,1] → uint8 [0,255]
        return (arr * 255).clip(0, 255).astype(np.uint8)

    def add(self, sample: dict):
        """添加单条样本（per-agent per-step dict）。"""
        self._ensure_init(sample)
        idx = self._head

        self._obs_patches[idx]       = self._patch_to_uint8(sample.get("obs_patch"))
        self._next_obs_patches[idx]  = self._patch_to_uint8(sample.get("next_obs_patch"))

        sc = sample.get("obs_scalar")
        self._obs_scalars[idx]       = np.asarray(sc, dtype=np.float32) if sc is not None else 0.0
        nsc = sample.get("next_obs_scalar")
        self._next_obs_scalars[idx]  = np.asarray(nsc, dtype=np.float32) if nsc is not None else 0.0

        self._actions_expert[idx]    = int(sample.get("action_expert", 0))
        self._rewards[idx]           = float(sample.get("reward", 0.0))
        self._dones[idx]             = bool(sample.get("done", False))

        am = sample.get("action_mask")
        self._action_masks[idx]      = np.asarray(am, dtype=bool) if am is not None else True
        nam = sample.get("next_action_mask")
        self._next_action_masks[idx] = np.asarray(nam, dtype=bool) if nam is not None else True

        self._margin_excluded[idx]   = bool(sample.get("margin_excluded", False))
        self._margin_weights[idx]    = float(sample.get("margin_weight", 1.0))

        self._head = (self._head + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, samples: List[dict]):
        """批量添加样本。"""
        for s in samples:
            self.add(s)

    def __len__(self) -> int:
        return self._size

    def sample(
        self,
        batch_size: int,
        sequence_len: int = 32,
        burn_in_len: int = 8,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, np.ndarray]:
        """
        采样一个 batch，返回 float32 tensors。

        返回字段：
            obs_patches:      (B, seq_len, C, K, K) float32 [0,1]
            obs_scalars:      (B, seq_len, 8)
            actions_expert:   (B, seq_len) int64
            rewards:          (B, seq_len)
            next_obs_patches: (B, seq_len, C, K, K) float32 [0,1]
            next_obs_scalars: (B, seq_len, 8)
            dones:            (B, seq_len) bool
            action_masks:     (B, seq_len, 5) bool
            next_action_masks:(B, seq_len, 5) bool
            margin_excluded:  (B, seq_len) bool
            margin_weights:   (B, seq_len)
        """
        if not self._arrays_inited:
            raise ValueError("Buffer empty, cannot sample")
        if rng is None:
            rng = self._rng

        total_needed = batch_size * sequence_len
        if self._size < total_needed:
            raise ValueError(
                f"Buffer has {self._size} samples, need at least {total_needed}"
            )

        C, K = self.obs_channels, self.patch_size

        # 有效写入区域：[0, _size)，circular 时需要考虑绕回
        # 简化：直接从 [0, _size - sequence_len] 采样起始点
        # （不跨越写入头，避免采到部分覆盖的旧数据）
        max_start = self._size - sequence_len
        start_indices = rng.integers(0, max_start, size=batch_size)

        # 利用 numpy fancy indexing 批量取连续段（避免 Python 循环）
        # idx_mat: (B, seq_len) 每行是一段连续索引
        idx_mat = (start_indices[:, None] + np.arange(sequence_len)[None, :]) % self.capacity

        # uint8 → float32 归一化（在采样时统一做，不在存储时）
        obs_p  = self._obs_patches[idx_mat].astype(np.float32) / 255.0      # (B, T, C, K, K)
        nobs_p = self._next_obs_patches[idx_mat].astype(np.float32) / 255.0

        return {
            "obs_patches":      obs_p,
            "obs_scalars":      self._obs_scalars[idx_mat].copy(),
            "actions_expert":   self._actions_expert[idx_mat].copy(),
            "rewards":          self._rewards[idx_mat].copy(),
            "next_obs_patches": nobs_p,
            "next_obs_scalars": self._next_obs_scalars[idx_mat].copy(),
            "dones":            self._dones[idx_mat].copy(),
            "action_masks":     self._action_masks[idx_mat].copy(),
            "next_action_masks":self._next_action_masks[idx_mat].copy(),
            "margin_excluded":  self._margin_excluded[idx_mat].copy(),
            "margin_weights":   self._margin_weights[idx_mat].copy(),
        }

    def get_stats(self) -> dict:
        if self._size == 0:
            return {"size": 0}
        valid = slice(0, self._size)
        wait_ratio = float((self._actions_expert[valid] == 0).mean())
        excl_ratio = float(self._margin_excluded[valid].mean())
        return {
            "size": self._size,
            "wait_ratio": wait_ratio,
            "margin_excluded_ratio": excl_ratio,
        }

    def clear(self):
        self._size = 0
        self._head = 0
