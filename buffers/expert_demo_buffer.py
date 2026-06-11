"""
专家演示回放缓冲（**per-timestep × N-agent** 列式预分配，uint8 patches）。

存储单位由「单体单步」改为「单时间步的全部 N 体」（per-timestep group），
原因：去中心化通信模块（BiIntentAttention）需要**同一时刻多个共存 agent
的局部观测与全局坐标**才能计算邻居注意力。若每条样本只含单体（N=1），
通信候选集恒为空、comm_features 恒为 0，通信参数在 off-policy 阶段二
**永远得不到梯度**。改为 per-timestep 后，一次前向即覆盖该时刻全部 agent，
通信权重正常回传。

每行（= 一个时间步）字段，N = num_agents：
  obs_patches      uint8   (cap, N, C, K, K)
  next_obs_patches uint8   (cap, N, C, K, K)
  obs_scalars      float32 (cap, N, 8)
  next_obs_scalars float32 (cap, N, 8)
  positions        float32 (cap, N, 2)   ← 通信邻居距离用；finished agent 置远哨兵
  next_positions   float32 (cap, N, 2)
  actions_expert   int64   (cap, N)
  rewards          float32 (cap, N)
  dones            bool    (cap, N)       ← per-agent 到达标志（ever_done）
  valid            bool    (cap, N)       ← 该步该体是否参与 loss（finished→False）
  action_masks     bool    (cap, N, 5)
  next_action_masks bool   (cap, N, 5)
  margin_excluded  bool    (cap, N)
  margin_weights   float32 (cap, N)
  episode_ids      int64   (cap,)         ← 采样时保证序列段不跨 episode 边界

INVARIANT 13: 采样返回 (B, T, N, ...) 定长序列，配合 GRU burn-in。
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


# 远哨兵坐标：finished agent 的 position 置此值，使其与任何 agent 的距离 > comm_radius，
# 从而被通信模块的距离过滤自然排除（无需改 communication.py）。
# 取 1e4：远大于任何真实坐标(map_size≤64)与 comm_radius，且 < fp16 上限(65504)，
# 保证 AMP 混合精度下 position 差值不溢出为 inf/nan。
FINISHED_AGENT_SENTINEL = 1.0e4


class ExpertDemoBuffer:
    """
    专家演示回放缓冲（循环队列，per-timestep 列式预分配）。

    用法:
        buf = ExpertDemoBuffer(capacity=50_000)          # capacity = 时间步数
        buf.add_episode(ep_dict)                          # ep_dict 见 add_episode
        batch = buf.sample(batch_size=32, sequence_len=32, burn_in_len=8)
        # batch["obs_patches"]: (B, T, N, C, K, K) float32 [0,1]
    """

    def __init__(self, capacity: int = 50_000, num_agents: Optional[int] = None,
                 obs_channels: int = 4, patch_size: int = 11):
        self.capacity = capacity            # 时间步数（每步含 N 体）
        self.num_agents = num_agents
        self.obs_channels = obs_channels
        self.patch_size = patch_size

        self._size = 0                      # 当前有效时间步数
        self._head = 0                      # 下一次写入位置（循环）
        self._arrays_inited = False
        self._episode_counter = 0           # 全局递增 episode id 分配器

        self._rng = np.random.default_rng()

        # 底层存储数组的 torch 零拷贝视图缓存（GIL-free gather 用，见 sample/_torch_views）。
        self._tv_cache = None

    # ------------------------------------------------------------------
    # 数组初始化
    # ------------------------------------------------------------------

    def _init_arrays(self, num_agents: int, obs_channels: int, K: int):
        """第一次 add 时初始化所有列式数组。"""
        N, C, cap = num_agents, obs_channels, self.capacity
        self.num_agents = N
        self.obs_channels = C
        self.patch_size = K

        self._obs_patches       = np.zeros((cap, N, C, K, K), dtype=np.uint8)
        self._next_obs_patches  = np.zeros((cap, N, C, K, K), dtype=np.uint8)
        self._obs_scalars       = np.zeros((cap, N, 8),        dtype=np.float32)
        self._next_obs_scalars  = np.zeros((cap, N, 8),        dtype=np.float32)
        self._positions         = np.zeros((cap, N, 2),        dtype=np.float32)
        self._next_positions    = np.zeros((cap, N, 2),        dtype=np.float32)
        self._actions_expert    = np.zeros((cap, N),           dtype=np.int64)
        self._rewards           = np.zeros((cap, N),           dtype=np.float32)
        self._dones             = np.zeros((cap, N),           dtype=bool)
        self._valid             = np.zeros((cap, N),           dtype=bool)
        self._action_masks      = np.ones((cap, N, 5),         dtype=bool)
        self._next_action_masks = np.ones((cap, N, 5),         dtype=bool)
        self._margin_excluded   = np.zeros((cap, N),           dtype=bool)
        self._margin_weights    = np.ones((cap, N),            dtype=np.float32)
        self._episode_ids       = np.full(cap, -1,             dtype=np.int64)

        self._arrays_inited = True
        self._tv_cache = None   # 数组重建后失效 torch 视图缓存

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_to_uint8(p: np.ndarray) -> np.ndarray:
        """patch 转 uint8（输入可以是 float [0,1] 或 uint8）。"""
        arr = np.asarray(p)
        if arr.dtype == np.uint8:
            return arr
        return (arr * 255).clip(0, 255).astype(np.uint8)

    def add_episode(self, ep: Dict[str, np.ndarray], episode_id: Optional[int] = None):
        """
        添加一个完整 episode（T 个时间步，每步 N 体）。

        Args:
            ep: dict，含以下键（形状以 T=步数、N=体数为前缀）：
                obs_patches      (T, N, C, K, K)  float[0,1] 或 uint8
                next_obs_patches (T, N, C, K, K)
                obs_scalars      (T, N, 8)
                next_obs_scalars (T, N, 8)
                positions        (T, N, 2)
                next_positions   (T, N, 2)
                actions_expert   (T, N)
                rewards          (T, N)
                dones            (T, N)
                valid            (T, N)
                action_masks     (T, N, 5)
                next_action_masks(T, N, 5)
                margin_excluded  (T, N)
                margin_weights   (T, N)
            episode_id: 可选，外部指定的 episode 编号（用于 copy 时保持唯一）；
                        默认用内部递增计数器。
        """
        obs_p = np.asarray(ep["obs_patches"])
        T, N, C, K = obs_p.shape[0], obs_p.shape[1], obs_p.shape[2], obs_p.shape[-1]

        if not self._arrays_inited:
            self._init_arrays(N, C, K)
        elif N != self.num_agents:
            raise ValueError(
                f"Episode has {N} agents but buffer was initialized with "
                f"{self.num_agents}. All episodes must share the same N. "
                f"Regenerate data with a consistent num_agents."
            )

        if episode_id is None:
            episode_id = self._episode_counter
            self._episode_counter += 1
        else:
            self._episode_counter = max(self._episode_counter, episode_id + 1)

        cap = self.capacity
        # 若单个 episode 超过容量，只保留最后 cap 步
        if T > cap:
            sl = slice(T - cap, T)
            T = cap
        else:
            sl = slice(0, T)

        # 向量化块写入（最多两段，处理 circular 绕回）
        def _put(dst, src_key, transform=None):
            src = np.asarray(ep[src_key])[sl]
            if transform is not None:
                src = transform(src)
            head = self._head
            if head + T <= cap:
                dst[head:head + T] = src
            else:
                first = cap - head
                dst[head:cap] = src[:first]
                dst[0:T - first] = src[first:]

        _put(self._obs_patches,       "obs_patches",       self._patch_to_uint8)
        _put(self._next_obs_patches,  "next_obs_patches",  self._patch_to_uint8)
        _put(self._obs_scalars,       "obs_scalars")
        _put(self._next_obs_scalars,  "next_obs_scalars")
        _put(self._positions,         "positions")
        _put(self._next_positions,    "next_positions")
        _put(self._actions_expert,    "actions_expert")
        _put(self._rewards,           "rewards")
        _put(self._dones,             "dones")
        _put(self._valid,             "valid")
        _put(self._action_masks,      "action_masks")
        _put(self._next_action_masks, "next_action_masks")
        _put(self._margin_excluded,   "margin_excluded")
        _put(self._margin_weights,    "margin_weights")

        # episode_ids 块写入
        head = self._head
        if head + T <= cap:
            self._episode_ids[head:head + T] = episode_id
        else:
            first = cap - head
            self._episode_ids[head:cap] = episode_id
            self._episode_ids[0:T - first] = episode_id

        self._head = (self._head + T) % cap
        self._size = min(self._size + T, cap)

    def load_from_arrays(self, arrays: Dict[str, np.ndarray]):
        """
        从一组拼接的 per-timestep 数组批量载入（按 episode_ids 切分为 episode 逐个 add）。

        arrays 必须含 add_episode 的全部键 + "episode_ids" (T_total,)。
        各 episode 在数组中按时间连续排布。逐 episode 调用 add_episode（向量化块写入），
        天然处理容量上限与 circular 绕回。
        """
        ep_ids = np.asarray(arrays["episode_ids"])
        total = len(ep_ids)
        if total == 0:
            return
        keys = [
            "obs_patches", "next_obs_patches", "obs_scalars", "next_obs_scalars",
            "positions", "next_positions", "actions_expert", "rewards",
            "dones", "valid", "action_masks", "next_action_masks",
            "margin_excluded", "margin_weights",
        ]
        # 找 episode 边界（id 变化处）
        boundaries = np.where(np.diff(ep_ids) != 0)[0] + 1
        seg_starts = np.concatenate([[0], boundaries])
        seg_ends   = np.concatenate([boundaries, [total]])
        for s, e in zip(seg_starts, seg_ends):
            ep = {k: arrays[k][s:e] for k in keys}
            self.add_episode(ep)   # 内部分配新的全局 episode_id，避免跨文件冲突

    def __len__(self) -> int:
        return self._size

    @property
    def num_episodes(self) -> int:
        return self._episode_counter

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        sequence_len: int = 32,
        burn_in_len: int = 8,
        rng: Optional[np.random.Generator] = None,
        max_retries: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        采样一个 batch，返回 float32 tensors，形状均以 (B, T, N, ...) 为前缀。

        采样保证每条长度 sequence_len 的序列段**完整落在同一 episode 内**
        （端点 episode_id 相等 ⟹ 中间全相等，因 episode_id 全局递增且按 episode
        连续写入）。

        返回字段（patches 为 **uint8**，由消费方在 GPU 上 .float()/255，见 trainer）：
            obs_patches:       (B, T, N, C, K, K) uint8 [0,255]
            obs_scalars:       (B, T, N, 8)
            positions:         (B, T, N, 2)
            next_obs_patches:  (B, T, N, C, K, K) uint8 [0,255]
            next_obs_scalars:  (B, T, N, 8)
            next_positions:    (B, T, N, 2)
            actions_expert:    (B, T, N)
            rewards:           (B, T, N)
            dones:             (B, T, N)
            valid:             (B, T, N)
            action_masks:      (B, T, N, 5)
            next_action_masks: (B, T, N, 5)
            margin_excluded:   (B, T, N)
            margin_weights:    (B, T, N)
        """
        if not self._arrays_inited:
            raise ValueError("Buffer empty, cannot sample")
        if rng is None:
            rng = self._rng
        if self._size < sequence_len:
            raise ValueError(
                f"Buffer has {self._size} timesteps, need at least "
                f"sequence_len={sequence_len}"
            )

        max_start = self._size - sequence_len
        starts = []
        for _ in range(batch_size):
            chosen = None
            for _try in range(max_retries):
                s = int(rng.integers(0, max_start + 1))
                # 端点同 episode（不跨边界、不跨 circular 写头）
                if self._episode_ids[s] == self._episode_ids[s + sequence_len - 1] \
                        and self._episode_ids[s] >= 0:
                    chosen = s
                    break
            if chosen is None:
                # 退化：接受任意起点（极少发生，仅当 episode 普遍短于 sequence_len）
                chosen = int(rng.integers(0, max_start + 1))
            starts.append(chosen)
        start_indices = np.array(starts, dtype=np.int64)

        # (B, T) 连续索引（已保证不跨 episode，无需取模，但安全起见取模）
        idx_mat = (start_indices[:, None] + np.arange(sequence_len)[None, :]) % self.capacity

        # 用 torch 高级索引（index_select）替代 numpy fancy-index 做 gather。
        # 背景：DualGPU 异步预取线程的采样占 meta_step wall ~38%（profile 实测），且与
        # 两 GPU 线程争 GIL 难以完全重叠。**实测**真实 DualGPU 端到端：torch 版中位
        # 3001→2637 ms/iter（~12%）且方差更低（numpy 2817-3075 抖动 vs torch 2616-2909）。
        # 机制未完全坐实（torch 张量直接喂 _patch_to_gpu，省 numpy→torch 包裹；gather 更稳），
        # 但数值**逐位等价**（已验证 max|Δ|=0，同一 idx_mat）。注意：注释别再写"GIL-free"——
        # 校准测试显示 numpy fancy-index 反而更好地释放 GIL，torch CPU 索引持 GIL 更多；
        # 这里的收益不是来自 GIL 释放，而是端到端更快更稳的经验结果。真正吃掉那 38% 需进程级
        # 并行（共享内存 buffer + 每卡独立进程，破 GIL）——见 memory 提速记录。
        # torch.from_numpy 零拷贝包裹底层存储（缓存复用，见 _torch_views），index_select
        # 输出即新张量（语义等价原 numpy .copy()）。patches 仍保留 uint8（消费方 _patch_to_gpu
        # 在 GPU 上 .float()/255，省 4× 传输；见原注释）。
        import torch  # 惰性 import：env_worker 等 CPU-only 模块不会在导入期拉入 torch
        tv = self._torch_views()
        idx = torch.from_numpy(np.ascontiguousarray(idx_mat.reshape(-1)))  # (B*T,) int64
        B, T = idx_mat.shape
        N, C, K = self.num_agents, self.obs_channels, self.patch_size

        def _g(name, tail):
            return tv[name].index_select(0, idx).reshape(B, T, *tail)

        return {
            "obs_patches":       _g("obs_patches",       (N, C, K, K)),
            "obs_scalars":       _g("obs_scalars",       (N, 8)),
            "positions":         _g("positions",         (N, 2)),
            "actions_expert":    _g("actions_expert",    (N,)),
            "rewards":           _g("rewards",           (N,)),
            "next_obs_patches":  _g("next_obs_patches",  (N, C, K, K)),
            "next_obs_scalars":  _g("next_obs_scalars",  (N, 8)),
            "next_positions":    _g("next_positions",    (N, 2)),
            "dones":             _g("dones",             (N,)),
            "valid":             _g("valid",             (N,)),
            "action_masks":      _g("action_masks",      (N, 5)),
            "next_action_masks": _g("next_action_masks", (N, 5)),
            "margin_excluded":   _g("margin_excluded",   (N,)),
            "margin_weights":    _g("margin_weights",    (N,)),
        }

    def _torch_views(self) -> dict:
        """底层存储数组的 torch 零拷贝视图（缓存）。供 sample 的 GIL-free gather 用。

        torch.from_numpy 与 numpy 数组共享内存：循环 buffer 的就地写入对视图可见；
        阶段二专家 buffer 载入后只读，更无失效问题。_init_arrays 重建数组时清缓存。
        """
        if self._tv_cache is None:
            import torch
            self._tv_cache = {
                "obs_patches":       torch.from_numpy(self._obs_patches),
                "next_obs_patches":  torch.from_numpy(self._next_obs_patches),
                "obs_scalars":       torch.from_numpy(self._obs_scalars),
                "next_obs_scalars":  torch.from_numpy(self._next_obs_scalars),
                "positions":         torch.from_numpy(self._positions),
                "next_positions":    torch.from_numpy(self._next_positions),
                "actions_expert":    torch.from_numpy(self._actions_expert),
                "rewards":           torch.from_numpy(self._rewards),
                "dones":             torch.from_numpy(self._dones),
                "valid":             torch.from_numpy(self._valid),
                "action_masks":      torch.from_numpy(self._action_masks),
                "next_action_masks": torch.from_numpy(self._next_action_masks),
                "margin_excluded":   torch.from_numpy(self._margin_excluded),
                "margin_weights":    torch.from_numpy(self._margin_weights),
            }
        return self._tv_cache

    # ------------------------------------------------------------------
    # 统计与维护
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        if self._size == 0:
            return {"size": 0}
        v = self._valid[:self._size]
        acts = self._actions_expert[:self._size]
        valid_acts = acts[v]
        wait_ratio = float((valid_acts == 0).mean()) if valid_acts.size else 0.0
        excl = self._margin_excluded[:self._size][v]
        excl_ratio = float(excl.mean()) if excl.size else 0.0
        return {
            "size": self._size,
            "num_episodes": self._episode_counter,
            "num_agents": self.num_agents,
            "wait_ratio": wait_ratio,
            "margin_excluded_ratio": excl_ratio,
        }

    def clear(self):
        self._size = 0
        self._head = 0
        self._episode_counter = 0
