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

import os
import numpy as np
from numpy.lib.format import open_memmap
from typing import Dict, List, Optional


# mmap 模式下每写入这么多 episode 就强制 flush（msync）一次。
# 2026-08-10 事故：批量 load_from_arrays 把 15000/12000 episode 逐个写进磁盘映射
# buffer，Python 写入速度远快于磁盘实际写回速度，脏页在页缓存里无节制堆积，
# 直到把 62GB cgroup 硬限额吃满、被内核 OOM-kill（日志无任何输出/无 traceback，
# 是 SIGKILL 的典型症状；数据生成阶段没炸是因为 LaCAM 求解天然拖慢了写入节奏，
# 训练 buffer 加载是纯拷贝循环，没有类似的天然节流）。定期 flush 强制同步落盘，
# 用阻塞等 IO 换取脏页堆积有上限。
_MMAP_FLUSH_EVERY_EPISODES = 100

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
                 obs_channels: int = 4, patch_size: int = 11,
                 mmap_dir: Optional[str] = None):
        """
        Args:
            mmap_dir: 若给定，所有列式数组改为磁盘内存映射（`.npy`，落在此目录），
                不再一次性占用等于 capacity 大小的常驻 RAM——写入/采样时按需由
                OS 换入换出。用于 62GB 内存装不下整族数据、但磁盘够大的场景
                （见 CLAUDE.md 服务器迁移记忆）。默认 None = 原地 np.zeros
                常驻内存（数据量小、追求采样速度时用）。
        """
        self.capacity = capacity            # 时间步数（每步含 N 体）
        self.num_agents = num_agents
        self.obs_channels = obs_channels
        self.patch_size = patch_size
        self.mmap_dir = mmap_dir

        self._episodes_since_flush = 0      # mmap 模式下批量写入的节流计数器
        self._size = 0                      # 当前有效时间步数
        self._head = 0                      # 下一次写入位置（循环）
        self._arrays_inited = False
        self._episode_counter = 0           # 全局递增 episode id 分配器

        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # 数组初始化
    # ------------------------------------------------------------------

    def _alloc(self, name: str, shape: tuple, dtype, fill=0):
        """按 mmap_dir 是否设置，分配磁盘映射数组或常驻内存数组。"""
        if self.mmap_dir is None:
            arr = np.empty(shape, dtype=dtype)
        else:
            os.makedirs(self.mmap_dir, exist_ok=True)
            path = os.path.join(self.mmap_dir, f"{name}.npy")
            arr = open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        arr[...] = fill
        return arr

    def _init_arrays(self, num_agents: int, obs_channels: int, K: int):
        """第一次 add 时初始化所有列式数组。"""
        N, C, cap = num_agents, obs_channels, self.capacity
        self.num_agents = N
        self.obs_channels = C
        self.patch_size = K

        self._obs_patches       = self._alloc("obs_patches",      (cap, N, C, K, K), np.uint8)
        self._next_obs_patches  = self._alloc("next_obs_patches", (cap, N, C, K, K), np.uint8)
        self._obs_scalars       = self._alloc("obs_scalars",      (cap, N, 8),       np.float32)
        self._next_obs_scalars  = self._alloc("next_obs_scalars", (cap, N, 8),       np.float32)
        self._positions         = self._alloc("positions",        (cap, N, 2),       np.float32)
        self._next_positions    = self._alloc("next_positions",   (cap, N, 2),       np.float32)
        self._actions_expert    = self._alloc("actions_expert",   (cap, N),          np.int64)
        self._rewards           = self._alloc("rewards",          (cap, N),          np.float32)
        self._dones              = self._alloc("dones",           (cap, N),          bool)
        self._valid              = self._alloc("valid",           (cap, N),          bool)
        self._action_masks      = self._alloc("action_masks",     (cap, N, 5),       bool, fill=True)
        self._next_action_masks = self._alloc("next_action_masks", (cap, N, 5),      bool, fill=True)
        self._margin_excluded   = self._alloc("margin_excluded",   (cap, N),         bool)
        self._margin_weights    = self._alloc("margin_weights",    (cap, N),         np.float32, fill=1)
        self._episode_ids       = self._alloc("episode_ids",       (cap,),           np.int64, fill=-1)

        self._arrays_inited = True

    # ------------------------------------------------------------------
    # 零拷贝挂载（大字段直接只读 mmap 源目录，不重新分配/搬运）
    # ------------------------------------------------------------------

    # 源目录（scripts/generate_lacam_data.py 落盘格式）里 dtype 与本 buffer 一致的
    # 字段——可以直接对源 .npy 只读 mmap，不必再拷一份。互不相同的字段（源用更省盘
    # 的窄 dtype：float16/int8/int32）体积很小（合计约数据总量的 6%），用小的常驻
    # RAM 数组即可，没必要也 mmap。
    _ZERO_COPY_KEYS = ("obs_patches", "next_obs_patches", "dones", "valid",
                       "action_masks", "next_action_masks", "margin_excluded")
    _SMALL_CAST_KEYS = ("obs_scalars", "next_obs_scalars", "positions", "next_positions",
                        "actions_expert", "rewards", "margin_weights", "episode_ids")
    _TARGET_DTYPES = {
        "obs_scalars": np.float32, "next_obs_scalars": np.float32,
        "positions": np.float32, "next_positions": np.float32,
        "actions_expert": np.int64, "rewards": np.float32,
        "margin_weights": np.float32, "episode_ids": np.int64,
    }

    def attach_disk_dir(self, base_path: str, load_next: bool = False) -> int:
        """
        零拷贝挂载一个 `expert_{fam}/` 目录（每字段一个独立 .npy，见
        scripts/generate_lacam_data.py 的落盘格式）为这个（空）buffer 的存储。

        2026-08-10 事故：原来的路径是 `_init_arrays` 建一份新的 mmap，再靠
        `add_episode` 逐 episode 从源目录拷过来——大字段(obs_patches 等)源/目标
        dtype 本来就相同，这一步拷贝纯属浪费：多占一倍磁盘（148GB 数据在 75GB
        剩余空间的盘上放不下）、且拷贝速度远超磁盘真实写回速度，脏页无界堆积把
        62GB 内存硬限额打穿、被内核 OOM-kill。
        现在：dtype 相同的大字段直接对源文件开只读 mmap，压根不拷贝、不占用额外
        磁盘、不产生任何脏页；只有窄 dtype 的小字段（合计约数据总量的 6%）才实际
        读进内存做 dtype 提升。仅适用于"整族一次性只读加载"场景（IL 训练标准用
        法）——载入后不能再 add_episode（大字段是只读 mmap）。
        """
        if self._arrays_inited:
            raise ValueError("attach_disk_dir 只能在空 buffer 上调用一次")

        def _src(key):
            return np.load(os.path.join(base_path, f"{key}.npy"), mmap_mode="r")

        obs_patches = _src("obs_patches")
        T, N, C, K = obs_patches.shape[0], obs_patches.shape[1], obs_patches.shape[2], obs_patches.shape[-1]
        self.num_agents, self.obs_channels, self.patch_size = N, C, K
        self.capacity = T

        # next_* 字段只有 DQfD 的 TD 项要用；纯 IL（v3+ 主线）从不读，且 next_obs_patches
        # 单项就占每族 ~39GB。故这些字段**允许在磁盘上不存在**：缺失时置 None，
        # sample(need_next=True) 会明确报错，而不是在这里就加载失败。
        # load_next=False（默认）：连"存在的" next_* 文件也不挂载 —— 纯 IL 不需要，
        # 而 mmap 住一个 39GB/族 的文件会让它无法被删除释放（被打开的 inode 不释放空间）。
        missing = []
        for key in self._ZERO_COPY_KEYS:
            if key == "obs_patches":
                setattr(self, f"_{key}", obs_patches); continue
            if key.startswith("next_") and not load_next:
                setattr(self, f"_{key}", None); missing.append(key); continue
            try:
                setattr(self, f"_{key}", _src(key))
            except FileNotFoundError:
                setattr(self, f"_{key}", None); missing.append(key)
        for key in self._SMALL_CAST_KEYS:
            if key.startswith("next_") and not load_next:
                setattr(self, f"_{key}", None); missing.append(key); continue
            try:
                setattr(self, f"_{key}", np.asarray(_src(key)).astype(self._TARGET_DTYPES[key]))
            except FileNotFoundError:
                setattr(self, f"_{key}", None); missing.append(key)
        if missing:
            assert all(k.startswith("next_") for k in missing), \
                f"缺失的必需字段（非 next_*）: {[k for k in missing if not k.startswith('next_')]}"
            self._next_fields_missing = tuple(missing)

        self._head = 0
        self._size = T
        self._episode_counter = int(self._episode_ids[-1]) + 1 if T else 0
        self._arrays_inited = True
        return T

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

        if self.mmap_dir is not None:
            self._episodes_since_flush += 1
            if self._episodes_since_flush >= _MMAP_FLUSH_EVERY_EPISODES:
                self._flush_mmap()

    def _flush_mmap(self):
        """强制把所有磁盘映射数组的脏页同步落盘（msync），节流写入速度，防止脏页无界堆积。"""
        for name in ("_obs_patches", "_next_obs_patches", "_obs_scalars", "_next_obs_scalars",
                     "_positions", "_next_positions", "_actions_expert", "_rewards",
                     "_dones", "_valid", "_action_masks", "_next_action_masks",
                     "_margin_excluded", "_margin_weights", "_episode_ids"):
            arr = getattr(self, name, None)
            if isinstance(arr, np.memmap):
                arr.flush()
        self._episodes_since_flush = 0

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
        if self.mmap_dir is not None:
            self._flush_mmap()   # 收尾：确保载入循环结束时脏页已完全落盘

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
        need_next: bool = True,
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

        # patches 保留 uint8（不在 CPU 上 astype/255）：
        # 实测 CPU 端 astype/255 占采样 ~75% 耗时（334/441 ms），且 float32 传输量是 uint8 的 4×。
        # 改为传 uint8 到 GPU、在 GPU 上 .float()/255（实测整条 patch 管道快 ~5.9×）。
        # 消费方（trainer）负责 GPU 端归一化（见 _patch_to_gpu）。
        obs_p  = self._obs_patches[idx_mat]        # (B,T,N,C,K,K) uint8

        out = {
            "obs_patches":       obs_p,
            "obs_scalars":       self._obs_scalars[idx_mat].copy(),
            "positions":         self._positions[idx_mat].copy(),
            "actions_expert":    self._actions_expert[idx_mat].copy(),
            "valid":             self._valid[idx_mat].copy(),
            "action_masks":      self._action_masks[idx_mat].copy(),
            "margin_excluded":   self._margin_excluded[idx_mat].copy(),
            "margin_weights":    self._margin_weights[idx_mat].copy(),
        }
        # next_obs/rewards/dones 仅 DQfD 的 TD 项需要；纯 IL 用不到 → 默认可跳过，
        # 省掉 ~一半 CPU gather/copy（next_obs_patches 单项即 ~148MB/步）。
        if need_next and getattr(self, "_next_fields_missing", None):
            raise ValueError(
                f"need_next=True 但该数据集未存 next_* 字段（缺 {self._next_fields_missing}）。"
                "纯 IL 请用 need_next=False；要跑 DQfD/TD 需重新生成含 next_* 的数据。")
        if need_next:
            out.update({
                "rewards":           self._rewards[idx_mat].copy(),
                "next_obs_patches":  self._next_obs_patches[idx_mat],
                "next_obs_scalars":  self._next_obs_scalars[idx_mat].copy(),
                "next_positions":    self._next_positions[idx_mat].copy(),
                "dones":             self._dones[idx_mat].copy(),
                "next_action_masks": self._next_action_masks[idx_mat].copy(),
            })
        return out

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
