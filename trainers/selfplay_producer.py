"""阶段二自游走 producer：用**当前 θ_meta** 在四族对齐地图上采样，填充 sp_buf。

设计要点（回应"别又变成策略迟滞"）：
- actor 每次 rollout **前**同步 `load_weights(当前 θ_meta)` → 即取最新元网络，
  rollout 期间仅作冻结快照（多步批量推理不能让权重中途变），**非长期滞后副本**。
- sp_buf 跨 meta-iter 复用是 **off-policy replay**（INVARIANT 9 阶段二本意），且 θ_meta
  外更新极小、K 轮内几乎不变，故复用数据≈当前策略产生，**非阶段三那种有害权重滞后**。

对齐（回应"地图对齐"）：
- **每族一个 VecEnvManager**（regime 锁 num_agents=96 / map_size=64，与专家数据一致），
  避免改 env_worker；每次 rollout 遍历四族 → sp_buf 覆盖全部家族，与专家 buffer 同分布。
- reward 由 env_worker 从各族 env 配置 reward 块构建，与专家生成同标度。

段格式由 VecEnvManager._build_seg 产出，直接喂 collate_selfplay_segments → dqfd_sequence_loss。
"""
from __future__ import annotations

import threading
from copy import deepcopy
from typing import Dict, List, Optional

import torch

from models.biintent_drqn import BiIntentGATDRQN


class SelfPlayProducer:
    """周期性用 θ_meta 采四族自游走段，推入 sp_buf。enable=False 时全为 no-op。"""

    def __init__(self, net_template: BiIntentGATDRQN, cfg: dict, sp_buf, device):
        sp_cfg = cfg.get("selfplay", {})
        self.enabled: bool = bool(sp_cfg.get("enable", False))
        self.sp_buf = sp_buf
        self.managers: Dict[str, object] = {}
        self.actor = None
        # 异步重叠：后台 daemon 线程持续 rollout→推 sp_buf，主循环只非阻塞 sync_weights。
        # _weight_lock 护 _pending_weights 快照；_new_weights 事件触发一次 rollout 周期
        # （≤1 周期/次 sync → 数据率 = 原同步版的「1 周期/K iter」，但与训练重叠）。
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._new_weights = threading.Event()
        self._weight_lock = threading.Lock()
        self._pending_weights: Optional[dict] = None
        self._async = False
        if not self.enabled:
            return

        self.K: int = int(sp_cfg.get("rollout_interval", 4))         # 每隔 K 个 meta-iter 采一次
        self.n_env: int = int(sp_cfg.get("n_env_per_family", 2))     # 每族并行 env 数
        self.num_agents: int = int(sp_cfg.get("num_agents", 96))     # 与专家数据对齐
        self.map_size: int = int(sp_cfg.get("map_size", 64))         # 与专家数据对齐
        self.seg_len: int = int(sp_cfg.get("segment_len", cfg.get("sequence_len", 32)))
        self.obs_radius: int = int(sp_cfg.get("obs_radius", 5))
        self.families: List[str] = list(sp_cfg.get("families", ["random", "warehouse", "maze", "room"]))
        self.env_config_tpl: str = sp_cfg.get("env_config_tpl", "configs/env_{fam}.yaml")
        self.device = torch.device(device)
        self._seed = int(sp_cfg.get("seed_base", 1_000_003))

        # actor：θ_meta 的冻结快照（每次 rollout 前同步），跑在 device（rollout 与 inner loop 不重叠）
        from trainers.seed_actor import SeedActor
        actor_net = deepcopy(net_template).to(self.device)
        use_amp = bool(cfg.get("parallel", {}).get("use_amp", False))
        self.actor = SeedActor(actor_net, self.device, use_amp=use_amp)

        # 四族各一个 VecEnvManager（spawn 一池 CPU worker，绝不碰 CUDA）
        from envs.env_worker import VecEnvManager
        for fam in self.families:
            worker_cfg = {
                "env_config": self.env_config_tpl.format(fam=fam),
                "map_size": self.map_size,
            }
            self.managers[fam] = VecEnvManager(
                n_env=self.n_env, num_agents=self.num_agents,
                segment_len=self.seg_len, obs_radius=self.obs_radius,
                worker_cfg=worker_cfg,
            )

    def _next_seeds(self, n: int) -> List[int]:
        seeds = [self._seed + i for i in range(n)]
        self._seed += n
        return seeds

    def _rollout_once(self) -> int:
        """遍历四族各采一次 rollout → 段推入 sp_buf。返回新增段数。actor 权重须已 load。"""
        n_added = 0
        for fam, mgr in self.managers.items():
            seeds = self._next_seeds(self.n_env)
            try:
                segs = mgr.rollout_selfplay(self.actor, seeds)
            except Exception as e:  # noqa: BLE001
                print(f"[selfplay] {fam} rollout failed: {e!r}", flush=True)
                continue
            for s in segs:
                if s is not None:
                    self.sp_buf.add(s)
                    n_added += 1
        return n_added

    def maybe_rollout(self, meta_iter: int, theta_meta_state: dict) -> int:
        """【同步路径，保留兼容】每隔 K 个 meta-iter：同步 θ_meta → 四族 rollout。"""
        if not self.enabled or meta_iter % self.K != 0:
            return 0
        self.actor.load_weights(theta_meta_state)   # 取**当前** θ_meta（杜绝滞后副本）
        return self._rollout_once()

    # ===== 异步重叠路径（推荐）=====
    def start(self):
        """启动后台 rollout 线程（enable=False 时 no-op）。"""
        if not self.enabled or self._thread is not None:
            return
        self._async = True
        self._thread = threading.Thread(target=self._run_loop, name="selfplay-producer", daemon=True)
        self._thread.start()

    def sync_weights(self, theta_meta_state: dict):
        """主循环非阻塞调用：存一份 θ_meta 的 CPU 快照，触发后台一次 rollout 周期。

        多次 sync 间若后台还没消费，只保留最新快照（事件去重）→ 后台用最新权重、丢中间帧，
        仍是 off-policy replay（K-stale 内），符合 INVARIANT 9 阶段二容忍度。
        """
        if not self.enabled:
            return
        # CPU 快照（detach + clone），与训练张量解耦，跨线程安全
        snap = {k: v.detach().to("cpu", copy=True) for k, v in theta_meta_state.items()}
        with self._weight_lock:
            self._pending_weights = snap
        self._new_weights.set()

    def _run_loop(self):
        """后台线程：等新权重 → load → 一次四族 rollout → 循环。"""
        while not self._stop_event.is_set():
            # 等 sync_weights 信号（超时轮询以便响应 stop）
            if not self._new_weights.wait(timeout=0.5):
                continue
            self._new_weights.clear()
            with self._weight_lock:
                snap = self._pending_weights
                self._pending_weights = None
            if snap is None:
                continue
            try:
                self.actor.load_weights(snap)
                self._rollout_once()
            except Exception as e:  # noqa: BLE001
                print(f"[selfplay] async rollout failed: {e!r}", flush=True)

    def close(self):
        self._stop_event.set()
        self._new_weights.set()   # 唤醒可能在 wait 的线程
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        for mgr in self.managers.values():
            try:
                mgr.close()
            except Exception:  # noqa: BLE001
                pass
