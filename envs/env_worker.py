"""
阶段三 SEED-RL：CPU env-worker + 共享内存同步批量采样基础设施。

设计（见设计文档 §4.5 / seedrl-plan）：
- **同步批量 actor**（非完整异步 SEED-RL 起步版）：N_env 个 CPU worker 各步进
  自己的 env + 构建观测，主进程把所有 env 的观测拼成一个 batch 在 GPU 上**一次**
  前向得动作，散回各 worker。
- **worker 绝不创建 CUDA 上下文**（CLAUDE.md 硬约束）：本模块**禁止 import torch**，
  只做 env / 观测 / 奖励 / LaCAM（均 CPU）。
- IPC：逐步交换的观测/动作张量走 `multiprocessing.shared_memory`（spawn 安全、零拷贝、
  uint?→这里用 float32 与单进程 _rollout_with_policy 完全同格式，避免量化误差）；
  专家段长度可变 → 走 Pipe 一次性回传。

同步协议（每个自游走 segment，长度 T = rollout_segment_len）：
    worker 第 t 步: 写 obs[slot,t] → obs_barrier → act_barrier → 读 action[slot,t]、
                    env.step、算 reward → 写 reward/done[slot,t]
    parent 第 t 步: obs_barrier → 读 obs[:,t]、批量前向、写 action[:,t] → act_barrier
    循环结束后双方各再 obs_barrier 一次作为「最后一步 reward 已写入」的完成同步。
专家 segment：parent 发 ('expert', seed)，worker 自主跑 LaCAM + build_dataset_from_rollout
            （不进 barrier），经 Pipe 回传 segment dict。
"""
from __future__ import annotations

import numpy as np
from collections import OrderedDict
from multiprocessing import shared_memory
from typing import Dict, List, Optional

from envs.pogema_env_wrapper import PogemaWrapper
from envs.observation_builder import ObservationBuilder, build_observations
from envs.reward_builder import RewardBuilder
from envs.action_mask import build_action_mask_batch
from buffers.expert_demo_buffer import FINISHED_AGENT_SENTINEL


# 自游走 segment 的列式键（与 RecentOnlineBuffer.SEGMENT_KEYS / build_dataset_from_rollout 同构）
def _make_specs(n_env: int, T: int, N: int, K: int, ch: int = 4) -> "OrderedDict[str, tuple]":
    """共享内存数组规格（双方确定性一致，故 worker 可凭 dims 重建）。"""
    return OrderedDict([
        ("patch",  ((n_env, T, N, ch, K, K), np.float32)),
        ("scalar", ((n_env, T, N, 8),        np.float32)),
        ("pos",    ((n_env, T, N, 2),        np.float32)),
        ("mask",   ((n_env, T, N, 5),        np.bool_)),
        ("action", ((n_env, T, N),           np.int64)),
        ("reward", ((n_env, T, N),           np.float32)),
        ("done",   ((n_env, T, N),           np.bool_)),
        ("valid",  ((n_env, T, N),           np.bool_)),
    ])


class SharedSeg:
    """
    一组 (N_env, T, N, ...) 的共享内存数组，供 worker↔parent 零拷贝交换。

    用 multiprocessing.shared_memory（spawn 安全）：parent create=True 持有句柄并在
    结束时 unlink；worker create=False 凭 names 附着。views() 在各进程本地按 buffer
    构建 numpy（不跨进程 pickle）。
    """

    def __init__(self, n_env: int, T: int, N: int, K: int, ch: int = 4,
                 create: bool = True, names: Optional[Dict[str, str]] = None):
        self.specs = _make_specs(n_env, T, N, K, ch)
        self.shm: Dict[str, shared_memory.SharedMemory] = {}
        self.names: Dict[str, str] = {}
        for k, (shape, dtype) in self.specs.items():
            if create:
                nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
                shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
            else:
                shm = shared_memory.SharedMemory(name=names[k])
            self.shm[k] = shm
            self.names[k] = shm.name

    def views(self) -> Dict[str, np.ndarray]:
        return {
            k: np.ndarray(shape, dtype=dtype, buffer=self.shm[k].buf)
            for k, (shape, dtype) in self.specs.items()
        }

    def close(self, unlink: bool = False):
        for shm in self.shm.values():
            try:
                shm.close()
                if unlink:
                    shm.unlink()
            except FileNotFoundError:
                pass


# ===========================================================================
# worker 进程主循环（必须是 top-level 函数以便 spawn pickle）
# ===========================================================================

def env_worker_loop(slot, cmd_conn, obs_barrier, act_barrier, shm_names, dims, worker_cfg):
    """
    单个 env worker。CPU only，**绝不 import / 触碰 torch / CUDA**。

    cmd_conn: 与 parent 的双工 Pipe 端，接收命令、回传专家段/错误。
    命令: ('selfplay', seed) / ('expert', seed) / ('close', None)
    """
    n_env, T, N, K, ch = dims["n_env"], dims["T"], dims["N"], dims["K"], dims["ch"]
    shared = SharedSeg(n_env, T, N, K, ch, create=False, names=shm_names)
    v = shared.views()

    # 构建 env / 观测 / 奖励 / 专家（均 CPU）
    wrapper = PogemaWrapper(worker_cfg["env_config"])
    wrapper.map_size = worker_cfg["map_size"]
    wrapper.num_agents = N
    obs_builder = ObservationBuilder(num_agents=N, obs_radius=worker_cfg["obs_radius"])
    # reward 必须与专家数据同标度（专家由 generate_lacam_data 经 RewardBuilder.from_config
    # 按 env 配置 reward 块生成）。优先从 env 配置读 reward 块；缺失才回退 lambda_f。
    _rcfg = getattr(wrapper, "_raw_cfg", {}).get("reward")
    reward_builder = (RewardBuilder.from_config(_rcfg) if _rcfg
                      else RewardBuilder(lambda_f=worker_cfg.get("lambda_f", 0.1)))

    runner = None  # 延迟构建 LaCAMRunner（仅专家路径需要）

    try:
        while True:
            cmd = cmd_conn.recv()
            kind = cmd[0]
            if kind == "close":
                break
            elif kind == "selfplay":
                seed = cmd[1]
                try:
                    _run_selfplay(slot, obs_barrier, act_barrier, v, wrapper,
                                  obs_builder, reward_builder, T, seed)
                except Exception as e:  # noqa: BLE001
                    # 中止 barrier，让 parent 的 wait 抛 BrokenBarrierError 而非死锁
                    obs_barrier.abort()
                    act_barrier.abort()
                    cmd_conn.send(("err", f"worker {slot} selfplay: {e!r}"))
            elif kind == "expert":
                seed = cmd[1]
                if runner is None:
                    from lacam.lacam_runner import LaCAMRunner
                    runner = LaCAMRunner(backend="auto")
                seg = _run_expert(wrapper, obs_builder, reward_builder, runner,
                                  worker_cfg["map_size"], seed)
                cmd_conn.send(("seg", seg))
            else:
                cmd_conn.send(("err", f"worker {slot}: unknown cmd {kind}"))
    finally:
        shared.close(unlink=False)


def _run_selfplay(slot, obs_barrier, act_barrier, v, wrapper, obs_builder,
                  reward_builder, T, seed):
    """
    自游走一段，逐步与 parent 的批量 actor 同步。

    与单进程 finetune_target_map._rollout_with_policy 的逐步语义**逐位一致**：
    obs 取 step 前状态、pos 对 finished agent 置哨兵、valid=未完成、reward step 后算。
    动作由 parent 写入 shared，本进程只负责 env 推进与 reward。
    """
    N = wrapper.num_agents
    obs_builder.reset()
    obs_list, _ = wrapper.reset(seed=int(seed))
    obstacles = wrapper.global_grid
    prev_actions = np.zeros(N, dtype=np.int64)
    prev_goals = wrapper.get_targets_xy()
    ever_done = [False] * N

    reward_builder.begin_episode()   # 归一 shaping：本段起点距离缓存
    for t in range(T):
        prev_ever = list(ever_done)
        agents_xy = wrapper.get_agents_xy()
        patches, scalars = build_observations(wrapper, obs_list, prev_actions, obs_builder)

        if obstacles is not None:
            agents_arr = np.array(agents_xy, dtype=np.int32)[np.newaxis]
            mask_np = build_action_mask_batch(agents_arr, obstacles)[0]
        else:
            mask_np = np.ones((N, 5), dtype=bool)

        pos = np.array(agents_xy, dtype=np.float32)
        for i in range(N):
            if prev_ever[i]:
                pos[i] = FINISHED_AGENT_SENTINEL

        v["patch"][slot, t] = patches
        v["scalar"][slot, t] = scalars
        v["pos"][slot, t] = pos
        v["mask"][slot, t] = mask_np
        for i in range(N):
            v["valid"][slot, t, i] = not prev_ever[i]

        # —— 与 parent 同步：obs 就绪 → 等动作 ——
        obs_barrier.wait()
        act_barrier.wait()
        actions = np.asarray(v["action"][slot, t], dtype=np.int64).copy()

        obs_list, rewards, dones, _ = wrapper.step(actions.tolist())

        curr_agents = wrapper.get_agents_xy()
        curr_goals = wrapper.get_targets_xy()
        if obstacles is not None:
            shaped, _ = reward_builder.compute(
                obstacles=obstacles,
                prev_agents_xy=agents_xy,
                agents_xy=curr_agents,
                targets_xy=curr_goals,
                prev_targets_xy=prev_goals,
                prev_actions=prev_actions,
                prev_dones=prev_ever,
            )
            v["reward"][slot, t] = np.asarray(shaped, dtype=np.float32)
        else:
            v["reward"][slot, t] = np.asarray([float(r) for r in rewards], dtype=np.float32)

        step_dones = dones if isinstance(dones, list) else [bool(dones)] * N
        for i in range(N):
            if step_dones[i]:
                ever_done[i] = True
        v["done"][slot, t] = np.array(ever_done, dtype=bool)

        prev_actions = actions
        prev_goals = curr_goals

    # 完成同步：保证最后一步 reward/done 已写入后 parent 才读取
    obs_barrier.wait()


def _run_expert(wrapper, obs_builder, reward_builder, runner, map_size, seed):
    """专家路径：LaCAM 求解 + 重放构段（与单进程 finetune 专家分支一致）。CPU only。"""
    from lacam.lacam_dataset_builder import build_dataset_from_rollout

    wrapper.reset(seed=int(seed))
    obstacles = wrapper.global_grid
    if obstacles is None:
        return None
    starts = wrapper.get_agents_xy()
    goals = wrapper.get_targets_xy()
    result = runner.solve(obstacles, starts, goals,
                          map_type=wrapper.map_type, map_size=map_size)
    if not result.success:
        return None
    wrapper.reset(seed=int(seed))
    return build_dataset_from_rollout(
        wrapper, obs_builder, reward_builder, obstacles,
        result.paths, wrapper.map_type, f"ft_{seed}",
    )


# ===========================================================================
# parent 侧管理器
# ===========================================================================

class VecEnvManager:
    """
    spawn 一池 CPU env-worker，编排同步批量自游走 / 并行专家采样。

    **本类不 import torch**；批量前向由调用方传入的 actor 对象（持 GPU 网络）完成，
    actor 需提供 reset(n_env, n_agents) 与 step(patches, scalars, positions, masks)->(n_env,N) int。
    """

    def __init__(self, n_env: int, num_agents: int, segment_len: int,
                 obs_radius: int, worker_cfg: dict, ctx=None,
                 barrier_timeout: float = 120.0):
        import multiprocessing as mp
        self.ctx = ctx or mp.get_context("spawn")
        self.n_env = n_env
        self.num_agents = num_agents
        self.T = segment_len
        self.K = 2 * obs_radius + 1
        self.ch = 4
        self.barrier_timeout = barrier_timeout

        self.shared = SharedSeg(n_env, self.T, num_agents, self.K, self.ch, create=True)
        self.views = self.shared.views()

        # parties = n_env workers + 1 parent
        self.obs_barrier = self.ctx.Barrier(n_env + 1)
        self.act_barrier = self.ctx.Barrier(n_env + 1)

        dims = {"n_env": n_env, "T": self.T, "N": num_agents, "K": self.K, "ch": self.ch}
        worker_cfg = {**worker_cfg, "obs_radius": obs_radius, "map_size": worker_cfg["map_size"]}

        self.conns = []
        self.procs = []
        for slot in range(n_env):
            parent_conn, child_conn = self.ctx.Pipe()
            p = self.ctx.Process(
                target=env_worker_loop,
                args=(slot, child_conn, self.obs_barrier, self.act_barrier,
                      self.shared.names, dims, worker_cfg),
                daemon=True,
            )
            p.start()
            self.conns.append(parent_conn)
            self.procs.append(p)

    # ---- 自游走（同步批量）----------------------------------------------
    def rollout_selfplay(self, actor, seeds: List[int]) -> List[Optional[dict]]:
        for slot, conn in enumerate(self.conns):
            conn.send(("selfplay", int(seeds[slot])))

        actor.reset(self.n_env, self.num_agents)
        v = self.views
        to = self.barrier_timeout
        try:
            for t in range(self.T):
                self.obs_barrier.wait(timeout=to)             # obs[:,t] 就绪
                actions = actor.step(
                    v["patch"][:, t], v["scalar"][:, t],
                    v["pos"][:, t], v["mask"][:, t],
                )                                              # (n_env, N) int64
                v["action"][:, t] = actions
                self.act_barrier.wait(timeout=to)              # 放行 worker 推进
            self.obs_barrier.wait(timeout=to)                  # 完成同步
        except Exception:
            # barrier 被某 worker abort：收集错误信息后上抛
            errs = self._drain_errors()
            self._reset_barriers()
            raise RuntimeError(f"selfplay rollout failed: {errs}")

        return [self._build_seg(slot) for slot in range(self.n_env)]

    # ---- 专家（并行自主）------------------------------------------------
    def rollout_expert(self, seeds: List[int]) -> List[Optional[dict]]:
        for slot, conn in enumerate(self.conns):
            conn.send(("expert", int(seeds[slot])))
        out: List[Optional[dict]] = []
        for conn in self.conns:
            kind, payload = conn.recv()
            out.append(payload if kind == "seg" else None)
        return out

    # ---- 内部 -----------------------------------------------------------
    def _build_seg(self, slot: int) -> Optional[dict]:
        v = self.views
        S_patch  = v["patch"][slot].copy()
        S_scalar = v["scalar"][slot].copy()
        S_pos    = v["pos"][slot].copy()
        S_action = v["action"][slot].copy()
        S_reward = v["reward"][slot].copy()
        S_done   = v["done"][slot].copy()
        S_valid  = v["valid"][slot].copy()
        S_mask   = v["mask"][slot].copy()

        if not S_valid.any():
            return None

        next_patch  = np.zeros_like(S_patch)
        next_scalar = np.zeros_like(S_scalar)
        next_pos    = np.full_like(S_pos, FINISHED_AGENT_SENTINEL)
        next_mask   = np.ones_like(S_mask)
        if self.T > 1:
            next_patch[:-1]  = S_patch[1:]
            next_scalar[:-1] = S_scalar[1:]
            next_pos[:-1]    = S_pos[1:]
            next_mask[:-1]   = S_mask[1:]

        return {
            "obs_patches":       S_patch,
            "obs_scalars":       S_scalar,
            "positions":         S_pos,
            "next_obs_patches":  next_patch,
            "next_obs_scalars":  next_scalar,
            "next_positions":    next_pos,
            "actions_expert":    S_action,
            "rewards":           S_reward,
            "dones":             S_done,
            "valid":             S_valid,
            "action_masks":      S_mask,
            "next_action_masks": next_mask,
            "margin_excluded":   np.ones((self.T, self.num_agents), dtype=bool),
            "margin_weights":    np.zeros((self.T, self.num_agents), dtype=np.float32),
        }

    def _drain_errors(self) -> List[str]:
        errs = []
        for conn in self.conns:
            if conn.poll(0.5):
                try:
                    kind, payload = conn.recv()
                    if kind == "err":
                        errs.append(str(payload))
                except EOFError:
                    pass
        return errs

    def _reset_barriers(self):
        self.obs_barrier.reset()
        self.act_barrier.reset()

    def close(self):
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.shared.close(unlink=True)
