"""
离线 IL + Reptile 训练器（v3）。取代 ReptileDQfDTrainer（DQfD/TD/target/selfplay）。

- 任务 = 地图族（Reptile 跨族学一个"易适应到新族"的 init，对应 v3 把 Reptile 重定位为
  "通信策略对新环境的快速适应"）。
- inner loop：从 θ_meta 起，在某族离线专家数据上跑 K 步交叉熵 SGD（masked、按 valid 屏蔽）。
- outer：θ ← θ_meta + ε · avg_task(θ_inner − θ_meta)，meta_lr 线性衰减。
- 无 reward / 无 TD / 无 target 网络 / 无 selfplay。
"""
from __future__ import annotations

import threading
import queue
import numpy as np
import torch
from copy import deepcopy
from typing import Dict, List

from losses.il_losses import il_cross_entropy_loss
from envs.d4_augment import d4_augment_batch

_SENTINEL = object()  # 预取队列结束标记


def _pick_families(fams, rng, n, weights=None):
    """按族抽 n 个 task。weights=None 时与历史实现**逐位一致**（同一 rng 消耗），保证旧 run 可复现。

    weights: {族名: 非负权重}。追加新数据批次（新族名）时用它保持「地图类型 × 体数」组级混合比例不变，
    否则均匀抽族会让批次多的组权重变大，和对照臂不再是唯一变量。所有族都必须显式给权重（防漏写成 0）。
    """
    if weights is None:
        return [fams[i] for i in rng.integers(0, len(fams), size=n)]
    missing = [f for f in fams if f not in weights]
    if missing:
        raise KeyError(f"family_weights 缺少这些族的权重：{missing}")
    w = np.asarray([float(weights[f]) for f in fams], dtype=np.float64)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"family_weights 必须非负且和 > 0：{dict(zip(fams, w))}")
    return [fams[i] for i in rng.choice(len(fams), size=n, p=w / w.sum())]


def _prefetch_batches(buf, n_batches, sample_kwargs, q):
    """后台线程：预采 n_batches 个 batch 入队。CPU gather 与主线程 GPU 计算重叠
    （_inner_step 里的 CUDA kernel 释放 GIL，采样线程得以并行推进）。"""
    for _ in range(n_batches):
        try:
            q.put(buf.sample(**sample_kwargs))
        except (ValueError, IndexError):
            q.put(None)   # 采样失败占位，主线程跳过
    q.put(_SENTINEL)


class ILReptileTrainer:
    def __init__(self, net, cfg: dict, device: torch.device = None):
        self.net = net
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.inner_lr = cfg.get("inner_lr", 5e-4)
        self.meta_lr_start = cfg.get("meta_lr_start", 0.1)
        self.meta_lr_end = cfg.get("meta_lr_end", 0.01)
        self.inner_steps = cfg.get("inner_steps", 5)
        self.tasks_per_meta = cfg.get("tasks_per_meta", 2)
        self.batch_size = cfg.get("batch_size", 32)
        self.sequence_len = cfg.get("sequence_len", 16)
        self.grad_clip = cfg.get("grad_clip", 10.0)
        self.excluded_weight = cfg.get("excluded_weight", 0.3)
        self.label_smoothing = cfg.get("label_smoothing", 0.0)
        self.wait_weight = cfg.get("il_wait_weight", 1.0)   # wait 类别平衡（<1 降权）
        # M1：D4 对称增强（8 群元，逐 batch 元素随机取）。网格 MAPF 在旋转/镜像下等变，
        # 一条专家样本等价 8 条 → 缓解数据量瓶颈。正确性见 scripts/test_d4_augment.py。
        self.d4_augment = bool(cfg.get("d4_augment", False))
        # reptile=False → 普通 joint 训练（消融）：不保存/恢复 θ_meta、不做外层平均、
        # **保留优化器动量**（Reptile 路径每 task 清动量；joint 若也清会被不公平削弱）。
        self.use_reptile = bool(cfg.get("reptile", True))
        # 一阶 MAML（FOMAML，路线图 Q2 的"若可实现则加 MAML"）。与 Reptile 的唯一区别在**外层**：
        #   Reptile : θ ← θ_meta + ε·mean(θ_inner − θ_meta)      —— 乘的是"位移"
        #   FOMAML  : θ ← θ_meta − lr_out·mean(∇L(θ_inner; query)) —— 乘的是"梯度"
        # 两者量纲不同：位移 ≈ K·inner_lr·梯度，故直接沿用 ε=0.1 会让 MAML 的有效步长大约
        # K·inner_lr 分之一倍（本配置下约 400 倍）。为使两臂**有效步长同量级、可公平对比**，
        # 这里取 lr_out = meta_lr × inner_steps × inner_lr。
        # （教训来源：7.45M Reptile 臂曾因"等梯度步数≠等优化预算"而被误判，见 SCHEDULE 已定论 17。）
        self.use_maml = bool(cfg.get("maml", False))
        self.cfg_maml_outer_lr = cfg.get("maml_outer_lr")
        if self.use_maml:
            self.use_reptile = False
        self.total_meta_iters = cfg.get("meta_iterations", 20000)
        self.family_weights = cfg.get("family_weights")   # None = 均匀抽族（历史行为）

        self.use_amp = bool(cfg.get("use_amp", False)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.inner_lr)
        self.meta_iter = 0

    def _current_meta_lr(self) -> float:
        t = self.meta_iter / max(1, self.total_meta_iters - 1)
        return self.meta_lr_start + (self.meta_lr_end - self.meta_lr_start) * t

    def _to_gpu(self, batch: dict) -> dict:
        dev = self.device
        def t(x, dtype=None):
            y = torch.as_tensor(x)
            if dtype is not None:
                y = y.to(dtype)
            return y.to(dev, non_blocking=True)
        patches = t(batch["obs_patches"])
        patches = patches.float().div_(255.0) if patches.dtype == torch.uint8 else patches.float()
        out = {
            "patches": patches,
            "scalars": t(batch["obs_scalars"]).float(),
            "positions": t(batch["positions"]).float(),
            "actions": t(batch["actions_expert"], torch.long),
            "valid": t(batch["valid"], torch.bool),
            "masks": t(batch["action_masks"], torch.bool),
        }
        out["margin_excluded"] = (t(batch["margin_excluded"], torch.bool)
                                  if batch.get("margin_excluded") is not None else None)
        return out

    def _inner_step(self, batch: dict) -> dict:
        self.net.train()
        b = self._to_gpu(batch)
        if self.d4_augment:
            b["patches"], b["scalars"], b["actions"], b["masks"] = d4_augment_batch(
                b["patches"], b["scalars"], b["actions"], b["masks"])
        amp_ctx = torch.autocast(device_type=self.device.type, enabled=self.use_amp)
        with amp_ctx:
            logits, info = self.net(b["patches"], b["scalars"], b["positions"],
                                    action_masks=b["masks"], reset_comm=True)
            loss, linfo = il_cross_entropy_loss(
                logits, b["actions"], valid=b["valid"], margin_excluded=b["margin_excluded"],
                excluded_weight=self.excluded_weight, label_smoothing=self.label_smoothing,
                wait_weight=self.wait_weight,
            )
        self.opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        linfo["edge_flip_rate"] = info.get("edge_flip_rate", 0.0)
        return linfo

    def run_inner_loop(self, buf, theta_meta: Dict[str, torch.Tensor], seed: int) -> dict:
        rng = np.random.default_rng(seed)
        if self.use_reptile or self.use_maml:
            self.net.load_state_dict({k: v.to(self.device) for k, v in theta_meta.items()})
            self.opt.state.clear()   # 持久优化器，每 task 清动量（等价每 task 新建，无跨 task 残留）
        # joint 模式：从当前权重继续、动量不清 —— 即普通连续 SGD
        losses, accs, flips, waits = [], [], [], []
        # 预取：后台线程采样 inner_steps 个 batch，主线程只做 GPU 计算（need_next=False：
        # 纯 IL 不需要 next_obs/rewards/dones，省 ~一半 CPU gather）。
        q: queue.Queue = queue.Queue(maxsize=2)
        sk = dict(batch_size=self.batch_size, sequence_len=self.sequence_len,
                  burn_in_len=0, rng=rng, need_next=False)
        th = threading.Thread(target=_prefetch_batches,
                              args=(buf, self.inner_steps, sk, q), daemon=True)
        th.start()
        while True:
            batch = q.get()
            if batch is _SENTINEL:
                break
            if batch is None:
                continue
            info = self._inner_step(batch)
            losses.append(info["loss_il"]); accs.append(info["acc_expert"])
            flips.append(info.get("edge_flip_rate", 0.0))
            waits.append(info.get("wait_pred_frac", 0.0))
        th.join()
        theta_inner = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        _avg = lambda xs: float(np.mean(xs)) if xs else 0.0
        return {"theta_inner": theta_inner, "loss": _avg(losses),
                "acc": _avg(accs), "edge_flip": _avg(flips), "wait_pred": _avg(waits)}

    def meta_step(self, family_buffers: Dict[str, object], seed: int = None) -> dict:
        if self.use_maml:
            return self._maml_step(family_buffers, seed)
        if not self.use_reptile:
            return self._joint_step(family_buffers, seed)
        theta_meta = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        fams = list(family_buffers.keys())
        rng = np.random.default_rng(seed or (10_000 + self.meta_iter))
        tasks = _pick_families(fams, rng, self.tasks_per_meta, self.family_weights)

        results = []
        for j, fam in enumerate(tasks):
            r = self.run_inner_loop(family_buffers[fam], theta_meta,
                                    seed=(seed or self.meta_iter * 1000) + j * 99991)
            results.append(r)

        cur_meta_lr = self._current_meta_lr()
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                deltas = [r["theta_inner"][name].to(self.device) - theta_meta[name].to(self.device)
                          for r in results if name in r["theta_inner"]]
                avg = torch.stack(deltas).mean(0)
                param.copy_(theta_meta[name].to(self.device) + cur_meta_lr * avg)
        # buffers (BN/running stats) 直接载入最后一个 task 的（IL 无 BN，此处仅兜底）
        self.net.load_state_dict({k: (v if k in dict(self.net.named_parameters())
                                      else results[-1]["theta_inner"][k].to(self.device))
                                  for k, v in self.net.state_dict().items()})

        self.meta_iter += 1
        _avg = lambda key: float(np.mean([r[key] for r in results])) if results else 0.0
        return {"meta_iter": self.meta_iter, "tasks": tasks, "meta_lr": cur_meta_lr,
                "loss": _avg("loss"), "acc": _avg("acc"), "edge_flip": _avg("edge_flip"),
                "wait_pred": _avg("wait_pred")}

    def _maml_step(self, family_buffers: Dict[str, object], seed: int = None) -> dict:
        """一阶 MAML：内层与 Reptile 完全一致（每 task 从 θ_meta 起跑 K 步、清动量），
        外层改为用**适应后参数上的 query batch 梯度**更新 θ_meta（忽略二阶项）。"""
        theta_meta = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        fams = list(family_buffers.keys())
        rng = np.random.default_rng(seed or (10_000 + self.meta_iter))
        tasks = _pick_families(fams, rng, self.tasks_per_meta, self.family_weights)

        results, grads_sum, n_ok = [], None, 0
        for j, fam in enumerate(tasks):
            r = self.run_inner_loop(family_buffers[fam], theta_meta,
                                    seed=(seed or self.meta_iter * 1000) + j * 99991)
            results.append(r)
            # query batch（与 support 不同的采样流）→ 在 θ_inner 上求梯度
            qrng = np.random.default_rng((seed or self.meta_iter * 1000) + j * 99991 + 555_666)
            try:
                qb = family_buffers[fam].sample(
                    batch_size=self.batch_size, sequence_len=self.sequence_len,
                    burn_in_len=0, rng=qrng, need_next=False)
            except (ValueError, IndexError):
                continue
            self.net.train(); self.opt.zero_grad(set_to_none=True)
            b = self._to_gpu(qb)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits, _ = self.net(b["patches"], b["scalars"], b["positions"],
                                     action_masks=b["masks"], reset_comm=True)
                loss, _ = il_cross_entropy_loss(
                    logits, b["actions"], valid=b["valid"],
                    margin_excluded=b["margin_excluded"], excluded_weight=self.excluded_weight,
                    label_smoothing=self.label_smoothing, wait_weight=self.wait_weight)
            # ★ 不能复用训练用的 self.scaler：这里只求梯度、不做 step/update，
            #   调了 unscale_ 会把 scaler 卡在"已 unscale"状态，下个 task 的内层再 unscale_ 就报错。
            #   改用固定静态缩放，手工还原，完全不碰 scaler 状态。
            _S = 2.0 ** 16
            (loss * _S).backward()
            gs = {n_: (p.grad.detach().cpu().clone() / _S if p.grad is not None else None)
                  for n_, p in self.net.named_parameters()}
            if any(g is not None and not torch.isfinite(g).all() for g in gs.values()):
                continue                      # AMP 溢出批次直接丢弃，不污染外层
            grads_sum = gs if grads_sum is None else {
                n_: (gs[n_] if grads_sum[n_] is None else
                     (grads_sum[n_] + gs[n_] if gs[n_] is not None else grads_sum[n_]))
                for n_ in grads_sum}
            n_ok += 1

        # 外层用 **Adam**（而非裸 SGD）。2026-09-23 实测：起初取
        #   lr_out = meta_lr × inner_steps × inner_lr（想让"梯度×lr"与 Reptile 的"位移"同量级），
        # 500 步时 loss 1.122/acc 0.671/wait_pred 0.997，而同 seed 的 Reptile 对照是 0.174/0.920
        # ——几乎没学动。错在折算：Reptile 的位移是 K 步 **Adam** 更新的累积，其量级 ≈ K×inner_lr
        # （被二阶矩归一化，与梯度大小脱钩），而裸梯度的模远小于归一化更新 → 有效步长被压小一两个
        # 数量级。改为外层 Adam 后内外层经过同样的归一化，步长才真正可比。
        if not hasattr(self, "_outer_opt"):
            self._outer_opt = torch.optim.Adam(list(self.net.parameters()), lr=self.inner_lr)
        # maml_outer_lr：显式指定外层 Adam 学习率（不给则按 inner_lr × meta_lr 衰减比例）。
        # 2026-09-24 实测：默认值 5e-4 会发散（loss 500→2000 步 1.52→4.10，而同 seed 的
        # Reptile 是 0.17→0.10），故该项需按实际扫描结果设定。
        base_out = self.cfg_maml_outer_lr if self.cfg_maml_outer_lr else self.inner_lr
        lr_out = base_out * (self._current_meta_lr() / max(self.meta_lr_start, 1e-12))
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                param.copy_(theta_meta[name].to(self.device))
        if grads_sum is not None and n_ok > 0:
            for pg in self._outer_opt.param_groups:
                pg["lr"] = lr_out
            self._outer_opt.zero_grad(set_to_none=True)
            for name, param in self.net.named_parameters():
                g = grads_sum.get(name)
                if g is not None:
                    param.grad = (g.to(self.device) / n_ok)
            self._outer_opt.step()
        self.net.load_state_dict({k: (v if k in dict(self.net.named_parameters())
                                      else results[-1]["theta_inner"][k].to(self.device))
                                  for k, v in self.net.state_dict().items()})
        self.meta_iter += 1
        _avg = lambda key: float(np.mean([r[key] for r in results])) if results else 0.0
        return {"meta_iter": self.meta_iter, "tasks": tasks, "mode": "maml",
                "meta_lr": lr_out, "loss": _avg("loss"), "acc": _avg("acc"),
                "edge_flip": _avg("edge_flip"), "wait_pred": _avg("wait_pred")}

    def _joint_step(self, family_buffers: Dict[str, object], seed: int = None) -> dict:
        """Reptile 消融：普通 joint 训练。直接在当前权重上连续跑 SGD，
        无 θ_meta 保存/恢复、无外层平均、动量不清。tasks_per_meta 只决定每次采几个族的批。"""
        fams = list(family_buffers.keys())
        rng = np.random.default_rng(seed or (10_000 + self.meta_iter))
        tasks = _pick_families(fams, rng, self.tasks_per_meta, self.family_weights)
        res = [self.run_inner_loop(family_buffers[fam], None,
                                   seed=(seed or self.meta_iter * 1000) + j * 99991)
               for j, fam in enumerate(tasks)]
        self.meta_iter += 1
        _avg = lambda k: float(np.mean([r[k] for r in res])) if res else 0.0
        # ★ mode=joint：无外层平均、无 θ_meta 回滚、动量不清；lr=inner_lr（恒定 SGD 学习率，
        #   非 meta_lr）。字段沿用 meta_lr 仅为日志兼容，值是 inner_lr。
        return {"meta_iter": self.meta_iter, "tasks": tasks, "mode": "joint",
                "meta_lr": self.inner_lr,
                "loss": _avg("loss"), "acc": _avg("acc"), "edge_flip": _avg("edge_flip"),
                "wait_pred": _avg("wait_pred")}

    def state_dict(self) -> dict:
        # ★ 2026-09-24：加存 optimizer state。此前 ckpt 只有 {type, net, meta_iter}，
        #   `--resume` 续训时 Adam 一阶/二阶动量从零重启，续训结果与"一口气训完"不严格等价
        #   （chameleon-0b 指出）。新增键向后兼容：旧 ckpt 无 "opt" 时 load 侧跳过。
        return {"net": self.net.state_dict(), "meta_iter": self.meta_iter,
                "opt": self.opt.state_dict()}

    def load_state_dict(self, state: dict):
        self.net.load_state_dict(state["net"])
        self.meta_iter = state.get("meta_iter", 0)
        if "opt" in state:
            try:
                self.opt.load_state_dict(state["opt"])
            except (ValueError, KeyError) as e:   # 参数组形状变了（换网络/换臂）时静默跳过
                print(f"[resume] optimizer state 不匹配，动量从零起：{e}")


# =============================================================================
# 双 GPU task-并行 IL + Reptile（§4.4/4.5 思路，IL 版更简：无 target/selfplay/presample）
# =============================================================================
class DualGPUILReptileTrainer:
    """两卡各跑不同族的 K 步内循环（线程并行，CUDA kernel 释放 GIL），meta-iter 末平均增量
    做一次 outer 更新（非逐步 DDP）。总 task = n_gpus × tasks_per_gpu。"""

    def __init__(self, net_template, cfg: dict, devices: List[str] = None):
        devices = devices or ["cuda:0", "cuda:1"]
        self.devices = [torch.device(d) for d in devices]
        self.sub = [ILReptileTrainer(deepcopy(net_template), cfg, device=d) for d in self.devices]
        self._meta_state = {k: v.detach().cpu().clone()
                            for k, v in net_template.state_dict().items()}
        self.meta_lr_start = cfg.get("meta_lr_start", 0.1)
        self.meta_lr_end = cfg.get("meta_lr_end", 0.01)
        self.tasks_per_gpu = cfg.get("tasks_per_gpu", 1)
        self.total_meta_iters = cfg.get("meta_iterations", 20000)
        self.family_weights = cfg.get("family_weights")
        self.meta_iter = 0

    @property
    def net(self):
        return self.sub[0].net

    def _meta_lr(self) -> float:
        t = self.meta_iter / max(1, self.total_meta_iters - 1)
        return self.meta_lr_start + (self.meta_lr_end - self.meta_lr_start) * t

    def meta_step(self, family_buffers: Dict[str, object], seed: int = None) -> dict:
        theta = self._meta_state
        fams = list(family_buffers.keys())
        rng = np.random.default_rng(seed or (10_000 + self.meta_iter))
        n_gpus, tpg = len(self.devices), self.tasks_per_gpu
        task_fams = _pick_families(fams, rng, n_gpus * tpg, self.family_weights)

        results: Dict[int, list] = {g: [] for g in range(n_gpus)}
        errors: Dict[int, Exception] = {g: None for g in range(n_gpus)}

        def _run(g: int):
            try:
                for j in range(tpg):
                    ti = g * tpg + j
                    r = self.sub[g].run_inner_loop(
                        family_buffers[task_fams[ti]], theta,
                        seed=(seed or self.meta_iter * 1000) + ti * 99991)
                    results[g].append((task_fams[ti], r))
            except Exception as exc:  # noqa: BLE001
                errors[g] = exc

        threads = [threading.Thread(target=_run, args=(g,)) for g in range(n_gpus)]
        for t in threads: t.start()
        for t in threads: t.join()
        for g, e in errors.items():
            if e is not None:
                raise RuntimeError(f"GPU{g} inner loop failed") from e

        allr = [r for g in range(n_gpus) for (_, r) in results[g]]
        tasks = [fam for g in range(n_gpus) for (fam, _) in results[g]]

        lr = self._meta_lr()
        with torch.no_grad():
            new = {}
            for k in theta:
                deltas = [r["theta_inner"][k] - theta[k] for r in allr if k in r["theta_inner"]]
                new[k] = theta[k] + lr * torch.stack(deltas).mean(0)
            self._meta_state = new
        for s in self.sub:
            s.net.load_state_dict({k: v.to(s.device) for k, v in new.items()})

        self.meta_iter += 1
        _avg = lambda key: float(np.mean([r[key] for r in allr])) if allr else 0.0
        return {"meta_iter": self.meta_iter, "tasks": tasks, "meta_lr": lr,
                "loss": _avg("loss"), "acc": _avg("acc"), "edge_flip": _avg("edge_flip"),
                "wait_pred": _avg("wait_pred")}

    def state_dict(self) -> dict:
        return {"net": self._meta_state, "meta_iter": self.meta_iter}

    def load_state_dict(self, state: dict):
        self._meta_state = {k: v.cpu() for k, v in state["net"].items()}
        self.meta_iter = state.get("meta_iter", 0)
        for s in self.sub:
            s.net.load_state_dict({k: v.to(s.device) for k, v in self._meta_state.items()})
