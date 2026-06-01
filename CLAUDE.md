# CLAUDE.md

本文件供 Claude Code 在本仓库工作时阅读。它是 `Lacam_DQfD_Reptile_MAPF_Plan_v2.md`（设计文档，唯一事实来源）的**操作性摘要**。当本文件与设计文档冲突时，以设计文档为准；修改设计决策时**两者一并更新**。

---

## Claude 交互与语言约定（最高优先级）

针对 Claude 内部 Agent 在处理多语言上下文时可能产生的“跨语言幻觉漂移” Bug，特制定以下硬性交互约束：

1. 回复语言限制：Claude 在与用户交互、生成命令行输出、输出解释或撰写日志时，必须且只能使用简体中文（Simplified Chinese）或英文（English）。
2. 绝对禁用语言：在任何情况下，严禁使用韩文（Korean）或日文（Japanese）进行回复。如果检测到自身输出中包含韩文字符，请立即擦除并重写。
3. 代码与注释规范：代码本身（变量、类名、函数名）必须使用英文；新增或修改的代码注释必须与项目既有风格保持一致（即使用简体中文）。

---

## 项目概述

基于 **Pogema** 网格环境的**部分可观测多智能体路径规划（MAPF）**。流水线：
`LaCAM 专家（训练期特权）→ DQfD 模仿（阶段二，off-policy）→ Reptile 跨四地图族元学习 → 目标地图低滞后近 on-policy 微调（阶段三）`。

网络是**共享参数的去中心化 Dueling DRQN（BiIntent-GAT-DRQN）**：每体只用自身局部观测输出 5 维动作价值 Q。**不用 QMIX、不构造 $Q_{tot}$。**

目标硬件：**2×RTX 5090 + 47 核 CPU**。吞吐目标 **≥ 4×10³ env-step/s**。

---

## 不可违反的硬约束（INVARIANTS）

> 改动任何一条前，先在设计文档找到对应小节并同步更新。这些是项目里最容易被改坏、且改坏后症状隐蔽的地方。

1. **Pogema 动作顺序（已实测，勿凭记忆）**：
   `0=wait(0,0)`, `1=up(-1,0)`, `2=down(+1,0)`, `3=left(0,-1)`, `4=right(0,+1)`。
   **wait 在第 0 位**。one-hot、动作掩码、LaCAM 专家动作转换、Q 头输出语义**全部以此为准**。改顺序会让模仿学习整体错位且不报错。

2. **部分可观测：policy 只吃局部信息**。policy 网络输入**仅**为 `(4,11,11) patch + 8 维标量特征`。**严禁**把全局地图、全局距离场、LaCAM 路径、其他体未来轨迹喂进 policy。

3. **特权信号只在训练期、只进 reward/专家，绝不进 policy**：BFS 测地距离与 LaCAM 都是训练期特权信号。BFS 距离**只用于环境计算 reward shaping**；LaCAM **只产专家动作**。部署期无 reward、无 LaCAM。→ 这不破坏部分可观测性。

4. **Reward = BFS 势函数 shaping（PBRS）**：$F=\gamma\Phi(s')-\Phi(s)$，$\Phi=-d_\text{BFS}(p_i,\text{goal}_i)$。
   **禁止**重新引入"动作方向对齐点积奖励"（非势函数、会偏置最优策略、是 maze 失败根因，已删除）。**禁止**用曼哈顿距离做 shaping 的势函数（梯度穿墙）。

5. **观测通道**：Pogema 原生 `(3,11,11)` = `[障碍 / 智能体(自身恒在中心格) / 目标边界投影]`。本仓库追加 **channel 3 = 时间衰减自身足迹**（范围 [0,0.8]，当前位置=1.0），共 4 通道。
   **禁止**再加一个"目标投影通道"——与原生 channel 2 重复。自身位置已由 channel 1 中心格表达，不需单独通道。

6. **目标相对方向是合法的局部特征**：智能体始终知道自己的 goal（`get_targets_xy`），归一化方向向量可作输入。它不是全局地图泄露。

7. **非法动作掩码 = `-float('inf')`**，不是有限大负常数（否则 mellowmax/logsumexp 数值出错）。须保证每体至少 1 个合法动作（wait 通常恒合法）。

8. **Reptile 的 target 网络**：inner loop 内**冻结于 θ_meta**，每个 task 起始重置为当前 θ_meta。**阶段三**微调才用周期性 target 同步（标准 DQN）。

9. **on/off-policy 分工**：
   - 阶段二 DQfD = **off-policy**（replay 复用专家+自游走），不可能做成纯 on-policy（专家轨迹按定义非当前策略产生）。
   - 阶段三 = **同步式近 on-policy**，`max_policy_version_lag=1`，每段 rollout 在冻结权重下采样、用完即同步、基本只用 1 次。

10. **通信约束**：`comm_radius=4 < obs_radius=5`，**只与 FOV 内可见邻居通信**。邻居选择用**软稀疏注意力(entmax)+迟滞($\tau_{out}<\tau_{in}$)**，**不要用 hard top-k 做选择**——hard top-k 会引起通信图闪烁(边界排名交换→边反复通断→特征跳变/梯度噪声/让行振荡)。`comm_topk=6` 仅作**算力上限**(封顶 GAT 计算量)，不是"抑制雪崩的机制"。规模由 FOV+稀疏控制，时间稳定性由迟滞/软化保证。不可放宽成全局通信。监控 `EdgeFlipRate` 防闪烁。

11. **专家 margin 样本过滤**：同一 `(局部patch+足迹+目标方向)` 下专家动作不唯一（对称让行）的样本，**剔除出 margin loss**（其 TD 项可留）。wait 动作 margin 按比例降权。

12. **mellowmax α**：当前实现中 **α 越小越接近 hard max、越易高估**。默认 `α=1.0`，调参在 `{0.5,1,2}`。**勿用 0.1。**

13. **循环回放 burn-in**：训练前先用 `burn_in_len=8` 步只前向重建 GRU 隐状态、不回传，再计 loss（θ 变化后存储的 `hidden_init` 已陈旧）。

14. **诊断用曼哈顿距离 ≠ reward**：`NoProgressRate` 等事后诊断指标可用曼哈顿距离，但**绝不对齐进策略内部 reward**。

---

## 语言与性能策略（务必遵守）

- **默认全 Python（PyTorch）。Rust 不是默认杠杆。** 网络计算全是 GPU kernel，Rust 无用；env 是 Pogema 内核，不重写；LaCAM 已是 C++ 绑定。
- **唯一可能用编译加速的是 CPU 观测/奖励热点**，顺序严格为 **numpy 向量化 → numba → 最后才 Rust**，且 Rust 仅在 profile 确认热点、且向量化不掉、并能 zero-copy 操作共享 numpy 缓冲时才考虑。
- **热路径禁止 per-agent / per-step 的 Python 循环**：全体 agent × 全部并行 env 合成**一次**前向；GAT 用批量 entmax 软稀疏注意力（非 hard top-k）；GRU 一次性展开整段 `sequence_len`。
- 标配：`torch.compile` + CUDA graphs（砍 launch 开销）、AMP 混合精度、patch 用 uint8/fp16、共享内存 zero-copy 传 patch、**禁用 pickle 大对象**。
- **profile 优先**。诊断分流：
  - GPU 利用率上不去 → launch 开销/Python 循环（**Rust 救不了**，做批量化/编译/重叠）。
  - GPU 满载但仍慢 → 网络太重，按设计文档 §4.6 降级 GAT（每 N 步通信 → mean-pool → random 图关通信）。
  - profiler 明确指向某 CPU 观测构建热点且 numba 压不下去 → 才轮到 Rust。

---

## 并行与进程布局

**inner loop 是 K 步串行 SGD，单 task 内不可并行。** 并行只在两轴：

- **轴 1｜CPU 多进程 env 采样**：`start_method=spawn`；env-worker **只做 CPU/env/观测构建，绝不创建 CUDA 上下文**；产 numpy 写入共享内存环形缓冲。
- **轴 2｜双 GPU task-并行 Reptile**：两卡各跑一个**不同 task** 的完整 K 步 inner loop，meta-iter 末平均增量做一次 outer 更新。**用 task-并行而非逐步 DDP**（`batch_size=32`，且 Reptile 每 K 步才同步一次）。

布局：
```
阶段二（预训练）
  GPU0 trainer  ── task A inner loop (K 步 DQfD) ┐
  GPU1 trainer  ── task B inner loop (K 步 DQfD) ┘→ 平均 Δ → outer 更新
  ~40 env-worker（自游走 rollout+观测→共享内存 uint8）
  ~4 LaCAM-worker（C++ 绑定）
  ~3 核 prefetch/collator/主控

阶段三（单目标地图微调，无 task 并行）
  GPU0 = learner
  GPU1 = batched 推理 actor（给数十并行 env 跑策略前向）
  ~40 核并行 env（同步段采样，lag≤1）
```

---

## 仓库结构与各模块职责

```
configs/      env_{random,warehouse,maze,room}.yaml, train_reptile_dqfd.yaml,
              finetune_online.yaml, network_biintent_drqn.yaml
envs/         pogema_env_wrapper（random 用 pogema 原生 generator；warehouse/maze/room 经 pogema-toolbox 加载）
              / observation_builder(原生3通道+足迹通道+标量)
              / action_mask(-inf) / reward_builder(BFS-PBRS) / potential_field(BFS 距离场)
lacam/        lacam_runner(C++绑定) / lacam_dataset_builder / trajectory_converter(动作映射+可解释性过滤)
              / hard_case_manager
models/       encoders / communication(FOV内 软稀疏entmax+迟滞 双向意图注意力, top-k仅算力上限) / recurrent_core(GRU)
              / q_heads(Dueling) / biintent_drqn(组装)
losses/       q_target(masked mellowmax) / td_losses(1-step+n-step) / dqfd_losses(总损失)
buffers/      expert_demo / selfplay / recent_online(lag≤1) / shared_rollout(连续内存)
              / batch_collator
trainers/     train_reptile_dqfd(双GPU task-并行) / finetune_online / evaluator(关专家介入) / scheduler(地图课程)
utils/        metrics / profiler / checkpoint(meta/pretrained/finetuned/best) / seed / logger
scripts/      generate_lacam_data / train_meta / finetune_target_map / evaluate_policy
```

DQfD 总损失：`L = λ_td·L_TD + λ_nstep·L_n + λ_margin·L_margin + λ_l2·||θ||²`。

---

## 命令（预期入口；实现后请校准）

> 项目处于设计→实现阶段，以下为**预期**命令；新增/改名脚本时同步此节。

```bash
# 环境（已实测 pogema 1.4.0，注意 numpy 兼容）
# pogema 自带 generator 只能生成 random 地图；
# warehouse / maze / room 需经 pogema-toolbox 加载/生成
git clone https://github.com/Cognitive-AI-Systems/pogema-toolbox.git
cd pogema-toolbox
pip install -e .          # 会一并拉入 pogema 依赖
cd ..
pip install torch numpy numba pyyaml
# LaCAM 需单独构建并提供 Python 绑定（C++）

# 阶段一：生成 LaCAM 专家数据（多进程）
python scripts/generate_lacam_data.py --config configs/env_random.yaml --num-workers 4

# 阶段二：DQfD + Reptile 元预训练（双 GPU task-并行 + 多进程 env）
python scripts/train_meta.py --config configs/train_reptile_dqfd.yaml

# 阶段三：目标地图近 on-policy 微调
python scripts/finetune_target_map.py --config configs/finetune_online.yaml --init <meta_ckpt>

# 评估（务必关闭专家介入）
python scripts/evaluate_policy.py --ckpt <finetuned_ckpt> --config configs/env_maze.yaml
```

---

## 关键超参（默认值，来自设计文档）

| 项 | 值 | 备注 |
|---|---|---|
| `obs_radius` / patch K | 5 / 11 | K = 2·r+1 |
| `patch_channels` | 4 | 原生3 + 足迹1 |
| `scalar_feat_dim` | 8 | 方向(2)+上一步动作(5)+到达(1) |
| `num_actions` | 5 | wait,up,down,left,right |
| `comm_radius` / `comm_topk` | 4 / 6 | r_c < obs_radius，FOV 内；选择用软稀疏+迟滞，topk 仅算力上限 |
| `gru_hidden_dim` | 256 | DRQN 记忆 |
| `inner_steps` / `inner_lr` / `meta_lr` | 5 / 3e-4 / 0.05 | Reptile |
| `n_step` / `gamma` | 5 / 0.99 | |
| `sequence_len` / `burn_in_len` | 32 / 8 | R2D2 burn-in |
| `mellowmax_alpha` | 1.0 | 调参 {0.5,1,2}，勿用 0.1 |
| `expert_ratio` | 0.9→0.5 | 阶段二课程式混合 |
| `max_policy_version_lag` | 1 | 阶段三 |

---

## 开发约定

- 所有可调项进 `configs/*.yaml`，勿硬编码。
- 全程固定随机种子（`utils/seed.py`），保证可复现。
- 任何性能改动**先跑 `utils/profiler.py`**，按占比改，改完用 §5 指标确认 SR/Deadlock/WaitRate 不退化。
- 指标必须能定位问题来源（任务效果/路径质量/冲突死锁/学习稳定性/Q诊断/模仿/通信/系统八类）；新增模块同时补对应指标。
- checkpoint 分四类保存：meta-init / pretrained-DQfD / finetuned / best-eval。

---

## 高危易错点速查（CHECKLIST）

- [ ] 动作 0 是 wait 吗？（不是最后一位）
- [ ] policy 输入是否混进了任何全局/未来/他体信息？
- [ ] reward 是否用了 BFS 势函数形式、而**不是**点积对齐或曼哈顿势函数？
- [ ] 观测是否误加了与原生 ch2 重复的目标投影通道？
- [ ] Reptile inner loop 的 target 是否冻结于 θ_meta？
- [ ] 阶段三是否真做到 lag≤1、用完即同步？
- [ ] 非法动作是否用 `-inf` 掩码？
- [ ] env-worker 里是否误建了 CUDA 上下文？
- [ ] 热路径是否残留 per-agent/per-step 的 Python 循环？
- [x] 通信邻居选择是否用了软稀疏+迟滞（而非 hard top-k 做选择）？EdgeFlipRate 是否在监控？
- [ ] 是否在没 profile 的情况下就动手写 Rust？（不要）
