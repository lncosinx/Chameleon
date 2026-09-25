# CLAUDE.md

本文件供 Claude Code 在本仓库工作时阅读。它是 `Lacam_DQfD_Reptile_MAPF_Plan_v2.md`（设计文档，**内容已是 v5，文件名沿用旧名**）的**操作性摘要**。

**三份文档的分工**（2026-09-25 明确）：
- `Lacam_DQfD_Reptile_MAPF_Plan_v2.md` —— **方法与不变式**的事实来源（网络、训练、协议、INVARIANTS）。
- `SCHEDULE.md` §6.5「已定论」—— **实验事实与结论**的事实来源（滚动维护）。
- 本文件 —— 每次工作都要知道的**操作细节**（环境、路径、进程、易错点）。

冲突时：**实验事实以 SCHEDULE 为准，方法与不变式以设计文档为准，操作细节以本文件为准**；
修改设计决策时**三处一并更新**。

---

## Claude 交互与语言约定（最高优先级）

针对 Claude 内部 Agent 在处理多语言上下文时可能产生的“跨语言幻觉漂移” Bug，特制定以下硬性交互约束：

1. 回复语言限制：Claude 在与用户交互、生成命令行输出、输出解释或撰写日志时，必须且只能使用简体中文（Simplified Chinese）或英文（English）。
2. 绝对禁用语言：在任何情况下，严禁使用韩文（Korean）或日文（Japanese）进行回复。如果检测到自身输出中包含韩文字符，请立即擦除并重写。
3. 代码与注释规范：代码本身（变量、类名、函数名）必须使用英文；新增或修改的代码注释必须与项目既有风格保持一致（即使用简体中文）。
4. 当因为网络问题导致卡住时，可以将127.0.0.1:20171端口设置为代理，例如：
export http_proxy="http://127.0.0.1:20171"
export https_proxy="http://127.0.0.1:20171"
export all_proxy="socks5://127.0.0.1:20171"

---

## 项目概述

基于 **POGEMA** 网格环境的**部分可观测多智能体路径规划（MAPF）**。方法细节见设计文档。

**现行方案（v5，2026-09-25）**：
- **流水线只有两段**：① LaCAM 专家数据（训练期特权，只产专家动作）→ ② **离线模仿学习**（交叉熵）。
  **没有在线阶段三、没有 reward、没有 DQfD 损失、没有 Q 头。**
- **头条贡献 = 通信**（FOV 受限 + entmax 软稀疏 + 迟滞），**不**比"泛化"。
- **骨干 = MADCL / `MAPFCommNet`**：残差 CNN → FOV 软稀疏通信（残差写回）→ 可插拔时间核 → 动作 logits。
  时间核实测**用 GRU**（`temporal_backend: gru`）；Transformer / Mamba 都不是主线
  （Mamba 的增益已查明是**前馈**的，与时序无关，见 SCHEDULE 已定论 21）。
- **主模型 = MADCL-L，7.45M，`ckpt_il/capbig_iter_20000.pt`**；
  推断 `sample` + `T=1.0` + `wait_bias=1.0` + `TTA`，12 档总分 **9.375**。
- **元学习**：Reptile 保留（SQ2 零样本跨族迁移有效）；joint 是主模型用的模式；FOMAML 已放弃。

**目标硬件（实测）**：**单卡 RTX 4080 SUPER 32G + 16 vCPU**（cgroup 80G 内存）。
> ⚠️ 旧文写"2×RTX 5090 + 47 核"已不成立；双 GPU task-并行的设计随之废弃。

**★ 2026-09-25：857G 专家数据已按用户要求删除。** 保留训练日志、`ckpt_il/`、`eval_v5/`、`dev_maps/`。
⇒ **任何需要训练的作业先读 `DATA_REGEN.md` 重生成数据（约 14h，必须 lacam310）**；纯评测可直接跑。
⇒ 重生成的数据与原数据**同分布但非同一份**，新臂**不能与历史臂做唯一变量对比**。

---

## 不可违反的硬约束（INVARIANTS）

> 改动任何一条前，先在设计文档找到对应小节并同步更新。这些是项目里最容易被改坏、且改坏后症状隐蔽的地方。

> **[v5 适用性，2026-09-25]** **仍是铁律**：**1**(动作序 wait=0)、**2**(只吃局部)、**3**(特权只训练期；
> cost2go 已放宽为合法局部特征)、**5**(观测通道，现 **5 通道**)、**6**(目标方向合法)、**7**(-inf 掩码)、
> **10**(通信，**头条**)、**13**(整段展开 + episode 边界 `reset_comm_state()`)、**14**(曼哈顿只做诊断)。
> **因转离线 IL 而废弃**：**4**(reward shaping)、**8**(Reptile 的 target 网络)、**9**(on/off-policy 分工)、
> **11**(margin 过滤；其动机仍解释"单步准确率不是 CSR 的好代理")、**12**(mellowmax)。
>
> **设计文档 v5 新增三条硬约束（都是真金白银换来的）**：
> **15 评测口径写死**（nothing/128 步/soft/validation-* 128 图/split=all；推断统一 sample+TTA）；
> **16 超参不得跨臂强加**（`lr=1e-4` 是为 7.45M 选的，小网对照臂被迫跟用而被低估 0.227 ★）；
> **17 默认值即协议**（探针脚本必须显式写死全部口径参数，已因此三次翻车）。

1. **Pogema 动作顺序（已实测，勿凭记忆）**：
   `0=wait(0,0)`, `1=up(-1,0)`, `2=down(+1,0)`, `3=left(0,-1)`, `4=right(0,+1)`。
   **wait 在第 0 位**。one-hot、动作掩码、LaCAM 专家动作转换、Q 头输出语义**全部以此为准**。改顺序会让模仿学习整体错位且不报错。

2. **部分可观测：policy 只吃局部信息**。policy 网络输入**仅**为 `(C,11,11) patch + 8 维标量特征`（C=4，或启用 cost2go 时 5，见 INVARIANT 5）。**严禁**把全局地图、**整幅**全局距离场、LaCAM 路径、其他体未来轨迹喂进 policy。（cost2go 只取 FOV 内裁剪，属合法局部特征，见 INVARIANT 3 [v4] 放宽。）

3. **特权信号只在训练期、只进 reward/专家，绝不进 policy**：LaCAM 是训练期特权信号，**只产专家动作**，绝不进 policy；部署期无 LaCAM。
   > **[v4 放宽，2026-07-22]** **cost2go（FOV 内 BFS 测地距离场的局部裁剪）现为合法局部特征、可进 policy 观测**（channel 4，见 INVARIANT 5）。理由：EPH(heuri_map)、MAPF-GPT(cost2go)、SCRIMP 等**所有对比 baseline 都喂障碍感知的目标梯度**；只给穿墙方向向量导致 mazes SR=0、纯 IL 塌缩到直线策略（实测双稳态：wait_pred 非 0 即 1）。cost2go 仅取**每体 FOV 内**的 BFS 距离（从全局静态障碍图算、相对自身居中），**不泄露全局地图/他体轨迹/未来**，不破坏部分可观测性。**全局 BFS 距离场整幅、他体路径、LaCAM 路径仍严禁进 policy。** 部署期 cost2go 由已知静态地图现算（与 baseline 惯例一致）。

4. **Reward = BFS 势函数 shaping（PBRS）**：$F=\gamma\Phi(s')-\Phi(s)$，$\Phi=-d_\text{BFS}(p_i,\text{goal}_i)$。
   **禁止**重新引入"动作方向对齐点积奖励"（非势函数、会偏置最优策略、是 maze 失败根因，已删除）。**禁止**用曼哈顿距离做 shaping 的势函数（梯度穿墙）。

5. **观测通道**：Pogema 原生 `(3,11,11)` = `[障碍 / 智能体(自身恒在中心格) / 目标边界投影]`。本仓库追加 **channel 3 = 时间衰减自身足迹**（范围 [0,0.8]，当前位置=1.0）。**[v4] channel 4 = cost2go**（FOV 内 BFS 到目标测地距离，相对自身居中到 [0,1]，0.5=同距，见 INVARIANT 3 放宽）。启用 cost2go 时 patch=**5 通道**，否则 4 通道（`ObservationBuilder(use_cost2go=...)`，`patch_channels` 配置须匹配数据）。
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

## 并行与进程布局（单卡，2026-09-25 实测）

**inner loop 是 K 步串行 SGD，单 task 内不可并行。** 现行只有一轴并行：

- **CPU 多进程专家数据生成**：`start_method=spawn`；worker 只做 CPU/env/LaCAM，**绝不创建 CUDA 上下文**。
  ★ **worker 数不超过 `nproc-4`**（本机 16 vCPU ⇒ ≤12），并配 `nice -n 19` / `ionice`
  —— 曾开 15 个进程把用户的 vscode-ssh 挤掉线。
- **训练是单卡单进程**（`--device cuda:0`）。双 GPU task-并行的设计已废弃（只有一张卡）。
- **显存**：7.45M 训练峰值约 19.6 GiB（接入官方 CUDA scan 核后，此前 29.7 GiB）；
  2.16M 约 8 GiB。**两个训练并跑会贴边 32 GiB，容易 OOM ⇒ 串行排队**。

---

## 仓库结构与各模块职责（2026-09-25 与代码核对）

```
configs/      env_*.yaml｜network_*.yaml（gru_c2g 2.16M / **big 7.45M 主模型** / small_mamba /
              big_mamba / small_gru_prenorm / 消融变体 small_mamba_{dtfix,fnorm,both,dconv1}）
              train_il_*.yaml（离线 IL 配方；**主模型用 train_il_cap_lr1e4.yaml**）
envs/         pogema_env_wrapper / observation_builder(原生3ch+足迹+cost2go) / action_mask(-inf)
              potential_field(现只为 cost2go 与诊断) / map_registry / dataset_split / d4_augment / env_worker
lacam/        lacam_runner(C++绑定，★必须 lacam310；已加 solver_seed 支持确定性求解)
              lacam_dataset_builder / trajectory_converter / hard_case_manager
models/       **mapf_comm_net（主网络组装入口）** / spatial_encoder(ResidualCNNEncoder)
              communication(BiIntentAttention：entmax+迟滞) / temporal_core(可插拔时间核)
              comm_baselines(6 个外部通信基线移植到同骨干) / obs_tokenizer / transformer_*
losses/       **il_losses（现行：交叉熵 + wait 权重 + label smoothing）**
buffers/      **expert_demo_buffer（现行，memmap 零拷贝）** / batch_collator
trainers/     **train_il_reptile（ILReptileTrainer：joint / reptile / maml 三模式）** / evaluator / scheduler
utils/        metrics / profiler / checkpoint / seed / logger
scripts/      **train_il_meta（训练入口）** / **eval_bench（统一评测台）** / make_dev_maps(独立 dev 地图集)
              generate_lacam_data、generate_expert_data(`--seed-start` 防重复) / make_sq1_table
              run_sq2_adaptation / run_sq4_continual / run_mapfgpt_bench / check_episode_overlap / test_*
```

**⚠️ v2 遗留文件保留但不使用**：`losses/{dqfd,td,q_target}`、`buffers/{selfplay,recent_online,shared_rollout}`、
`models/{biintent_drqn,q_heads,encoders,recurrent_core}`、`trainers/{train_reptile_dqfd,finetune_online,
seed_actor,selfplay_producer}`、`envs/reward_builder`、`configs/{train_reptile_dqfd,finetune_online}.yaml`。
**不要以为它们在跑。**

---

## 环境与协议（2026-09-25 实测，最高操作优先级）

- **3 个 conda 环境**（注意路径，`ls /root/miniconda3/envs` 只看得到一个）：
  | 环境 | 路径 | 用途 |
  |---|---|---|
  | **base** | `/root/miniconda3`（py3.12） | **训练 / 评测**，pogema 2.0.0a |
  | `marlgpt` | **`/root/autodl-tmp/envs/marlgpt`** | MAPF-GPT / pogema-benchmark 复现，**pogema==1.4.0** |
  | `lacam310` | `/root/miniconda3/envs/lacam310`（py3.10） | **只**生成专家数据 |
  > 旧文写的 `gat` 环境**不存在**。
- **★ LaCAM 致命坑**：`lacam/_lacam.so` 是 **py3.10 编译**，**py3.12 下 import 失败 →
  `backend=auto` 静默回退到差得多的 prioritized 规划器**（污染专家数据、不报错）。
  所有 LaCAM 数据生成**必须** `lacam310`。
- **评测协议（写死）**：`on_target=nothing`、`max_episode_steps=128`、`collision_system=soft`、
  `validation-*` 全 128 图、`split="all"`。主指标 CSR/ISR/SoC/makespan
  （★ makespan **右censored 于 128**，失败记 128）。
- **解码规则（本文的推断规则，基线用各自默认）**：`--action-mode sample`（**不是 argmax**，
  argmax 会让共享参数策略在对称局面互锁）、`--base-temp 1.0`、`--wait-bias 1.0`、`--tta`。
- **选模隔离**：检查点与解码超参**只在独立 dev 集上选**（`scripts/make_dev_maps.py`，64 图/族，
  `DEV_SEED_BASE=7_000_000`），report 只用 `validation-*`。**不得用测试集选配置。**
- **判性能只看 rollout，不看 loss**：loss ~2000 步就平台震荡，rollout CSR 涨到 20000+。
  **别在 ≤25% 训练下结论。**
- **root 所有不可写**：`checkpoints/`、`data/`、`logs/` → 用 `ckpt_il/`、`/root/autodl-tmp/`、scratchpad。
- **进程**：多进程任务用 `setsid` 独立进程组，`kill -9 -- -PGID`
  （`pkill -f 脚本名` 杀不掉 spawn 子进程，会留孤儿写坏数据）。
  **等作业不要用 `pgrep -f 名字`**（会匹配到自己的 heredoc，曾空等 12 小时）——用 PID 文件 + `kill -0` 或产物文件。

---

## 命令（2026-09-25 校准）

```bash
# 阶段一：生成 LaCAM 专家数据 —— ★必须 lacam310，★worker ≤ nproc-4
conda activate lacam310
nice -n 19 python scripts/generate_lacam_data.py --map-type maze --num-agents 64 \
  --num-episodes 20000 --seed-start <不相交段> --num-workers 12 --out <目录>
# 数据已删除，重生成配方与逐族参数见 DATA_REGEN.md（含「校正 A」：c2g_a64 的 stats 标错了 map_type）

# 阶段二：离线 IL 训练（单卡）
python -u scripts/train_il_meta.py --config configs/train_il_cap_lr1e4.yaml \
  --network-config configs/network_big.yaml --data-dir <数据目录> \
  --meta-iters 20000 --ckpt-milestones 2000,4000,...,20000 --seed 42 \
  --tag <tag> --out ckpt_il/<tag>_latest.pt

# 评测（统一评测台；★口径全部显式写死，不依赖默认值）
python -u scripts/eval_bench.py --ckpt ckpt_il/capbig_iter_20000.pt \
  --network-config configs/network_big.yaml \
  --families random,mazes --agent-counts 8,16,24,32,48,64 \
  --num-maps 128 --seeds 0,1,2 --max-steps 128 --on-target nothing \
  --action-mode sample --base-temp 1.0 --wait-bias 1.0 --tta \
  --out /root/autodl-tmp/eval_v5/<name>.json
# 调超参时加 --maps-file /root/autodl-tmp/dev_maps（dev 集，64 图/族）

# 主表 + 逐基线判决
python scripts/make_sq1_table.py --ours <json>:MADCL --metric CSR
```

---

## 关键超参（主模型 `network_big.yaml` + `train_il_cap_lr1e4.yaml`）

| 项 | 值 | 备注 |
|---|---|---|
| `obs_radius` / patch K | 5 / 11 | K = 2·r+1 |
| `patch_channels` | **5** | 原生3 + 足迹 + **cost2go** |
| `scalar_feat_dim` | 8 | 方向(2)+上一步动作(5)+到达(1) |
| `num_actions` | 5 | wait,up,down,left,right |
| `d_model` / `cnn_channels` / `n_res_blocks` | 320 / 128 / 3 | |
| `comm_radius` / `comm_topk_cap` | 4 / 6 | r_c < obs_radius；topk **仅算力上限** |
| `entmax_alpha` | 1.5 | 软稀疏 |
| `tau_in` / `tau_out` / `stickiness_bonus` | 0.05 / 0.02 / 0.1 | 迟滞 |
| `comm_dim` / `intent_dim` / `intent_mode` | 192 / 192 / `action` | |
| `temporal_backend` / `temporal_layers` | `gru` / 2 | |
| `batch_size` / `sequence_len` | 32 / 32 | |
| `inner_lr` | **1e-4** | ★**只对 7.45M 合适**；2.16M 偏好 **5e-4**（见 INVARIANT 16） |
| `inner_steps` / `tasks_per_meta` | 5 / 2 | |
| `grad_clip` / `use_amp` | 10.0 / true | |
| `reptile` | `false`（主模型 joint） | Reptile 臂用 true；`meta_lr` 0.1→0.01 |

> **废弃的旧超参**：`gru_hidden_dim`、`n_step`/`gamma`、`burn_in_len`、`mellowmax_alpha`、
> `expert_ratio`、`max_policy_version_lag` —— 都属 DQfD / 在线阶段三，v5 不用。

---

## 开发约定

- 所有可调项进 `configs/*.yaml`，勿硬编码。
- 全程固定随机种子（`utils/seed.py`），保证可复现。
- 任何性能改动**先跑 `utils/profiler.py`**，按占比改，改完用 §5 指标确认 SR/Deadlock/WaitRate 不退化。
- 指标必须能定位问题来源（任务效果/路径质量/冲突死锁/学习稳定性/模仿/通信/系统**七类**；
  **Q 值诊断整类已废弃**——没有 Q 头）。新增模块同时补对应指标。
- checkpoint 按 `--ckpt-milestones` 存里程碑 + `_latest.pt`（v2 的四类命名已不用）。
- **阴性结论要给上界，不要只说"不显著"**（例："效应 ≤ 基准的 8%，95% 置信"）。
- **差值必须报合并 CI**；落在 CI 内写"打平"，不得写"反超"。
- **先算功效再花算力**：SQ3 曾计划 512 图重跑，功效计算显示某臂需 968× 样本 ⇒ 直接放弃，省 20h。

---

## 高危易错点速查（CHECKLIST）

- [ ] 动作 0 是 wait 吗？（不是最后一位）
- [ ] policy 输入是否混进了任何全局/未来/他体信息？
- [ ] ~~reward 形式~~（v5 无 reward；纯 IL 只有交叉熵）
- [ ] 观测是否误加了与原生 ch2 重复的目标投影通道？
- [ ] ~~Reptile target 网络~~ / ~~阶段三 lag≤1~~（v5 均已废弃）
- [ ] **评测是否显式写死了全部口径参数**（不依赖任何默认值）？dev 与 test 有没有混用？
- [ ] **两臂是否各自在自己的最优 lr 下**？（别把治疗组的超参强加给对照组）
- [ ] 引用历史结果前**查过文件 mtime** 吗？（记录的 args ≠ 当时生效的行为，已三次翻车）
- [ ] 等作业是否用了 PID 文件 + `kill -0`，而**不是** `pgrep -f 名字`？（自匹配曾空等 12h）
- [ ] 多进程 worker 数是否 ≤ `nproc-4` 并加了 `nice`？
- [ ] 非法动作是否用 `-inf` 掩码？
- [ ] env-worker 里是否误建了 CUDA 上下文？
- [ ] 热路径是否残留 per-agent/per-step 的 Python 循环？
- [x] 通信邻居选择是否用了软稀疏+迟滞（而非 hard top-k 做选择）？EdgeFlipRate 是否在监控？
- [ ] 训练/评估地图是否互不相交？（地图直接取自 when-to-switch 全部 231 张 `when-to-switch/pomapf_env/maps.yaml`，**全池不分类别**、全局 8:2 划 train/test（184/47），见 `envs/map_registry.py`。配置 `map_name: "all"`；想只用某族改成正则前缀如 `random-`。`PogemaWrapper(split=...)` 默认 'train'，评估脚本须传 `split='test'`。`pick_map` 用乘法哈希按 seed 取图（连续 seed 跨族分布、确定性）。`envs/dataset_split.py` 把 train/test 的 seed 段隔开保证评估可复现。)
- [ ] 是否在没 profile 的情况下就动手写 Rust？（不要）
- [ ] 需要训练的作业：**数据已删**，是否先读了 `DATA_REGEN.md`（含 `c2g_a64` 的 map_type 标错）？
