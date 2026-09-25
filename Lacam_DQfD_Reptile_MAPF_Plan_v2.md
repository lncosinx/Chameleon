# MADCL：部分可观测多智能体路径规划方案说明（v5，2026-09-25）

> **本文件是设计与方法的事实来源。** 滚动的实验记录与结论在 `SCHEDULE.md`（§6.5「已定论」），
> 每次工作都要读的操作性摘要在 `CLAUDE.md`。三者冲突时：**实验事实以 SCHEDULE 为准，
> 方法与不变式以本文件为准，操作细节以 CLAUDE.md 为准**；修改设计决策时三处一并更新。
>
> **⚠️ v2→v5 的落差提示**：本文件在 2026-06-26 到 2026-09-25 之间未同步，
> 而这段时间方法已换过两次范式。旧 v2 正文（DQfD / 在线 RL / Dueling Q / 双 GPU 布局）
> 已整体移入 §9 归档，**不再是现行方案**。

---

## 变更摘要（v2 → v5）

| # | 项 | v2（2026-06） | **v5（现行）** |
|---|---|---|---|
| 1 | **训练范式** | DQfD（TD + n-step + margin）+ 在线 RL 三阶段 | **纯离线模仿学习**：对 LaCAM 专家动作做交叉熵 |
| 2 | **输出头** | Dueling Q（5 维 Q 值） | **动作 logits（5 维）+ 交叉熵** |
| 3 | **奖励** | BFS 势函数 shaping（PBRS） | **不用奖励**（纯 IL 无 reward 项） |
| 4 | **阶段三** | 在线 near-on-policy 微调（lag ≤ 1） | **废弃**，改为离线 few-shot 适应评估（SQ2） |
| 5 | **观测通道** | 4 通道（原生 3 + 足迹） | **5 通道**（+ cost2go，INVARIANT 3 已放宽） |
| 6 | **骨干** | Patch CNN + 单轮 GAT + GRU | **残差 CNN + FOV 软稀疏通信 + 时间核**（时间核可插拔：gru / gru_prenorm / mamba / transformer） |
| 7 | **硬件** | 2×RTX 5090 + 47 核 | **单卡 4080S 32G + 16 vCPU**（cgroup 80G 内存） |
| 8 | **主模型** | 未定 | **MADCL-L：GRU 7.45M @ iter 20000** |
| 9 | **评测协议** | `on_target=finish`、512 步 | **`on_target=nothing`、128 步、soft 碰撞、validation-* 128 图**（写死不再改） |
| 10 | **元学习** | Reptile（含 target 网络处理） | Reptile 保留（SQ2），**新增 joint 与 FOMAML 两个对照臂**；FOMAML 已放弃 |
| 11 | **选模** | 未规定 | **独立 dev 地图集**（`make_dev_maps.py`）选检查点与解码超参，report 只用 validation-* |
| 12 | **科学问题** | 无 | **SQ1~SQ4 四条主线**（见 §7） |

---

## 0. 定位（v5 现行）

基于 **POGEMA** 网格环境的**部分可观测多智能体路径规划**。每体只用自身 FOV 内信息决策，
去中心化、共享参数、**不构造 $Q_{tot}$、不用 QMIX**。

**流水线只有两段**：

```
① LaCAM 专家数据生成（训练期特权，只产专家动作）
      ↓
② 离线模仿学习（交叉熵）—— joint 或 Reptile 元初始化
      ↓
   评测：POGEMA 标准协议 + 解码规则（采样 + wait_bias + TTA）
```

**头条贡献 = 通信**（FOV 受限 + entmax 软稀疏 + 迟滞），不比"泛化"。
理由与竞品分析见 §9.1（v3 战略转向原文，仍然有效）。

**诚实边界**：主模型对 SCRIMP 在 mazes 赢、random 平；对 MAPF-GPT（2M / 85M）输，
按**参数 × 数据双轴 Pareto** 陈述而非"我们更强"。详见 `SCHEDULE.md` §6.5「当前主表」。

---

## 1. 不可违反的硬约束（INVARIANTS，v5 现行）

> 改动任何一条前，先在本文件找到对应小节并同步更新。带 ~~删除线~~ 的是因范式转换而废弃的。

1. **POGEMA 动作顺序（实测，勿凭记忆）**：`0=wait`, `1=up(-1,0)`, `2=down(+1,0)`, `3=left(0,-1)`, `4=right(0,+1)`。
   **wait 在第 0 位。** one-hot、动作掩码、LaCAM 动作转换、logits 语义全部以此为准。
   改顺序会让模仿学习整体错位**且不报错**。
2. **部分可观测**：policy 输入**仅** `(C,11,11) patch + 8 维标量`（C=5）。
   严禁全局地图、**整幅**全局距离场、LaCAM 路径、他体未来轨迹进 policy。
3. **特权信号只在训练期**：LaCAM 只产专家动作，部署期无 LaCAM。
   **[v4 放宽]** **cost2go（FOV 内 BFS 测地距离场的局部裁剪）是合法局部特征、可进 policy**（channel 4）。
   理由：所有对比 baseline（EPH / MAPF-GPT / SCRIMP）都喂障碍感知的目标梯度；
   只给穿墙方向向量会导致 mazes SR=0、纯 IL 塌缩成直线策略。
   **全局 BFS 距离场整幅、他体路径、LaCAM 路径仍严禁进 policy。**
4. ~~**Reward = BFS 势函数 shaping（PBRS）**~~ —— **v5 废弃**：纯 IL 无 reward 项。
   历史约束保留在 §9 供追溯（其中「禁止用曼哈顿距离做势函数」的理由仍有参考价值）。
5. **观测通道**：POGEMA 原生 `(3,11,11)` = 障碍 / 智能体（自身恒在中心格）/ 目标边界投影；
   本仓库追加 **ch3 = 时间衰减自身足迹**、**ch4 = cost2go**（[0,1]，0.5=同距）。足迹通道定义：

   $$\text{ch3}(u)=\max_{\tau\le t}\;\mathbb 1\big[\,p_i^{\tau}\ \text{落在以当前位置为中心的 patch 坐标 } u\,\big]\cdot\rho^{\,t-\tau},
     \qquad \rho\in(0,1)$$

   取值范围 [0, 0.8]，当前位置恒为 1.0。
   ⇒ `patch_channels=5`。**禁止再加与原生 ch2 重复的"目标投影通道"**；自身位置已由 ch1 中心格表达。
6. **目标相对方向是合法局部特征**（`get_targets_xy`）——不是全局泄露。
7. **非法动作掩码 = `-float('inf')`**，不是有限大负常数。须保证每体至少 1 个合法动作（wait 通常恒合法）。
8. ~~**Reptile 的 target 网络冻结于 θ_meta**~~ —— **v5 废弃**（无 target 网络，纯 IL）。
9. ~~**on/off-policy 分工**~~ —— **v5 废弃**（无在线阶段）。
10. **通信约束**：`comm_radius=4 < obs_radius=5`，**只与 FOV 内可见邻居通信**。
    邻居选择用**软稀疏注意力（entmax α=1.5）+ 迟滞（τ_out=0.02 < τ_in=0.05）**，
    **不用 hard top-k 做选择**——hard top-k 会引起通信图闪烁（边界排名交换 → 边反复通断）。
    `comm_topk_cap=6` 仅是**算力上限**。监控 `EdgeFlipRate`。
    > **⚠️ 实测修正（已定论 24）**：出货配置下 **迟滞把 entmax 的稀疏性抵消了** ——
    > 真实度数 dense 5.911 / entmax 无迟滞 4.118 / **entmax+迟滞 5.877（仅比 dense 少 0.6%）**。
    > 故「软稀疏省带宽」在出货配置下**不成立**；可立住的是通信必要性与迟滞降低闪烁（同机制内 −19.6%）。
11. ~~**专家 margin 样本过滤**~~ —— margin loss 已废弃；但其动机（同一局部观测下专家动作不唯一）
    仍解释为何**单步模仿准确率不是 CSR 的好代理**（见 §7.2）。
    现行的对应旋钮是 `il_wait_weight`（专家数据 wait 占 ~50%），**实测降权无效**（已定论 15）。
12. ~~**mellowmax α**~~ —— 废弃（无 Q 目标）。
13. **循环回放 burn-in**：GRU 类时间核在训练时按 `sequence_len=32` 整段展开；
    评测期逐步推进并在 episode 边界 `reset_comm_state()`（同时重置通信迟滞与时间核状态）。
14. **诊断用曼哈顿距离 ≠ 训练信号**：事后诊断可用，绝不进策略内部目标。
15. **[v5 新增] 评测口径写死**：`on_target=nothing`、`max_episode_steps=128`、`collision_system=soft`、
    `validation-*` 全 128 图、`split="all"`（不重映射 seed，保证与 leaderboard 逐实例配对）。
    **推断规则统一为采样**（`--action-mode sample`，`base_temp=1.0`）**+ TTA**，所有臂一致（含基线消融）。
16. **[v5 新增] 超参不得跨臂强加**：为治疗组选的超参不能强加给对照组。
    实测教训：`inner_lr=1e-4` 是为 7.45M 选的（大网 5e-4 第 250 步塌缩），
    小网对照臂被迫跟用而被系统性低估 **0.227（显著）**，导致「等步数加容量就赢」这一结论失效（已定论 23）。
17. **[v5 新增] 默认值即协议**：评测台每个默认值都是一份隐式协议。
    探针脚本必须**显式写死全部口径参数**，不得依赖默认值。已因此三次翻车（见 §7.1）。

---

## 2. 目录结构与模块职责（2026-09-25 与代码核对）

```
configs/      env_*.yaml（random/mazes/maze/warehouse/room/street/sc1/wc3 + *_small/*_nothing/*_bench）
              network_*.yaml   骨干与规模：gru_c2g(2.16M) / big(7.45M，主模型) / xl(16.9M，弃)
                               small_mamba(2.22M) / big_mamba(7.57M) / small_gru_prenorm / transformer
                               消融变体：small_mamba_{dtfix,fnorm,both,dconv1}、arm{T,S}
              train_il_*.yaml  离线 IL 训练配方（cap_lr{1e4,2e4,5e4} / joint / reptile / v5* / v6* / waitw）
              train_reptile_dqfd.yaml、finetune_online.yaml  ← v2 遗留，已不用
envs/         pogema_env_wrapper（random 用 POGEMA 原生 generator；其余经 pogema-toolbox）
              observation_builder（原生 3ch + 足迹 + cost2go，`use_cost2go` 开关）
              action_mask（-inf）/ potential_field（BFS 距离场，现只为 cost2go 与诊断）
              map_registry / dataset_split（train/test seed 段隔离）/ d4_augment（D4 对称增强）
              env_worker（多进程采样）/ reward_builder ← v2 遗留，纯 IL 不用
lacam/        lacam_runner（C++ 绑定，★必须用 lacam310 环境）/ lacam_dataset_builder
              trajectory_converter（动作映射）/ hard_case_manager / _lacam.so（py3.10 编译）
models/       mapf_comm_net（**主网络 MAPFCommNet，v5 组装入口**）
              spatial_encoder（ResidualCNNEncoder）/ communication（BiIntentAttention：entmax + 迟滞）
              temporal_core（可插拔时间核：gru / gru_prenorm / mamba / transformer）
              comm_baselines（6 个外部通信基线移植到同骨干）/ obs_tokenizer / transformer_*
              biintent_drqn / q_heads / encoders / recurrent_core ← v2 遗留
losses/       il_losses（**现行**：交叉熵 + wait 权重 + label smoothing）
              dqfd_losses / td_losses / q_target ← v2 遗留，已不用
buffers/      expert_demo_buffer（**现行**，支持 memmap 零拷贝挂载）/ batch_collator
              selfplay_buffer / recent_online_buffer / shared_rollout_buffer ← v2 遗留
trainers/     train_il_reptile（**现行 ILReptileTrainer**：joint / reptile / maml 三模式）
              evaluator / scheduler / seed_actor / selfplay_producer / finetune_online / train_reptile_dqfd ← 遗留
utils/        metrics / profiler / checkpoint / seed / logger
scripts/      train_il_meta（**训练入口**）/ eval_bench（**统一评测台**）
              generate_lacam_data、generate_expert_data（`--seed-start` 防重复）/ generate_dagger_data
              make_dev_maps（**独立 dev 地图集**，DEV_SEED_BASE=7_000_000）
              make_sq1_table（主表 + 逐基线判决）/ run_sq2_adaptation / run_sq4_continual
              run_mapfgpt_bench（marlgpt 环境复现 MAPF-GPT）/ probe_generalization
              check_episode_overlap（首帧哈希查重）/ test_*（单测与阶段门）
```

**⚠️ v2 遗留文件保留但不使用**：`losses/{dqfd,td,q_target}`、`buffers/{selfplay,recent_online}`、
`models/{biintent_drqn,q_heads}`、`trainers/{train_reptile_dqfd,finetune_online}`、`envs/reward_builder`。
保留是为可追溯与将来做「IL + 轻量 TD grounding」消融；**不要以为它们在跑**。

### 三个 conda 环境（不可混用）

| 环境 | 用途 | 关键约束 |
|---|---|---|
| 默认（`/root/miniconda3`，py3.12） | 训练 / 评测 | pogema 2.0.0a；`mamba-ssm 2.3.2.post1` 已装（内核可用） |
| `marlgpt` | MAPF-GPT / pogema-benchmark 复现 | **pogema==1.4.0** |
| `lacam310` | **只**生成专家数据 | **py3.10**；`_lacam.so` 是 py3.10 编译 |

**★ LaCAM 致命坑**：py3.12 下 import `_lacam.so` 失败 → `backend=auto` **静默回退到差得多的
prioritized 规划器**，污染专家数据且不报错。所有 LaCAM 数据生成必须用 `lacam310`。

---

## 3. 网络架构（MADCL / `MAPFCommNet`）

### 3.1 前向（与代码一致）

```
patch (5,11,11) + scalar (8)
      ↓  ResidualCNNEncoder（n_res_blocks 个残差块，cnn_channels 通道）
   e : (M, d_model)                          M = batch × agents
      ↓  comm_rounds 次（默认 1 轮）
   intent = comm_in( e )                     intent_mode=feature
          = comm_in( [e ; prev_action_onehot] )   intent_mode=action（主模型用）
   comm_feat = BiIntentAttention(intent, 位置, FOV 掩码)
   e = e + comm_out(comm_feat)               ← 残差写回
      ↓
   TemporalCore（backend ∈ {gru, gru_prenorm, mamba, transformer}，temporal_layers 层）
      ↓
   head: Linear(d_model, head_hidden) → GELU → Linear(head_hidden, 5)
      ↓
   logits (5)   —— 非法动作置 -inf 后按 §5 的解码规则取动作
```

### 3.1.1 策略与解码的形式化

共享参数网络 $f_\theta$ 对每体输出 5 维 logits；非法动作先置 $-\infty$：

$$\ell_i^t = f_\theta\big(o_i^t,\ h_i^{t-1},\ \mathbf c_i^t\big)\in\mathbb R^{5},
\qquad
\tilde\ell_{i,a}^t=\begin{cases}\ell_{i,a}^t & a\in\mathcal A_i^{\text{legal}}\\[2pt] -\infty & \text{否则}\end{cases}$$

**v5 的动作选取不是 argmax，而是带 wait 偏置的温度采样**（见 §5.2；argmax 会让共享参数策略
在对称局面互锁）：

$$a_i^t \sim \mathrm{softmax}\Big(\tfrac{1}{T}\big(\tilde\ell_i^t - b\,\mathbf e_0\big)\Big),
\qquad T=1.0,\quad b=1.0,\quad \mathbf e_0=\text{wait 的 one-hot}$$

**TTA（D4 测试期集成）**：对 8 个二面体对称 $g\in D_4$ 各前向一次，把 logits 经动作置换
$\pi_g$ 逆映射回原动作空间后**对概率平均**：

$$p_i^t=\frac{1}{|D_4|}\sum_{g\in D_4}\pi_g^{-1}\!\Big[\mathrm{softmax}\big(f_\theta(g\cdot o_i^t)\big)\Big]$$

> 逆置换的正确性用"人工构造的等变模型"单测验证，误差 1.5e-8（`scripts/test_d4_augment.py`）。

### 3.2 通信模块（`BiIntentAttention`，头条）

候选邻接：$j \in \mathcal{N}_i^{\text{cand}} \iff \lVert p_i - p_j \rVert \le r_c \wedge j \in \mathrm{FOV}_i$。
在候选集上算多头注意力，用 **entmax(α=1.5)** 代替 softmax 得到可**精确为零**的权重：

$$\alpha_{ij}=\mathrm{entmax}_{1.5}\big(\mathrm{score}(\mathbf z_i,\mathbf z_j)\big),\qquad
  \mathbf c_i=\sum_{j}\alpha_{ij}\,W\mathbf z_j$$

**迟滞**：已存在的边按较低阈值 `tau_out=0.02` 保留，新边需越过 `tau_in=0.05` 才建立，
并对已有边加 `stickiness_bonus=0.1`。无邻居则 $\mathbf c_i=\mathbf 0$。

上报三个指标：`avg_neighbors`（= `W>0` 的真实度数，**即带宽口径**）、`edge_flip_rate`、`attention 熵`。

> **★ fp16 溢出坑（已修，`8f4bbb4`）**：注意力 logits 在 AMP 下会溢出到 65504 → NaN。
> **只写 `.float()` 无效**——autocast 会拦截 matmul 把 fp32 入参强制降回 fp16。
> 必须显式 `with torch.autocast(..., enabled=False)` 包住该 matmul。
> 回归测试：`scripts/test_comm_fp16_overflow.py`（comm_dim × 激活幅度 12 组合全过）。

### 3.3 时间核（`TemporalCore`，可插拔）

通信残差写回后进时间核；记 $x_i^t$ 为进入时间核的特征：

$$x_i^t=\underbrace{e_i^t}_{\text{CNN 编码}}+\ \mathrm{comm\_out}\big(\mathbf c_i^t\big),
\qquad
h_i^t=\mathcal T_\theta\big(x_i^t,\ h_i^{t-1}\big),
\qquad
\ell_i^t=\mathrm{head}\big(h_i^t\big)$$

其中 $\mathcal T_\theta$ 按 `temporal_backend` 取 GRU / pre-norm GRU / Mamba(S6) / Transformer。
训练时按 `sequence_len=32` **整段展开**；评测时逐步推进，并在 episode 边界
`reset_comm_state()`（同时清通信迟滞与 $h$）。

| backend | 说明 |
|---|---|
| `gru`（**主模型**） | 标准 `nn.GRU`，`temporal_layers=2` |
| `gru_prenorm` | GRU 外套 pre-norm + 逐层残差（归因臂，实测 −0.060 n.s.） |
| `mamba` | S6 选择性 SSM；`mamba-ssm` 官方 CUDA 内核可用（块级快 6.7×，但只占训练单步 1.4%） |
| `transformer` | 未作为主线 |

> **★ Mamba 的实测结论（已定论 18/21）**：该块的增益**不来自时序机制**。
> SSM 因 `dt` 初始化缺失而无记忆（dA≈2.5e-3，scan 坍缩）；`d_conv` 的 4 步局部卷积贡献 +0.018 n.s.；
> 把卷积降到 kernel=1（跨时步泄漏 0.63 → 4.6e-3）后相对裸 GRU 仍保留 +0.315。
> ⇒ 赢在**门控前馈块**（`in_proj` 2× 扩张 + SiLU(z) 门控 + `out_proj`），不是时间核。
> 反向印证：修好 `dt` 初始化并补出口 LayerNorm（`dt_init: official` + `final_norm: true`）
> 终点 −0.042 n.s. —— 救活时序通路同样买不到分。
> ⚠️ 上述优势量级**与 lr 不可分离**（各臂皆 lr=1e-4，最优 lr 未测，见 INVARIANT 16）。

### 3.4 主模型超参（`configs/network_big.yaml`，7.45M）

| 项 | 值 | | 项 | 值 |
|---|---|---|---|---|
| `patch_channels` | 5 | | `comm_radius` / `comm_topk_cap` | 4 / 6 |
| `scalar_feat_dim` | 8 | | `entmax_alpha` | 1.5 |
| `obs_radius` / patch K | 5 / 11 | | `tau_in` / `tau_out` / `stickiness_bonus` | 0.05 / 0.02 / 0.1 |
| `d_model` | 320 | | `comm_dim` / `intent_dim` | 192 / 192 |
| `cnn_channels` / `n_res_blocks` | 128 / 3 | | `intent_mode` | `action` |
| `head_hidden` | 256 | | `temporal_backend` / `temporal_layers` | `gru` / 2 |
| `num_actions` | 5 | | `num_attention_heads` | 4 |

---

## 4. 训练（离线模仿学习）

### 4.1 阶段一：LaCAM 专家数据

`scripts/generate_lacam_data.py` / `generate_expert_data.py`，**必须 `lacam310` 环境**。
`--seed-start` 必传并与 `TEST_SEED_BASE` 做范围检查（**曾因 seed 写死产出 66~98% 重复 episode**，
使「数据扩充无用」与「泛化 gap 0.009」两条结论作废，见 §7.1）。
落盘为 `expert_{family}/` 目录 + memmap，`ExpertDemoBuffer` 零拷贝挂载。
查重工具：`scripts/check_episode_overlap.py`（首帧哈希）。

### 4.2 阶段二：离线 IL

**损失**（`losses/il_losses.py`）：对专家动作的交叉熵，可选 `wait_weight`（专家 wait 占 ~50%）
与 `label_smoothing`。**无 TD、无 n-step、无 margin、无 reward。**

$$\mathcal L = \mathbb E_{(o,a^E)}\big[\, w(a^E)\cdot \mathrm{CE}\big(\text{logits}_\theta(o),\, a^E\big)\,\big]$$

设 inner loop 从 $\theta$ 出发做 $K$ 步（`inner_steps=5`）Adam 得到 $\theta_K$，
$\mathcal B$ 为本 meta-iter 的 task 集合（`tasks_per_meta=2`）。**三种外层模式**
（`trainers/train_il_reptile.py::ILReptileTrainer`，共用同一份 inner loop）：

$$\textbf{Reptile:}\quad
  \theta \leftarrow \theta + \varepsilon\cdot\frac{1}{|\mathcal B|}\sum_{j\in\mathcal B}
  \big(\theta_K^{(j)}-\theta\big),
  \qquad \varepsilon:\ 0.1 \to 0.01$$

$$\textbf{FOMAML:}\quad
  \theta \leftarrow \mathrm{Adam}\Big(\theta,\ \frac{1}{|\mathcal B|}\sum_{j\in\mathcal B}
  \nabla_{\theta}\mathcal L^{(j)}_{\text{query}}\big(\theta_K^{(j)}\big)\Big)$$

$$\textbf{joint:}\quad \text{无外层}\ —\ \text{连续 SGD，保留优化器动量}$$

⚠️ Reptile 的 $\varepsilon$ 作用在**位移**上（已被 inner Adam 归一化到 $\sim$lr 量级），
FOMAML 的作用在**原始梯度**上 —— 量纲差几个数量级，这是外层必须换 Adam 的原因。


| 模式 | 配置 | 外层更新 |
|---|---|---|
| **joint**（主模型用） | `reptile: false` | 普通连续 SGD，**保留优化器动量** |
| **Reptile**（SQ2） | `reptile: true` | $\theta \leftarrow \theta + \text{meta\_lr}\,(\theta_K-\theta)$；每 task 起点重置、清动量 |
| **FOMAML**（SQ2 第三臂） | `maml: true` | 查询集在 $\theta_K$ 处的梯度；**外层必须用 Adam**（见下） |

> **★ FOMAML 的两个实现坑（`a138c09`）**：
> ① 不能复用 `self.scaler` 做查询梯度缩放——inner loop 已对它 `unscale_()` 过，
>    再调会抛 `unscale_() has already been called`。改用静态缩放 `2**16`。
> ② **外层必须用 Adam**：Reptile 的 `meta_lr` 作用在**位移**上（已被 inner Adam 归一化），
>    MAML 的作用在**原始梯度**上，量纲差几个数量级；沿用 `meta_lr=0.1` 会瞬间发散。
> **结论：已放弃**（外层 lr 扫 4 档，最好 acc 0.754 vs Reptile 0.90~0.92，已定论 20）。

**主模型训练配方**（`configs/train_il_cap_lr1e4.yaml` + `network_big.yaml`）：
`batch_size=32`、`sequence_len=32`、`inner_steps=5`、`tasks_per_meta=2`、`inner_lr=1e-4`、
`grad_clip=10`、`use_amp=true`、12 个族（random/maze × {32,48,64} × {a,b}）、`reptile: false`。
**主检查点 = `ckpt_il/capbig_iter_20000.pt`。**

> **★ 学习率的规模律（已定论 22/23）**：**最优 lr 随模型规模下降。**
> 2.16M 偏好 **5e-4**（比 1e-4 高 +0.227 ★显著）；7.45M 在 **1e-4** 附近
> （2e-4 打平 +0.031/−0.047 n.s.、**3e-4 第 15 步即塌缩成全 wait**）。
> ⇒ 跨规模对照**必须每臂各自调 lr**，否则对照组被系统性低估（INVARIANT 16）。

### 4.3 判性能只看 rollout，不看 loss

训练 loss 约 2000 步就平台震荡，而 rollout CSR 到 20000+ 仍在涨。**别在 ≤25% 训练进度下结论**
（v4 曾在 iter 1000/5000 下过三次相反结论，全被 20000 打脸）。

---

## 5. 评测协议与解码规则（写死，不再改）

### 5.1 协议

`on_target=nothing`（标准 MAPF，到达即守位）、`max_episode_steps=128`、`collision_system=soft`、
`observation_type=MAPF`、`validation-*` 全 128 图、`split="all"`。
主指标 **CSR**（全体到达才算成功）、**ISR**、**SoC**、**makespan**（★右censored 于 128，失败记 128）。

**与 leaderboard 的可比性（三条前提已逐一核对）**：地图逐字节相同；协议相同；
实例相同（不重映射 seed）⇒ 与 SCRIMP/DCC/LaCAM **逐实例配对**可比。
CSR/ISR 与 POGEMA 自带 `info['metrics']` **逐 episode 对拍一致**。

**各族协议不同**（`--auto-protocol` 内置官方网格）：warehouse 是 1 图 × 128 seed、
agents 32~192；movingai 是 256 步。混用会产出不可比数字。

### 5.2 解码规则（本文的推断规则，须在论文声明；基线用各自默认解码）

| 组件 | 值 | 增益 |
|---|---|---|
| 动作选取 | **按策略分布采样**（非 argmax） | argmax 会让共享参数策略在对称局面互锁，CSR 0.30→0.88（已定论 1） |
| 温度 | `base_temp = 1.0` | 温度不敏感；T=0.3 时 dev 从 2.094 崩到 0.656 |
| `wait_bias` | **1.0**（wait 的 logit 减 1.0） | 同检查点 9.232 → **9.375（+0.143）**，零训练成本 |
| **TTA** | D4 八视角逆置换后概率平均 | 第二个免费大幅提升（已定论 3），代价 8× 推断算力，须明说 |

> **★ 选模隔离**：检查点与解码超参**只在独立 dev 集上选**（`make_dev_maps.py`，
> 64 图/族，`DEV_SEED_BASE=7_000_000`，与训练/测试 seed 段均不重叠），report 只用 `validation-*`。
> **曾失效过**：`--maps-file` 自引入起未接到 `load_maps()`，使 32 个脚本的「dev」实际是
> validation 的偶数下标 64 图子集（已修，去污染净代价 −0.018，见已定论 19）。

> **★ makespan 不是越小越好（已定论 13 补注）**：三个解码变体中 **makespan 最短的 CSR 最低**
> （分层温度 62.7 / 9.326），**最长的 CSR 最高**（wait_bias 1.5 → 75.8 / 9.393）。
> 唯一奏效的干预是让智能体**多等**而非更快 ⇒ 约束是**冲突消解效率**，不是原始速度。
> 「把 makespan 放进训练目标」这个方向的前提因此存疑。

---

## 6. 评价指标（v5 精简，删去 Q 值类）

v2 的八类指标中，**Q 值诊断（原 §5.5）整类废弃**（无 Q 头）；`L_TD/L_n/L_margin` 走势改为单一 CE loss。
其余保留。**先给形式化定义**（v2 的公式在此保留并按 v5 口径更新），再列在 v5 中起决定性作用的几项。

### 6.1 任务效果

记一个 episode 有 $N$ 体，$T_i$ 为第 $i$ 体的到达时刻（未到达则 $T_i=\infty$），$H=128$ 为步数上限。

$$\textbf{ISR}=\frac{1}{N}\sum_{i=1}^{N}\mathbb 1\big[T_i\le H\big],
\qquad
\textbf{CSR}=\mathbb 1\Big[\textstyle\bigwedge_{i=1}^{N}\ T_i\le H\Big]$$

**CSR 是主指标**（全体到达才算成功），在 $n$ 个实例上取均值。
**ISR 高而 CSR 低 ⟹ 失败集中在少数体**——本项目 `ISR@384 = 0.999` 正是由此判出"不是死锁"。

### 6.2 路径质量

$$\textbf{Makespan}=\max_i \hat T_i,\qquad
\textbf{SoC}=\sum_i \hat T_i,\qquad
\hat T_i=\min\big(T_i,\ H\big)$$

**★ $\hat T_i=\min(T_i,H)$ 即右删失**：未到达的体按 $H=128$ 计入。
⇒ **CSR 越高、被删失的 episode 越少，均值本应下降**；若 CSR 升而 makespan 也升，
说明成功的那些 episode **确实变慢了**（见 §5.2 末对 `wait_bias` 的解读）。

$$\textbf{Inflation}=\frac{\text{len}_{\text{policy}}}{\text{len}_{\text{LaCAM}}}
\qquad(\gg 1 \Rightarrow \text{绕路/等待过多})$$

本项目实测 makespan 比专家慢 **2.0~3.7×**，这是"失败源于超时"的直接证据（已定论 13）。

### 6.3 冲突与死锁

顶点冲突数、边冲突数（交换位置）、**WaitRate**（未移动步数占比）、
**NoProgressRate**（★仅用曼哈顿做**事后诊断**，绝不进训练目标）、
**RepeatRate**（反复经过自身足迹）、**死锁判定**（连续 $W$ 步无新增到达且平均位移极低）。

### 6.4 通信图监控（规模 + 时间稳定性）

在 $r_c$（FOV 内、软稀疏选择）下构建动态图 $G^t=(V,E^t)$，边集取注意力权重非零者：

$$E^t=\big\{(i,j)\ :\ W_{ij}^t>0\big\}$$

$$\textbf{avg\_neighbors}=\frac{1}{N}\sum_i\big|\{j:W_{ij}^t>0\}\big|
\qquad(\textbf{即带宽/消息量口径，topk 封顶后的真实度数})$$

$$\textbf{EdgeFlipRate}=\frac{\big|E^t\ \triangle\ E^{t+1}\big|}{\big|E^t\ \cup\ E^{t+1}\big|}
\qquad(\text{高} \Rightarrow \text{通信图闪烁，时间稳定性失控})$$

$$\textbf{Attention 熵}=-\sum_j \alpha_{ij}\log\alpha_{ij}
\qquad(\approx\log k \Rightarrow \text{无差别关注};\ \text{低熵} \Rightarrow \text{聚焦关键冲突邻居})$$

> **⚠️ EdgeFlipRate 跨臂比较不可比**（已定论 24）：分母是边集并集，而各臂度数相差 30% 以上
> ⇒ 归一化基数不同。更要紧的是 **dense softmax 的 $W_{ij}>0$ 恒为真**，其 $E^t$ 只随 FOV 进出与
> topk 排名交换变化，与 entmax 的"注意力驱动通断"**不是同一种量**。
> 迟滞的稳定作用只能在**同一稀疏机制内部**比较（madcl 0.1458 vs 无迟滞 0.1813 = −19.6%）。
>
> **⚠️ 另**：`MaxCC`（最大连通分支规模）在 v2 用于监控"拓扑雪崩"，v5 因 FOV + `topk_cap=6`
> 已把规模封住，该指标未再作为判据。

### 6.5 统计判据（v5 新增，必须遵守）

差值必须报**合并 95% CI**。设两臂各格的 CI95 为 $c^{(1)}_k,c^{(2)}_k$，在 $m$ 个格上求和后：

$$\mathrm{CI}_{\text{合并}}=1.96\sqrt{\sum_{k=1}^{m}\Big(\tfrac{c^{(1)}_k}{1.96}\Big)^2+\sum_{k=1}^{m}\Big(\tfrac{c^{(2)}_k}{1.96}\Big)^2}$$

- $|d|>\mathrm{CI}_{\text{合并}}$ ⟹ 显著；否则**写"打平"，不得写"反超"**。
- **阴性要给上界**，不要只说"不显著"：$|\text{效应}|\le |d|+\mathrm{CI}_{\text{合并}}$。
- **先算功效再花算力**。由 $\mathrm{CI}\propto n^{-1/2}$，令差值 $d$ 达到显著所需样本：

$$n_{\text{需要}}=n_{\text{现有}}\cdot\Big(\frac{\mathrm{CI}_{\text{合并}}}{d}\Big)^{2}$$

> 实例：SQ3 内部消融曾计划 512 图重跑（12× 样本），按上式迟滞那一臂需 **968×** ⇒ 直接判定
> **阴性是结论本身、不是样本不足**，省下约 20 小时（已定论 24）。

### 6.6 在 v5 中起决定性作用的几项

- **CSR / ISR 联读**：`ISR@384 = 0.999` 而 `CSR` 低 ⇒ 失败集中在少数体、且不是死锁（已定论 13）。
- **makespan（右censored 于 128）**：与专家比 2.0~3.7×。★注意删失：CSR 越高被记 128 的越少，
  均值本该下降；若 CSR 升而 makespan 也升，说明成功 episode 确实变慢了。
- **`avg_neighbors`**（带宽口径）与 **`edge_flip_rate`**：定义与跨臂比较的陷阱见 §6.4。
  实测结论：**迟滞把 entmax 的稀疏性抵消了** —— dense 5.911 / entmax 无迟滞 4.118 /
  出货配置 5.877（仅比 dense 少 0.6%）⇒「软稀疏省带宽」在出货配置下**不成立**（已定论 24）。
- **单步模仿准确率**：**不是 CSR 的好代理**——四个模型留出集准确率差 ≤0.005，而测试 CSR 差 0.339。
  机理见 INVARIANT 11 的说明（同一局部观测下专家动作不唯一，准确率天花板在 0.90~0.96，
  胜负取决于少数关键步）。

---

## 7. 四条科学问题（论文骨架 = 实验骨架）

各条的完整结果与判决在 `SCHEDULE.md` §6.5；此处只记问题、贡献点与**当前立得住的边界**。

| | 问题 | 贡献点 | 当前边界 |
|---|---|---|---|
| **SQ1** | 去中心化 PO 策略在标准 POGEMA 协议下能到什么位置 | MADCL 整体 | 主表 **9.375**（12 档和）。对 DCC/MAMBA/QMIX/VDN/IQL/QPLEX **两族全 12 档胜**；对 SCRIMP **mazes 赢（+0.194，5/6 档）、random 平（+0.018，噪声内）**；对 MAPF-GPT 输 |
| **SQ2** | 元学习是否带来跨场景与快速适应 | Reptile | **零样本**跨族迁移全 6 档优于 joint 初始化（3.352 vs 3.062）；**"快速适应"不成立**——适应对两臂方向相反且收敛到同一水平。FOMAML 未能稳定训练。局限：基于 2.16M 档 |
| **SQ3** | 通信机制是否提升去中心化协作 | FOV + 软稀疏 + 迟滞（**头条**） | **通信必要性极强**（去掉后 3.461 → 0.031）；迟滞降低闪烁（同机制内 −19.6%）；@64 档带宽 Pareto 优于 TarMAC（−32%，CSR 差在 CI 内）。**不可立住**：稀疏本身涨 CSR（效应 ≤ 基准 8%）、软稀疏省带宽（被迟滞抵消） |
| **SQ4** | 持续分布漂移下是否保持性能 | Replay + Meta Update | 跨族 replay 显著抑制遗忘（BWT −0.027 → −0.008） |

### 7.1 方法学陷阱清单（本项目最贵的教训，每条都真实翻过车）

1. **「记录值 ≠ 生效值」（三次）**：
   `--map-size` 默认 64 覆盖配置的 20；`--base-temp` 默认 0.3 **在 sample 模式下从未生效**
   （历史"T=0.3"实际都是 T=1.0）；`--maps-file` 自引入起未接到调用点。
   ⇒ **引用任何历史结果前先查文件 mtime**，不能只信 json 里记录的 args。
2. **「共享超参 ≠ 公平对照」（第四次同类）**：见 INVARIANT 16。比前三次更隐蔽，
   因为命令行里两臂确实只差一个变量。
3. **seed 写死 → 专家数据重复**：使两条结论作废。`--seed-start` + 首帧哈希查重是防线。
4. **把「相关」当成可干预的「因果」**：探针显示专家解冲突时"有动量者胜 74.8%"（z=+95.7），
   但解码端加惯性**单调有害**（2.359 → 1.828）。相关性不等于可干预。
5. **清零式记忆消融会系统性高估时序重要性**：「每 N 步清零隐状态」测出记忆很重要
   （+0.922 / +0.500），但**从零训练一个没有时序通路的模型并不掉分** ⇒ 清零度量的是
   **分布外冲击**而非记忆价值。该类结论只能作为**上界**。
6. **阴性结论要给上界，不要只说"不显著"**：例如"entmax 与迟滞对 CSR 的影响 ≤ 基准的 8%
   （95% 置信）"比"不显著"有信息量，且不需要更多算力。
7. **先算功效再花算力**：SQ3 内部消融曾计划 512 图重跑，功效计算显示迟滞那一臂需
   **968×** 样本 ⇒ 阴性是结论本身，不是样本不足。省下约 20 小时。
8. **单 seed 是全项目的共同软肋**：所有臂只训一个 seed，而判"显著"的效应量在 0.05~0.40。
   训练 seed 方差**从未测过**，审稿最易问此处。

### 7.2 两个有效杠杆 vs 十项阴性

**有效**：① **加容量 + 更长训练**（2.16M→7.45M 且 8k→14k，+0.391 ★；⚠️ **等步数下只有 +0.164 n.s.**，
见 INVARIANT 16）② **解码优化**（+0.143，**零训练成本**）。

**阴性**（均唯一变量对照）：D4 训练增强、DAgger、更长训练（8k→30k）、2.56× 真·新数据、
小幅加容量 +17%（CNN / GRU 两侧各一次）、Mamba 骨干（7.45M 档）、训练端 wait 降权、
通信 2 轮、解码端动作惯性、一阶 MAML。

---

## 8. 判决原则（本项目最贵的教训，继续遵守）

1. 让数据判决，不靠论证；每改一项 → 同图台重测 → 看 rollout CSR。
2. **≤25% 训练进度不下结论。**
3. **loss / 单步准确率 ≠ rollout CSR**（拟合更好但 rollout 更差是常态）。
4. Deadlock 二值指标不可单独解读，必须与 ISR / WaitRate 联读。
5. 任何一臂的结论必须在**唯一变量**下取得（warm 配 warm、同 iter、同 seed 集、**同 lr 调优待遇**）。
6. **差值必须报合并 CI**；落在 CI 内就写"打平"，不得写"反超"。
7. **不得用测试集选配置**：即使某个未被 dev 选中的配置测试分更高，也不能改报它。

---

## 9. 归档（v2 / v3 原文，**已不是现行方案**）

保留目的：追溯设计决策的由来。**其中的 DQfD 损失、在线阶段三、reward shaping、
Dueling Q 头、双 GPU 布局均已废弃**，不要据此写代码。

### 9.1 v3 战略转向（2026-06-15）—— 方向性判断仍然有效

**触发**：同实验室（CogAI/AIRI）的 `When-to-Switch` / `MAPF-GPT`（~10 亿 (obs,action) 离线 IL）/
`LC-MAPF`（MAPF-GPT + 多轮通信）已占据本赛道；本项目 `meta_iter_20000` 实测 ≈ 经典 MARL 档。
⇒ **用 CNN+GRU + ~10× 更少数据、在人家造的 benchmark 上比"泛化/刷 SR"，赢不了。**

**因此（仍然有效的判断）**：
- 头条押**通信**（软稀疏 + 迟滞的图时间稳定性 / 通信效率 / 受限带宽鲁棒性），不比"泛化"。
- **换骨干是入场券，不是贡献。**
- 训练范式转**离线 IL**，去掉在线 RL 机器。
- Reptile 保留但重定位为"对新环境的快速适应"，且**必须保留一个适应评估**。
- 阶段三在线微调 → 离线 few-shot 适应评估，对位 MAPF-GPT active fine-tuning。

**v3 当时的"待定"现已定**：
- Transformer vs Mamba → **都不是主线**，主模型用 GRU；Mamba 的增益已查明是前馈的（§3.3）。
- 是否保留 DQfD 价值项 → **未保留**，纯 IL。
- 通信贡献的窄口 → **通信必要性 + 迟滞的图稳定性 + 带宽 Pareto**（见 §7 SQ3 边界）。

### 9.2 v2 已废弃部分索引

| v2 章节 | 内容 | v5 状态 |
|---|---|---|
| §1.2 `train_reptile_dqfd.yaml` / `finetune_online.yaml` | DQfD 与在线微调配方 | 文件保留，**不用** |
| §2.7 Dueling Q Head | 5 维 Q 值 + 非法位 -inf | **换成动作 logits + CE**（-inf 掩码保留） |
| §3.3.2~3.3.5 | 1-step TD / masked mellowmax target / n-step / margin | **全部废弃** |
| §3.3.1 | 专家/自游走课程式混合 | 废弃（无自游走） |
| §3.6 / §3.7 / §3.8 | 阶段三在线微调 / LaCAM 实时介入 / 损失退火 | **废弃** |
| §4.4 / §4.5 | 双 GPU task-并行 Reptile / 2×5090 + 47 核布局 | **废弃**（单卡 4080S 32G + 16 vCPU） |
| §5.5 | Q 值诊断整类 | **废弃** |
| §6.4 | "Q 值升高但 SR 不升"的定位流程 | 废弃 |
| §7 | v2 的量化预期 | 已被实测取代，见 `SCHEDULE.md` |

> v2 正文全文见 git 历史：`git show 04faaaa:Lacam_DQfD_Reptile_MAPF_Plan_v2.md`。

---

## 10. 仍然开放的方向（截至 2026-09-25）

| 方向 | 成本 | 说明 |
|---|---|---|
| **训练 seed 方差标定** | ~3.8h × 2 | 性价比最高。所有臂单 seed，而效应量 0.05~0.40，审稿最易问 |
| 架构 A/B 每臂各扫 lr | ~30~45h | `smamba` / `dconv1` / `gprenorm` 最优 lr 未测 ⇒ 对 GRU 的优势量级与 lr 不可分离。纯严谨性，**不抬主表** |
| 通信 2 轮从零训 | ~23h | 唯一只做了筛选、未做完整实验的轴 |
| SQ2 在 7.45M 规模复现 | ~105h | 已决定**接受局限**并在论文写明（等步数对 Reptile 不公平，需两臂各训 40k 步） |
| ~~SQ3 512 图重跑~~ | — | **已判定不做**：功效计算显示迟滞需 968× 样本（§7.1 第 7 条） |
| ~~makespan 进目标~~ | — | **已降级**：前提被自家数据削弱（§5.2 末） |
