# SCHEDULE.md — v5 实验总纲（四条科学问题主线 / MADCL）

> 本文件是**当前唯一的执行计划**，取代 v4 的 T0–T7 任务清单（v4 备份见 git 历史 `836de16:SCHEDULE.md`）。
> 组织原则来自实验路线图：**不围绕 baseline 组织实验，围绕四个科学问题（SQ）组织**，
> 每个 SQ 对应论文的一个贡献点，每张表/图只回答一个 SQ。
>
> 方法名统一为 **MADCL**（Meta-Adaptive Decentralized Communication Learning）：
> 局部观测（5ch cost2go patch + 8 标量）→ 残差 CNN → **FOV 内软稀疏(entmax)+迟滞通信** →
> GRU 时间核 → 动作 logits；离线 IL(LaCAM 专家 CE) + **DAgger 式在线纠错** + Reptile 元适应 +
> replay/meta-update 的持续学习。仓库里旧称 `il_joint` / `MAPFCommNet`，论文里一律叫 MADCL。

---

## 0. 硬件与路径（2026-09-09 换机后实测，最高操作优先级）

| 项 | 值 |
|---|---|
| GPU | **1× RTX 5090 D 32GB**（sm_120，需 torch cu128+；`base` 环境 torch 2.8.0+cu128 已验证可用） |
| CPU / RAM | **25 vCPU** / cgroup 上限 **90 GB**（`/sys/fs/cgroup/memory.max`，**不要信 `free -h` 的 754G**） |
| 系统盘 | `/` overlay **30G**（`checkpoints/ data/ logs/` 是 root 只读遗留目录，**别写**） |
| 数据盘 | `/root/autodl-tmp` **250G，已用 226G，剩 25G** ← **瓶颈，已向用户申请扩容** |
| conda | `base`(py3.12, torch2.8+cu128, pogema 2.0.0a) = 训练/评测；`lacam310`(py3.10) = **只生成专家数据** |
| 现有数据 | `c2g_small_15k/`(147G, 小图 20×20/21×21, 48体, nothing, 5ch, random+maze 各 15k ep ≈ 2.59M steps)；`c2g_nothing/`(79G, 64×64, 96体) |
| 现有主 ckpt | `ckpt_il/il_smallwarm100k_iter_100000.pt`（2.16M 参数，warm-start，joint，100k iter） |

**★ 三条不可忘的坑**（v4 血泪，全部保留）：
1. LaCAM `_lacam.so` 是 py3.10 编译 → **专家数据生成必须 `conda activate lacam310`**，否则静默回退 prioritized 规划器污染数据。
2. 长任务一律 `setsid nohup ... > log 2>&1 &`，收尾 `kill -9 -- -PGID`（`pkill -f` 杀不掉 spawn 子进程）。
3. **判性能只看 rollout CSR，不看 train loss**；**≤25% 训练进度不下结论**。

---

## 1. 现状基线（我们自己的起点，2026-08-28 实测）

`il_smallwarm100k_iter_100000.pt`，marl-gpt validation 图 20 张/族，`on_target=nothing`：

| family | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|
| random CSR | 0.95 | 0.65 | 0.60 | 0.25 | 0.10 | 0.00 |
| mazes CSR | 0.85 | 0.70 | 0.45 | 0.35 | 0.15 | 0.05 |
| random Deadlock | 0.05 | 0.25 | 0.25 | **0.60** | **0.70** | **0.85** |

**诊断**：ISR 在高密度仍有 0.65–0.79（多数体到达），CSR 却塌到 0 ——
**瓶颈是"少数体死锁"而非"不会走路"**。这决定了 v5 的改进方向（见 §2）：
不是加参数量，而是**治 BC 的复合误差与对称僵持**。

---

## 2. 目标与诚实边界（先把话说清楚）

路线图 Q1 的 baseline 里混了三类不同性质的方法，**不能用同一句"要赢"来概括**：

| 类别 | 方法 | 我们的目标 | 理由 |
|---|---|---|---|
| **A. 集中式满信息规划器（天花板行，不作对手）** | CBS、LaCAM | **不比**，作为 upper bound 报出 | 它们有全局地图+全体起终点+联合搜索；PO 去中心化方法赢不了是**定义上的**，MAPF-GPT/SCRIMP 论文同样把它们列为 ceiling |
| **B. 去中心化 PO 学习方法（真对手，必须全赢）** | PRIMAL、PRIMAL2、SCRIMP、DCC、EPH、SACHA、ALPHA、经典 MARL(QMIX/VDN/IQL/QPLEX) | **每一档 agent 数、每族地图都要 ≥** | 同观测假设、同去中心化约束、同量级参数 → 公平可比，这是论文的主张 |
| **C. Foundation-model 尺度方法** | MAPF-GPT-2M / -6M / -85M、MAPF-GPT-DDG(85M) | **在 2M 尺度对齐点上赢 MAPF-GPT-2M**；对 85M/-DDG 报**参数-性能 Pareto**（我们 2.16M vs 它 85M，≈40× 参数差） | 85M+1B steps+DAgger 的绝对 CSR 不是 2M 模型靠调结构能追平的；诚实的卖点是 **同尺度赢 + 跨尺度 Pareto 不被 dominate** |

**判决线（论文能立住的最低要求）**：
- ✅ B 类**全档全赢**（尤其 SCRIMP：random@64 CSR 0.70、mazes@32 0.55 是硬指标）；
- ✅ 同尺度赢 MAPF-GPT-2M；
- ✅ 四个 SQ 各有一张自洽的表/图，消融把"通信"这个头条贡献钉死。

---

## 3. 方法 v5 改进清单（做什么才可能赢 SCRIMP）

现状 CSR 落后 SCRIMP 的根因排序（有实测支撑，非猜测）：
① **纯 BC 的复合误差**：偏离专家分布后无纠正信号 → 高密度死锁 0.6–0.85；
② **数据量**：2.59M steps vs MAPF-GPT 1B（~400×）；
③ **推断期无死锁逃逸**：贪心 argmax，对称僵持无法自破（SCRIMP 有 tie-breaking + priority，EPH 有 escape）；
④ 训练分布单一（只 random+maze 小图 48 体）。

对应五项改进（**M1–M3 是关键，M4–M5 是加分**）：

| ID | 改进 | 预期收益 | 成本 | 状态 |
|---|---|---|---|---|
| **M1** | **D4 对称数据增强**（8× 旋转/镜像，同步置换动作标签+旋转目标方向向量+置换 prev-action one-hot） | 等效 8× 数据，直击根因② | 半天实现 + 0 额外显存 | ⏳ |
| **M2** | **推断期死锁逃逸**：per-agent 进度监视器，连续 k 步无进展 → 从 masked softmax 按温度采样（而非 argmax），带随机优先级打破对称 | 直击根因③，高密度 CSR 预计最大单项收益 | 半天实现，纯推断期，所有臂公平共用 | ⏳ |
| **M3** | **DAgger 式在线纠错**（policy rollout 采状态 → `lacam310` 子进程用 LaCAM 重标注 → 混进 buffer 继续训） | 直击根因①，是 MAPF-GPT-DDG 相对 MAPF-GPT 的增益来源 | 2 天实现（跨 py 版本 worker）+ 每轮 ~6h 采集 | ⏳ |
| **M4** | **训练加速 + 扩尺度**：bf16/torch.compile/更大 batch，把 1.3 s/it 压到 ≤0.7；数据生成去掉 IL 用不到的 `next_obs`（存储直接减半 → 同盘装 2× 数据） | 让 M3 的多轮迭代在预算内跑完 | 1 天 | ⏳ |
| **M5** | 扩训练分布：加 warehouse 族专家数据（需磁盘扩容） | 治根因④，同时给 Q1 的 warehouse 列提供分布内训练 | 依赖磁盘 | ⏸ 待磁盘 |

**不做的事**（明确排除，避免浪费预算）：换 Transformer/Mamba 骨干（v3 遗留待定项，收益不确定且重训成本高）、等变卷积 escnn（依赖重、与提速冲突）、任何把全局信息喂进 policy 的"捷径"（违反 INVARIANT 2/3）。

---

## 4. 四条科学问题主线（论文骨架 = 实验骨架）

### 统一实验协议（所有 SQ、所有方法共用，写死不再改）
- 环境：POGEMA，`collision_system=soft`，`on_target=nothing`（标准 MAPF：到达即守位）。
- 地图：marl-gpt/POGEMA `validation-*`，**全 128 张/族**（降方差；调试期可 20 张，出表必须 128）。
- 族：`01-random`、`02-mazes`（主表），`03-warehouse` 作泛化/外分布列。
- agent 数：8 / 16 / 24 / 32 / 48 / 64；`max_episode_steps=128`（**两族统一 128**，v4 的 random=256 是不一致，已废止）。
- 指标：**CSR**（终局全员在目标）、**ISR**（终局个体到达率）、SoC/makespan（路径质量）、Deadlock、
  通信侧 **AvgActiveNeighbors / EdgeFlipRate / 消息字节数**。
- 种子：3 seed × 128 图，报均值（主表）+ 95% CI。

---

### **SQ1｜整体框架在标准 MAPF benchmark 上是否有竞争力？** → 贡献点：MADCL 整体

**产出**：主表 T1（random/mazes × 6 档 agent × CSR/ISR），+ 参数-性能 Pareto 图 F1。

| 对比对象 | 获取方式 | 状态 |
|---|---|---|
| LaCAM（天花板） | pogema-benchmark `raw_data_MAPF` 引用 / 本地 `lacam310` 现跑 | ⏳ |
| CBS（天花板，小规模） | pogema-toolbox 适配器或引用 POGEMA 论文 | ⏳ |
| PRIMAL / PRIMAL2 | POGEMA benchmark 已有数字（引用）；无则用 leaderboard 同协议数 | ⏳ |
| SCRIMP / DCC / MAMBA / QMIX/VDN/IQL/QPLEX | pogema-benchmark leaderboard 引用（同协议已算好） | ⏳ |
| **MAPF-GPT-2M / -6M / -85M** | HF `aandreychuk/MAPF-GPT` 权重，本地 `marl-gpt` 管线跑同协议 | ⏳ |
| **MAPF-GPT-DDG** | `MAPF-GPT-DDG/benchmark.py` + HF 权重 | ⏳ |
| ALPHA (2025) | 找官方权重；无则引用论文同协议数字并注明 | ⏳ |

**判决门**：MADCL 在 B 类全档 ≥，且 ≥ MAPF-GPT-2M。

---

### **SQ2｜元学习是否带来跨场景与快速适应能力？** → 贡献点：Reptile

**已知**（v4 实测，不重做）：Reptile 进预训练会**拖累峰值 CSR**（joint 完胜）。
→ 论文里 Reptile **不放预训练**，只放**适应阶段**，这正是路线图 Q2 的问法。

**产出**：图 F2（held-out 族 few-shot 适应曲线：x=适应样本量/梯度步，y=CSR）+ 表 T2。

| 臂 | 说明 |
|---|---|
| MADCL-Reptile-init → k 步适应 | 元训练在 {random, maze}，held-out = **warehouse**（外分布） |
| MADCL-joint-init → k 步适应 | 同初始化预算的普通 joint 预训练（消融"init 是否更易适应"） |
| MAPF-GPT active fine-tuning | 官方 `finetune_sf.py` 同预算对位 |
| PRIMAL2 / zero-shot 列 | 不适应的 zero-shot CSR 作参照 |

**判决门**：同适应预算下 Reptile-init 的 CSR 提升速度显著快于 joint-init（k ∈ {0, 10, 50, 200} 步）。

---

### **SQ3｜通信机制是否提升去中心化协作？** → 贡献点：FOV + 软稀疏注意力（**头条**）

**产出**：表 T3（消融）+ 图 F3（通信量-性能 Pareto）+ 图 F4（EdgeFlipRate vs 密度）。

**A. 内部机制消融**（唯一变量，全部 warm-start 自同一 init，训练预算相同）：

| 臂 | 开关 |
|---|---|
| **MADCL（完整）** | entmax(1.5) + 迟滞(τ_in .05/τ_out .02) + FOV 内 + topk cap 6 |
| 无通信 | `--set-net comm_rounds=0` |
| dense softmax | `--set-net entmax_alpha=1.0` |
| 无迟滞 | `--set-net tau_in=0 tau_out=0 stickiness_bonus=0` |
| hard top-k 选择 | 用 top-k 做**选择**（而非算力上限）→ 验证 INVARIANT 10 的"闪烁"论断 |
| 全局通信（无 FOV 限制） | `comm_radius=∞` → 验证"FOV 限制不是妥协而是有益归纳偏置" |

**B. 外部通信基线**（路线图 Q3 指定，全部在**我们的骨干**上替换通信模块，只换通信，保证唯一变量）：
CommNet（mean-pool 广播）、DIAL（离散可微消息）、ATOC（门控发起）、TarMAC（targeted 注意力）、
IC3Net（gating 决定通不通）、DGN（多头图卷积，全连接 k-近邻）。
→ 这是本项目**代码量最大的一块新实现**（6 个通信模块，但都可挂在同一 `BiIntentAttention` 接口上，
每个 50–120 行），必须做，因为它是头条贡献的正面证据。

**判决门**：完整版 ≥ 所有 6 个外部通信基线；且在**同通信量（AvgActiveNeighbors/消息字节）**下更优（Pareto 图）；
EdgeFlipRate 显著低于 dense/hard-topk 臂。

---

### **SQ4｜持续分布漂移下框架是否保持性能？** → 贡献点：Replay + Meta Update

**产出**：表 T4（任务序列上的 final CSR / 平均遗忘 / 后向迁移 BWT）+ 图 F5（遗忘曲线）。

**设置**：任务序列 = `random → mazes → warehouse → (movingai)`，每段训练固定预算，段末在**全部已见族**上测 CSR。

| 臂 | 说明 |
|---|---|
| **MADCL（replay + meta update）** | 经验回放缓冲跨族保留 + Reptile 式外层平均 |
| 无 replay（纯顺序微调） | 测灾难性遗忘幅度 |
| 有 replay 无 meta update | 拆开两个组件的贡献 |
| joint 上界（同时训全族） | multi-task 上界参照 |
| MAPF-GPT / PRIMAL2 顺序微调 | 外部对比（同任务序列，同预算） |

**判决门**：MADCL 的平均遗忘显著低于无 replay 臂，final CSR 接近 joint 上界，且优于外部方法顺序微调。

---

## 5. 排期（单卡串行，按依赖排序；每格给出预计墙钟）

> 训练现价：1.3 s/it × 100k = 36h。M4 提速后目标 ≤0.7 s/it → 50k iter ≈ 10h，可支撑多臂消融。

| 阶段 | 内容 | 预计 | 依赖 |
|---|---|---|---|
| **P0** | 评测协议统一（128 图、128 步、CSR/ISR/SoC/通信量）、复现现状数、训练提速 M4 | 1 天 | — |
| **P1** | M1(D4 增强) + M2(死锁逃逸) 实现 + 单测 + 小预算 A/B（20k iter，判是否有效） | 1 天 | P0 |
| **P2** | v5 主模型长训（50–100k iter，含 M1/M2） | 1.5 天 | P1 |
| **P3** | 基线落地：leaderboard 数字、MAPF-GPT-2M/6M/85M、MAPF-GPT-DDG、LaCAM | 与 P2 并行（CPU/间隙 GPU） | P0 |
| **P4** | M3 DAgger 纠错（实现 + 2–3 轮采集/重训） | 3 天 | P2 |
| **P5** | **SQ3** 全部消融 6 臂 + 6 个外部通信基线实现与训练 | 4 天 | P2 |
| **P6** | **SQ2** Reptile few-shot 适应实验 | 1 天 | P2 |
| **P7** | **SQ4** 持续学习序列实验 4 臂 | 2 天 | P2、M5 数据 |
| **P8** | 出全部表/图 + 结果汇总报告 | 1 天 | 全部 |

**总计 ≈ 13–15 天单卡墙钟**（不含磁盘扩容后重新生成数据的时间）。
优先级：**P0→P1→P2→P3 出 SQ1 主表 → P5 出头条 SQ3 → P4 提性能 → P6/P7**。
若中途预算吃紧，砍的顺序是：P7 → P6 → P4 的第 3 轮 → 外部通信基线里的 DIAL/ATOC。

---

## 6. 资源缺口（需用户决策）

1. **磁盘**：`/root/autodl-tmp` 只剩 25G。M3(DAgger 数据)、M5(warehouse 族)、去 `next_obs` 重生成数据都需要空间。
   **申请扩容到 1 TB**（最低 500 GB）。备选：删 `c2g_nothing/`(79G，64×64 旧数据，warm-start ckpt 已产出) 换 79G。
2. **单卡**：所有实验串行。SQ3 的 12 个训练臂是最大开销；若能加一卡，P5 直接对折。

---

## 6.5 进展与已定论（2026-09-10 更新，滚动维护）

### ★ 已定论 1：推断规则 = **纯策略采样**（不是贪心 argmax）
同一个 v4 checkpoint、同协议同实例、128 图，只换动作选取方式：

| 推断规则 | random 32/48/64 | mazes 32/48/64 |
|---|---|---|
| 贪心 argmax（原做法） | 0.30 / 0.10 / 0.05 | 0.15 / 0.00 / 0.05 |
| **按策略分布采样 τ=1.0** | **0.88 / 0.62 / 0.41** | **0.52 / 0.24 / 0.13** |
| 只在卡住时采样(escape,k=3) | 0.84 / 0.51 / 0.21 | 0.42 / 0.17 / 0.09 |
| 采样 + 到目标即守位(greedy) | 0.76 / 0.56 / 0.25 | 0.43 / 0.20 / 0.09 |

**机理**：共享参数 + 贪心 ⇒ 对称局面下所有体选同一动作 ⇒ 互锁且无法自破。
**escape 输给纯采样**说明关键不是"卡住后逃逸"，而是**持续的随机性让僵局根本形不成**。
到目标即守位反而更差 —— `nothing` 协议下守位体有时必须主动让路。
→ **所有臂一律用 `--action-mode sample --escape-temp 1.0`**（含所有基线消融，保证公平）。

### ★ 已定论 2：leaderboard 数字可直接引用（三条前提已逐一核对）
1. 地图逐字节相同（`pogema-benchmark/raw_data_MAPF` vs `marl-gpt/eval_configs`，两族各 128/128）；
2. 协议相同（nothing / 128 步 / soft / seed 0 / agents 8..64）；
3. **实例相同**：`eval_bench.py` 已改 `split="all"`，不再重映射 seed，起终点布局与 leaderboard 一致
   → 我们与 SCRIMP/DCC/LaCAM 是**逐实例配对**可比，不是"同图不同实例"。

### ★ 当前站位（v4 模型 + 采样推断，128 图，同实例）
- **已全档胜**：DCC、MAMBA、QMIX、VDN、IQL、QPLEX（random 与 mazes 六档全胜）。
- **仍落后**：SCRIMP（random 差距大：0.88/0.62/0.41 vs 0.96/0.84/0.70；mazes 差距小：
  0.52/0.24/0.13 vs 0.56/0.28/0.15）、MAPF-GPT-2M。
- 表由 `scripts/make_sq1_table.py` 自动生成（含"对每个对手是否全档胜"的判决行）。

### ★ 已就绪的实验基础设施
| 组件 | 位置 | 状态 |
|---|---|---|
| 统一评测台（CSR/ISR/SoC/通信量，批量 rollout，TTA） | `scripts/eval_bench.py` | ✅ |
| SQ1 合表（leaderboard 引用 + 判决） | `scripts/make_sq1_table.py` | ✅ |
| D4 对称增强（M1）+ 单测 | `envs/d4_augment.py`, `scripts/test_d4_augment.py` | ✅ 96 组合全过 |
| DAgger 纠错管线（M3，两阶段跨 py 版本） | `scripts/generate_dagger_data.py` | ✅ 端到端跑通 |
| SQ3 六个外部通信基线（同骨干移植）+ 公平性单测 | `models/comm_baselines.py`, `scripts/test_comm_baselines.py` | ✅ 7 臂全过 |
| SQ3 内部消融（无通信/dense/无迟滞/hard-topk/全局） | `models/communication.py` + `--set-net` | ✅ |
| SQ3 驱动脚本（12 臂串行训练+评测） | `/root/autodl-tmp/run_sq3.sh` | ✅ |
| SQ2 few-shot 适应驱动 | `scripts/run_sq2_adaptation.py` | ✅ |
| SQ4 持续学习驱动（4 臂 + forgetting/BWT） | `scripts/run_sq4_continual.py` | ✅ |

### ★ 进行中
- 专家数据扩充：`random@64` 已完成（1.93M 步 / 19,915 ep）；`maze@64` 生成中。
  32 体两族推迟（性价比低于 64 体，评测弱项在高密度）。
- v5 主训练双臂（D4 on/off，唯一变量）：数据齐即由 `/root/autodl-tmp/night_driver.sh` 自动拉起。

---

## 7. 判决原则（沿用 v4，是本项目最贵的教训）

1. 让数据判决，不靠论证；每改一项 → 同图台重测 → 看 rollout CSR。
2. **≤25% 训练进度不下结论**（v4 在 iter1000/5000 下过三次相反结论，全被 20000 打脸）。
3. **loss/acc ≠ rollout CSR**（拟合更好但 rollout 更差是常态，见 v4 cost2go 案例）。
4. Deadlock 二值指标不可单独解读，必须与 ISR/WaitRate 联读。
5. 任何一臂的结论必须在**唯一变量**下取得（warm 配 warm，同 iter，同 seed 集）。
