# SCHEDULE.md — 当前工作计划（v4：追平 SOTA baseline）

---
## ★★★ 服务器迁移交接（2026-07-23，换服务器前必读）★★★

### 一句话现状
`il_joint`（MAPFCommNet，2.16M，5ch cost2go + full-flatten 编码器，joint 训练，`nothing` 协议）
在 POGEMA 标准 MAPF benchmark 上，**与 DCC(2021)~SCRIMP(2023) 同档，低于 MAPF-GPT-2M(AAAI'25)**。
**头条贡献 = 通信消融**（通信在拥堵区带来 +0.2~0.5 CSR）。**不是 SOTA，是"诚实落后 foundation model + 通信机制有清晰价值"的机制论文。**

### 最终结果表（CSR，nothing，validation-*，20图；il_joint=il_smallwarm@20000）
| RANDOM @agents | il_joint(warm) | comm-OFF | MAPF-GPT-2M | SCRIMP | DCC |
|---|---|---|---|---|---|
| 32 | 0.45 | 0.15 | 0.99 | 0.96 | 0.73 |
| 48 | 0.35 | 0.00 | 0.95 | 0.84 | 0.30 |
| 64 | 0.20 | 0.00 | 0.84 | 0.70 | 0.06 |
| MAZES @agents | il_joint(warm) | comm-OFF | MAPF-GPT-2M | SCRIMP | DCC |
| 32 | 0.45 | 0.05 | 0.80 | 0.55 | 0.36 |
| 48 | 0.20 | 0.00 | 0.66 | 0.28 | 0.12 |
| 64 | 0.10 | 0.00 | 0.34 | 0.15 | 0.02 |

### 环境（新服务器必须重建，3 个 conda 环境）
- **`gat`**：本仓库训练/评测（il_*）。pogema 2.0.0a、numpy 2.x、torch 2.11+cu130、entmax(本地函数非包)。
- **`marlgpt`**：跑 marl-gpt/MAPF-GPT/pogema-benchmark 系。**pogema==1.4.0**、numpy 1.26、torch 2.11+cu130、pogema-toolbox、cppimport、pybind11、huggingface_hub、socksio。
- **`lacam310`**：**只为生成专家数据**（python 3.10）。numpy、pyyaml、pogema 2.0.0a、pogema-toolbox。
- **★ 致命坑**：`lacam/_lacam.so` 是 **py3.10 编译**的原生 LaCAM 绑定。py3.11 下 `import _lacam` 失败 → `backend=auto` **静默回退到差得多的 prioritized 规划器**（wait 2%→42%、SoC 2×），污染数据且不报错。**所有 LaCAM 数据生成必须用 lacam310 环境**。已在 `lacam_runner.py` 加了大声告警。新服务器若 py 版本变，需重编译 `_lacam.so`（源在 `lacam/lacam_bind.cpp` + CMakeLists_pybind.txt，依赖 /tmp/lacam 的 liblacam.a——已丢，可能要重建 LaCAM C++）。

### 数据（在 `/data/lncosinx/chameleon_data/`，~180GB，换服务器多半带不走→需重生成）
- `c2g/`：when-to-switch 64×64，96体，finish 协议，5ch cost2go（旧）。
- `c2g_nothing/`：when-to-switch 64×64，96体，**nothing** 协议，5ch。
- `c2g_small/`：**小图 20×20/21×21，48体，nothing，5ch** ← **il_small/il_smallwarm 用的**，最重要。
- 重生成命令（lacam310 环境）：`python scripts/generate_expert_data.py --families random_small,maze_small --num-episodes 3000 --num-agents 48 --num-workers 40 --cost2go --output-dir <dir> --config-tpl "configs/env_{fam}.yaml"`
- **旧 4ch 专家数据在 `data/`（root 所有，可能读不到；里面是 comm2/DQfD 的 finish 数据）**。

### 关键 checkpoint（在 `ckpt_il/`，本地盘，可能要一并迁）
- **`il_smallwarm_iter_20000.pt`** = 主模型（warm-start，comm ON，小图 nothing）。
- `il_noComm_iter_20000.pt` = 通信消融的 comm-OFF（**scratch，非warm→配对不完全干净**）。
- `il_nothing_iter_20000.pt` = warm-start 的初始化源（when-to-switch nothing）。
- `il_small_iter_5000.pt` = 从零版（干净 @5000 消融的 comm-ON 半边）。

### 复现/继续用的脚本（本仓库，随 git）
- `scripts/eval_marlgpt_il.py` — il 模型在 marl-gpt validation-* 图上评测（--on-target nothing --max-steps 128 对齐 leaderboard；输出 CSR/ISR）。
- `scripts/train_il_meta.py` — 训练（新增 `--init <ckpt>` 热启动、`--set-net K=V` 网络覆盖；日志区分 [joint]/[reptile]）。
- `configs/train_il_joint_small.yaml`（joint, reptile:false, families [random,maze]）+ `network_gru_c2g.yaml`（patch5, spatial_pool:0）。
- MAPF-GPT 基线：`MAPF-GPT/run_nothing_bench.py`（--families 01-random,02-mazes；权重 hf_weights/ 自动下载）。
- pogema-benchmark leaderboard 数字：`pogema-benchmark/raw_data_MAPF/{01-random,02-mazes}/*.json`（DCC/SCRIMP/MAMBA/LaCAM/QMIX/VDN/IQL/QPLEX，nothing 协议，直接引用）。

### 血泪教训（都踩过，别重踩）
1. **别在训练早期(≤25%)下结论**——今天在 iter1000/5000 下过多次相反结论，全被 20000 打脸。**只在收敛后判断。**
2. **loss ≠ rollout**：训练 loss 早平（~2000）后一直震荡 ~0.06，但 rollout CSR 一路涨到 20000（mazes@32: 0.20→0.45）。**判性能只看 rollout，不看 loss。且 20000 时 rollout 还在爬，值得试 30k-50k。**
3. **协议 = `nothing`**（标准 MAPF，到达守位）：这是 leaderboard/MAPF-GPT/SCRIMP/DCC 的协议。`finish`(到达消失) 训的模型在 `nothing` 下崩(不守位→CSR≈0)。`nothing` 数据生成自动捕获守位(valid=1、wait~50%)，`il_wait_weight=1.0` 不要降权。
4. **评测公平性**：validation-*(20×21) 是 MAPF-GPT 的训练分布、il_joint 的外分布；差距主因是**密度/拥堵**(64体/20×21 vs il_joint 训练的 64体/64×64)，不是地图大小(FOV固定11×11)。→ 用小图高密度(c2g_small)训练才公平。
5. **`pkill -f 脚本名` 杀不掉 spawn 子进程** → 用 `setsid` 独立进程组，`kill -9 -- -PGID`。
6. **数据生成去压缩**(`np.savez` 非 savez_compressed)：合并 13min→1min，加载也快。写 `/data` 大盘不占主盘。
7. **`checkpoints/`、`data/`、`logs/` 是 root 所有不可写** → 用 `ckpt_il/`、`/data/...`、scratchpad。

### 下一步（新服务器上按优先级）
1. **补干净通信消融**：`il_noComm_warm`（`--init il_nothing_20000 --set-net comm_rounds=0`）配 `il_smallwarm`，warn 配 warm。
2. **机制消融** + EdgeFlipRate：entmax(1.5) vs softmax(1.0)、迟滞 tau on/off。证明"稀疏+迟滞"这个具体设计的价值（`--set-net entmax_alpha=1.0` / `--set-net tau_in=0 --set-net tau_out=0`）。
3. **全 128 图**降方差（--num-maps-per-family 0）。
4. **试更长训练**(30k-50k)，rollout 20000 时还在涨。
5. 定投稿会议（RA-L/IROS/AAMAS 现实；TNNLS 需机制做到很完整）。

---


> **背景（2026-07-22 诊断结论）**
> comm2（0.98M，CNN+GRU+GAT+DQfD）在**同一套 marl-gpt 地图 + 同起终点 + 同 horizon** 上，
> 被 **EPH(2.9M, IROS'24)** 和 **MAPF-GPT(86M, AAAI'25)** 全面压制（SR 与通信节俭度双输）。
> 详见 `logs/` 与 scratchpad 的同图对比。根因按影响排序：
> 1. **观测**：只喂穿墙的目标方向向量，**没有障碍感知的 cost2go**（EPH 有 heuri_map，MAPF-GPT 有 cost2go）。mazes 上 SR=0 的直接原因。
> 2. **架构**：`PatchEncoder` 的 `AdaptiveAvgPool2d((4,4))` 把 11×11 空间信息平均糊掉，对精确寻路是坏归纳偏置。
> 3. **训练**：
>    - "IL 不如 DQfD" 是**被污染的结论**——IL 损失无 wait 类别平衡，而专家数据 wait 占 **49–57%** → IL 塌缩到到处等待（eval WaitRate 0.51）。反证：MAPF-GPT 纯 CE IL 就是这里最强。
>    - 数据量鸿沟：~1.5M timesteps vs MAPF-GPT 1B（~600×）。
>    - Reptile 放预训练大概率拖累峰值 SR（优化的是适应性，非峰值）。
>
> **策略结论**：不追 SR 榜首（打不过 86M），但必须把 comm2 从"被 dominated"拉到"可辩护的
> Pareto 前沿"（同参数量/通信预算/观测假设下有竞争力），头条贡献仍是**通信机制**。

---

## 任务清单（按顺序，每步用同图台重测）

### T0 训练提速（先做，全程回本）← **进行中**
- **动机**：T1–T5 每步都要重训（现 ~8–9h/次）。GPU 利用率忽高忽低 = 输入管道 stall（已定位）。
- **做什么**：
  - [ ] **跳过 IL 用不到的 next_obs**：`buffers/expert_demo_buffer.py` 加轻量采样路径（或 `need_next=False`），IL 只 gather `obs/scalars/positions/actions/valid/masks/margin`，不拷 `next_obs_*/rewards/dones/next_masks`（省 ~一半 CPU 拷贝，148MB/步 的 next_obs_patches 是大头）。
  - [ ] **预取 + pinned memory**：`trainers/train_il_reptile.py` 后台线程预取下一 batch，`_to_gpu` 源用 pinned buffer 使 `non_blocking=True` 真异步，重叠 CPU 采样与 GPU 计算。
  - [ ] **不碰**通信逐-t 迟滞循环（正确性敏感）。
- **判决门**：nvidia-smi 利用率平稳、s/it 从 ~1.5 降到 ~0.7–0.9（目标 8–9h→4–5h）；SR/acc 与提速前一致（只改数据管道，不改数值）。
- **备注**：显存忽高忽低是通信 `(B,T,N,N)=O(T·N²)` 张量 churn，非 bug，暂不处理。

### T1 修 IL 的 wait 塌缩 + 公平复测 ← **代码已改，待 T0 后启动长训**
- **动机**：证明"IL<DQfD"是假结论（缺 wait 平衡所致），决定训练范式走向。
- **做什么**：
  - [x] 量化：专家数据 wait 占比 49–57%（moves 各 ~12%）。
  - [ ] `losses/il_losses.py` 加 `wait_weight`（对 expert==wait 样本降权，≈逆频率 0.23–0.3）。
  - [ ] 接入 `trainers/train_il_reptile.py` + `configs/train_il_reptile.yaml`。
  - [ ] 同 `MAPFCommNet` 骨干重训 IL（仅改 wait 平衡这一个变量），与旧 broken-IL 同 iter 对比。
  - [ ] 在 warehouse（原 IL/DQfD 对比用的 held-out 族）复测：看 WaitRate 是否从 0.51 掉下来、AR 是否回升。
- **判决门**：wait 平衡后 IL 的 AR 若追平/反超 DQfD(0.927)、WaitRate 显著下降 → "用 DQfD 不用 IL" 结论被推翻，转纯监督路线。

### T2 观测加 cost2go 通道 ← **已确认有效（iter5000 定论）**
> **✅ 定论（2026-07-22, iter5000, 64 agents, 50 eps, 唯一变量=cost2go）**
> | run | family | SR | AR | NoProgress |
> |---|---|---|---|---|
> | **cost2go** | mazes | **0.360** | **0.912** | 0.318 |
> | control | mazes | 0.020 | 0.747 | 0.518 |
> | **cost2go** | random | **0.400** | **0.885** | 0.308 |
> | control | random | 0.060 | 0.734 | 0.532 |
>
> SR 提升 18×/6.7×，AR +22%/+21%。**对比 comm2/DQfD(20000 步旧架构) mazes@64 AR=0.705 / SR=0.000
> → IL 在 25% 训练量下全面超越。**
> **Deadlock 指标陷阱**：cost2go 的 Deadlock(0.64) 高于 control(0.40)，但 control 的 WaitRate 高达 0.75–0.79
> （全员原地打转、AR 仅 0.73）。该二值指标把"多数到达+少数卡住"与"全员打转"混为一类，**不可单独解读**。
> **教训**：iter1000 的读数（当时判 cost2go 有害）是**过早结论**；本轮 3 次反转说明 **5% 训练进度不足以判决，至少要 25%**。
>
> ---
> **历史记录（已被上面推翻的中间结论）**：下方 "+21%/+17%" 是在**旧编码器(AdaptiveAvgPool2d(1)，空间信息被全局平均掉)**
> 下测得的。修好池化(T3，全展平)后重做同一 A/B，**cost2go 优势消失甚至变负**：
> | run | family | AR(旧池化) | AR(全展平) | Deadlock(全展平) |
> |---|---|---|---|---|
> | cost2go | mazes | 0.637 | 0.654 | **0.550** |
> | cost2go | random | 0.651 | **0.605↓** | **0.800** |
> | control | mazes | 0.525 | **0.701↑↑** | 0.200 |
> | control | random | 0.555 | 0.562 | 0.100 |
>
> **解读**：cost2go 先前的收益**大部分是在代偿编码器丢失的空间信息**；空间信息恢复后它不再有增益，
> 且**死锁率激增**(0.55/0.80 vs 0.10/0.20)——推测是所有 agent 对测地最优路径判断高度一致 → 挤同一走廊 → 对称僵持。
> 训练指标反向(acc 0.880 vs 0.855、loss 0.265 vs 0.349)⇒ **拟合更好但 rollout 更差**。
> **待办**：跑到 5000/20000、episodes 提到 50 复核；若死锁病理持续，需重新设计 cost2go 的用法（或放弃）。

> **结果（2026-07-22，64 agents / 20 eps / iter1000，唯一变量=cost2go 通道）**
> | run | family | AR | WaitRate | NoProgress |
> |---|---|---|---|---|
> | cost2go(5ch) | mazes | **0.637** | 0.262 | 0.613 |
> | control(4ch) | mazes | 0.525 | 0.351 | 0.681 |
> | cost2go(5ch) | random | **0.651** | 0.235 | 0.578 |
> | control(4ch) | random | 0.555 | 0.269 | 0.615 |
>
> **AR +21%(mazes) / +17%(random) 相对提升**；NoProgress、WaitRate 同向改善。
> **方法论**：训练 acc 几乎无差（0.842 vs 0.837）但 rollout 差 17–21% —— 专家状态上"猜下一步"
> 用方向向量就够，cost2go 只在**偏离专家轨迹后自主绕障恢复**时才显价值。**判断此类改动必须看 rollout，不能只看训练 acc。**
> SR 仍为 0（iter1000 太早）。待 5000/20000 里程碑复测。

- **动机**：mazes SR=0 的直接根因；所有强 baseline 都有障碍感知目标信号。
- **做什么**：
  - [ ] `envs/observation_builder.py` 加 FOV 内 cost2go 通道（BFS 到目标的测地距离/梯度，clip 到 value_limit）。复用 `envs/potential_field.py` 的 BFS。
  - [ ] 松动 **INVARIANT 3**：把"FOV 内 cost2go"从特权信号改判为合法局部特征，同步改 `CLAUDE.md`（与所有对比 baseline 一致，也是公平性必需）。
  - [ ] patch 通道数 4→5（或独立 cost2go 分支），改 `patch_channels` 配置。
- **判决门**：mazes 的 AR/SR 是否明显抬升。

### T3 修 CNN 池化 ← **已完成，是目前最大的单项收益**
- **发现**：`spatial_encoder.py`（IL 用的 MAPFCommNet）原为 `AdaptiveAvgPool2d(1)`，把 11×11 **全局平均成 1×1**。
  conv 平移等变 + 全局平均 ⇒ 输出对平移近似**不变**：模型分不清"障碍/邻居在左还是右"，
  FOV 空间布局被彻底丢弃，唯一方向信息只剩标量目标方向向量。（`encoders.py` 的 comm2 路径是 pool(4,4)，问题较轻但同类。）
- **已做**：改为**全展平**（`spatial_pool: 0` 可配置），参数 0.68M→**2.16M**（进入 EPH 2.9M 同量级）。
  结构验证：平移 patch 内容会改变输出（旧版构造上不变）。
- **结果（iter1000, 64 agents, 20 eps）**：对照臂 mazes AR **0.525 → 0.701 (+34%)**；
  **对照**：comm2/DQfD 训 20000 步的 mazes@64 AR = 0.705 —— **IL 只训 1000 步(5%) 就追平**。
- **含义**：**"IL 不如 DQfD" 主要是被这个池化坑的**，不是范式问题。这是对最初那个战略问题的实质回答。

### T3.5 D4 对称利用（先数据增强，等变卷积后置）
- **动机**：MAPF 网格有 D4 对称（4 旋转 × 镜像），`up/down/left/right` 在旋转下循环、`wait` 不变。
  正确的对称先验能提样本效率+泛化——**正好补数据量鸿沟（T5）**。
- **先做（便宜、高 ROI）**：
  - [ ] **D4 数据增强**：训练时对 patch 做 8 种旋转/镜像，**同步置换动作标签**（按 INVARIANT 1 的旋转/镜像置换）、**旋转标量里的目标方向向量(2D)**、置换 prev-action one-hot；`arrived` 不变。等价 8× 有效数据，零依赖、几乎无速度成本。
  - **约束**：动作置换必须与 INVARIANT 1 精确对齐（旋转 90°：up→right→down→left→up；镜像另算），错一个整批标签错位且不报错。写单测校验置换表。
- **后置（可选新颖点）**：
  - [ ] **全等变卷积（escnn/e2cnn，C4/D4 steerable）**：观测 + 动作头 + 通信(相对位置)全等变。**escnn 更慢 + 加依赖，与 T0 提速冲突**，只在基础修复跑通、且要把等变性当论文卖点时再上。
- **判决门**：D4 增强后 held-out 泛化 SR/AR 是否提升；等变卷积仅在增强收益不足且想要架构新颖点时评估。

### T4 去掉预训练里的 Reptile ← **已完成：joint 完胜，预训练去掉 Reptile**
> **✅ 定论（2026-07-22, iter20000, 64 agents, 50 eps, held-out, 唯一变量=`reptile` 开关）**
> | arm | mazes SR/AR | random SR/AR | Deadlock(m/r) |
> |---|---|---|---|
> | Reptile | 0.680 / 0.955 | 0.620 / 0.927 | 0.28 / 0.38 |
> | **joint** | **0.880 / 0.991** | **0.840 / 0.981** | 0.12 / 0.16 |
>
> joint 每项都赢（SR +0.20/+0.22，死锁减半），训练指标同向（acc 0.925 vs 0.872）。
> **预训练改用 joint**（`configs/train_il_joint.yaml`，`reptile: false`）。
> **注意**：这只证明 Reptile 拖累**峰值 SR**，未否定其 **few-shot 适应**价值 → Reptile 留到 T7 单独验证，不进预训练。

- **完整链路（held-out, mazes@64）**：comm2/DQfD(旧架构) SR0/AR0.705 → +修池化+cost2go+去Reptile → **SR0.88/AR0.991**。
  "IL 不如 DQfD" 被彻底推翻，是旧池化+错误 wait 降权造成的假象。

### T5 扩数据 / 重训
- **做什么**：评估是否需要生成更多 LaCAM 专家数据（当前 5 族 ~1.5M steps）；或直接复用 MAPF-GPT 公开数据集（POGEMA 同分布）。

### T6 baseline 对比基础设施
- **状态**：
  - [x] 同图台：`scripts/convert_pogema_to_eph_instances.py` + `EPH/eval_instances.py` + `scripts/eval_comm2_commcost.py`（EPH/comm2 逐张同实例）。
  - [x] MAPF-GPT(86M) 已在 marl-gpt 地图上跑通（`marl-gpt/run_light_bench.py`）。
  - [x] EPH 已跑通（纯策略 + 通信量）。
  - [ ] MAPF-GPT-2M（尺度对齐点，HF `aandreychuk/MAPF-GPT`）。
  - [ ] SRMT（`SRMT/`，2025 通信同侪，需下权重）。
  - [ ] SCRIMP：直接引用 `pogema-benchmark/raw_data_MAPF` 的 leaderboard 数字。
  - [ ] CS-PIBT / SACHA（用户已克隆）跑通。
- **通信效率 Pareto**：comm2 真实度数（topk 后 `avg_active_neighbors`）vs EPH comm 事件 vs EdgeFlipRate。

### T6.5 同图台重评估 il_joint ← **已完成：从被压制到追平/反超**
> **✅ 结果（2026-07-22, marl-gpt validation-* 地图, on_target=finish, 20图/族, SR）**
> | agents | comm2(0.98M,旧) | **il_joint(2.16M,新)** | EPH(2.9M) | MAPF-GPT(86M) |
> |---|---|---|---|---|
> | random 16 | 0.50 | **1.00** | 1.00 | 0.90 |
> | random 64 | 0.00 | **0.50** | 0.30 | 0.50 |
> | mazes 8 | 0.40 | **1.00** | 0.85 | 0.75 |
> | mazes 32 | 0.15 | **0.80** | 0.35 | 0.35 |
> | mazes 64 | 0.00 | **0.30** | 0.00 | 0.15 |
>
> il_joint 在 random 全程贴 EPH/MAPF-GPT，**mazes 每一档最高**。参数是 MAPF-GPT 的 1/40、EPH 同量级。
> **⚠️ 协议 confound**：三方跑三套 env（il_joint=我们 POGEMA soft-collision；EPH=自己的 DHC env；
> MAPF-GPT=自己的管线）。同图但动态学不同 → SR 非严格同一实现。**必须先做 T6.6 消除。**
> 脚本：`scripts/eval_marlgpt_il.py`。

### T6.6 消除 env confound（同一环境跑所有方法）← **当前任务 (b)**
- **动机**：T6.5 的对比同图但**不同 env 动态**，审稿人会打回。金标准=所有方法在**同一个 env** 评测。
- **统一协议（对齐 pogema-benchmark leaderboard，MAPF）**：POGEMA soft-collision，`on_target: nothing`
  （标准 MAPF：到达即守位、变静态障碍），`max_episode_steps: 128`，validation-* 全 128 图/族，agent [8,16,24,32,48,64]，
  指标 **CSR/ISR**（终局口径）。il_joint 须在 `nothing` 下**重训**（`finish` 训的模型到达后不守位 → CSR≈0，实测 c 已证）。

---

## ★ 对比算法阵容（BASELINE ROSTER）

> 全部在**同一 POGEMA env / 同协议（nothing, 128步, validation 图）/ 同指标（CSR/ISR）**下比较。
> **来源**三类：`leaderboard`=pogema-benchmark/raw_data_MAPF 已算好直接引用；`adapter`=pogema-benchmark/algorithms 有适配器可跑；`cloned`=已克隆需自己跑。

**A. Planner 上界（集中式满信息，非 PO，不作"对手"只作天花板）**
| 方法 | 年份 | 来源 | 状态 |
|---|---|---|---|
| LaCAM | 2023 | leaderboard | ✅ 引用（random/mazes 全档≈1.0） |
| RHCR | — | adapter | 可选 |

**B. 2025 学习方法（技术审查硬要求，必须有）**
| 方法 | 年份/venue | 参数 | 来源 | 状态 |
|---|---|---|---|---|
| **MAPF-GPT** | AAAI'25 | 86M(+2M变体) | cloned(POGEMA-native) | 🔄 nothing 下跑中；待补 2M 尺度对齐点 |
| **MAPF-GPT-DDG** | IROS'25 | ~85M | cloned(benchmark.py, 权重 HF) | ⏳ 待下权重+跑 |
| **SRMT** | 2025(NeurIPS'24 workshop) | 小 | cloned(eval.py) | ⏳ 待查权重来源 |
| **CS-PIBT** | ICRA'25 | 轻 | cloned(ML-MAPF-with-Search, 非POGEMA框架) | ⏳ 独立框架，单独排期 |

**C. 学习式通信（il_joint 的"真同侪"，重点对标）**
| 方法 | 年份 | 参数 | 来源 | 状态 |
|---|---|---|---|---|
| **SCRIMP** | 2023 | ~1M | leaderboard | ✅ 引用（强上界：random64 CSR 0.70） |
| **DCC** | 2021 | ~1M | leaderboard | ✅ 引用（纯PO同类，il_joint 应明确赢：random64 CSR 0.06） |
| **EPH** | IROS'24 | 2.9M | cloned(自有env,已跑) | ⚠️ 需移植进 POGEMA 消 confound（≠DCC架构，要 EPH模型+DCC式obs适配器） |
| SACHA | 2023 | 小 | cloned | 可选 |

**D. 其它近期学习/混合**
| 方法 | 年份 | 来源 | 状态 |
|---|---|---|---|
| MAMBA | 2024 | leaderboard | ✅ 引用（高密度崩到0） |
| MATS-LP | 2024 | adapter | 可选（MCTS+学习混合） |
| Follower | AAAI'24 | adapter/cloned | 可选（规划+RL） |

**E. 经典 MARL（老基线，秒杀档，凑技术审查的历史线）**
| QMIX / VDN / IQL / QPLEX | 2018–21 | leaderboard | ✅ 引用（高密度全 0） |

**本方**：il_joint（MAPFCommNet，2.16M，5ch cost2go，full-flatten，joint，nothing重训后）。

**及格线（不是屠榜）**：① 明确压过纯PO同类 **DCC**；② 对强上界 **SCRIMP** 有竞争力（考虑参数/规模）；
③ 覆盖 ≥3 个 2025 方法过技术审查。**LaCAM 是天花板不比，MARL 老基线是历史线。**

**优先级**：MAPF-GPT(跑中) → MAPF-GPT-DDG(最低工作量) → SRMT(通信同侪,卖点相关) → EPH移植 → CS-PIBT(独立框架,最后)。

---

### T7 头条实验 + framing（重训/基线齐活后）
- 全 baseline 同 env 同协议出 CSR/ISR 主表（random/mazes，可加 warehouse/movingai）。
- 头条：通信消融（软稀疏+迟滞 vs dense vs 无通信）+ 通信稳定性（EdgeFlipRate）+ 通信量 Pareto。
- Reptile 卖点：held-out 族 few-shot 适应速度（vs MAPF-GPT active fine-tuning）——Reptile 已确认拖累峰值(T4)，只在此处体现价值。

---

## 已建工具索引
- `scripts/eval_marlgpt_benchmark.py` — comm2/任意 comm_rounds 在 marl-gpt 地图上评测（fixed_map）。
- `scripts/convert_pogema_to_eph_instances.py` — POGEMA 地图 → EPH 实例集（同起终点）。
- `EPH/eval_instances.py` — EPH 纯策略在外部实例集上评测（同 horizon）。
- `scripts/eval_comm2_commcost.py` — comm2 同图评测 + 通信开销（真实 topk 后度数/EdgeFlip/消息量）。
- `marl-gpt/run_light_bench.py` — 官方 MAPF-GPT/marl-gpt 管线轻量子集评测。
- 独立环境：conda `marlgpt`（pogema 1.4.0 + CUDA torch），跑 marl-gpt/EPH 系；`gat` 环境跑本仓库训练。

## 判决原则
让数据判决，不靠论证。每改一项 → 同图台重测 → SR/AR/通信量动没动，再决定下一步。
