# 基于 LaCAM-DQfD-Reptile 的部分可观测多智能体路径规划方案说明（修订版 v2）

> **本版相对 v1 的主要改动（changelog）**
>
> 1. **奖励**：删除"动作方向对齐点积奖励"（非势函数、会偏置最优策略）。改用 **BFS 真实测地距离的势函数 shaping（PBRS）**，并给出"严格局部奖励"的退化备选。明确论证 reward shaping 使用全局信息 **不破坏部分可观测性**。
> 2. **观测**：按 Pogema 1.4.0 **实测**的原生观测格式重写。原生观测为 `(3, 11, 11)`：障碍物 / 智能体（自身恒在中心）/ 目标边界投影。v1 自创的"channel 4 目标投影"与 Pogema 原生 channel 2 **重复**，已删除。足迹通道与标量特征为 wrapper 追加。
> 3. **动作空间**：按 Pogema 实测顺序修正为 `0=wait, 1=up, 2=down, 3=left, 4=right`（wait 在 **第 0 位**，而非 v1 的最后一位）。掩码、专家动作转换、one-hot 全部对齐。
> 4. **Reptile + target 网络**：明确 target 在每个 inner loop 内 **冻结于 θ_meta**，每个 task 起始重置；阶段三微调采用周期性 target 同步。
> 5. **阶段三**：由"replay + 版本滞后 ≤2"改为 **低滞后同步式近 on-policy（lag ≤ 1）**，避免权重滞后导致的不收敛/慢收敛。明确阶段二（DQfD）必须 off-policy、阶段三转 near-on-policy 的分工。
> 6. **mellowmax**：删除 `alpha=0.1`（过小→近 hard max→易高估）。改为扫描 `α∈{0.5,1,2}`，按 `Q_max/Q_mean` 走势选取。
> 7. **通信**：约束 `r_c < obs_radius`，仅与视野内智能体通信；邻居选择改为 **软稀疏注意力(entmax)+迟滞**，`top-k` 降级为纯算力上限（更正早期"top-k 抑制雪崩"的措辞——hard top-k 会引入通信图闪烁）。规模由 FOV+稀疏控制，时间稳定性由迟滞/软化保证。
> 8. **循环回放**：新增 **R2D2 式 burn-in**，缓解隐状态陈旧。
> 9. **预训练混合比例**：由固定 70/30 改为 **课程式**（早期专家为主，后期逐步增加自游走）。
> 10. **大幅扩写第 4 章「训练速度优化」**：新增 **语言选型（为何绝大部分用 Python、Rust 非杠杆，编译顺序 numpy→numba→Rust）**、**元学习两层并行（CPU 多进程采样 + 双 GPU task-并行 Reptile）**、**目标硬件 2×RTX 5090 + 47 核的进程布局（阶段二/阶段三）**，以及 launch-受限的诊断分流；保留吞吐目标、瓶颈顺序与 GAT 降级备选。
> 11. **预期结果**：所有量化数字降级为 **待验证假设**；新增 **held-out 地图族** 以真正检验泛化。
> 12. 正文公式全部改写为可渲染 LaTeX（替代 v1 的图片占位符）。

---

## 0. 方案定位

本方案面向基于 **Pogema** 网格环境的部分可观测多智能体路径规划（MAPF）任务。环境部分可观测：每个智能体仅能使用自身局部视野（FOV）内的信息决策，**不能访问全局地图、全局最短路距离场、或全局规划器输出的未来轨迹**。

训练不从纯随机探索开始，而是：利用 **LaCAM** 生成专家轨迹，通过 **DQfD** 损失进行模仿学习；在四种地图分布上用 **Reptile** 元学习得到可快速适应的初始化参数 $\theta^\*$；最终在特定目标地图上以 **低滞后近 on-policy** 微调，降低离线回放导致的策略滞后。

网络采用**共享参数的去中心化 DQN**：每个智能体根据自身局部观测输出动作价值函数 $Q$。**不使用 QMIX，不构建中心化联合值函数 $Q_{tot}$。**

> **已知局限（如实声明）**：去中心化独立 Q 学习（IQL 式）在多智能体下存在 **非平稳性**——他人策略变化使每个智能体面对的环境非平稳，破坏单体 Q 学习的收敛保证。本方案以**双向意图通信**部分缓解，但不消除该问题。最接近的先行工作是 PRIMAL/PRIMAL2（IL+RL+共享参数去中心化），建议作为 baseline 对照。

> **目标规模**（与 v1 一致，沿用 Pogema 配置）：`map_size=64, num_agents=96, obs_radius=5, max_episode_steps=512, obstacle_density=0.3`。`obs_radius=5` ⟹ 局部 patch 边长 $K=2\cdot5+1=11$。

---

## 1. 文件命名与作用

### 1.1 总体目录结构

```
mapf_lacam_dqfd_reptile/
├── configs/
│   ├── env_random.yaml
│   ├── env_warehouse.yaml
│   ├── env_maze.yaml
│   ├── env_room.yaml
│   ├── train_reptile_dqfd.yaml
│   ├── finetune_online.yaml
│   └── network_biintent_drqn.yaml
│
├── envs/
│   ├── pogema_env_wrapper.py
│   ├── observation_builder.py     # 在 Pogema 原生 3 通道之上追加足迹通道与标量特征
│   ├── action_mask.py
│   ├── reward_builder.py          # BFS 势函数 shaping（训练时特权信号）
│   └── potential_field.py         # 训练时按地图预计算 BFS 测地距离场（每个 goal 一张）
│
├── lacam/
│   ├── lacam_runner.py
│   ├── lacam_dataset_builder.py
│   ├── trajectory_converter.py
│   └── hard_case_manager.py
│
├── models/
│   ├── encoders.py
│   ├── communication.py
│   ├── recurrent_core.py
│   ├── q_heads.py
│   └── biintent_drqn.py
│
├── losses/
│   ├── td_losses.py
│   ├── dqfd_losses.py
│   └── q_target.py
│
├── buffers/
│   ├── expert_demo_buffer.py
│   ├── selfplay_buffer.py         # 阶段二自游走数据
│   ├── recent_online_buffer.py    # 阶段三低滞后近 on-policy
│   ├── shared_rollout_buffer.py
│   └── batch_collator.py
│
├── trainers/
│   ├── train_reptile_dqfd.py
│   ├── finetune_online.py
│   ├── evaluator.py
│   └── scheduler.py
│
├── utils/
│   ├── metrics.py
│   ├── profiler.py
│   ├── checkpoint.py
│   ├── seed.py
│   └── logger.py
│
└── scripts/
    ├── generate_lacam_data.py
    ├── train_meta.py
    ├── finetune_target_map.py
    └── evaluate_policy.py
```

### 1.2 configs/

#### configs/env_random.yaml
```yaml
map_type: random
map_size: 64
num_agents: 96
obs_radius: 5          # ⟹ patch 边长 K = 11
max_episode_steps: 512
obstacle_density: 0.3
```

#### configs/env_warehouse.yaml / env_maze.yaml / env_room.yaml
分别定义 Pogema 的 warehouse（规则货架、窄通道、瓶颈）、maze（长通道、多转角，考验记忆与死锁恢复）、room（房间+门洞，考验门口协同）三类地图。**注意：pogema 自带 generator 只能生成 random 地图；warehouse/maze/room 需经 `pogema-toolbox` 加载/生成**（见安装说明）。结构参数沿用 pogema-toolbox 各地图的默认/标定值。

#### configs/train_reptile_dqfd.yaml
```yaml
meta_iterations: 20000
inner_steps: 5
inner_lr: 0.0003
meta_lr: 0.05
batch_size: 32
sequence_len: 32
burn_in_len: 8         # 新增：R2D2 式隐状态重建步数
gamma: 0.99
n_step: 5
lambda_td: 1.0
lambda_nstep: 1.0
lambda_margin: 1.0
lambda_l2: 0.00001
margin: 0.5
mellowmax_alpha: 1.0   # 修正：删除 0.1，先取 1.0，按 Q_max/Q_mean 在 {0.5,1,2} 中调
# 阶段二专家/自游走混合比例改为课程式（见 3.3.1）
expert_ratio_start: 0.9
expert_ratio_end: 0.5
expert_ratio_decay_iters: 8000
# 并行（见 §4.4 / §4.5），目标硬件 2×RTX 5090 + 47 核
parallel:
  task_parallel_gpus: [0, 1]   # 双卡 task-并行 Reptile：两卡各跑一个 task 的 inner loop
  num_env_workers: 40          # CPU 多进程 env/观测构建（spawn，不碰 CUDA）
  num_lacam_workers: 4         # C++ 绑定，按需生成/介入
  start_method: spawn
  shared_buffer_dtype: uint8   # patch 共享内存 zero-copy，禁用 pickle
  use_amp: true
  use_torch_compile: true      # + CUDA graphs，砍 kernel launch 开销
```

#### configs/finetune_online.yaml
```yaml
# 阶段三：低滞后同步式近 on-policy
rollout_segment_len: 32     # 每段同步采样长度
num_parallel_envs: 64
max_policy_version_lag: 1   # 修正：由 2 收紧到 1（同步段内权重冻结）
sync_after_each_segment: true
reuse_epochs_per_segment: 1 # 近 on-policy：每段数据基本只用 1 次
expert_prob_start: 0.5
expert_prob_end: 0.02
expert_decay_steps: 200000
lambda_margin_start: 1.0
lambda_margin_end: 0.0       # 末期专家项归零，退化为纯近 on-policy Q 学习
```

#### configs/network_biintent_drqn.yaml
```yaml
obs_radius: 5
patch_channels: 4      # Pogema 原生 3 通道 + 追加足迹通道 1
cnn_channels: [32, 64, 64]
scalar_feat_dim: 8     # 目标相对方向(2)+上一步动作 one-hot(5)+到达标志(1)
intent_dim: 64
comm_dim: 64
comm_radius: 4         # 修正：r_c < obs_radius(=5)，且仅与视野内邻居通信
comm_topk: 6           # 仅算力上限(封顶 GAT 计算量)；邻居选择用软稀疏+迟滞，非"抑制雪崩"
gru_hidden_dim: 256
dueling_hidden_dim: 256
num_actions: 5
num_attention_heads: 4
```

**动作空间（按 Pogema 1.4.0 实测，`GridConfig.MOVES`）**：

| action_id | 含义 | (drow, dcol) |
|---|---|---|
| 0 | wait | (0, 0) |
| 1 | up | (−1, 0) |
| 2 | down | (+1, 0) |
| 3 | left | (0, −1) |
| 4 | right | (0, +1) |

> ⚠️ 注意 wait 是 **action 0**，与 v1 文档相反。所有 one-hot、掩码、专家转换均以此为准。

### 1.3 envs/

#### envs/pogema_env_wrapper.py
封装 Pogema，统一训练 / 专家数据生成 / 评估接口。功能：创建 random/warehouse/maze/room 四类地图；设置 agent 数、尺寸、障碍密度、最大步数；返回每体局部观测；记录 episode 成功率、碰撞、停滞、路径长度；支持 policy rollout 与 LaCAM expert rollout。

环境建模为部分可观测多智能体决策过程 $\langle N, \mathcal{S}, \{\mathcal{A}_i\}, \{O_i\}, P, R, \gamma\rangle$，其中 $N$ 为智能体数，$o_i^t = O_i(s^t)$ 为第 $i$ 体在 $t$ 时刻的局部观测。

#### envs/observation_builder.py — **按 Pogema 实测重写**

**Pogema 1.4.0 原生观测**为每体一个 `(3, K, K)` float32 张量（实测确认）：

- **channel 0｜障碍物**：FOV 内障碍（含越界 padding 为 1）；
- **channel 1｜智能体**：FOV 内所有智能体，**自身恒在中心格** `[K//2, K//2]`，其余为可见邻居；
- **channel 2｜目标投影**：自身目标在 FOV 内则标在真实格；在 FOV 外则**沿射线投影到 FOV 边界格**（这正是 v1 自创"channel 4"想做的事——已由 Pogema 原生提供，故删除自创通道）。

本方案在原生 3 通道之上 **追加 1 个自定义通道**（仍是纯局部、不含全局/未来信息）：

- **channel 3｜自身历史足迹（时间衰减）**：当前位置置 1.0，历史足迹随时间衰减，落在 $[0, 0.8]$。

$$\text{channel3}(u) = \max_{\tau \le t}\; \mathbb{1}[\,p_i^\tau \text{ 落在以当前位置为中心的 patch 坐标 } u\,]\cdot \rho^{\,t-\tau},\quad \rho\in(0,1)$$

足迹通道只记录走过的位置，**不含未来路径、不含全局规划器信息**，因此不破坏部分可观测。最终 patch 为 `(4, 11, 11)`。

**追加的标量/向量特征**（均为局部或自身已知信息，经 MLP 编码）：

- **目标相对方向** $\mathbf{g}_i^t \in \mathbb{R}^2$：归一化的"指向自身目标"的方向向量。
  > 合法性说明：智能体**始终知道自己的目标坐标**（Pogema `get_targets_xy` 确认每体有自身 goal），这是它的任务定义，不是全局地图泄露。该向量只给直线几何朝向，不含绕障信息。
- **上一步动作** $a_i^{t-1}$：5 维 one-hot（或 embedding）。
- **到达标志** $\text{done}_i^t \in \{0,1\}$：是否已到达自身目标。

#### envs/action_mask.py
对智能体 $i$，合法动作集合 $\mathcal{A}_i^{\text{legal}}$ 排除：(1) 越界；(2) 进入静态障碍；(3) 环境明令禁止的动作。**动态碰撞不做硬掩码**（多体冲突需策略学习协调），改由 reward 惩罚并记为诊断指标。掩码实现：非法位置在 Q 上填 $-\infty$（工程上 `-float('inf')`，不用有限大负常数）。

#### envs/potential_field.py — **新增**
训练时为每个 `(map, goal)` 用 **BFS 在四连通可行图上**预计算真实测地距离场 $d_{\text{BFS}}(\cdot, \text{goal})$（不可达格设为大常数）。该距离场**只用于环境计算 reward shaping，不进入任何 agent 的观测**。每张地图每个 goal 只算一次，开销可忽略。

#### envs/reward_builder.py — **重写（核心修订）**

> **设计原则（回应"全局信息=破坏部分可观测"的疑虑）**：部分可观测性是 **policy 输入** 的属性，而 reward 由模拟器在训练时计算、**永不进入 policy 网络**；部署时根本不计算 reward。因此 reward 使用全局距离 **不改变 policy 的部分可观测性**。这是标准的"特权/非对称训练"。况且本方案已接受更强的特权信号（LaCAM 用全局图直接产出被模仿的动作），用全局距离做 reward 只是更弱的特权。

**主方案：BFS 测地距离的势函数 shaping（PBRS）**

$$\Phi_i(s) = -\,d_{\text{BFS}}\big(p_i,\ \text{goal}_i\big),\qquad F_i^t = \gamma\,\Phi_i(s^{t+1}) - \Phi_i(s^{t})$$

总奖励：

$$R_i^t = \underbrace{R_{\text{goal}}\mathbb{1}[\text{到达}]}_{\text{稀疏达标}} \;-\; \underbrace{c_{\text{step}}}_{\text{每步}} \;-\; \underbrace{c_{\text{coll}}\,\mathbb{1}[\text{碰撞}]}_{\text{碰撞}} \;-\; \underbrace{c_{\text{wait}}\,\mathbb{1}[\text{无意义等待}]}_{\text{无障碍却 wait}} \;+\; \underbrace{\lambda_F\,F_i^t}_{\text{BFS 势函数}}$$

- **为何用 BFS 而非曼哈顿**：曼哈顿忽略墙，其梯度"穿墙"指向，在 maze/room 中误导学习；BFS 测地距离的梯度沿真实最短路，绕墙时也单调下降。
- **为何用势函数形式**：由 Ng–Harada–Russell (1999)，$F=\gamma\Phi(s')-\Phi(s)$ 形式的 shaping **策略不变**——绕行造成的临时变化只影响学习动力学，**不改变最优策略**，迷宫亦然。（v1 的点积对齐奖励 **不是** 势函数，会把最优策略偏向直线运动，是 maze 失败的根因之一，已删除。）

**退化备选（若坚持 reward 也必须纯局部）**：去掉一切稠密方向 shaping，仅保留 `稀疏达标 − 每步 − 碰撞 − 无意义等待`，依靠 **n-step 回报 + DQfD 专家 margin** 传播价值。此路收敛更慢，但不需要任何全局量。**无论哪种，都不要用点积对齐奖励。**

### 1.4 lacam/

#### lacam/lacam_runner.py
对给定地图、起终点调用 LaCAM 生成多体路径。LaCAM 可用全局地图——它只作训练期专家，**不进入学生策略输入**。输出：
```json
{ "map_type": "random|warehouse|maze|room", "map_size": 64, "num_agents": 96,
  "starts": [[r,c],...], "goals": [[r,c],...], "success": true,
  "paths": [[[r,c],...],...], "makespan": 0, "sum_of_costs": 0, "runtime": 0.0 }
```

#### lacam/lacam_dataset_builder.py
成功样本 → DQfD 专家监督（含完整路径与动作序列）；失败样本 → hard case 池（保存地图、起终点、失败原因、运行信息），**不进入专家 margin loss**，避免学习失败动作。

#### lacam/trajectory_converter.py — **动作映射按 Pogema 修正**
设 $\Delta = p^{t+1} - p^{t} = (\Delta r, \Delta c)$，专家动作：

```
(0, 0)  -> 0  wait
(-1,0)  -> 1  up
(1, 0)  -> 2  down
(0,-1)  -> 3  left
(0, 1)  -> 4  right
```

> **关键**：观测 $o_i^t$ **必须由 observation_builder 按局部可观测规则重新生成**，不能直接用 LaCAM 的全局状态。

> **专家可解释性过滤（针对中心化→去中心化的时序信息丢失）**：LaCAM 是全局时间步同步的——某体 wait 是因为"它知道"另一体将在精确时刻让开。去中心化策略无此上下文，盲目模仿会退化为死锁。因此：**若同一 `(局部patch + 足迹 + 目标方向)` 下专家动作不唯一（典型如长廊对称让行），将该样本剔除出 margin loss**（否则 margin 永远拉不开，对应 §5.1 的失败模式），其 TD 项仍可保留。

#### lacam/hard_case_manager.py
hard case 来源：LaCAM 失败、策略 rollout 失败、高碰撞、高停滞、长时间未达标、warehouse 拥堵、room 门口阻塞、maze 长廊往返。对困难任务提高在环境回放中被重采样的权重。

### 1.5 models/

#### models/encoders.py
- **PatchEncoder**：CNN 对 `(4,11,11)` patch 快速提特征。
- **GoalEncoder / ActionEncoder / StatusEncoder**：对目标相对方向、上一步动作、到达标志编码（合为 `scalar_feat_dim`）。

#### models/communication.py — **FOV 约束 + 防闪烁的稀疏邻居选择**

> **更正(相对 v1/早期措辞)**:hard top-k 截断**会引入通信图闪烁(flickering)**——当第 k 与第 k+1 名邻居得分接近时,任一体微动一步就使排名交换,边在相邻时间步反复通断,导致 (a) 聚合特征 $\mathbf{c}_i$ 跳变、GRU 输入不连续,(b) hard 选择不可导、边界梯度来回翻、训练方差大,(c) 让行决策走/停振荡。因此 **top-k 只压住了通信图的*规模*(雪崩),却用*时间稳定性*换规模(闪烁)**——这是两个不同的病,不能用 top-k 同时声称解决。下面把邻居选择改为**软稀疏 + 迟滞**,把 top-k 降级为纯粹的算力上限。
- **IntentEncoder**：生成自身意图向量 $\mathbf{z}_i\in\mathbb{R}^{64}$。
- **NeighborSelector**：候选邻居须**同时**满足 (a) 距离 $\le r_c$ 且 $r_c < \text{obs\_radius}$，(b) **在自身 FOV 内可见**。在候选集上**不做 hard top-k**，改为：
  - **主方案——软稀疏注意力**：对所有候选邻居用带温度的**稀疏 softmax(entmax/sparsemax)** 算权重,让不相关邻居权重被自然压到 0。稀疏性由可导的权重衰减实现,**不闪烁**;
  - **迟滞(hysteresis)**:维持/纳入邻居用阈值 $\tau_{in}$、踢出用更宽松的 $\tau_{out}<\tau_{in}$,已在集合内的邻居要明显变差才被踢,避免边界反复横跳;
  - **粘滞项(可选)**:对"上一步已在集合内"的邻居给一个小加分,制造切换惯性;
  - **`comm_topk=6` 仅作算力上限**:当软稀疏后非零邻居仍超过 k 时,取权重最高的 k 个以封顶 GAT 计算量($O(Nk)$),**不再把它当作"抑制雪崩的机制"**。
- **BiIntentAttention**：多头注意力聚合选中邻居意图,得 $\mathbf{c}_i$。配合上面的软稀疏 + 迟滞,通信图在时间上平滑;必要时对 $\mathbf{c}_i$ 跨步做轻度 EMA 平滑进一步抑制高频抖动。

> 通信只在 FOV 内、软稀疏选择(算力上限 k),从机制上限制级联*规模*,同时用迟滞/软化保证时间*稳定性*。"和谁通信"(选择层)与"通信内容/如何聚合"(意图+注意力)是两个不同层面——见 §7 的消融阶梯(H5)验证后者是否值得其复杂度。

#### models/recurrent_core.py
DRQN 循环记忆，建议 GRU：$h_i^t = \text{GRU}(x_i^t, h_i^{t-1})$，$x_i^t$ 为融合局部观测+通信后的特征。对 maze/room 尤为重要。

#### models/q_heads.py
Dueling 头：$Q(s,a)=V(s)+\big(A(s,a)-\frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a')\big)$。

#### models/biintent_drqn.py
组合完整网络，命名 **BiIntent-GAT-DRQN**。

### 1.6 losses/
- **q_target.py**：masked mellowmax target（数值安全切片，见 §3.3.3）。
- **td_losses.py**：1-step 与 n-step TD loss。
- **dqfd_losses.py**：DQfD 总损失 $L = \lambda_{td}L_{TD} + \lambda_{n}L_{n} + \lambda_{m}L_{margin} + \lambda_{l2}\lVert\theta\rVert^2$。

### 1.7 buffers/
- **expert_demo_buffer.py**：存 LaCAM 成功专家轨迹。每条至少含 `obs, hidden_init, action_expert, reward, next_obs, done, action_mask, map_type, task_id`。
- **selfplay_buffer.py**：阶段二自游走探索数据（含噪声）。
- **recent_online_buffer.py**：阶段三短窗口近 on-policy 数据，记 `policy_version, sample_time, task_id, is_expert_action`；仅当 $v_{\text{learner}} - v_{\text{sample}} \le \text{lag}_{\max}(=1)$ 时可训练，否则丢弃。
- **shared_rollout_buffer.py**：连续内存、避免 pickle、支持切片与 pinned/异步拷贝。结构化数组：
  ```
  obs_buffer:     [E, T, N, C, K, K]   (K=11, C=4, uint8/float16)
  action_buffer:  [E, T, N]
  reward_buffer:  [E, T, N]
  done_buffer:    [E, T]
  mask_buffer:    [E, T, N, A]
  version_buffer: [E]
  ```
- **batch_collator.py**：构造 batch，保证 padding/对齐一致，支持 `batch.to(device, non_blocking=True)`。

### 1.8 trainers/
- **train_reptile_dqfd.py**：DQfD + Reptile 联合训练（见 §3.3–3.4）；按 §4.4–4.5 实现 **双 GPU task-并行 Reptile**（两卡各跑一个 task 的 inner loop，meta-iter 末平均增量做 outer 更新）+ **CPU 多进程 env-worker** 采样。
- **finetune_online.py**：目标地图低滞后近 on-policy 微调（见 §3.6–3.8）。
- **evaluator.py**：评估时**必须关闭专家介入**，否则不反映策略本身能力。
- **scheduler.py**：管理课程与地图采样概率（见 §3.5）。

### 1.9 utils/
metrics（足够细以定位问题来源）、profiler（env/forward/batch/IPC/序列化/拷贝/backward/optimizer/LaCAM 各段耗时）、checkpoint（meta init / pretrained DQfD / finetuned / best 分别保存）、seed、logger。

---

## 2. 网络架构

### 2.1 问题建模
每体共享参数网络 $f_\theta$，依局部观测 $o_i^t$、隐状态 $h_i^{t-1}$、邻居通信 $\mathbf{c}_i^t$ 选动作：

$$a_i^t = \arg\max_{a\in\mathcal{A}_i^{\text{legal}}} Q_\theta\big(o_i^t, h_i^{t-1}, \mathbf{c}_i^t,\ a\big),\qquad \theta \text{ 全体共享}$$

输入不同 ⟹ 输出不同。

### 2.2 输入表示
每体输入由四部分组成：`patch (4,11,11)` + `目标相对方向 (2)` + `上一步动作 one-hot (5)` + `到达标志 (1)`，后三者合为标量特征。详见 §1.3。

### 2.3 Patch Encoder
```
Conv2d(4, 32, k=3, pad=1) + ReLU
Conv2d(32, 64, k=3, pad=1) + ReLU
Conv2d(64, 64, k=3, pad=1) + ReLU
Flatten -> Linear -> d_patch (=128)
```

### 2.4 状态特征编码
目标方向/上一步动作/到达标志经 MLP 得 $d_{\text{scalar}}$，与 patch 特征拼接。

### 2.5 双向意图通信
意图 $\mathbf{z}_i = g_\theta(\text{patch}\oplus\text{scalar})$。候选邻接：$j\in\mathcal{N}_i^{\text{cand}} \iff \lVert p_i-p_j\rVert\le r_c \wedge j\in\text{FOV}_i$。在候选集上用**稀疏 softmax(entmax)** 算权重(不相关邻居权重→0),配合迟滞 $\tau_{out}<\tau_{in}$ 保证时间稳定;`comm_topk` 仅在非零邻居超额时封顶算力。多头注意力：

$$\alpha_{ij}=\text{entmax}_j\big(\text{score}(\mathbf{z}_i,\mathbf{z}_j)\big),\quad \mathbf{c}_i=\sum_{j\in\mathcal{N}_i}\alpha_{ij}\,W\mathbf{z}_j$$

无邻居则 $\mathbf{c}_i = \mathbf{0}$。

### 2.6 DRQN 记忆
$h_i^t=\text{GRU}(x_i^t, h_i^{t-1})$，$x_i^t=\text{patch}\oplus\text{scalar}\oplus\mathbf{c}_i^t$。

### 2.7 Dueling Q Head
输出 5 维 $Q$（对应 wait/up/down/left/right）。选动作时非法位置置 $-\infty$。

### 2.8 完整前向
```
patch -> CNN -> patch feature
scalar(目标方向/上一步动作/到达) -> MLP -> state feature
patch⊕state -> intent z
z -> (FOV内 top-k) 双向 attention -> communication feature c
patch⊕state⊕c -> GRU -> hidden h
h -> dueling Q head -> Q(5)
```

---

## 3. 训练方式

### 3.1 总体流程（三阶段）
- **阶段一**：LaCAM 专家数据离线积累 + 困难库标定。
- **阶段二**：DQfD + Reptile 联合预训练（**off-policy**，混合自游走防 OOD 高估）。
- **阶段三**：目标地图 **低滞后近 on-policy** 微调（lag≤1，专家介入概率 + margin 权重退火）。

> **关于 on/off-policy 的分工（回应你的点 3）**：DQfD 的本质是从专家示范学习——专家轨迹按定义**不是**当前策略产生的，所以**阶段二必须 off-policy（replay 复用专家+自游走）**，无法做成纯 on-policy。你担心的"版本滞后导致不收敛"主要发生在**阶段三**的自主 RL；因此阶段三改为**同步式近 on-policy**：并行 env 在冻结权重下采一段 rollout，learner 仅在这段新鲜数据上更新，随即同步权重，**lag ≤ 1 段**。专家 margin 项随退火→0，末期即纯近 on-policy Q 学习。这样既拿到你要的低滞后稳定性，又能靠批量并行采样维持吞吐（与 PPO 的"按段 on-policy"同理）。

### 3.2 阶段一：LaCAM 专家数据生成
对每种地图类型采样任务 $\tau=(m, \text{size}, N, \text{struct}, \text{starts}, \text{goals})$，LaCAM 求解；成功转动作入专家 buffer，失败入 hard case 池。

### 3.3 阶段二：DQfD + Reptile 联合预训练

#### 3.3.1 防高估混合策略（**课程式**）
纯专家预训练会令策略一旦偏离 LaCAM 轨迹就进入未见过的"脏状态"，Q 值在那里塌缩/高估、退火即崩。故每个 inner-loop batch 按比例混合专家与自游走：

- 早期 `expert_ratio≈0.9`（策略尚弱，自游走多为碰撞噪声，价值有限）；
- 随 meta 迭代**线性降至 0.5**（逐步引入真实惩罚传播，收敛 Q 边界）。

（取代 v1 的固定 70/30。）

#### 3.3.2 1-step TD
$$y_i^{1}=r_i^t+\gamma\,(1-\text{done})\,V_{\bar\theta}^{\text{mm}}(o_i^{t+1}),\qquad L_{TD}=\big(Q_\theta(o_i^t,a_i^t)-y_i^{1}\big)^2$$

#### 3.3.3 数值安全的 Masked Mellowmax Target
对合法动作集 $\mathcal{A}^{\text{legal}}$，温度 $\alpha$：

$$V^{\text{mm}}_{\bar\theta}(s)=\alpha\Big(\log\!\sum_{a\in\mathcal{A}^{\text{legal}}} e^{\,Q_{\bar\theta}(s,a)/\alpha}-\log|\mathcal{A}^{\text{legal}}|\Big)$$

工程实现（非法项以 $-\infty$ 隔离在求导路径外；保证至少 1 个合法动作——wait 通常恒合法）：
```python
safe_q = q_values / alpha
safe_q = safe_q.masked_fill(~action_masks, -float('inf'))
v_target = alpha * (torch.logsumexp(safe_q, dim=-1)
                    - torch.log(valid_counts + 1e-8))
```

> **α 选取（修正）**：本实现中 $\alpha$ 越小越接近 hard max、越易高估。删除 v1 的 `α=0.1`（过小）。默认 `α=1.0`，并在 `{0.5, 1, 2}` 中按 `Q_max/Q_mean` 走势与 TD error 分布选取。

#### 3.3.4 n-step TD
$$y_i^{n}=\sum_{l=0}^{n-1}\gamma^l r_i^{t+l}+\gamma^{n}V^{\text{mm}}_{\bar\theta}(o_i^{t+n}),\qquad L_n=\big(Q_\theta(o_i^t,a_i^t)-y_i^{n}\big)^2$$

#### 3.3.5 专家大间隔损失
对专家动作 $a^E$（已剔除可解释性不足的样本，§1.3）：

$$L_{margin}=\max_{a\in\mathcal{A}^{\text{legal}}}\big[Q_\theta(s,a)+\ell(a^E,a)\big]-Q_\theta(s,a^E),\quad \ell(a^E,a)=\begin{cases}0 & a=a^E\\ m & a\ne a^E\end{cases}$$

非法动作已置 $-\infty$，自动排除。

#### 3.3.6 wait 权重
专家 wait 过多易致过度等待，对 margin 按 wait 比例加权（避碰/瓶颈让行的 wait 保留较高权重，过渡性静止 wait 降权）。

### 3.4 Reptile 元学习（**含 target 网络处理，回应你的点 2**）
学习易适应的初始化 $\theta^\*$。每个 task $\tau\sim P(m)$：

```
initialize meta-parameter θ
for meta_iter = 1 ... M:
    sample map type m ~ P(m); sample task τ from m
    θ_inner ← θ
    θ_target ← θ            # ★ target 在本 task 内冻结于 θ_meta（你的方案）
    for k = 1 ... K:
        sample mixed batch (course expert/self-play ratio) from τ
        burn-in 前 burn_in_len 步重建隐状态（只前向，不计 loss）
        compute DQfD loss with masked-mellowmax target using θ_target
        θ_inner ← θ_inner − β ∇L(θ_inner)
    θ ← θ + ε (θ_inner − θ)
```

> **target 选择（采纳你的判断）**：inner loop 内 target **冻结于 θ_meta**，每个 task 起始重置。理由：inner 只有 $K=5$ 步，若 target 在 inner 内同步几乎不动、语义混乱；冻结于 θ_meta 给出每个 task 一致、稳定的自举基准，与 Reptile"围绕 θ 做局部适应"的语义一致。阶段三微调则改用**周期性 target 同步**（标准 DQN）。
>
> 外层更新 $\theta\leftarrow\theta+\varepsilon(\theta_{\text{inner}}-\theta)$，$\beta$=inner_lr，$\varepsilon$=meta_lr，$K$=inner_steps。

> **隐状态陈旧（R2D2）**：专家/自游走序列存了 `hidden_init`，但 θ 变化后旧 hidden 失效。故每段训练前做 `burn_in_len=8` 步 burn-in，仅前向重建 GRU 隐状态、不回传，再在其后计 loss。

### 3.5 地图类型采样课程
仅用 `random / warehouse / maze / room`，$\sum_m P(m)=1$。

| 阶段 | random | warehouse | maze | room |
|---|---|---|---|---|
| 前期（学基本避障达标） | 0.45 | 0.20 | 0.20 | 0.15 |
| 中期（提结构化比例） | 0.30 | 0.30 | 0.25 | 0.15 |
| 后期（强瓶颈/迷宫泛化） | 0.20 | 0.30 | 0.30 | 0.20 |

某类失败率显著偏高时启用自适应采样 $P(m)\propto \exp(\eta\cdot\text{failrate}(m))$。

### 3.6 阶段三：目标地图近 on-policy 微调
初始化 $\theta\leftarrow\theta^\*$。采用短窗口 recent buffer + 同步段采样，丢弃 $v_{\text{learner}}-v_{\text{sample}}>1$ 的样本（见 §3.1 分工说明）。

### 3.7 LaCAM 实时专家介入
按概率 $p(t)$（`p_start=0.5 → p_min=0.02` 衰减）调用 LaCAM 帮脱困；或仅在困难态触发（多步无人达标 / 同区反复冲突 / 反复踏自身足迹 / 剩余步数不足 / 通道门口长时拥堵）。

### 3.8 微调损失退火
$\lambda_E$ 退火：前期 1.0 → 中期 0.3 → 后期 0.05 或 0；前期靠专家脱困，后期靠自主 RL。

---

## 4. 训练速度优化（**新增**）

### 4.1 吞吐目标与瓶颈定性
对标"When to Switch …"（PPO，约 8h/10 亿步，≈ 4.3×10⁴ env-step/s）。可接受最多慢 10×，故 **目标 ≥ 4×10³ env-step/s**。需先认清：本方案比纯 PPO 重在两处——**(a) GRU 循环展开**、**(b) 每步对 96 体做 top-k 邻居注意力（GAT）**；这两项是预算大头。该 PPO 论文能跑那么快，恰说明 **Pogema 在向量化时步进很快、不是主瓶颈**——本方案慢在更重的网络与数据管道，**不在 env**。

本方案模型很小、目标硬件 GPU 很大（见 §4.5），因此大概率处于 **kernel-launch 受限**区间：GPU 算力没吃满，时间耗在零碎 kernel 启动与 Python 逐体/逐步循环上。**这一点决定了下面所有取舍——必须先 profile 再动手。**

### 4.2 语言选型：绝大部分用 Python，Rust 不是杠杆
真正的开销分两类：**GPU 上的网络计算** 与 **CPU 上的 env 步进 + 数据搬运**。据此定语言：

- **网络前向/反向、DQfD 损失、mellowmax、TD target、Dueling/GRU/GAT**：全是 GPU 张量运算（CUDA kernel），PyTorch 调用即可，**Rust 毫无帮助**。
- **Pogema env 步进**：env 内核，不应"用 Rust 重写"；向量化后本就不慢（见 §4.1）。
- **LaCAM**：Okumura 原版即 **C++**，通过绑定调用即达 native 速度，**不用动**。
- **observation_builder / reward_builder(BFS 查表) / action_mask / top-k 邻居选择**：唯一**可能**值得编译加速的 CPU 热点。但**正确顺序是 numpy 向量化 → numba → 最后才 Rust**：跨 Python↔Rust 边界每步搬大数组，若非 zero-copy，FFI 开销会吃掉收益；对研究项目，PyO3/maturin 的构建与维护成本通常不划算。

> **Rust 唯一可能真正赚到的场景**：profile 后确认"96 体 × 数十并行 env 的 per-agent 不规则 patch 拼装 + 足迹衰减 + 不规则 top-k 邻接"在 CPU 上占大头，且因不规则结构 numpy/numba 都向量化不掉——此时写一个**操作共享 numpy 缓冲区的 zero-copy 小扩展**才有意义。这是**最后手段**，多数情况用不到。

**比换语言高得多的杠杆**（都在 Python 内即可实现）：① 把"全体 agent × 全部并行 env"合成**一次**前向，杜绝 per-agent Python 循环；② `torch.compile` + CUDA graphs 砍 launch 开销（小模型大卡，launch 常比算力还贵）；③ AMP 混合精度，patch 用 uint8/fp16；④ 共享内存 zero-copy 传 uint8 patch、禁用 pickle；⑤ 采样与训练流水线重叠。

### 4.3 瓶颈定位顺序（先测后改）
用 `utils/profiler.py` 拆分：env step / NN forward / batch 构造 / IPC 等待 / 序列化 / CPU→GPU 拷贝 / backward / optimizer / LaCAM 调用。按占比从大到小处理：

1. **张量拷贝/序列化**：patch 以 **uint8 连续存储**于共享内存，actor 只传 `slot_id, episode_len, policy_version, task_id`，learner 按 slot 读连续数据；避免 pickle 大对象。
2. **CPU→GPU 搬运**：`pin_memory=True` + `non_blocking=True` 异步拷贝；下一 batch 预取与本 batch 计算重叠。
3. **96 体 GAT 注意力**：把"逐体逐步"改为**批量稀疏注意力**（一次处理整个 batch×time×agent 的 top-k 邻接，用 `index_select`/`scatter`，杜绝 Python 循环）；top-k 截断（k=6）把复杂度从 $O(N^2)$ 降到 $O(Nk)$。
4. **GRU 展开**：用 cudnn 加速的 GRU 一次性展开整段 `sequence_len`，而非逐步 Python 循环；burn-in 与训练段一起前向。
5. **整网编译**：`torch.compile` + CUDA graphs；AMP 混合精度。
6. **向量化/并行环境**：见 §4.4 的多进程布局。
7. **LaCAM 实时介入开销**：阶段三困难态触发阈值不要过敏，避免频繁唤醒 C++/Python 边界；对同一困难场景缓存 LaCAM 结果。

> **诊断分流**：若 **GPU 利用率上不去** ⟹ 病因是 launch 开销/Python 循环（Rust 救不了，做①②⑤）；若 **GPU 满载但整体仍慢** ⟹ 网络太重（按 §4.6 简化 GAT）；只有 **profiler 明确指向某 CPU 观测构建热点且 numba 也压不下去** ⟹ 才轮到 Rust（§4.2）。

### 4.4 元学习并行：两层并行（多进程是对的）
**关键约束**：Reptile 的 inner loop 是 **K 步串行 SGD**（第 k 步依赖 k−1 步），**单 task 内部无法并行**。并行只能在两个轴上展开：

- **轴 1｜CPU 多进程做环境采样。** 阶段二自游走数据靠 env rollout 产生；Python **GIL 会使多线程 env 步进串行化，必须用多进程**（`spawn` 启动）。worker **只做 CPU/env/观测构建、不碰 CUDA**，产出 numpy 写入共享内存环形缓冲；trainer 进程独占 CUDA 上下文。
- **轴 2｜双 GPU 做 task-并行（batched Reptile）。** 两卡各跑**一个不同 task** 的完整 K 步 inner loop，结束后平均增量做一次 outer 更新：

$$\theta \leftarrow \theta + \varepsilon \cdot \frac{1}{|\mathcal{B}|}\sum_{j\in \mathcal{B}}\big(\theta_{\text{inner}}^{(j)} - \theta\big),\qquad \mathcal{B}=\{\text{GPU0 task},\ \text{GPU1 task}\}$$

  这天然契合 Reptile：**同步只在每个 meta-iter 发生一次**（K 步才同步一回），不像 DDP 每步 all-reduce。本方案 `batch_size=32`，逐步 DDP 多半不划算，故**用 task-并行而非 DDP**。它同时把元吞吐翻倍、降低元梯度方差。

> 坑：用 `spawn`；CUDA 上下文只放在 2 个 trainer 进程，**不要在 env-worker 里做 GPU 计算**；worker 产 numpy，trainer 负责所有张量/CUDA 工作。

### 4.5 目标硬件的进程布局（2×RTX 5090 + 47 核 CPU）
显存对这个小模型绰绰有余，**始终盯吞吐与 GPU 利用率，而非显存**。

**阶段二（DQfD + Reptile 预训练）**——用 §4.4 两层并行：
```
GPU0 trainer 进程   ── task A 的 inner loop（K 步 DQfD）  ┐
GPU1 trainer 进程   ── task B 的 inner loop（K 步 DQfD）  ┘→ 平均 Δ → outer 更新
~40 env-worker 进程 ── 自游走 rollout + 观测构建 → 共享内存(uint8 环形缓冲)
~4 LaCAM-worker 进程 ── 数据生成/困难态按需调用（C++ 绑定）
剩余 ~3 核          ── prefetch / collator / 主控 / OS
```

**阶段三（单目标地图、近 on-policy 微调）**——此时**没有 task 并行可用**，改 Ape-X / SEED-RL 式分工，避免一张卡闲置：
```
GPU0 = learner（训练）
GPU1 = batched 推理 actor（为数十个并行 env 跑策略前向）
~40 核 = 并行 env（同步段采样，lag≤1）
```

### 4.6 速度兜底（若预算被打爆）
若 profile 显示 GAT 仍是主瓶颈且压不进预算，**按以下顺序降级**：
1. 通信改为**每 N 步**做一次（而非每步）；
2. 注意力降级为**对 top-k 邻居意图带权 mean-pool**（去掉多头 softmax）；
3. 在 random 地图上关闭通信（拥堵少），只在 warehouse/room 开启。

每降一级都用 §5 指标确认 SR/Deadlock 不显著恶化。

---

## 5. 评价指标（必须能定位问题来源）

仅看成功率不够。指标分八类：任务效果 / 路径质量 / 冲突死锁 / 学习稳定性 / Q 值诊断 / 专家与模仿 / 通信 / 系统性能。

### 5.1 任务效果
- **Episode 成功率 SR**：全体到达的 episode 占比。
- **平均到达率 AR**：每 episode 到达智能体占比。AR 高但 SR 低 ⟹ 少数体卡死。
- **最差体未达数 U**：统计 `mean(U), max(U), P(U≤1), P(U≤5)`，判断失败是否集中在少数体。

### 5.2 路径质量
- **Makespan** $=\max_i T_i$；**SoC** $=\sum_i \text{len}_i$（SR 高但 SoC 大 ⟹ 能完成但低效）。
- **膨胀率 Inflation** $=\text{len}_{\text{policy}}/\text{len}_{\text{LaCAM}}$（≫1 ⟹ 绕路/等待过多）。
- **平均步数利用率**（接近 1 且 SR 低 ⟹ 常耗尽步数）。

### 5.3 冲突与死锁
顶点冲突数、边冲突数（交换位置）、**WaitRate**（未移动比例）、**NoProgressRate**（仅用曼哈顿做**事后诊断**，不进 reward）、**RepeatRate**（反复经过自身足迹）、**死锁判定**（连续 $W$ 步无新增到达且平均位移极低 ⟹ 记 DeadlockCount/Duration/FirstTime）。

### 5.4 学习稳定性
$L_{TD}, L_n, L_{margin}$ 走势、梯度范数（max/mean/裁剪比）、参数更新幅度。

### 5.5 Q 值诊断
Q 的 mean/max/min/std；**Expert Q advantage** $=Q(s,a^E)-\max_{a\ne a^E}Q(s,a)$（<0 ⟹ 网络认为非专家更好）；TD error 分布。

### 5.6 专家与模仿
LaCAM 成功率、模仿匹配准确度 `Acc_E`、专家 wait 比率 `Wait_E`、微调实时干预占比 `InterventionRate`。

### 5.7 通信图监控（规模 + 时间稳定性）
在 $r_c$（FOV 内、软稀疏选择）下构建动态图 $G^t$：
- **最大连通分支规模 MaxCC**：频繁接近 $N$ ⟹ 局部通信退化为大范围无序网络（拓扑雪崩，*规模*失控）。
- **平均邻居数**：长期≈0 ⟹ $r_c$ 过小/分布过稀；过大 ⟹ 计算开销与噪声上升。
- **边翻转率 EdgeFlipRate（闪烁指标，新增）**：相邻两步 $G^t$ 与 $G^{t+1}$ 的边集对称差占比 $\frac{|E^t \triangle E^{t+1}|}{|E^t \cup E^{t+1}|}$。高 ⟹ 通信图闪烁（*时间稳定性*失控），应加强迟滞/软化或调 $\tau_{in},\tau_{out}$。这是判断"是否被 hard 选择的边界抖动困扰"的直接证据。
- **Attention 熵**：高熵（≈$\log k$）⟹ 无差别关注、避障混乱；低熵 ⟹ 聚焦关键冲突邻居（窄门口堵路者），通常对应更高效避碰。

### 5.8 系统性能
Actor FPS、Learner Update FPS、IPC 等待、序列化耗时、显存拷贝、策略版本滞后分布。

---

## 6. 如何根据指标定位问题

### 6.1 SR 低 且 Acc_E 也低
可能：channel 2 目标投影编码失效；局部观测不足以区分两个专家动作（对称让行 ⟹ margin 拉不开，正是 §1.3 要剔除的样本）；网络容量不足；通信无效（$r_c$ 过小或 attention 崩溃）；margin 梯度占比过小。
先查：`Acc_E, L_margin, Adv_E, NeighborCount, AttnEntropy, channel2 投影正确性`。

### 6.2 Acc_E 高 但 rollout SR 低
可能：OOD 脆弱（自游走比例不足，偏离专家即崩）；纠偏/回退能力弱（TD 未有效传播）；专家 wait 被滥用（无全局时序同步 ⟹ 局部死锁）；Reptile 偏置到 random、对 warehouse/maze 适应差。
先查：`SR/AR, RepeatRate, DeadlockCount, TD error 分布, InterventionRate, Acc_E(move) vs Acc_E(wait)`。

### 6.3 WaitRate 很高
可能：$c_{\text{coll}}$ 过高（move 的 Q 普遍低于 wait）；wait margin 权重过高；通信级联（MaxCC≈$N$ 的 Cascade Wait）；$\lambda_F$ 偏小不足以克服每步惩罚。
先查：`Wait_E, WaitRate, Q_mean(wait) vs Q_mean(move), Adv_E(wait), MaxCC, NoProgressRate`。

### 6.4 Q 值持续升高 但 SR 不升
可能：非法动作未严格切片致 mellowmax 受污染；reward 尺度过大；target 更新过慢；replay 滞后致离线自举；$\alpha$ 过小退化为 hard max（见 §3.3.3）。
先查：`Q_max/Q_mean, TD error 分布, policy version lag, reward scale, mellowmax α`。

### 6.5 Loss 下降 但 Reward 也下降
可能：策略版本滞后（拟合已被抛弃的旧动作分布）；recent buffer 过大混入过时序列；RL 对齐与模仿梯度冲突（$\lambda_E$ 退火过慢）。
先查：`Policy Version Lag, L_TD/L_margin, Average Reward, Recent Buffer Age, InterventionRate`。

### 6.6 训练速度低
见 §4.3 的瓶颈定位顺序与 §4.2 的诊断分流。
先查：`ActorFPS/UpdateFPS, T_IPC, T_ser, T_batch, T_copy, NN Forward Time, LaCAM Runtime`。

---

## 7. 预期与验证（**全部降级为待验证假设**）

> 以下为**假设**，需以对照实验证伪，不是承诺结论。

- **H1（训练稳定性）**：阶段二引入课程式自游走混合后，相比纯专家 DQfD，微调退火期的性能抖动显著下降、收敛更平稳。**验证**：对比 `自游走比例 ∈ {0, 课程}` 的 SR/Reward 方差与收敛步数。
- **H2（泛化）**：Reptile 元参数 $\theta^\*$ 在少量微调（目标 < 30 epoch）后即可适应。**验证**：必须**留出一个训练中从未出现的地图族（held-out family）**作真正的泛化测试，而非用训练同族的新布局冒充"完全陌生"。
- **H3（奖励）**：BFS 势函数 shaping 相比无稠密 shaping，在 maze/room 上降低 DeadlockCount 与 RepeatRate。**验证**：`{无稠密, BFS-PBRS}` 在 maze/room 上对比这两个指标；并验证 BFS-PBRS 不改变收敛到的策略质量（策略不变性的经验检查）。
- **H4（速度）**：在 §4 优化后达到 ≥ 4×10³ env-step/s（对标的 1/10）。**验证**：profiler 报告 + 端到端吞吐。
- **H5（通信复杂度是否值得）**：双向意图 + 注意力相比更朴素的聚合,在强协同地图(maze/room)上更优。**注意**:"和谁通信"(选择层:软稀疏/迟滞)与"通信内容+如何聚合"(意图向量+注意力)是两个不同层面,本假设只检验后者是否值得其复杂度。**验证——消融阶梯**(与 §4.6 降级路线挂钩,一举两得):

  | 档 | 邻居选择 | 内容 | 聚合 |
  |---|---|---|---|
  | A 全量 | 软稀疏(entmax)+迟滞 | 双向意图 $\mathbf{z}_j$ | 多头注意力 |
  | B 去注意力 | 软稀疏+迟滞 | 双向意图 | 带权 mean-pool |
  | C 去意图 | 软稀疏+迟滞 | 仅相对位置/速度 | 多头注意力 |
  | D 无通信 | — | — | — |

  在 maze/room 上对比 **SR / DeadlockCount / WaitRate / EdgeFlipRate**：A≈D ⟹ 通信整体无用应砍；A≈B ⟹ 注意力多余,退 mean-pool（即 §4.6 档位 2）；A≈C ⟹ 意图向量多余,只传位置；A 显著最优 ⟹ 复杂度值得。**让数据判决"是否多余",不靠设计阶段论证。**

---

## 8. 总结

核心：**LaCAM 专家（特权，仅训练期）→ DQfD 模仿（off-policy 阶段二）→ Reptile 跨四地图族元初始化 → 目标地图低滞后近 on-policy 微调（阶段三）**。

网络为**共享参数去中心化 Dueling DRQN**，不使用 QMIX、不构造 $Q_{tot}$；输入为 **Pogema 原生 3 通道（障碍/智能体含自身/目标边界投影）+ 追加足迹通道 + 标量特征（目标方向/上一步动作/到达）**；用 **FOV 内软稀疏(entmax)+迟滞的双向意图注意力**实现避让（控制通信图*规模*与*时间稳定性*,top-k 仅作算力上限）；用**数值安全 masked mellowmax**抑制高估；用 **BFS 测地距离的势函数 shaping**（策略不变、不破坏部分可观测）替代被删除的点积对齐奖励，规避迷宫绕行陷阱。已如实声明 IQL 非平稳性、中心化→去中心化的时序模仿丢失、hard top-k 闪烁等核心风险，并配以指标（含 EdgeFlipRate）与机制缓解，通信复杂度由 H5 消融判决。
