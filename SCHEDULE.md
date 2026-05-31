# SCHEDULE.md — Claude Code 实现顺序

> **怎么用这份文件**
> Claude Code 按 **Phase 0 → 7 的顺序**逐任务实现。每个任务都有：**依赖**（必须先完成的前置任务）、**交付物**（要产出的文件）、**约束**（引用 `CLAUDE.md` 的 INVARIANT 编号，不可违反）、**验收门**（必须通过才能进入下一任务）。
> 规则：
> 1. **不跳过验收门**。门没过就不写下一个任务的代码。
> 2. 每个任务先写实现，再写它的验收脚本（放 `tests/` 或 `scripts/` 下的 smoke 脚本），跑通后再继续。
> 3. 任何与 `CLAUDE.md` / `Lacam_DQfD_Reptile_MAPF_Plan_v2.md` 冲突的设计，**停下来问人**，不要擅自改设计。
> 4. 先在小规模（`num_agents=4, size=16`）验证逻辑，再放大到目标规模（`96/64`）。
> 5. 每个任务完成后在本文件对应 `[ ]` 打勾并简注 commit。

---

## Phase 0 — 脚手架与环境（无依赖）

### T0.1 安装与环境自检
- **依赖**：无
- **交付物**：`scripts/env_check.py`、`requirements.txt`、`README.md`（最简）
- **做什么**：
  - 按 `CLAUDE.md` 命令安装 `pogema-toolbox`（含 pogema）+ `torch numpy numba pyyaml`。
  - `env_check.py`：打印 pogema 版本、`GridConfig().MOVES`、双 GPU 可见性（`torch.cuda.device_count()==2`）、CPU 核数。
- **约束**：INVARIANT 1（动作顺序必须实测确认）
- **验收门**：
  - `env_check.py` 实测打印的 `MOVES` 等于 `[[0,0],[-1,0],[1,0],[0,-1],[0,1]]`（wait=0）。**若不符，停下问人**——后续所有动作语义依赖它。
  - `torch.cuda.device_count()` 返回 2。

### T0.2 仓库骨架
- **依赖**：T0.1
- **交付物**：`CLAUDE.md` 仓库结构里列出的全部空目录 + `__init__.py`；`configs/` 下 7 个 yaml（值照搬设计文档 §1.2）；`utils/seed.py`、`utils/logger.py`。
- **约束**：所有可调项进 yaml，勿硬编码（开发约定）。
- **验收门**：`python -c "import yaml; [yaml.safe_load(open(f)) for f in <7 configs>]"` 全部成功解析；`seed.py` 能固定 torch/numpy/random 种子。

---

## Phase 1 — 环境与观测（项目地基，错了全盘错）

### T1.1 Pogema 封装 + 四地图族
- **依赖**：T0.2
- **交付物**：`envs/pogema_env_wrapper.py`
- **做什么**：统一 reset/step 接口；random 走 pogema 原生 generator，warehouse/maze/room 走 pogema-toolbox 加载；暴露每体 `get_targets_xy`、起终点、地图栅格（**后者仅供训练期 reward/LaCAM，不进 policy**）。
- **约束**：INVARIANT 2、3
- **验收门**：四种 `env_*.yaml` 各能 reset+随机走 20 步不崩；打印每体原生观测 shape `(3,K,K)`。

### T1.2 观测构建器
- **依赖**：T1.1
- **交付物**：`envs/observation_builder.py`
- **做什么**：在原生 3 通道上追加 **channel 3 时间衰减足迹**（当前=1.0，历史按 ρ 衰减到 [0,0.8]）；构建 8 维标量特征 = 目标相对方向(2)+上一步动作 one-hot(5)+到达(1)。输出 `(4,11,11)` patch + `(8,)` 标量。
- **约束**：INVARIANT 2、5、6（**不得**再加重复的目标投影通道；自身位置已在 ch1 中心格）
- **验收门**：
  - patch 恰为 4 通道；ch1 中心格恒为 1（自身）。
  - 单元测试：足迹通道随时间单调衰减、当前位置=1.0。
  - **断言**：observation_builder 的输入不含全局地图/距离场（用代码审查 + 显式断言）。

### T1.3 动作掩码
- **依赖**：T1.1
- **交付物**：`envs/action_mask.py`
- **做什么**：越界/静态障碍/环境禁止 → 非法；非法位填 `-float('inf')`；保证每体≥1 合法动作。**动态碰撞不硬掩码**。
- **约束**：INVARIANT 7
- **验收门**：构造贴墙场景，验证非法动作为 `-inf`、wait 恒合法；mellowmax 对全 `-inf` 之外的输入数值稳定（无 nan）。

### T1.4 BFS 距离场 + 势函数奖励
- **依赖**：T1.1
- **交付物**：`envs/potential_field.py`、`envs/reward_builder.py`
- **做什么**：
  - `potential_field`：对每个 `(map,goal)` 在四连通可行图上 BFS 算测地距离场，缓存；不可达填大常数。**只在环境内用，不进观测。**
  - `reward_builder`：`R = 稀疏达标 − 每步 − 碰撞 − 无意义wait + λ_F·(γΦ(s')−Φ(s))`，`Φ=−d_BFS`。
- **约束**：INVARIANT 3、4、14（**禁止点积对齐奖励**；**禁止曼哈顿做势函数**）
- **验收门**：
  - 在一张 maze 上，沿 BFS 最短路前进每步 shaping 项 > 0、绕墙时不出现"穿墙正梯度"。
  - 代码审查确认 reward 路径未被 policy 引用（grep 断言）。

### Phase 1 总验收门（必须全过再进 Phase 2）
随机策略在四地图族各跑 50 episode：能产出 `(obs, action, reward, done, mask)` 五元组、reward 有限、无 nan、足迹/掩码/距离场行为符合预期。

---

## Phase 2 — 专家数据（LaCAM）

### T2.1 LaCAM 绑定与运行器
- **依赖**：T1.1
- **交付物**：`lacam/lacam_runner.py`、`scripts/generate_lacam_data.py`
- **做什么**：构建 LaCAM（C++）Python 绑定；对给定 map/起终点求路径，输出设计文档 §1.4 的 JSON；`generate_lacam_data.py` 用 `num_lacam_workers=4` 多进程批量生成。
- **约束**：INVARIANT 3（LaCAM 仅训练期专家）；env-worker/LaCAM-worker 不碰 CUDA。
- **验收门**：在 4 地图族各成功生成 ≥10 条成功轨迹；失败样本进 hard case 池。

### T2.2 轨迹→动作转换 + 可解释性过滤
- **依赖**：T2.1, T1.2
- **交付物**：`lacam/trajectory_converter.py`、`lacam/lacam_dataset_builder.py`、`lacam/hard_case_manager.py`
- **做什么**：
  - 位移→动作严格按 INVARIANT 1 映射；
  - **观测必须由 observation_builder 按局部规则重生成**，不得用 LaCAM 全局状态；
  - 标记"同一 (patch+足迹+目标方向) 下专家动作不唯一"的样本为 `margin_excluded=True`。
- **约束**：INVARIANT 1、2、11
- **验收门**：
  - 反向校验：把转换出的动作施加到 env，能复现 LaCAM 路径（位置逐步一致）。
  - 对称让行场景确实被标 `margin_excluded`。

### T2.3 专家回放缓冲
- **依赖**：T2.2
- **交付物**：`buffers/expert_demo_buffer.py`
- **做什么**：存 `obs, hidden_init, action_expert, reward, next_obs, done, action_mask, map_type, task_id, margin_excluded`。
- **验收门**：能按 `sequence_len=32` 采样定长序列，字段齐全。

---

## Phase 3 — 网络

### T3.1 编码器
- **依赖**：T1.2
- **交付物**：`models/encoders.py`（PatchEncoder + Goal/Action/Status 编码）
- **验收门**：`(B,4,11,11)`→patch 特征、`(B,8)`→标量特征，shape 正确，可反传。

### T3.2 通信模块（FOV 内 软稀疏+迟滞 双向注意力）
- **依赖**：T3.1
- **交付物**：`models/communication.py`
- **做什么**：意图编码；候选邻居须**同时**满足 `dist≤comm_radius(=4)` 且 `comm_radius<obs_radius` 且 **FOV 内可见**；邻居选择用**软稀疏注意力(entmax/sparsemax)+迟滞($\tau_{out}<\tau_{in}$)**，**不要用 hard top-k 做选择**；`comm_topk=6` 仅作算力上限(非零邻居超额时封顶)；批量稀疏多头注意力（`index_select`/`scatter`，**禁止 per-agent Python 循环**）。
- **约束**：INVARIANT 10；语言与性能策略（批量稀疏、无 per-agent 循环）
- **验收门**：
  - 邻居全部在 FOV 内、生效邻居数 ≤ 算力上限；无邻居时 `c_i=0`。
  - **防闪烁检验**：构造两个邻居得分接近的边界场景，令一体微动一步，验证 **hard top-k 会翻转、而软稀疏+迟滞下 `EdgeFlipRate` 显著更低**（这是本任务的核心验收点，不是可选）。
  - 用 `num_agents=96` 跑一次前向，**无 Python 逐体循环**（代码审查）；记录耗时作 §4 基线。

### T3.3 循环核 + Dueling Q 头 + 组装
- **依赖**：T3.2
- **交付物**：`models/recurrent_core.py`、`models/q_heads.py`、`models/biintent_drqn.py`
- **做什么**：GRU 一次性展开整段序列；Dueling 输出 5 维 Q；组装 BiIntent-GAT-DRQN。
- **约束**：语言与性能策略（GRU 整段展开，不逐步循环）
- **验收门**：端到端前向 `(B,T,N,...)`→`Q (B,T,N,5)`，可反传；选动作时 `-inf` 掩码生效。

---

## Phase 4 — 损失

### T4.1 Masked Mellowmax Target
- **依赖**：T3.3, T1.3
- **交付物**：`losses/q_target.py`
- **做什么**：实现设计文档 §3.3.3 的数值安全版本（`masked_fill(-inf)` + `logsumexp`）。
- **约束**：INVARIANT 7、12（默认 `α=1.0`，**勿用 0.1**）
- **验收门**：含非法动作的输入不产生 nan；`α` 减小时 V 单调趋近 max（验证语义方向）。

### T4.2 TD（1-step + n-step）+ DQfD 总损失
- **依赖**：T4.1, T2.3
- **交付物**：`losses/td_losses.py`、`losses/dqfd_losses.py`
- **做什么**：n_step=5；margin loss 排除 `margin_excluded` 样本、wait 按比例降权；`L = λ_td·L_TD+λ_n·L_n+λ_margin·L_margin+λ_l2·||θ||²`。
- **约束**：INVARIANT 11
- **验收门**：在一小批专家数据上，margin loss 推动 `Q(s,a_expert)` 高于其他合法动作至少 `margin`；`margin_excluded` 样本不进 margin 项。

---

## Phase 5 — 缓冲与采样基础设施

### T5.1 共享内存 rollout 缓冲 + collator
- **依赖**：T1.2
- **交付物**：`buffers/shared_rollout_buffer.py`、`buffers/selfplay_buffer.py`、`buffers/batch_collator.py`
- **做什么**：连续内存结构化数组（patch uint8）；actor 只传 `slot_id/len/version/task_id`；collator 支持 `pin_memory + non_blocking`。**禁用 pickle 大对象。**
- **约束**：语言与性能策略
- **验收门**：写入/切片/读取一致；跨进程传递只传索引、不传大数组（审查）。

### T5.2 多进程 env-worker（CPU）
- **依赖**：T5.1, T1.*
- **交付物**：`buffers/` 内 worker 逻辑 + `utils/profiler.py`
- **做什么**：`spawn` 启动 `num_env_workers=40` 个 env-worker，只做 CPU/env/观测，写共享内存。
- **约束**：INVARIANT（env-worker 不建 CUDA 上下文）；并行布局 §4.4/4.5
- **验收门**：40 worker 稳定产数据 ≥5 分钟无死锁；`nvidia-smi` 确认 worker 进程**无 CUDA 上下文**；profiler 输出各段耗时。

---

## Phase 6 — 训练器

### T6.1 DQfD + Reptile（双 GPU task-并行）— 先单卡跑通
- **依赖**：T4.2, T5.2, T2.3
- **交付物**：`trainers/train_reptile_dqfd.py`、`trainers/scheduler.py`、`scripts/train_meta.py`、`utils/checkpoint.py`
- **做什么**：
  - **先单 GPU、单 task** 跑通 inner loop（K=5 步串行 SGD）+ outer 更新 + burn-in（8 步只前向）+ 课程式 expert_ratio(0.9→0.5)；
  - target **冻结于 θ_meta**，每 task 重置；
  - scheduler 实现地图课程采样。
- **约束**：INVARIANT 8、13；on/off 分工（阶段二 off-policy）
- **验收门（单卡）**：小规模 200 meta-iter，专家匹配准确度 `Acc_E` 上升、`L_margin` 下降、Q 不发散；checkpoint 四类可存可载。

### T6.2 升级为双 GPU task-并行
- **依赖**：T6.1
- **交付物**：T6.1 文件的并行版本
- **做什么**：两卡各跑一个不同 task 的完整 inner loop，meta-iter 末**平均增量**做一次 outer 更新（非逐步 DDP）。
- **约束**：并行布局 §4.4/4.5
- **验收门**：双卡吞吐相对单卡近翻倍；两卡 Δ 平均后 θ 一致；与单卡结果在可控方差内一致。

### Phase 6 性能门
跑 profiler 做诊断分流（GPU 利用率 / GPU 满载但慢 / CPU 热点），记录 env-step/s。**目标 ≥ 4×10³**。未达标按 §4.3/4.6 优化（先批量化/编译，**不要先写 Rust**）。

---

## Phase 7 — 微调与评估

### T7.1 近 on-policy 微调
- **依赖**：T6.2
- **交付物**：`buffers/recent_online_buffer.py`、`trainers/finetune_online.py`、`scripts/finetune_target_map.py`
- **做什么**：`θ←θ*`；同步段采样（GPU0 learner / GPU1 推理 actor），丢弃 `lag>1` 样本；专家介入概率 0.5→0.02、margin 权重 1.0→0 退火；周期性 target 同步。
- **约束**：INVARIANT 8（阶段三 target）、9（lag≤1）
- **验收门**：单目标 maze 上 <30 epoch 内 SR 上升；buffer 中无 `lag>1` 样本进训练。

### T7.2 评估器（关闭专家介入）
- **依赖**：T7.1
- **交付物**：`trainers/evaluator.py`、`scripts/evaluate_policy.py`、`utils/metrics.py`
- **做什么**：实现设计文档 §5 八类指标；评估**必须关专家介入**。
- **约束**：开发约定（指标定位问题来源）
- **验收门**：输出 SR/AR/Makespan/SoC/Deadlock/WaitRate/Q诊断/通信指标；与设计文档 §6 的排障表能对应。

### T7.3 泛化验证（held-out 地图族）
- **依赖**：T7.2
- **交付物**：`scripts/evaluate_policy.py` 的 held-out 模式
- **做什么**：**留一个训练中从未出现的地图族**做泛化测试（对应设计文档 H2）。
- **验收门**：报告 held-out 上的 SR，作为泛化的诚实证据（不得用训练同族新布局冒充）。

### T7.4 通信消融阶梯（H5：双向意图通信是否值得其复杂度）
- **依赖**：T7.2, T3.2
- **交付物**：`scripts/evaluate_policy.py` 的消融模式 + 结果表
- **做什么**：跑设计文档 §7-H5 的四档（A 全量软稀疏+意图+注意力 / B 去注意力换 mean-pool / C 去意图只传位置 / D 无通信），在 maze/room 上对比 **SR / DeadlockCount / WaitRate / EdgeFlipRate**。
- **约束**：评估关专家介入；与 §4.6 降级路线挂钩（B 即降级档位 2）。
- **验收门**：输出四档对照表并给出判决：A≈D⟹砍通信；A≈B⟹退 mean-pool；A≈C⟹只传位置；A 显著最优⟹保留全量。**让数据判决，不靠论证。**

---

## 全局完成定义（DoD）
- [ ] Phase 0–7 全部验收门通过
- [ ] `CLAUDE.md` 的 14 条 INVARIANT 在代码中都有对应断言或测试
- [ ] env-step/s ≥ 目标，profiler 报告归档
- [ ] 四地图族训练 + held-out 泛化结果归档
- [ ] 通信消融阶梯（H5）结果归档并据此定型通信复杂度
- [ ] 全程固定种子可复现；四类 checkpoint 齐全

---

## 进度勾选
- [ ] T0.1  - [ ] T0.2
- [ ] T1.1  - [ ] T1.2  - [ ] T1.3  - [ ] T1.4  - [ ] Phase1门
- [ ] T2.1  - [ ] T2.2  - [ ] T2.3
- [ ] T3.1  - [ ] T3.2  - [ ] T3.3
- [ ] T4.1  - [ ] T4.2
- [ ] T5.1  - [ ] T5.2
- [ ] T6.1  - [ ] T6.2  - [ ] Phase6性能门
- [ ] T7.1  - [ ] T7.2  - [ ] T7.3  - [ ] T7.4
- [ ] DoD
