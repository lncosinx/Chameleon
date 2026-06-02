# Chameleon — 基于 LaCAM-DQfD-Reptile 的部分可观测 MAPF

## 镜像构建

```bash
cd Chameleon
docker build \
  --network=host \
  --build-arg http_proxy=http://127.0.0.1:20171 \    #代理端口设置，不需要可以删掉
  --build-arg https_proxy=http://127.0.0.1:20171 \   #代理端口设置，不需要可以删掉
  -t chameleon:05301957 .

```

## 依赖安装

```bash
# Pogema 环境 + toolbox（仓库/迷宫/房间地图）
# pip install pogema==2.0.0a 安装pogema-toolbox会自动安装pogema
git clone https://github.com/Cognitive-AI-Systems/pogema-toolbox.git
cd pogema-toolbox && pip install -e . && cd ..

# 核心依赖
pip install torch numpy numba pyyaml

# 可选：LaCAM C++ 绑定（需单独构建，见 lacam/ 目录）
```

## 快速自检

```bash
python scripts/env_check.py
```

---

## 训练流程（三阶段）

训练分三个阶段，按顺序执行。所有可调超参均在 `configs/` 下的 YAML 文件中，**不要在命令行硬编码**。

---

### 阶段一 — 生成 LaCAM 专家数据

对四种地图族批量生成专家轨迹，保存为 `data/expert_<map>.npz`，供阶段二加载。

`--num-workers -1` 自动使用全部 CPU 核心（47 核约 4 ep/s，500 ep ≈ 2 分钟/地图族）。
四种地图族顺序执行即可；若同时执行请手动拆分核数（如 `--num-workers 10`）避免竞争。

```bash
python scripts/generate_lacam_data.py \
    --config configs/env_random.yaml \
    --num-episodes 500 \
    --num-agents 96 \
    --map-size 64 \
    --backend prioritized \
    --output-dir data \
    --num-workers -1

python scripts/generate_lacam_data.py \
    --config configs/env_warehouse.yaml \
    --num-episodes 500 \
    --num-agents 96 \
    --map-size 64 \
    --backend prioritized \
    --output-dir data \
    --num-workers -1

python scripts/generate_lacam_data.py \
    --config configs/env_maze.yaml \
    --num-episodes 500 \
    --num-agents 96 \
    --map-size 64 \
    --backend prioritized \
    --output-dir data \
    --num-workers -1

python scripts/generate_lacam_data.py \
    --config configs/env_room.yaml \
    --num-episodes 500 \
    --num-agents 96 \
    --map-size 64 \
    --backend prioritized \
    --output-dir data \
    --num-workers -1
```

完成后 `data/` 目录应包含四个 npz 文件（每个约 500–800 MB 压缩后）：
```
data/expert_random.npz
data/expert_warehouse.npz
data/expert_maze.npz
data/expert_room.npz
```

> **回合数说明（为什么是 500 而不是更多）**
>
> 目标规模（96 agents，64×64）下每个 episode 约产生 19,000 个单体样本。
> 500 × 0.65（成功率） × 19,000 ≈ **620 万**有效样本/地图族，buffer 设为 200 万上限后保留最多样的 200 万条。
> 阶段二训练共消耗约 7200 万专家样本（20000 iters × 5 steps × batch × 0.7 ratio），依赖 replay buffer 对同一样本重用约 10 次，这是 off-policy 方法的标准设计。
> - 500 episode 足够满足数据多样性要求，超过 1000 收益递减。
> - `--backend prioritized`（非真 LaCAM）在大地图/多 agent 上成功率约 60–70%，不建议依赖更多 episode 来弥补质量问题；应尽快切换到 `--backend lacam`（需构建 C++ 绑定）。

> **其他说明**
> - 阶段二 `train_meta.py` 会自动从 `--data-dir` 加载这四个 npz 文件；若文件不存在则退为小规模在线生成（仅供调试）。
> - 小规模调试可使用 `--num-agents 8 --map-size 16 --num-episodes 50`（几分钟完成）。

---

### 阶段二 — DQfD + Reptile 元预训练

在四种地图族上元预训练，输出 `checkpoints/meta_init.pt`。

```bash
# 双 GPU task-parallel（推荐，2×RTX 5090）
python scripts/train_meta.py \
    --config configs/train_reptile_dqfd.yaml \
    --network-config configs/network_biintent_drqn.yaml \
    --meta-iters 20000 \
    --gpus 0,1 \
    --data-dir data \
    --seed 42

# 单 GPU（调试 / 单卡机器）
python scripts/train_meta.py \
    --config configs/train_reptile_dqfd.yaml \
    --network-config configs/network_biintent_drqn.yaml \
    --meta-iters 20000 \
    --gpus 0 \
    --data-dir data \
    --seed 42
```

> **说明**
> - 超参均在 `configs/train_reptile_dqfd.yaml` 中调整；关键项：
>   - `meta_lr_start` / `meta_lr_end`：outer update 学习率线性衰减（默认 0.01 → 0.001）
>   - `tasks_per_gpu`：每卡串行任务数（默认 2，双卡合计 4 tasks/meta-iter）
>   - `inner_steps`、`inner_lr`、`burn_in_len`、`expert_ratio_*` 等
> - 双 GPU 模式下两卡各串行跑 `tasks_per_gpu` 个 inner loop，meta-iter 末平均增量做一次 outer update（详见设计文档 §4.4）。
> - 性能诊断：将 `debug: true` 写入 yaml，每 `profile_interval` 个 iter 自动打印 `presample` / `inner_loop_wall` / `outer_update` / `sync_meta` 各段耗时。
> - 训练中途可用 `--meta-iters` 缩短进行快速验证，checkpoint 保存在 `checkpoints/meta_init.pt`。
> - 监控指标：`inner_loss_mean` 应持续下降；`expert_ratio` 按课程从 0.9 线性降至 0.5；`meta_lr` 随 iter 线性递减。

---

### 阶段三 — 目标地图近 on-policy 微调

从 Reptile meta checkpoint 初始化，在指定目标地图上低滞后近 on-policy 微调。

```bash
# 在 maze 地图上微调（示例）
python scripts/finetune_target_map.py \
    --config configs/finetune_online.yaml \
    --network-config configs/network_biintent_drqn.yaml \
    --env configs/env_maze.yaml \
    --init checkpoints/meta_init.pt \
    --num-segments 5000 \
    --num-agents 96 \
    --map-size 64 \
    --device cuda:0 \
    --seed 42 \
    --output checkpoints/finetuned_maze.pt

# 在 warehouse 地图上微调（示例）
python scripts/finetune_target_map.py \
    --config configs/finetune_online.yaml \
    --network-config configs/network_biintent_drqn.yaml \
    --env configs/env_warehouse.yaml \
    --init checkpoints/meta_init.pt \
    --num-segments 5000 \
    --num-agents 96 \
    --map-size 64 \
    --device cuda:0 \
    --seed 42 \
    --output checkpoints/finetuned_warehouse.pt
```

> **说明**
> - 超参（专家介入概率 `expert_prob_*`、margin 退火 `lambda_margin_*`、`rollout_segment_len` 等）在 `configs/finetune_online.yaml` 中调整。
> - 阶段三 lag 约束为 `max_policy_version_lag=1`（已在配置中设定），每 segment 数据用完即同步。
> - 每 20 个 segment 打印一次 `loss`、`expert_prob`、`lambda_margin`。`lambda_margin` 退火到 0 后退化为纯近 on-policy Q 学习。

---

### 评估

```bash
# 评估指定 checkpoint（关闭专家介入）
python scripts/evaluate_policy.py \
    --ckpt checkpoints/finetuned_maze.pt \
    --config configs/env_maze.yaml \
    --network-config configs/network_biintent_drqn.yaml \
    --episodes 200 \
    --num-agents 96 \
    --map-size 64 \
    --device cuda:0 \
    --seed 100 \
    --output results/eval_maze.yaml

# held-out 泛化测试（在训练未见过的地图族上评估）
python scripts/evaluate_policy.py \
    --ckpt checkpoints/finetuned_maze.pt \
    --config configs/env_room.yaml \
    --episodes 100 \
    --num-agents 96 \
    --map-size 64 \
    --held-out
```

> **说明**
> - 评估时**绝不调用 LaCAM**，所有决策仅依赖策略网络（INVARIANT 3）。
> - 主要关注指标：`SR_mean`（成功率）、`Deadlock_mean`（死锁率）、`WaitRate_mean`（等待率）、`EdgeFlipRate`（通信图稳定性）。
> - 完整指标定义见设计文档 §5。

---

### 小规模冒烟验证（开发调试用）

```bash
# 快速端到端验证（不需要 GPU，< 3 分钟）
python scripts/train_meta.py \
    --meta-iters 5 \
    --gpus cpu \
    --seed 42

python scripts/finetune_target_map.py \
    --init checkpoints/meta_init.pt \
    --num-segments 5 \
    --num-agents 4 \
    --map-size 16 \
    --device cpu \
    --seed 42 \
    --output checkpoints/smoke_finetuned.pt

python scripts/evaluate_policy.py \
    --ckpt checkpoints/smoke_finetuned.pt \
    --config configs/env_random.yaml \
    --episodes 3 \
    --num-agents 4 \
    --map-size 16
```

---

## 硬件要求

- **开发/冒烟**：1×GPU（CUDA 11.8+），≥8 GB VRAM，≥4 核 CPU
- **目标训练**：2×RTX 5090（32 GB），47 核 CPU，≥250 GB RAM

## 项目结构

```
Chameleon/
├── configs/         # 7 个 YAML 配置文件
├── envs/            # 环境封装、观测构建、掩码、奖励
├── lacam/           # LaCAM 专家（C++ 绑定 + 优先规划 fallback）
├── models/          # BiIntent-GAT-DRQN 网络
├── losses/          # mellowmax target、TD、DQfD 损失
├── buffers/         # 专家/自游走/在线 replay 缓冲
├── trainers/        # Reptile、微调、评估器
├── utils/           # 指标、profiler、checkpoint、seed、logger
└── scripts/         # 入口脚本
```

## 文档

- `Lacam_DQfD_Reptile_MAPF_Plan_v2.md` — 设计文档（唯一事实来源）
- `CLAUDE.md` — 操作摘要 + 14 条 INVARIANT
- `SCHEDULE.md` — 实现顺序（Phase 0→7）
