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

## 硬件要求

- **开发**：1×GPU（CUDA 11.8+），≥8 GB VRAM，≥4 核 CPU
- **训练**：2×RTX 5090（32 GB），47 核 CPU，≥250 GB RAM

## 项目结构

```
Chameleon/
├── configs/         # 7 个 YAML 配置文件
├── envs/            # 环境封装、观测构建、掩码、奖励
├── lacam/           # LaCAM 专家（C++ 绑定）
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
