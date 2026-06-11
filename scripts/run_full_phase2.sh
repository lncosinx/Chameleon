#!/usr/bin/env bash
# 全量阶段二：四族 LaCAM 专家数据重生成 → 校验 → DQfD+Reptile 全量元预训练。
# 整条链路串行，单条 nohup 后台运行，断 SSH 不影响。
set -u
cd /root/Chameleon

DATA_DIR=data
FAMILIES=(random warehouse maze room)
NUM_AGENTS=96
MAP_SIZE=64
NUM_EPISODES=3000
NUM_WORKERS=40
TS=$(date '+%m%d_%H%M')
GEN_LOG=logs/gen_full_${TS}.log
TRAIN_LOG=logs/train_full_${TS}.log

echo "==== [$(date '+%F %T')] 阶段二全量开始 ====" | tee -a "$GEN_LOG"

# ---- 1. 四族专家数据生成 ----
for fam in "${FAMILIES[@]}"; do
  echo "---- [$(date '+%T')] 生成 $fam (N=$NUM_AGENTS size=$MAP_SIZE ep=$NUM_EPISODES) ----" | tee -a "$GEN_LOG"
  python scripts/generate_lacam_data.py \
    --config configs/env_${fam}.yaml \
    --num-episodes $NUM_EPISODES --num-agents $NUM_AGENTS --map-size $MAP_SIZE \
    --backend lacam --num-workers $NUM_WORKERS --output-dir "$DATA_DIR" \
    >> "$GEN_LOG" 2>&1
done

# ---- 2. 校验四族 npz 齐全 ----
missing=0
for fam in "${FAMILIES[@]}"; do
  if [ ! -f "${DATA_DIR}/expert_${fam}.npz" ]; then
    echo "!! 缺失 ${DATA_DIR}/expert_${fam}.npz —— 中止，不启动训练" | tee -a "$GEN_LOG"
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "==== 数据生成不完整，退出 ====" | tee -a "$GEN_LOG"
  exit 1
fi
echo "==== [$(date '+%T')] 四族数据齐全，启动全量训练 ====" | tee -a "$GEN_LOG"

# ---- 3. 全量 DQfD+Reptile 元预训练（self-play 开，bc 1000，20000 iter）----
python scripts/train_meta.py \
  --config configs/train_reptile_dqfd.yaml \
  --data-dir "$DATA_DIR" \
  --bc-iters 1000 \
  --meta-iters 20000 \
  --log-interval 50 \
  --seed 42 \
  --ckpt-milestones 1000,5000,10000,15000,20000 \
  >> "$TRAIN_LOG" 2>&1

echo "==== [$(date '+%F %T')] 阶段二全量结束 ====" | tee -a "$TRAIN_LOG"
