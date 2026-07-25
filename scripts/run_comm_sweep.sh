#!/usr/bin/env bash
# 通信层数消融 sweep：依次训练 comm_rounds = 0/1/2/3（无通信 / 1 / 2 / 3 层）。
#
# 结构（按用户要求）：
#   - Reptile 任务 = when-to-switch 原生 5 族 (random/mazes/sc1/street/wc3)
#   - 一个 meta-iter 4 个 task，两卡各 2 个 task，两卡**顺序执行不并发**(trainer 内已实现)
#   - 前期纯专家 BC = 2000 个 meta-iter (--bc-iters)
#   - 默认 meta-iters == bc-iters → 全程纯 BC（纯模仿）。要加 DQfD 价值阶段：
#       提高 META_ITERS（> BC_ITERS）并去掉 --no-selfplay。
#
# 用法:
#   bash scripts/run_comm_sweep.sh                 # 默认 comm 0,1,2,3
#   ROUNDS="1 2" bash scripts/run_comm_sweep.sh    # 只跑部分
#   META_ITERS=20000 SELFPLAY=1 bash scripts/run_comm_sweep.sh   # 加 DQfD 阶段
set -euo pipefail
cd "$(dirname "$0")/.."

ROUNDS="${ROUNDS:-0 1 2 3}"
BC_ITERS="${BC_ITERS:-2000}"
META_ITERS="${META_ITERS:-2000}"          # 默认 = BC_ITERS → 纯 BC
FAMILIES="${FAMILIES:-random,mazes,sc1,street,wc3}"
GPUS="${GPUS:-0,1}"
SEED="${SEED:-42}"
SELFPLAY="${SELFPLAY:-0}"                  # 0=纯 BC 关自游走；1=开（仅 DQfD 阶段有意义）
MILESTONES="${MILESTONES:-500,1000,1500,2000}"
LOGDIR="${LOGDIR:-logs/comm_sweep}"
mkdir -p "$LOGDIR" checkpoints

SP_FLAG="--no-selfplay"
[ "$SELFPLAY" = "1" ] && SP_FLAG=""

echo "[sweep] rounds=[$ROUNDS] bc_iters=$BC_ITERS meta_iters=$META_ITERS families=$FAMILIES gpus=$GPUS selfplay=$SELFPLAY"
declare -A STATUS
for R in $ROUNDS; do
  TAG="comm${R}"
  LOG="$LOGDIR/${TAG}.log"
  echo "===== [sweep] comm_rounds=$R  tag=$TAG  log=$LOG  $(date '+%F %T') ====="
  # 单配置失败不中断整条 sweep（set +e 局部包裹）；--resume auto 支持断点续跑。
  set +e
  python scripts/train_meta.py \
    --config configs/train_reptile_dqfd.yaml \
    --network-config configs/network_biintent_drqn.yaml \
    --comm-rounds "$R" \
    --families "$FAMILIES" \
    --meta-iters "$META_ITERS" \
    --bc-iters "$BC_ITERS" \
    --gpus "$GPUS" \
    --seed "$SEED" \
    --tag "$TAG" \
    --ckpt-milestones "$MILESTONES" \
    --resume auto \
    $SP_FLAG \
    > "$LOG" 2>&1
  rc=$?
  set -e
  STATUS[$R]=$rc
  echo "===== [sweep] comm_rounds=$R exit=$rc  $(date '+%F %T') ====="
done
echo "[sweep] ALL DONE. 结果:"
for R in $ROUNDS; do echo "  comm_rounds=$R -> exit=${STATUS[$R]}  (checkpoints/comm${R}_*.pt, $LOGDIR/comm${R}.log)"; done
