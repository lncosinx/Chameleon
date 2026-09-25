# 专家数据重生成配方（2026-09-25 删除前存档）

> **背景**：实验全部收口后删除了 857G 专家数据（训练日志与检查点保留）。
> 本文件是恢复 `c2g_mix12`（主模型 MADCL-L 7.45M 的训练集）的唯一凭据。

> **⚠️ 重生成的数据与原数据同分布但不同一份**，原因见文末「可复现性边界」。

## 一、`c2g_mix12` 的族映射与逐族配方

| mix12 中的族名 | 真实目录 | map_type | num_agents | num_episodes | 落盘步数 | 生成耗时 |
|---|---|---|---|---|---|---|
| `expert_maze32` | `c2g_dense32m` | maze | 32 | 20000 | 1,148,177 | 0.47h |
| `expert_maze32b` | `c2g_gen2_maze32` | maze | 32 | 15000 | 864,494 | 0.37h |
| `expert_maze48` | `c2g_small_15k` | maze | 48 | 15000 | 1,175,027 | 1.81h |
| `expert_maze48b` | `c2g_gen2_maze48` | maze | 48 | 15000 | 1,197,050 | 0.71h |
| `expert_maze64` | `c2g_a64` | **maze**（见校正 A） | 64 | 20000 | 1,927,548 | 3.48h |
| `expert_maze64b` | `c2g_gen2_maze64` | maze | 64 | 15000 | 1,436,375 | 5.93h |
| `expert_random32` | `c2g_dense32` | random | 32 | 20000 | 631,251 | 0.16h |
| `expert_random32b` | `c2g_gen2_random32` | random | 32 | 30000 | 945,384 | 0.23h |
| `expert_random48` | `c2g_dense48` | random | 48 | 20000 | 699,341 | 0.30h |
| `expert_random48b` | `c2g_gen2_random48` | random | 48 | 30000 | 1,048,803 | 0.43h |
| `expert_random64` | `c2g_dense64` | random | 64 | 20000 | 775,674 | 0.49h |
| `expert_random64b` | `c2g_gen2_random64` | random | 64 | 30000 | 1,165,771 | 0.68h |

**合计 13,014,895 专家步**（约 13.01M），原始生成墙钟合计约 **15.1 小时**（`num_workers` 多为 20；本机 16 vCPU 需按 CLAUDE.md 的 nproc−4 上限下调，耗时会更长）。

## 二、质量指标（用于校验重生成是否落在同一分布）

| 族 | makespan_mean | soc_mean | wait_ratio_mean | fail_count | 顶点冲突率 |
|---|---|---|---|---|---|
| `expert_maze32` | 62.3794 | 805.7656 | 0.5525 | 1503 | 0.0 |
| `expert_maze32b` | 62.9319 | 808.4582 | 0.5535 | 1146 | 0.0 |
| `expert_maze48` | 99.6496 | 1484.6823 | 0.6338 | 2673 | 0.0 |
| `expert_maze48b` | 100.6508 | 1491.6249 | 0.6343 | 2524 | 0.0 |
| `expert_maze64` | 96.7888 | 2955.7469 | 0.5196 | 85 | 0.0 |
| `expert_maze64b` | 157.6213 | 2474.2967 | 0.6945 | 4484 | 0.0 |
| `expert_random32` | 32.2511 | 550.5749 | 0.459 | 427 | 0.0 |
| `expert_random32b` | 32.2008 | 549.8992 | 0.4589 | 641 | 0.0 |
| `expert_random48` | 36.7803 | 896.7608 | 0.4806 | 986 | 0.0 |
| `expert_random48b` | 36.7653 | 896.1777 | 0.4809 | 1473 | 0.0 |
| `expert_random64` | 42.6953 | 1299.7549 | 0.5079 | 1831 | 0.0 |
| `expert_random64b` | 42.6649 | 1299.0827 | 0.5074 | 2674 | 0.0 |

## 三、重生成命令

```bash
# ★ 必须用 lacam310 环境（_lacam.so 是 py3.10 编译；py3.12 下会静默回退到
#   差得多的 prioritized 规划器，污染数据且不报错）
conda activate lacam310
cd /root/Chameleon

# 每族一条。--seed-start 必须与下表一致，且不得越过 TEST_SEED_BASE（脚本内有检查）。
# --num-workers 按 CLAUDE.md：不超过 nproc-4，并配 nice 19 / ionice。
nice -n 19 python scripts/generate_lacam_data.py \
  --map-type maze --num-agents 32 \
  --num-episodes 20000 --seed-start <见下表> \
  --num-workers 12 --out c2g_dense32m   # expert_maze32
nice -n 19 python scripts/generate_lacam_data.py \
  --map-type maze --num-agents 32 \
  --num-episodes 15000 --seed-start <见下表> \
  --num-workers 12 --out c2g_gen2_maze32   # expert_maze32b
nice -n 19 python scripts/generate_lacam_data.py \
  --map-type maze --num-agents 48 \
  --num-episodes 15000 --seed-start <见下表> \
  --num-workers 12 --out c2g_small_15k   # expert_maze48
nice -n 19 python scripts/generate_lacam_data.py \
  --map-type maze --num-agents 48 \
  --num-episodes 15000 --seed-start <见下表> \
  --num-workers 12 --out c2g_gen2_maze48   # expert_maze48b
nice -n 19 python scripts/generate_lacam_data.py \
  --map-type maze --num-agents 64 \
  --num-episodes 20000 --seed-start <见下表> \
  --num-workers 12 --out c2g_a64   # expert_maze64  ★注意：不是 random，见校正 A
nice -n 19 python scripts/generate_lacam_data.py \
  --map-type maze --num-agents 64 \
  --num-episodes 15000 --seed-start <见下表> \
  --num-workers 12 --out c2g_gen2_maze64   # expert_maze64b
nice -n 19 python scripts/generate_lacam_data.py \
  --map-type random --num-agents 32 \
  --num-episodes 20000 --seed-start <见下表> \
  --num-workers 12 --out c2g_dense32   # expert_random32
nice -n 19 python scripts/generate_lacam_data.py \
  --map-type random --num-agents 32 \
  --num-episodes 30000 --seed-start <见下表> \
  --num-workers 12 --out c2g_gen2_random32   # expert_random32b
nice -n 19 python scripts/generate_lacam_data.py \
  --map-type random --num-agents 48 \
  --num-episodes 20000 --seed-start <见下表> \
  --num-workers 12 --out c2g_dense48   # expert_random48
nice -n 19 python scripts/generate_lacam_data.py \
  --map-type random --num-agents 48 \
  --num-episodes 30000 --seed-start <见下表> \
  --num-workers 12 --out c2g_gen2_random48   # expert_random48b
nice -n 19 python scripts/generate_lacam_data.py \
  --map-type random --num-agents 64 \
  --num-episodes 20000 --seed-start <见下表> \
  --num-workers 12 --out c2g_dense64   # expert_random64
nice -n 19 python scripts/generate_lacam_data.py \
  --map-type random --num-agents 64 \
  --num-episodes 30000 --seed-start <见下表> \
  --num-workers 12 --out c2g_gen2_random64   # expert_random64b
```

**⚠️ `seed_start` 未被 `lacam_stats_*.yaml` 记录**（该字段只写进 `generate_lacam_data.py`
的 json 分支，而落盘用的是 yaml 分支）。各族必须给**互不相交**的 seed 段，否则会产出重复
episode —— 本项目曾因此让「数据扩充无用」与「泛化 gap 0.009」两条结论作废。
建议按族顺序分配不相交段，例如第 k 族用 `--seed-start k*100000`，并用
`scripts/check_episode_overlap.py`（首帧哈希）验证跨族无重合。

## 三·五、★ 校正 A：`c2g_a64` 的 stats 元数据是错的

`c2g_a64` 的数据目录是 `expert_maze`，但其 stats 文件名为 `lacam_stats_random.yaml`
且内容写 `map_type: random`。**数据本身是 maze，元数据是错的。**

删除前用数据自身判别（障碍通道的密度与空间相关，`obs_patches[:,0,0]`，uint8/255）：

| 数据集 | 障碍密度 | 水平相关 | 垂直相关 |
|---|---|---|---|
| 已知 maze64（`gen2_maze64`） | 0.311 | +0.212 | +0.204 |
| 已知 maze32（`gen2_maze32`） | 0.315 | +0.211 | +0.206 |
| 已知 random64（`gen2_random64`） | 0.227 | +0.113 | +0.112 |
| 已知 random32（`gen2_random32`） | 0.226 | +0.108 | +0.117 |
| **`c2g_a64/expert_maze`** | **0.312** | **+0.207** | **+0.205** |

迷宫走廊使障碍成条状（相关 ~+0.21、密度 ~0.31），random 障碍近 i.i.d.（相关 ~+0.11、密度 ~0.23）。
⇒ `c2g_a64` 与已知 maze 逐项吻合。**重生成该族必须用 `--map-type maze`。**
若照 stats 写成 random，会在 `expert_maze64` 槽位静默填入 random 数据 —— 训练不会报错，
但主模型的 12 族混合会变成「两份 random64、缺一份 maze64」。

## 四、可复现性边界（重要）

重生成的数据**与原数据同分布，但不是同一份**，有两处硬性不确定源：

1. **LaCAM 求解种子**（`lacam/lacam_runner.py`）：历史行为是 `np.random.randint(...)`
   从 numpy 全局 RNG 现抽，随进程数与调度顺序而变 → 同一实例可能得到**同样合法但不同的**
   最优解。**2026-09-25 已加 `solver_seed` 参数**：传整数即确定性求解，传 `None` 沿用历史行为。
   重生成时应显式传入由 episode seed 派生的值，使数据自身可复现。
2. **`time_limit`（默认 10s）**：LaCAM 是 anytime 搜索，换机器或负载不同仍可能返回
   不同质量的解。要完全可复现需固定种子**并**保证求解不触顶时限。

**后果**：在重生成数据上训练的新臂，**不能与历史臂做唯一变量对比**
（`capsmall` 8.682 / `smamba` 9.016 / `mdconv1` 8.997 / `caplr5e4` 8.909 等数字都建立在原数据上）。
若要继续这些对照，必须**整组重训**，或改为只在新数据内部自成一套对照。

## 五、未保留的其他数据集（已收口实验用过，未记配方）

`c2g_gen3_*`（各族）、`c2g_wh96` / `c2g_wh192`（warehouse）、`dagger_r2`、`c2g_heldout`、
`c2g_small_15k` 以外的 small 系列等，合计约 477G。对应实验均已收口且结论已写入 `SCHEDULE.md`，
如需复现请按各自已定论条目里记的族与规模重新生成。
