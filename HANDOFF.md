# 换机交接（2026-09-16，GPU 已全停）

## 一、必须带走的东西

| 内容 | 路径 | 大小 | 说明 |
|---|---|---|---|
| **全部实验结果** | `/root/autodl-tmp/handoff/eval_v5/` | 480 KB | 四个 SQ 的所有 json，论文表格的唯一数据源 |
| **关键 ckpt** | `/root/autodl-tmp/handoff/ckpt/` | 62 MB | it_act（原主模型）、capbig/capsmall（容量 A/B）、v6A/v6B（隔离实验） |
| **训练日志** | `/root/autodl-tmp/handoff/*.log` | <1 MB | 容量 A/B + lr 扫描的原始记录 |
| **代码** | 本仓库（git） | — | 已全部提交 |

**合计 62 MB。** 论文需要的一切都在这里，不依赖任何专家数据。

## 二、专家数据（整容器迁移 → 自动跟着走，无需操作）

`/root/autodl-tmp/chameleon_data/` 共 **526 GB**，整机迁移会一并带过去。

仅当你想**换更小的数据盘来省钱**时才需要清理。可安全删除的部分：

| 目录 | 大小 | 能不能删 |
|---|---|---|
| `c2g_mix12` 比 `c2g_mix6` 多出的 6 个族（`*b` 批次） | ~197 GB | ⚠️ 2026-09-17 更正：这批**不是**「没价值的新数据」，而是**大部分重复**（random 66%、maze 89~98% 与原批逐 episode 相同，seed 写死所致）。random 的 b 批仍有约 1/3 新 episode，删前先想清楚 |
| `dagger_r2` | 37 GB | **可删**。DAgger 经测定为 −0.141，是 v6 退化的主因 |
| `c2g_small_15k`、`c2g_a64` 等早期批次 | ~100 GB | 视情况，已不在任何现行配方里 |

保留 `c2g_mix6`（182 GB，it_act 主模型用的就是它）+ `c2g_whmix`（110 GB，SQ2 warehouse）
即可覆盖全部现行实验，约 **292 GB**。

注意 `c2g_mix6/mix12/whmix` 是**符号链接目录**，真实数据在 `c2g_dense*` / `c2g_gen2_*` /
`c2g_wh*` 下，清理前务必 `readlink -f` 解析，别把还在用的目标删掉。

## 三、停在哪里 / 下一步

**已完成**：SQ1/SQ2/SQ3/SQ4 全部结果、隔离实验（3 臂）、容量 A/B（4 里程碑 + 完整协议）。

**中断点**：容量臂续训 8k→24k，刚跑 9 分钟就停了。`ckpt_il/capbig_iter_8000.pt` 是完好断点，新机上用
`--resume ckpt_il/capbig_iter_8000.pt` 可无缝接上（里程碑计数会对齐）。

**它要回答的问题**（尚未有答案，不要当成已知结论）：
1. mazes 的 +0.185 优势能否随训练扩大到统计显著（目前 6/6 档同向，但只有 2 档超出 95% CI）；
2. random 高密度两档（−0.034 / −0.047，均未超出 CI）会不会随训练转正。

**换机后最省钱的做法**：先只带 62 MB 出结果和写论文；确需继续训练时再迁 mix6 或重生成。

## 四、本轮修掉的两个真问题（新机上仍然有效，已提交）

1. **fp16 通信注意力溢出 → NaN**（commit 8f4bbb4）。`matmul(Q,K^T)*scale` 缩放在乘之后，
   fp16 累加越过 65504 → +inf → entmax 里 inf-inf=NaN。修法是先缩放 + **显式
   `torch.autocast(enabled=False)`**（只写 `.float()` 无效，autocast 会强制降回 fp16）。
   回归测试：`scripts/test_comm_fp16_overflow.py`。
2. **大网从零训必须用 lr 1e-4**。5e-4 下 7.45M 模型第 250 步即塌缩到 `wait_pred=1.000`
   （全 wait，CSR=0），grad_clip 无关（10 → 1.0 无任何差别）。配置见
   `configs/train_il_cap_lr1e4.yaml`。小网从零训不受影响。
