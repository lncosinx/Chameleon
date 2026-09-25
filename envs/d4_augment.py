"""
D4 对称数据增强（M1）。

网格 MAPF 在二面体群 D4（4 旋转 × 是否镜像 = 8 个元素）下等变：把整个局部观测旋转/镜像，
专家的正确动作也随之置换。故一条专家样本等价于 8 条 —— 在数据量是硬瓶颈时（我们 2.6M steps
vs MAPF-GPT 1B）这是零成本的 8× 有效数据。

**必须同步变换的东西**（漏一个就是整批标签错位且不报错，见 CLAUDE.md INVARIANT 1）：
  1. patch (…,C,K,K)         —— 空间旋转/镜像
  2. 专家动作标签 (…,)        —— 动作置换
  3. 动作掩码 (…,5)           —— 同一置换
  4. scalars[…, 0:2]          —— 目标方向向量（2D 旋转）
  5. scalars[…, 2:7]          —— 上一步动作 one-hot（同一置换）
  6. scalars[…, 7]            —— arrived 标志，**不变**
  7. positions                —— 通信只用 pairwise 距离（旋转不变），故**不需要**变换

动作序（INVARIANT 1）：0=wait(0,0), 1=up(-1,0), 2=down(+1,0), 3=left(0,-1), 4=right(0,+1)。
坐标：r 向下、c 向右，patch 以自身居中。

变换定义（与 torch.rot90(x, k, dims=(-2,-1)) / torch.flip(x, dims=(-1,)) 严格对应，
推导见 tests：rot90 k=1 ⇒ 偏移 (r,c) → (-c, r)；水平镜像 ⇒ (r,c) → (r,-c)）。
"""
from __future__ import annotations

import torch

# 5 个动作的位移（INVARIANT 1）
ACTION_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _delta_to_action(dr: int, dc: int) -> int:
    for a, (r, c) in enumerate(ACTION_DELTAS):
        if (r, c) == (dr, dc):
            return a
    raise ValueError(f"位移 {(dr, dc)} 不是合法动作")


def _rot_delta(dr: int, dc: int, k: int):
    """逆时针旋转 k 次 90°：(r,c) → (-c, r)。"""
    for _ in range(k % 4):
        dr, dc = -dc, dr
    return dr, dc


def build_action_perm(k: int, flip: bool) -> torch.Tensor:
    """返回 perm，使得 new_action = perm[old_action]。"""
    perm = []
    for a, (dr, dc) in enumerate(ACTION_DELTAS):
        r, c = _rot_delta(dr, dc, k)
        if flip:
            c = -c                      # 水平镜像（翻列）
        perm.append(_delta_to_action(r, c))
    return torch.tensor(perm, dtype=torch.long)


# 8 个群元的置换表（模块级缓存，避免每 batch 重建）
_PERMS = [build_action_perm(k, f) for f in (False, True) for k in range(4)]
_ELEMS = [(k, f) for f in (False, True) for k in range(4)]


def transform_patch(patch: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """patch (..., C, K, K)。先旋转再镜像（与 build_action_perm 的顺序一致）。"""
    out = torch.rot90(patch, k=k, dims=(-2, -1)) if k % 4 else patch
    if flip:
        out = torch.flip(out, dims=(-1,))
    return out


def transform_dir(vec: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """目标方向向量 (..., 2) = (dr, dc)，与位移同变换。"""
    dr, dc = vec[..., 0], vec[..., 1]
    for _ in range(k % 4):
        dr, dc = -dc, dr
    if flip:
        dc = -dc
    return torch.stack([dr, dc], dim=-1)


def d4_augment_batch(patches: torch.Tensor, scalars: torch.Tensor,
                     actions: torch.Tensor, masks: torch.Tensor,
                     generator: torch.Generator = None):
    """
    对一个训练 batch 做 D4 增强。**按 batch 维逐样本随机取群元**（同一序列内保持同一变换，
    否则时间维语义会断裂）。

    Args:
        patches (B,T,N,C,K,K) / scalars (B,T,N,8) / actions (B,T,N) / masks (B,T,N,5)
    Returns: 同形状的增强版本
    """
    B = patches.shape[0]
    dev = patches.device
    g_idx = torch.randint(0, 8, (B,), device=dev, generator=generator)

    p_out = patches.clone()
    s_out = scalars.clone()
    a_out = actions.clone()
    m_out = masks.clone()

    for gi in range(8):
        sel = (g_idx == gi)
        if not bool(sel.any()):
            continue
        k, flip = _ELEMS[gi]
        if k == 0 and not flip:
            continue                        # 恒等元
        perm = _PERMS[gi].to(dev)
        p_out[sel] = transform_patch(patches[sel], k, flip)
        s = scalars[sel]
        new_s = s.clone()
        new_s[..., 0:2] = transform_dir(s[..., 0:2], k, flip)
        # prev-action one-hot：new_onehot[perm[a]] = old_onehot[a] ⇒ index_copy 到 perm 位置
        new_s[..., 2:7] = torch.zeros_like(s[..., 2:7]).index_copy(-1, perm, s[..., 2:7])
        s_out[sel] = new_s
        a_out[sel] = perm[actions[sel]]
        m_out[sel] = torch.zeros_like(masks[sel]).index_copy(-1, perm, masks[sel])

    return p_out, s_out, a_out, m_out
