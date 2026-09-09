#!/usr/bin/env python3
"""
D4 增强正确性单测（M1）。**独立验证**：不是"用同一套公式自己验自己"，而是
把**整张地图物理旋转/镜像**后重新走一遍真实观测构建管线，与 d4_augment 的变换结果比对。

校验三件事：
  1. transform_patch(观测) == 在物理旋转后的地图上重新构建的观测（含 cost2go/足迹/原生 3 通道）
  2. build_action_perm 把"专家动作"（指向 BFS 距离更小的邻格）映射到旋转后地图上的同一物理动作
  3. scalars 的方向向量、prev-action one-hot、动作掩码同步正确

跑：python scripts/test_d4_augment.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

from pogema import GridConfig, pogema_v0
from envs.observation_builder import ObservationBuilder
from envs.action_mask import build_action_mask_batch
from envs.potential_field import PotentialFieldCache as BFSDistanceCache
from envs.d4_augment import (transform_patch, transform_dir, build_action_perm,
                             _ELEMS, ACTION_DELTAS, d4_augment_batch)

R = 5


def rot_grid(chars: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(chars, k=k) if k % 4 else chars
    return np.fliplr(out) if flip else out


def rot_point(r: int, c: int, H: int, W: int, k: int, flip: bool):
    """点在整图旋转/镜像下的新坐标（与 np.rot90/np.fliplr 一致）。"""
    for _ in range(k % 4):
        r, c, H, W = W - 1 - c, r, W, H     # np.rot90: (r,c)->(W-1-c, r)，形状转置
    if flip:
        c = W - 1 - c
    return r, c, H, W


def make_obs(chars: np.ndarray, apos, gpos, prev_action: int):
    """在给定地图上放 1 个 agent + 目标，跑真实观测构建管线，返回 (patch, scalars, mask)。"""
    map_str = "\n".join("".join(row) for row in chars)
    gc = GridConfig(map=map_str, agents_xy=[list(apos)], targets_xy=[list(gpos)],
                    num_agents=1, obs_radius=R, on_target="nothing",
                    max_episode_steps=64, collision_system="soft")
    env = pogema_v0(gc)
    obs_list, _ = env.reset()
    ob = ObservationBuilder(num_agents=1, obs_radius=R, use_cost2go=True, cost2go_clip=10.0)
    ob.reset()
    obstacles = (chars == "#").astype(np.int32)
    patch, scal = ob.build(obs_list, [tuple(apos)], [tuple(gpos)],
                           np.array([prev_action]), obstacles=obstacles)
    mask = build_action_mask_batch(np.array([[apos[0], apos[1]]], dtype=np.int32)[None],
                                   obstacles)[0]
    return patch, scal, mask, obstacles


def expert_action_set(obstacles: np.ndarray, apos, gpos) -> set:
    """最优动作**集合** = 所有走向 BFS 距离最小合法邻格的动作。
    用集合而非单个动作：并列最优时"取第一个"本身不是等变的（是测试脚手架的
    tie-break 顺序，不是被测代码的问题），比较集合才是对置换正确性的干净检验。"""
    dist = BFSDistanceCache().get(obstacles, tuple(gpos))
    H, W = obstacles.shape
    cand = {}
    for a, (dr, dc) in enumerate(ACTION_DELTAS):
        r, c = apos[0] + dr, apos[1] + dc
        if not (0 <= r < H and 0 <= c < W) or obstacles[r, c]:
            continue
        cand[a] = float(dist[r, c])
    best = min(cand.values())
    return {a for a, d in cand.items() if d == best}


def main():
    rng = np.random.default_rng(7)
    n_checked = 0
    for trial in range(12):
        H = W = 17
        chars = np.where(rng.random((H, W)) < 0.22, "#", ".")
        free = np.argwhere(chars == ".")
        if len(free) < 2:
            continue
        i, j = rng.choice(len(free), size=2, replace=False)
        apos, gpos = tuple(free[i]), tuple(free[j])
        obstacles = (chars == "#").astype(np.int32)
        dist = BFSDistanceCache().get(obstacles, gpos)
        if dist[apos[0], apos[1]] > 1e8:
            continue                                   # 不可达，跳过
        prev = int(rng.integers(0, 5))

        p0, s0, m0, _ = make_obs(chars, apos, gpos, prev)
        a0 = expert_action_set(obstacles, apos, gpos)

        for gi, (k, flip) in enumerate(_ELEMS):
            # --- 参照系：物理旋转整张地图后重新构建观测 ---
            chars_t = rot_grid(chars, k, flip)
            ar, ac, _, _ = rot_point(apos[0], apos[1], H, W, k, flip)
            gr, gc_, _, _ = rot_point(gpos[0], gpos[1], H, W, k, flip)
            perm = build_action_perm(k, flip)
            prev_t = int(perm[prev])
            p1, s1, m1, obst_t = make_obs(chars_t, (ar, ac), (gr, gc_), prev_t)
            a1 = expert_action_set(obst_t, (ar, ac), (gr, gc_))

            # --- 待验：直接对观测做 D4 变换 ---
            p_aug = transform_patch(torch.from_numpy(p0), k, flip).numpy()
            s_aug = s0.copy()
            s_aug[..., 0:2] = transform_dir(torch.from_numpy(s0[..., 0:2]), k, flip).numpy()
            s_aug[..., 2:7] = np.zeros_like(s0[..., 2:7])
            s_aug[..., 2 + np.asarray(perm)] = s0[..., 2:7]
            m_aug = np.zeros_like(m0)
            m_aug[..., np.asarray(perm)] = m0

            assert np.allclose(p_aug, p1, atol=1e-5), \
                f"patch 不匹配 g={gi}(k={k},flip={flip}) 最大差 {np.abs(p_aug-p1).max()}"
            assert np.allclose(s_aug, s1, atol=1e-5), \
                f"scalars 不匹配 g={gi}: {s_aug} vs {s1}"
            assert np.array_equal(m_aug, m1), f"动作掩码不匹配 g={gi}"
            mapped = {int(perm[a]) for a in a0}
            assert mapped == a1, \
                f"动作置换错 g={gi}(k={k},flip={flip}): perm({a0})={mapped} != {a1}"
            n_checked += 1

    assert n_checked >= 40, f"有效校验样本太少 ({n_checked})，测试没覆盖到"

    # 批量接口形状/一致性
    B, T, N = 4, 3, 6
    pt = torch.rand(B, T, N, 5, 11, 11)
    sc = torch.rand(B, T, N, 8); sc[..., 2:7] = 0
    sc[..., 2 + torch.randint(0, 5, (B, T, N))] = 1.0
    ac = torch.randint(0, 5, (B, T, N))
    mk = torch.ones(B, T, N, 5, dtype=torch.bool)
    p, s, a, m = d4_augment_batch(pt, sc, ac, mk, generator=torch.Generator().manual_seed(0))
    assert p.shape == pt.shape and s.shape == sc.shape and a.shape == ac.shape and m.shape == mk.shape
    assert s[..., 7].equal(sc[..., 7]), "arrived 标志不应被变换"

    print(f"✅ D4 增强单测通过（{n_checked} 个 (样本 × 群元) 组合，8 个群元全覆盖）")


if __name__ == "__main__":
    main()
