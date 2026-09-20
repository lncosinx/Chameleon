#!/usr/bin/env python3
"""
v5 统一评测台（SCHEDULE.md §4 协议）。所有 SQ / 所有臂共用这一个入口。

协议（写死，不再各脚本各改）：
  POGEMA soft-collision，on_target=nothing（标准 MAPF：到达即守位），max_episode_steps=128，
  地图 = marl-gpt/POGEMA validation-*（默认全 128 张/族），agents ∈ {8,16,24,32,48,64}，
  指标 CSR / ISR / SoC / makespan / Deadlock + 通信侧 AvgActiveNeighbors / EdgeFlipRate。

与旧 scripts/eval_marlgpt_il.py 的区别：
  1. **两族统一 128 步**（旧脚本 random 用 256、mazes 用 128，不可比，已废止）；
  2. **批量 rollout**：同一 agent 数的 E 个 episode 合成一次 (E,1,N,...) 前向（GPU 利用率↑，
     热路径无 per-agent Python 循环，符合 CLAUDE.md 性能约定）；
  3. **推断期死锁逃逸（M2）**：`--action-mode {argmax,sample,escape}`。escape = 只对"卡住"的体
     按温度采样打破对称僵持；卡住判据用**自身**到**自身目标**的测地距离是否长时间无改善
     （与 cost2go 通道同一合法假设：已知静态地图 + 自身目标，不含他体/未来信息，INVARIANT 2/3）。
  4. 多 seed、通信统计、JSON 输出。

用法：
  python scripts/eval_bench.py --ckpt ckpt_il/xxx.pt --families random,mazes \
      --num-maps 128 --action-mode escape --out /root/autodl-tmp/eval/xxx.json
"""
from __future__ import annotations
import sys, os, argparse, json, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, yaml

from utils.seed import set_seed
from envs.pogema_env_wrapper import PogemaWrapper
from envs.observation_builder import ObservationBuilder, build_observations
from envs.action_mask import build_action_mask_batch
from scripts.train_il_meta import build_net
from envs.d4_augment import transform_patch, transform_dir, _PERMS, _ELEMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARLGPT = os.path.join(ROOT, "marl-gpt", "eval_configs")
FAMILY_DIR = {"random": "01-random", "mazes": "02-mazes",
              "warehouse": "03-warehouse", "movingai": "04-movingai"}
# 各族的官方网格（pogema-benchmark raw_data_MAPF/*/*.yaml）：族之间协议并不相同，
# 照搬 random 的设置去跑 warehouse/movingai 得到的数字**不可与 leaderboard 比**。
FAMILY_PROTOCOL = {
    "random":    dict(agents=[8, 16, 24, 32, 48, 64],    maps=128, seeds="0",     steps=128),
    "mazes":     dict(agents=[8, 16, 24, 32, 48, 64],    maps=128, seeds="0",     steps=128),
    "warehouse": dict(agents=[32, 64, 96, 128, 160, 192], maps=1,  seeds="0-127", steps=128),
    "movingai":  dict(agents=[64, 128, 192, 256],        maps=128, seeds="0",     steps=256),
}


# ---------------------------------------------------------------------------
# 地图
# ---------------------------------------------------------------------------
def load_maps(family: str, num_maps: int, maps_file: str = None):
    # maps_file：调参用的独立 dev 地图集（scripts/make_dev_maps.py 生成）。
    # 推断期超参必须在 dev 上调，报告只用 validation-*，否则是拟合测试集。
    src = maps_file or os.path.join(MARLGPT, FAMILY_DIR[family], "maps.yaml")
    maps = yaml.safe_load(open(src))
    names = sorted(maps.keys())
    if 0 < num_maps < len(names):
        idx = sorted(set(np.linspace(0, len(names) - 1, num_maps).round().astype(int).tolist()))
        names = [names[i] for i in idx]
    return [(n, maps[n]) for n in names]


def make_env(map_str: str, name: str, num_agents: int, max_steps: int, on_target: str, seed: int):
    rows = map_str.splitlines()
    cfg = dict(map_type="random", fixed_map=map_str, fixed_map_name=name,
               map_size=max(len(rows), len(rows[0])), num_agents=num_agents, obs_radius=5,
               max_episode_steps=max_steps, collision_system="soft",
               # split="all"：不做 train/test seed 重映射，直接用原始 seed —— 这样起终点
               # 布局与 pogema-benchmark leaderboard（seed: 0）**逐实例一致**，我们的数字
               # 与 SCRIMP/DCC/LaCAM 是配对可比的，而不是"同图不同实例"。
               # 地图本身是外部固定的 validation-* 集，训练用的是在线 generator，无泄漏风险。
               on_target=on_target, split="all")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f); p = f.name
    try:
        w = PogemaWrapper(p)
    finally:
        os.unlink(p)
    obs, _ = w.reset(seed=seed)
    return w, obs


# ---------------------------------------------------------------------------
# 批量 rollout（同 agent 数的 E 个 episode 一起跑）
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_chunk(net, specs, num_agents, max_steps, device, use_c2g, c2g_clip,
                  on_target="nothing", action_mode="argmax", escape_k=3, escape_temp=1.0,
                  escape_no_wait_after=8, base_temp=0.3, goal_hold="none", tta=False, rng=None,
                  reset_hidden_every=0, zero_channels=()):
    """specs: [(map_str, name, seed)] 长度 E，全部同 num_agents。返回每个 episode 的指标 dict。"""
    E, N = len(specs), num_agents
    rng = rng or np.random.default_rng(0)
    envs, obs_l, builders = [], [], []
    for map_str, name, seed in specs:
        w, obs = make_env(map_str, name, N, max_steps, on_target, seed)
        ob = ObservationBuilder(num_agents=N, obs_radius=5, use_cost2go=use_c2g,
                                cost2go_clip=c2g_clip)
        ob.reset()
        envs.append(w); obs_l.append(obs); builders.append(ob)

    prev = np.zeros((E, N), dtype=np.int64)
    net.reset_comm_state()

    goals = [np.array(w.goals or w.get_targets_xy(), dtype=np.int32) for w in envs]
    grids = [w.global_grid for w in envs]
    # 每体到自身目标的测地距离场（与 cost2go 通道同源、同假设；仅用于"卡住"判据与诊断）
    dist_fields = [[builders[e]._c2g_cache.get(grids[e], tuple(goals[e][i])) for i in range(N)]
                   for e in range(E)]

    best_d = np.full((E, N), np.inf, dtype=np.float32)   # 历史最优（最小）测地距离
    no_improve = np.zeros((E, N), dtype=np.int32)        # 连续无改善步数
    at_goal_now = np.zeros((E, N), dtype=bool)           # 当前是否站在自身目标上
    at_goal_hist = np.zeros((E, N, max_steps + 1), dtype=bool)
    n_escape = 0
    comm_neighbors, comm_flips = [], []

    for t in range(max_steps):
        patches = np.empty((E, N, (5 if use_c2g else 4), 11, 11), dtype=np.float32)
        scalars = np.empty((E, N, 8), dtype=np.float32)
        masks = np.ones((E, N, 5), dtype=bool)
        pos = np.zeros((E, N, 2), dtype=np.float32)
        for e, w in enumerate(envs):
            axy = w.get_agents_xy()
            p_, s_ = build_observations(w, obs_l[e], prev[e], builders[e])
            patches[e], scalars[e] = p_, s_
            a_arr = np.array(axy, dtype=np.int32)
            pos[e] = a_arr
            if grids[e] is not None:
                masks[e] = build_action_mask_batch(a_arr[np.newaxis], grids[e])[0]
            # 进度 / 到达统计
            H, W = grids[e].shape
            for i in range(N):
                r, c = int(a_arr[i, 0]), int(a_arr[i, 1])
                d = float(dist_fields[e][i][r, c]) if (0 <= r < H and 0 <= c < W) else np.inf
                at_goal_now[e, i] = (d <= 0.0)
                if d <= 0.0:
                    # 已在目标：nothing 协议下必须守位，**永不判定为卡住**（否则会被逃逸
                    # 机制赶离目标 → CSR 归零。这是第一版实现的 bug，勿回退）。
                    best_d[e, i] = 0.0; no_improve[e, i] = 0
                elif d < best_d[e, i]:
                    best_d[e, i] = d; no_improve[e, i] = 0
                else:
                    no_improve[e, i] += 1
            at_goal_hist[e, :, t] = np.all(a_arr == goals[e], axis=1)

        # 通道消融探针：把指定观测通道置零（3=时间衰减足迹，4=cost2go），
        # 用来看空间编码到底依赖哪些输入、以及它们与 GRU 记忆的分工。
        if zero_channels:
            patches[:, :, list(zero_channels)] = 0

        # 记忆消融探针：每 reset_hidden_every 步清零时间核状态（=1 相当于无记忆逐帧决策）。
        # 用来分离「瓶颈在空间编码还是在时序记忆」——通信迟滞状态不动，只砍 GRU 记忆。
        if reset_hidden_every and t % reset_hidden_every == 0:
            net.temporal.reset_state()

        pt = torch.from_numpy(patches).unsqueeze(1).to(device)      # (E,1,N,C,K,K)
        st = torch.from_numpy(scalars).unsqueeze(1).to(device)
        po = torch.from_numpy(pos).unsqueeze(1).to(device)
        mt = torch.from_numpy(masks).unsqueeze(1).to(device)
        if tta:
            # 测试期 D4 增强：把同一真实状态的 8 个对称视角一起前向，各自逆置换回原动作空间
            # 后对**概率**求平均。8 个视角各自维护自己的 GRU/通信状态（它们是同一条真实轨迹
            # 的不同表示），故沿 batch 维扩成 E*8，跨步保持对应关系。
            pv, sv, mv = [], [], []
            for k, flip in _ELEMS:
                pv.append(transform_patch(pt, k, flip))
                s2 = st.clone()
                s2[..., 0:2] = transform_dir(st[..., 0:2], k, flip)
                perm = _PERMS[_ELEMS.index((k, flip))].to(device)
                s2[..., 2:7] = torch.zeros_like(st[..., 2:7]).index_copy(-1, perm, st[..., 2:7])
                sv.append(s2)
                mv.append(torch.zeros_like(mt).index_copy(-1, perm, mt))
            logits, info = net(torch.cat(pv, 0), torch.cat(sv, 0),
                               po.repeat(8, 1, 1, 1), action_masks=torch.cat(mv, 0),
                               reset_comm=False, stateful=True)
            probs_sum = 0.0
            for gi, (k, flip) in enumerate(_ELEMS):
                perm = _PERMS[gi].to(device)
                lp = logits[gi * E:(gi + 1) * E, 0]                  # (E,N,A) 在变换后的动作空间
                pr = torch.softmax(lp, dim=-1)
                # 逆置换回原动作空间：perm[a_orig] = a_transformed ⇒ p_orig[a] = p_trans[perm[a]]
                probs_sum = probs_sum + pr[..., perm]
            probs_avg = probs_sum / 8.0
            lg = torch.log(probs_avg.clamp_min(1e-12))               # 回到 logit 尺度
            lg = lg.masked_fill(~torch.from_numpy(masks).to(device), float("-inf"))
        else:
            logits, info = net(pt, st, po, action_masks=mt, reset_comm=False, stateful=True)
            lg = logits[:, 0]                                        # (E,N,A)
        comm_neighbors.append(info.get("avg_neighbors", 0.0))
        comm_flips.append(info.get("edge_flip_rate", 0.0))

        if action_mode == "argmax":
            acts = lg.argmax(dim=-1).cpu().numpy()
        elif action_mode == "sample":
            probs = torch.softmax(lg / escape_temp, dim=-1)
            acts = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1
                                     ).reshape(E, N).cpu().numpy()
            if goal_hold != "none":
                # 已在目标的体不再采样（nothing 协议要守位；采样会把它随机走开）。
                # greedy=退回 argmax（仍可能主动让路）；wait=强制原地。
                hold = (np.zeros_like(acts) if goal_hold == "wait"
                        else lg.argmax(dim=-1).cpu().numpy())
                acts = np.where(at_goal_now, hold, acts)
        elif action_mode == "escape2":
            # 分层温度：没卡住的体用**低温**采样（近贪心，保住路径质量，但仍能打破对称并列），
            # 卡住的体用**高温**采样（真正跳出僵局）。动机：纯采样(T=1)虽然破死锁，但给所有体
            # 都加了噪声 → 路径变长（实测 SoC 比 MAPF-GPT-2M 高 50%）；纯 argmax 又会死锁。
            stuck = torch.from_numpy(no_improve >= escape_k).to(device)
            temp = torch.where(stuck, escape_temp, base_temp).unsqueeze(-1)
            lg2 = lg.clone()
            hard = torch.from_numpy(no_improve >= escape_no_wait_after).to(device)
            legal_move = torch.from_numpy(masks[:, :, 1:].any(-1)).to(device)
            lg2[:, :, 0] = torch.where(hard & legal_move,
                                       torch.full_like(lg2[:, :, 0], float("-inf")),
                                       lg2[:, :, 0])
            probs = torch.nan_to_num(torch.softmax(lg2 / temp, dim=-1), nan=0.0)
            bad = probs.sum(-1) <= 0
            if bad.any():
                probs = torch.where(bad.unsqueeze(-1), torch.from_numpy(masks).to(device).float(), probs)
            acts = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(E, N).cpu().numpy()
            n_escape += int(stuck.sum())
        else:  # escape：仅对卡住的体采样
            acts = lg.argmax(dim=-1).cpu().numpy()
            stuck = no_improve >= escape_k                          # (E,N)
            if stuck.any():
                lg2 = lg.clone()
                # 卡很久还没脱困 → 禁 wait，强制移动（打破"全员互等"）
                hard = torch.from_numpy(no_improve >= escape_no_wait_after).to(device)
                legal_move = torch.from_numpy(masks[:, :, 1:].any(-1)).to(device)
                lg2[:, :, 0] = torch.where(hard & legal_move,
                                           torch.full_like(lg2[:, :, 0], float("-inf")),
                                           lg2[:, :, 0])
                probs = torch.softmax(lg2 / escape_temp, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0)
                bad = probs.sum(-1) <= 0
                if bad.any():   # 全 -inf 兜底：均匀合法
                    fallback = torch.from_numpy(masks).to(device).float()
                    probs = torch.where(bad.unsqueeze(-1), fallback, probs)
                samp = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1
                                         ).reshape(E, N).cpu().numpy()
                acts = np.where(stuck, samp, acts)
                n_escape += int(stuck.sum())

        for e, w in enumerate(envs):
            obs_l[e], _, _, _ = w.step(acts[e].tolist())
        prev = acts.astype(np.int64)

    # ---- 终局指标 ----
    out = []
    for e, w in enumerate(envs):
        final = np.array(w.get_agents_xy(), dtype=np.int32)
        at_goal = np.all(final == goals[e], axis=1)
        # 到达时刻：最后一次"离开目标"之后的第一步（nothing 协议下守位）
        arrive_t = np.full(N, float(max_steps), dtype=np.float32)
        for i in range(N):
            if at_goal[i]:
                hist = at_goal_hist[e, i, :max_steps]
                notat = np.nonzero(~hist)[0]
                arrive_t[i] = float(notat[-1] + 1) if len(notat) else 0.0
        out.append(dict(
            CSR=float(at_goal.all()), ISR=float(at_goal.mean()),
            # SoC 用**总和**（对齐 pogema-benchmark 的 SoC 口径：8体 ep_length=22 → SoC=84），
            # makespan = 最后一个到达时刻（= 它们的 ep_length）
            SoC=float(arrive_t.sum()), makespan=float(arrive_t.max()),
            escape_frac=float(n_escape) / max(1, E * N * max_steps),
            avg_neighbors=float(np.mean(comm_neighbors)) if comm_neighbors else 0.0,
            edge_flip=float(np.mean(comm_flips)) if comm_flips else 0.0,
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--network-config", default="configs/network_gru_c2g.yaml")
    ap.add_argument("--set-net", action="append", default=[], metavar="K=V")
    ap.add_argument("--families", default="random,mazes")
    ap.add_argument("--agent-counts", default="8,16,24,32,48,64")
    ap.add_argument("--maps-file", default=None,
                    help="用独立 dev 地图集调参（见 scripts/make_dev_maps.py）；不给则用 validation-*")
    ap.add_argument("--num-maps", type=int, default=128, help="每族地图数，0/128=全部")
    ap.add_argument("--max-steps", type=int, default=128)
    ap.add_argument("--on-target", default="nothing", choices=["nothing", "finish"])
    ap.add_argument("--batch-envs", type=int, default=16, help="一次批量 rollout 的 episode 数")
    ap.add_argument("--action-mode", default="argmax",
                    choices=["argmax", "sample", "escape", "escape2"])
    ap.add_argument("--base-temp", type=float, default=0.3,
                    help="escape2：未卡住的体的采样温度（低温=近贪心，保路径质量）")
    ap.add_argument("--escape-k", type=int, default=3)
    ap.add_argument("--escape-temp", type=float, default=1.0)
    ap.add_argument("--escape-no-wait-after", type=int, default=8)
    ap.add_argument("--auto-protocol", action="store_true",
                    help="按各族官方协议跑（见 FAMILY_PROTOCOL），覆盖 --agent-counts/"
                         "--num-maps/--seeds/--max-steps")
    ap.add_argument("--zero-channels", type=str, default="",
                    help="观测通道消融：逗号分隔的通道号置零（3=足迹，4=cost2go）。")
    ap.add_argument("--reset-hidden-every", type=int, default=0,
                    help="记忆消融探针：推断时每 N 步清零时间核(GRU)状态；1=完全无记忆，0=不清（默认）。"
                         "用于分离瓶颈在空间编码还是时序记忆。")
    ap.add_argument("--tta", action="store_true",
                    help="测试期 D4 对称集成（8 视角平均动作分布）")
    ap.add_argument("--goal-hold", default="none", choices=["none", "greedy", "wait"],
                    help="sample 模式下已到达目标的体是否停止采样（nothing 协议守位）")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    set_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nc = yaml.safe_load(open(args.network_config))
    for kv in args.set_net:
        k, _, v = kv.partition("="); nc[k.strip()] = yaml.safe_load(v)
        print(f"[set-net] {k.strip()} = {nc[k.strip()]}")
    use_c2g = int(nc.get("patch_channels", 4)) >= 5
    net = build_net(nc)
    ck = torch.load(args.ckpt, map_location="cpu")
    state = {k.replace("_orig_mod.", ""): v for k, v in ck["net"].items()}
    net.load_state_dict(state); net.to(device).eval()
    print(f"[eval] {args.ckpt} | {sum(p.numel() for p in net.parameters())/1e6:.2f}M params | "
          f"cost2go={use_c2g} | iter={ck.get('meta_iter','?')} | mode={args.action_mode} "
          f"(k={args.escape_k},T={args.escape_temp},nowait={args.escape_no_wait_after}) | "
          f"protocol: on_target={args.on_target} steps={args.max_steps}", flush=True)

    def _parse_seeds(spec):
        out = []
        for tok in str(spec).split(","):
            tok = tok.strip()
            if "-" in tok and not tok.startswith("-"):
                a, _, b = tok.partition("-"); out += list(range(int(a), int(b) + 1))
            elif tok:
                out.append(int(tok))
        return out

    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    acs = [int(x) for x in args.agent_counts.split(",")]
    # 支持 "0-127" 区间语法：warehouse 族的官方协议是 **1 张图 × 128 个 seed**
    # （而 random/mazes 是 128 张图 × seed 0），必须能表达这种网格。
    seeds = []
    for tok in args.seeds.split(","):
        tok = tok.strip()
        if "-" in tok and not tok.startswith("-"):
            a, _, b = tok.partition("-"); seeds += list(range(int(a), int(b) + 1))
        elif tok:
            seeds.append(int(tok))
    results = {}
    for fam in fams:
        # --auto-protocol：按各族**官方**网格跑（agents/图数/seed/步数），避免产出
        # 与 leaderboard 不可比的数字。warehouse 是 1 图×128 seed、movingai 是 256 步。
        if args.auto_protocol and fam in FAMILY_PROTOCOL:
            pr = FAMILY_PROTOCOL[fam]
            f_acs, f_maps, f_seeds, f_steps = (pr["agents"], pr["maps"],
                                               _parse_seeds(pr["seeds"]), pr["steps"])
        else:
            f_acs, f_maps, f_seeds, f_steps = acs, args.num_maps, seeds, args.max_steps
        maps = load_maps(fam, f_maps)
        print(f"\n===== {fam}: {len(maps)} maps × {len(f_seeds)} seed(s) "
              f"× {f_steps} steps | agents={f_acs} =====", flush=True)
        results[fam] = {}
        for na in f_acs:
            t0 = time.time()
            specs = [(ms, nm, sd) for sd in f_seeds for nm, ms in maps]
            rows = []
            for i in range(0, len(specs), args.batch_envs):
                chunk = [(ms, nm, sd) for (ms, nm, sd) in specs[i:i + args.batch_envs]]
                rows += rollout_chunk(net, chunk, na, f_steps, device, use_c2g,
                                      float(nc.get("cost2go_clip", 10.0)),
                                      on_target=args.on_target, action_mode=args.action_mode,
                                      escape_k=args.escape_k, escape_temp=args.escape_temp,
                                      escape_no_wait_after=args.escape_no_wait_after,
                                      base_temp=args.base_temp,
                                      goal_hold=args.goal_hold, tta=args.tta,
                                      reset_hidden_every=args.reset_hidden_every,
                                      zero_channels=tuple(int(x) for x in args.zero_channels.split(",") if x.strip()),
                                      rng=np.random.default_rng(1234 + i))
            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            agg["n"] = len(rows)
            # CSR 的 95% CI（二项）
            p_, n_ = agg["CSR"], len(rows)
            agg["CSR_ci95"] = float(1.96 * np.sqrt(max(p_ * (1 - p_), 1e-9) / max(n_, 1)))
            results[fam][na] = agg
            print(f"  agents={na:>3}: CSR={agg['CSR']:.3f}±{agg['CSR_ci95']:.3f}  "
                  f"ISR={agg['ISR']:.3f}  SoC={agg['SoC']:.1f}  mkspn={agg['makespan']:.1f}  "
                  f"nbr={agg['avg_neighbors']:.2f} flip={agg['edge_flip']:.3f} "
                  f"esc={agg['escape_frac']:.3f}  (n={agg['n']}, {time.time()-t0:.0f}s)", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"ckpt": args.ckpt, "tag": args.tag, "args": vars(args), "results": results},
                  open(args.out, "w"), indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
