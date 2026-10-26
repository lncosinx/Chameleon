#!/usr/bin/env python3
"""专家动作分布 vs 模型动作分布（2026-10-12）。

**为什么需要**：已定论 15 发现解码端 `wait_bias`（压低 wait 的 logit）能涨 +0.143 CSR，
这**间接**说明模型等得比最优更多；但从未直接量过"模型到底比专家多等多少"。
本探针在**同一批测试实例**上直接对拍两个分布。

**口径（关键）**：
  - 实例完全复刻 `eval_bench.make_env`（validation-* 地图 + 同 seed ⇒ 起终点逐实例相同）。
  - 模型侧 **wait_bias=0**（用户要求：开着会直接把 wait 压掉，看不出原始偏差）。
  - 两种分布分别回答两个不同问题，都要报：
      (a) **行为分布**：各自沿**自己的**轨迹走，统计动作边缘分布。
          回答"模型的行为像不像专家"——这是 CSR 真正关心的。
      (b) **教师强制分布**：把模型放在**专家的**状态序列上（隐状态也由专家轨迹建立），
          统计它在专家状态下的动作分布与逐步一致率。
          回答"模型能不能预测专家"——已知一致率 0.90~0.96 而 CSR 差很多，故 (a) 才是新信息。

**三阶段**（LaCAM 的 _lacam.so 是 py3.10 编译，不能与 py3.12 的策略推断同进程，
见 CLAUDE.md 致命坑）：
  phase=instances  (base)      复刻实例，存 (map_str, starts, goals)
  phase=expert     (lacam310)  对每个实例跑 LaCAM，存专家动作序列
  phase=model      (base, GPU) 跑模型，产出 (a) 与 (b) 两套分布并与专家对比

用法：
  python scripts/probe_action_dist.py --phase instances --families random,mazes \
      --agent-counts 64 --num-maps 64 --out /root/autodl-tmp/actdist
  conda activate lacam310
  python scripts/probe_action_dist.py --phase expert --out /root/autodl-tmp/actdist --num-workers 12
  python scripts/probe_action_dist.py --phase model --out /root/autodl-tmp/actdist \
      --ckpt ckpt_il/capbig_iter_20000.pt --network-config configs/network_big.yaml
"""
from __future__ import annotations
import sys, os, argparse, json, tempfile, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ACT_NAMES = ["wait", "up", "down", "left", "right"]
FAMILY_DIR = {"random": "01-random", "mazes": "02-mazes"}
MARLGPT = "/root/Chameleon/marl-gpt/eval_configs"


def _load_maps(family, num_maps):
    import yaml
    src = os.path.join(MARLGPT, FAMILY_DIR[family], "maps.yaml")
    maps = yaml.safe_load(open(src))
    names = sorted(maps.keys())
    if 0 < num_maps < len(names):
        idx = sorted(set(np.linspace(0, len(names) - 1, num_maps).round().astype(int).tolist()))
        names = [names[i] for i in idx]
    return [(n, maps[n]) for n in names]


def _make_env(map_str, name, num_agents, max_steps, seed):
    """与 eval_bench.make_env 逐字一致（勿改，否则实例不同、对比作废）。"""
    import yaml
    from envs.pogema_env_wrapper import PogemaWrapper
    rows = map_str.splitlines()
    cfg = dict(map_type="random", fixed_map=map_str, fixed_map_name=name,
               map_size=max(len(rows), len(rows[0])), num_agents=num_agents, obs_radius=5,
               max_episode_steps=max_steps, collision_system="soft",
               on_target="nothing", split="all")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f); p = f.name
    try:
        w = PogemaWrapper(p)
    finally:
        os.unlink(p)
    obs, _ = w.reset(seed=seed)
    return w, obs


# ===========================================================================
# phase = instances
# ===========================================================================
def phase_instances(args):
    os.makedirs(args.out, exist_ok=True)
    recs = []
    for fam in args.families.split(","):
        for na in [int(x) for x in args.agent_counts.split(",")]:
            for name, mstr in _load_maps(fam, args.num_maps):
                for sd in [int(x) for x in args.seeds.split(",")]:
                    w, _ = _make_env(mstr, name, na, args.max_steps, sd)
                    # ★ 剥 pad 圈：global_grid 是 pad 过 obs_radius 的，而 get_agents_xy()
                    #   返回同一 padded 系。存盘必须剥掉，否则重建时 pogema 再 pad 一圈，
                    #   坐标整体偏移、起终点落到障碍上被静默挪走（见 generate_dagger_data._grid_to_str）。
                    g = w.global_grid
                    pad = 5
                    gs = g[pad:g.shape[0] - pad, pad:g.shape[1] - pad]
                    recs.append(dict(
                        fam=fam, na=na, name=name, seed=sd,
                        map_str="\n".join("".join("#" if v else "." for v in row) for row in gs),
                        starts=np.array(w.get_agents_xy(), dtype=np.int32).tolist(),
                        goals=np.array(w.get_targets_xy(), dtype=np.int32).tolist()))
    p = os.path.join(args.out, "instances.json")
    json.dump(recs, open(p, "w"))
    print(f"[instances] {len(recs)} 个实例 → {p}")


# ===========================================================================
# phase = expert
# ===========================================================================
def _expert_worker(a):
    payload, idxs, max_steps = a
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from lacam.lacam_runner import LaCAMRunner
    runner = LaCAMRunner(backend="auto", solver_seed=12345)
    if runner.backend != "lacam":
        raise RuntimeError(f"LaCAM 后端回落成 {runner.backend}！必须在 lacam310 环境跑"
                           "（见 CLAUDE.md 致命坑），否则专家动作被污染。")
    recs = json.load(open(payload))
    out = []
    for j in idxs:
        r = recs[j]
        w, _ = _make_env(r["map_str"], r["name"], r["na"], max_steps, r["seed"])
        got_s = np.array(w.get_agents_xy(), dtype=np.int32)
        got_g = np.array(w.get_targets_xy(), dtype=np.int32)
        # 不校验相等（本阶段是从 map_str 重建、seed 相同 ⇒ pogema 会重新按 seed 放置），
        # 而是**以重建后的实际布局为准**并一并存下，供 model 阶段核对。
        res = runner.solve(obstacles=w.global_grid, starts=got_s.tolist(),
                           goals=got_g.tolist(), map_type="random",
                           map_size=max(len(r["map_str"].splitlines()),
                                        len(r["map_str"].splitlines()[0])))
        if not res.success:
            out.append(dict(idx=j, ok=False)); continue
        # trajectory_converter 只有单体版 path_to_actions；逐体转换后按时间对齐成 (T, N)。
        # LaCAM 各体路径等长（到达后以 wait 补齐），这里仍显式按最短长度裁剪以防越界。
        from lacam.trajectory_converter import path_to_actions
        per_agent = [path_to_actions([tuple(p) for p in pth]) for pth in res.paths]
        T = min(len(x) for x in per_agent)
        acts = np.array([[per_agent[i][t] for i in range(len(per_agent))]
                         for t in range(T)], dtype=np.int8)   # (T, N)
        out.append(dict(idx=j, ok=True, starts=got_s.tolist(), goals=got_g.tolist(),
                        actions=np.asarray(acts, dtype=np.int8).tolist()))
    return out


def phase_expert(args):
    import multiprocessing as mp
    payload = os.path.join(args.out, "instances.json")
    n = len(json.load(open(payload)))
    nw = max(1, args.num_workers)
    chunks = [list(range(i, n, nw)) for i in range(nw)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(nw) as pool:
        res = pool.map(_expert_worker, [(payload, c, args.max_steps) for c in chunks])
    flat = [x for r in res for x in r]
    flat.sort(key=lambda d: d["idx"])
    p = os.path.join(args.out, "expert.json")
    json.dump(flat, open(p, "w"))
    ok = sum(1 for x in flat if x["ok"])
    print(f"[expert] {ok}/{n} 求解成功 → {p}  ({time.time()-t0:.0f}s)")




# ===========================================================================
# phase = model
# ===========================================================================
def phase_model(args):
    import torch, yaml
    from envs.observation_builder import ObservationBuilder, build_observations
    from envs.action_mask import build_action_mask_batch
    from scripts.train_il_meta import build_net

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc = yaml.safe_load(open(args.network_config))
    use_c2g = int(nc.get("patch_channels", 4)) >= 5
    net = build_net(nc)
    ck = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["net"].items()})
    net.to(dev).eval()
    print(f"[model] {args.ckpt} | iter={ck.get('meta_iter','?')} | wait_bias=0（用户要求）", flush=True)

    recs = json.load(open(os.path.join(args.out, "instances.json")))
    exp = {d["idx"]: d for d in json.load(open(os.path.join(args.out, "expert.json")))}
    rng = np.random.default_rng(0)
    snaps = []

    # 四个计数器：按族分开（random 与 mazes 的专家 wait 率差 0.2 以上，混在一起没意义）
    # ★ 同步数 + 分「全体」与「仅未到达体」两套计数（2026-10-12 用户指出的混淆）：
    #   (i) 步数：两侧都只统计 t < T_expert（专家需要的那个窗口），否则模型多跑的
    #       85 步会把分布带偏；
    #   (ii) 已到达体：on_target=nothing 下已到家的体**必须守位，而守位就是 wait 动作**，
    #       把它算成"模型爱等"是错的。故另算一套只含未到达体的**决策**分布。
    cnt = {}
    def _c(fam):
        return cnt.setdefault(fam, dict(
            expert=np.zeros(5, np.int64),      behav=np.zeros(5, np.int64),
            expert_act=np.zeros(5, np.int64),  behav_act=np.zeros(5, np.int64),  # 仅未到达
            tf=np.zeros(5, np.int64), tf_s=np.zeros(5, np.int64),
            agree=0, agree_s=0, tf_total=0, n_ep=0, steps=0,
            behav_am=np.zeros(5, np.int64), behav_am_act=np.zeros(5, np.int64)))

    @torch.no_grad()
    def _logits(w, ob, prev, grid):
        axy = np.array(w.get_agents_xy(), dtype=np.int32)
        p_, s_ = build_observations(w, w.last_obs if hasattr(w, "last_obs") else None, prev, ob)
        pt = torch.from_numpy(p_[None, None]).float().to(dev)
        st = torch.from_numpy(s_[None, None]).float().to(dev)
        po = torch.from_numpy(axy[None, None].astype(np.float32)).to(dev)
        m = build_action_mask_batch(axy[None], grid)[0]
        mt = torch.from_numpy(m[None, None]).to(dev)
        lg, _ = net(pt, st, po, action_masks=mt, stateful=True)
        return lg[0, 0], m    # (N,5) logits, (N,5) mask

    for j, r in enumerate(recs):
        fam = r["fam"]; N = r["na"]; c = _c(fam)
        e = exp.get(j)
        if e is None or not e["ok"]:
            continue
        ea = np.asarray(e["actions"], dtype=np.int64)          # (T, N) 专家动作
        Texp = min(len(ea), args.max_steps)                    # ★ 统一的步数窗口
        g = np.asarray(e["goals"], dtype=np.int32)
        # 由 starts + 动作增量重建专家逐步位置，用来判"该体此刻是否已到达"
        DELTA = np.array([[0,0],[-1,0],[1,0],[0,-1],[0,1]], dtype=np.int32)
        epos = np.asarray(e["starts"], dtype=np.int32).copy()
        for t in range(Texp):
            arrived = np.all(epos == g, axis=1)
            c["expert"] += np.bincount(ea[t], minlength=5)
            if (~arrived).any():
                c["expert_act"] += np.bincount(ea[t][~arrived], minlength=5)
            epos = epos + DELTA[ea[t]]
        c["n_ep"] += 1; c["steps"] += Texp

        # ---------- (a) 行为分布：模型沿**自己**的轨迹 ----------
        w, obs = _make_env(r["map_str"], r["name"], N, args.max_steps, r["seed"])
        ob = ObservationBuilder(num_agents=N, obs_radius=5, use_cost2go=use_c2g,
                                cost2go_clip=float(nc.get("cost2go_clip", 10.0)))
        ob.reset(); net.reset_comm_state()
        grid = w.global_grid
        prev = np.zeros(N, dtype=np.int64)
        w.last_obs = obs
        for t in range(Texp):          # ★ 与专家同步数，不再跑到 128
            lg, m = _logits(w, ob, prev, grid)
            lg = lg.clone(); lg[~torch.from_numpy(m).to(dev)] = float("-inf")
            # ★ wait_bias = 0（不压 wait），T = 1.0 —— 看模型**原始**的动作偏好
            pr = torch.softmax(lg / 1.0, dim=-1).cpu().numpy()
            a = np.array([rng.choice(5, p=pr[i] / pr[i].sum()) for i in range(N)], dtype=np.int64)
            a_am = lg.argmax(dim=-1).cpu().numpy()      # 同状态下的 argmax，供口径对照
            mpos = np.array(w.get_agents_xy(), dtype=np.int32)
            marr = np.all(mpos == g, axis=1)
            c["behav"] += np.bincount(a, minlength=5)
            c["behav_am"] += np.bincount(a_am, minlength=5)
            if (~marr).any():
                c["behav_act"] += np.bincount(a[~marr], minlength=5)
                c["behav_am_act"] += np.bincount(a_am[~marr], minlength=5)
            # ★ 快照：模型漂到的状态 + 它在该状态选的动作。
            #   供 phase=expert_on_model 让 LaCAM 从该状态重解，回答
            #   「同样的状态下专家会不会也这么等」（区分状态分布问题 vs 策略问题）。
            if args.dump_states and t % args.dump_states == 0:
                snaps.append(dict(fam=fam, map_str=r["map_str"], name=r["name"], na=int(N),
                                  seed=int(r["seed"]), t=int(t),
                                  pos=mpos.tolist(), goals=g.tolist(),
                                  model_act=a.tolist(),              # 采样（部署口径）
                                  model_act_argmax=a_am.tolist(),    # argmax（最佳猜测）
                                  arrived=marr.tolist()))
            # PogemaWrapper.step 返回 4 元组 (obs, rewards, dones, info)，非 gymnasium 5 元组
            obs, _, dones, _ = w.step(a.tolist())
            w.last_obs = obs; prev = a
            if np.all(np.asarray(dones)):
                break

        # ---------- (b) 教师强制：模型放在**专家**的状态序列上 ----------
        w2, obs2 = _make_env(r["map_str"], r["name"], N, args.max_steps, r["seed"])
        ob2 = ObservationBuilder(num_agents=N, obs_radius=5, use_cost2go=use_c2g,
                                 cost2go_clip=float(nc.get("cost2go_clip", 10.0)))
        ob2.reset(); net.reset_comm_state()
        prev2 = np.zeros(N, dtype=np.int64)
        w2.last_obs = obs2
        for t in range(min(len(ea), args.max_steps)):
            lg, m = _logits(w2, ob2, prev2, grid)
            lg = lg.clone(); lg[~torch.from_numpy(m).to(dev)] = float("-inf")
            # ★ 两种口径都算（2026-10-12 修）：此前只算 argmax，而行为侧用采样，两边不可比
            #   —— 已定论 19 同类坑（argmax vs sample 曾给出反向结论）。
            am = lg.argmax(dim=-1).cpu().numpy()
            prf = torch.softmax(lg / 1.0, dim=-1).cpu().numpy()
            asm = np.array([rng.choice(5, p=prf[i] / prf[i].sum()) for i in range(N)],
                           dtype=np.int64)
            c["tf"] += np.bincount(am, minlength=5)
            c["tf_s"] += np.bincount(asm, minlength=5)
            c["agree"] += int((am == ea[t]).sum())
            c["agree_s"] += int((asm == ea[t]).sum())
            c["tf_total"] += N
            # ★ 用**专家**动作推进环境（teacher forcing）
            obs2, _, dones2, _ = w2.step(ea[t].tolist())
            w2.last_obs = obs2; prev2 = ea[t]
            if np.all(np.asarray(dones2)):
                break
        if (j + 1) % 20 == 0:
            print(f"  ... {j+1}/{len(recs)}", flush=True)

    out = {}
    print()
    for fam, c in cnt.items():
        nrm = lambda v: v / max(v.sum(), 1)
        ex, bh = nrm(c["expert"]), nrm(c["behav"])
        exa, bha = nrm(c["expert_act"]), nrm(c["behav_act"])
        tf = nrm(c["tf"])
        print(f"===== {fam}  ({c['n_ep']} episode，同步数窗口共 {c['steps']} 步) =====")
        print(f"{'动作':>7}│{'专家':>8}{'模型':>8}│{'专家*':>8}{'模型*':>8}│{'教师强制':>9}")
        print(f"{'':>7}│{'(全体)':>8}{'(全体)':>8}│{'(未到达)':>8}{'(未到达)':>8}│{'(专家状态)':>9}")
        for k in range(5):
            print(f"{ACT_NAMES[k]:>7}│{ex[k]:>8.3f}{bh[k]:>8.3f}│{exa[k]:>8.3f}{bha[k]:>8.3f}│{tf[k]:>9.3f}")
        print(f"\n  wait 率（全体，含守位）  专家 {ex[0]:.3f} vs 模型 {bh[0]:.3f}  差 {bh[0]-ex[0]:+.3f}")
        print(f"  wait 率（仅未到达=真决策）专家 {exa[0]:.3f} vs 模型 {bha[0]:.3f}  差 {bha[0]-exa[0]:+.3f}  ← 看这个")
        print(f"  教师强制逐步一致率：argmax {c['agree']/max(c['tf_total'],1):.4f}  "
              f"采样 {c['agree_s']/max(c['tf_total'],1):.4f}   ★两种口径都报，勿混比\n")
        out[fam] = dict(expert=ex.tolist(), behavior=bh.tolist(),
                        expert_active=exa.tolist(), behavior_active=bha.tolist(),
                        teacher_forced=tf.tolist(), steps=int(c["steps"]),
                        agree=c["agree"]/max(c["tf_total"],1),
                        agree_sampled=c["agree_s"]/max(c["tf_total"],1),
                        behavior_argmax_active=nrm(c["behav_am_act"]).tolist(),
                        n_ep=c["n_ep"])
    p = os.path.join(args.out, "actdist.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"→ {p}")
    if snaps:
        sp = os.path.join(args.out, "model_states.json")
        json.dump(snaps, open(sp, "w"))
        print(f"→ {sp}  ({len(snaps)} 个模型状态快照)")




# ===========================================================================
# phase = expert_on_model（LaCAM 从**模型走到的状态**重解，取首动作）
# ===========================================================================
def _eom_worker(a):
    payload, idxs, max_steps, solver_seed = a
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import yaml, tempfile as _tf
    from envs.pogema_env_wrapper import PogemaWrapper
    from lacam.lacam_runner import LaCAMRunner
    from lacam.trajectory_converter import path_to_actions
    runner = LaCAMRunner(backend="auto", solver_seed=solver_seed)
    if runner.backend != "lacam":
        raise RuntimeError(f"LaCAM 后端回落成 {runner.backend}！必须在 lacam310 环境跑。")
    snaps = json.load(open(payload))
    out = []
    for j in idxs:
        sn = snaps[j]
        pos = np.asarray(sn["pos"], dtype=np.int32)
        gl = np.asarray(sn["goals"], dtype=np.int32)
        rows = sn["map_str"].splitlines()
        cfg = dict(map_type="random", fixed_map=sn["map_str"], fixed_map_name=sn["name"],
                   map_size=max(len(rows), len(rows[0])), num_agents=sn["na"], obs_radius=5,
                   max_episode_steps=max_steps, collision_system="soft",
                   on_target="nothing", split="all",
                   fixed_agents_xy=pos.tolist(), fixed_targets_xy=gl.tolist())
        with _tf.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f); cp = f.name
        try:
            w = PogemaWrapper(cp)
        finally:
            os.unlink(cp)
        w.reset(seed=0)
        got_p = np.array(w.get_agents_xy(), dtype=np.int32)
        got_g = np.array(w.get_targets_xy(), dtype=np.int32)
        # ★ pogema 遇到"起终点落在障碍上"只打 Warning 然后偷偷换格 ⇒ 必须显式拦截，
        #   否则重解的根本不是模型真正走到的那个状态（见 generate_dagger_data 同类注释）。
        if not (np.array_equal(got_p, pos) and np.array_equal(got_g, gl)):
            out.append(dict(idx=j, ok=False, why="inject_mismatch")); continue
        res = runner.solve(obstacles=w.global_grid, starts=pos.tolist(), goals=gl.tolist(),
                           map_type="random", map_size=cfg["map_size"])
        if not res.success:
            out.append(dict(idx=j, ok=False, why="unsolved")); continue
        first = []
        for pth in res.paths:
            acts = path_to_actions([tuple(q) for q in pth])
            first.append(int(acts[0]) if acts else 0)   # 路径只有一点 ⇒ 原地
        out.append(dict(idx=j, ok=True, expert_first=first))
    return out


def phase_expert_on_model(args):
    import multiprocessing as mp
    payload = os.path.join(args.out, "model_states.json")
    snaps = json.load(open(payload))
    n = len(snaps)
    nw = max(1, args.num_workers)
    chunks = [list(range(i, n, nw)) for i in range(nw)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(nw) as pool:
        res = pool.map(_eom_worker,
                       [(payload, c, args.max_steps, args.solver_seed) for c in chunks])
    lab = {}
    for r in res:
        for x in r:
            lab[x["idx"]] = x
    ok = sum(1 for x in lab.values() if x["ok"])
    print(f"[expert_on_model] {ok}/{n} 重解成功 ({time.time()-t0:.0f}s)")

    # 统计：在**同一批模型状态**上，专家 vs 模型的动作分布（只算未到达体）
    cnt = {}
    for j, sn in enumerate(snaps):
        x = lab.get(j)
        if x is None or not x["ok"]:
            continue
        fam = sn["fam"]
        c = cnt.setdefault(fam, dict(exp=np.zeros(5, np.int64), mod=np.zeros(5, np.int64),
                                     agree=0, tot=0, n=0))
        arr = np.asarray(sn["arrived"], dtype=bool)
        ma = np.asarray(sn["model_act" if not args.use_argmax else "model_act_argmax"],
                        dtype=np.int64)
        ea = np.asarray(x["expert_first"], dtype=np.int64)
        m = ~arr
        if not m.any():
            continue
        c["exp"] += np.bincount(ea[m], minlength=5)
        c["mod"] += np.bincount(ma[m], minlength=5)
        c["agree"] += int((ea[m] == ma[m]).sum()); c["tot"] += int(m.sum()); c["n"] += 1

    out = {}
    print()
    for fam, c in cnt.items():
        nrm = lambda v: v / max(v.sum(), 1)
        e, m = nrm(c["exp"]), nrm(c["mod"])
        print(f"===== {fam}：在**模型走到的状态**上（{c['n']} 个快照，{c['tot']} 个未到达体决策）=====")
        print(f"{'动作':>7}{'专家重解':>11}{'模型':>9}{'Δ(模型−专家)':>14}")
        for k in range(5):
            print(f"{ACT_NAMES[k]:>7}{e[k]:>11.3f}{m[k]:>9.3f}{m[k]-e[k]:>+14.3f}")
        print(f"\n  wait 率：专家(同状态重解) {e[0]:.3f}  vs  模型 {m[0]:.3f}   差 {m[0]-e[0]:+.3f}")
        print(f"  逐体首动作一致率：{c['agree']/max(c['tot'],1):.4f}\n")
        out[fam] = dict(expert_on_model=e.tolist(), model=m.tolist(),
                        agree=c["agree"]/max(c["tot"],1), n_snap=c["n"], n_dec=c["tot"])
    sfx = "" if args.solver_seed == 12345 else f"_s{args.solver_seed}"
    p = os.path.join(args.out, f"expert_on_model{sfx}.json")
    json.dump(out, open(p, "w"), indent=2)
    # ★ 同时落盘逐快照逐体的首动作，供「LaCAM 自己跟自己」的一致率天花板测量
    raw = os.path.join(args.out, f"eom_raw{sfx}.json")
    json.dump({str(k): (v["expert_first"] if v["ok"] else None) for k, v in lab.items()},
              open(raw, "w"))
    print(f"→ {p}\n→ {raw}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["instances", "expert", "model", "expert_on_model"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", default="random,mazes")
    ap.add_argument("--agent-counts", default="64")
    ap.add_argument("--num-maps", type=int, default=64)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--max-steps", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--ckpt", default="ckpt_il/capbig_iter_20000.pt")
    ap.add_argument("--network-config", default="configs/network_big.yaml")
    ap.add_argument("--use-argmax", action="store_true",
                    help="expert_on_model 用模型的 argmax 而非采样动作（口径对照，"
                         "采样自带熵会额外压低一致率）")
    ap.add_argument("--solver-seed", type=int, default=12345,
                    help="LaCAM 求解种子。换种子重解同一批状态 ⇒ 量"
                         "等价最优解之间的分歧 = 一致率的天花板")
    ap.add_argument("--dump-states", type=int, default=0,
                    help=">0 时 phase=model 每该步数快照一次 (位置,目标,模型动作)，"
                         "供 phase=expert_on_model 用 LaCAM 从该状态重解")
    args = ap.parse_args()
    if args.phase == "instances":
        phase_instances(args)
    elif args.phase == "expert":
        phase_expert(args)
    elif args.phase == "model":
        phase_model(args)
    else:
        phase_expert_on_model(args)
