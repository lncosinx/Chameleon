"""推断期双向意图通信（BiIntentAttention）注意力可视化。

把一条 episode 的逐步注意力权重矩阵抓出来，画成两张图：
  *_spatial.png  —— 网格地图上的空间视图：焦点体 → 其 FOV 内邻居的注意力边
                     （边的颜色/粗细 = 权重），多个时刻并排，看"聚焦对象如何转移"
  *_matrix.png   —— 左：某时刻的 N×N 注意力矩阵（看 entmax 稀疏性）
                     右：焦点体的 邻居×时间 热力图（看迟滞带来的时间稳定性）

同时把原始权重存成 npz，改图不必重跑 rollout。

★ 口径说明：
  - **不开 TTA**。TTA 下 8 个 D4 视角各自维护独立通信状态、产生 8 套不同的注意力矩阵，
    没有唯一的"注意力"可展示。此脚本用单一规范视角（与 eval_bench 的非 TTA 路径一致），
    故其轨迹与带 TTA 的主结果不是同一条，仅用于机制展示，不用于报告指标。
  - 注意力权重取的是 `_aggregate` 里的最终 W：**已经过 topk 封顶 + 迟滞 scan**，
    即真实生效的通信边，而不是 entmax 的原始输出。
  - 自身被 `is_candidate` 的 `dist > 1e-6` 排除，故对角线恒为 0。
"""
from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection

from utils.seed import set_seed
from envs.observation_builder import ObservationBuilder, build_observations
from envs.action_mask import build_action_mask_batch
from scripts.train_il_meta import build_net
from scripts.eval_bench import make_env, load_maps
from pogema.animation.config import AnimationStyle
from pogema.animation.svg import Drawing
from pogema.animation.svg_objects import Circle, RectangleHref

# ---------------------------------------------------------------------------
# 设计系统取值（sequential 单色蓝 ramp 100→700；注意力是 magnitude，不用彩虹色图）
# ---------------------------------------------------------------------------
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_MUTED = "#898781"
GRIDLINE  = "#e1e0d9"
BASELINE  = "#c3c2b7"
FOCAL     = "#0d366b"
FOCAL_RED = "#c1433c"      # pogema 自己的 ego 色
OBSTACLE  = "#dcdbd3"      # pogema 原生 #84A1AE 是蓝灰，会与蓝色权重 ramp 混淆，改中性灰
INACTIVE  = "#9a9890"      # 未被关注的体
COMM_TINT = "#e8eef7"      # 通信候选格底色（浅蓝，弱于权重 ramp 的最浅步）


@torch.no_grad()
def rollout_with_attn(net, w, obs, num_agents, max_steps, device, use_c2g, c2g_clip,
                      base_temp, wait_bias, action_mode):
    """单 env 逐步推断，记录每步的注意力矩阵 / 位置 / 动作。"""
    N = num_agents
    builder = ObservationBuilder(num_agents=N, obs_radius=5, use_cost2go=use_c2g,
                                 cost2go_clip=c2g_clip)
    builder.reset()
    net.reset_comm_state()
    net.communication.store_attn = True          # ← 打开权重留痕（默认关闭，零开销）

    grid = w.global_grid
    goals = np.array(w.goals or w.get_targets_xy(), dtype=np.int32)
    prev = np.zeros(N, dtype=np.int64)
    attn_steps, pos_steps = [], []

    for t in range(max_steps):
        axy = np.array(w.get_agents_xy(), dtype=np.int32)
        p_, s_ = build_observations(w, obs, prev, builder)
        mask = build_action_mask_batch(axy[np.newaxis], grid)[0]

        pt = torch.from_numpy(p_[np.newaxis, np.newaxis]).float().to(device)
        st = torch.from_numpy(s_[np.newaxis, np.newaxis]).float().to(device)
        po = torch.from_numpy(axy.astype(np.float32)[np.newaxis, np.newaxis]).to(device)
        mt = torch.from_numpy(mask[np.newaxis, np.newaxis]).to(device)

        logits, _ = net(pt, st, po, action_masks=mt, reset_comm=False, stateful=True)

        A = net.communication.last_attn        # (B,T,N,N)
        assert A is not None, "last_attn 为空：store_attn 没生效，或通信模块不是 madcl"
        attn_steps.append(A[0, 0].float().cpu().numpy().copy())
        pos_steps.append(axy.copy())

        lg = logits[:, 0].clone()
        lg[:, :, 0] -= wait_bias               # 与评测一致的 wait 抑制
        if action_mode == "sample":
            probs = torch.softmax(lg / base_temp, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            bad = probs.sum(-1) <= 0
            if bad.any():
                probs = torch.where(bad.unsqueeze(-1),
                                    torch.from_numpy(mask[np.newaxis]).to(device).float(), probs)
            acts = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(N).cpu().numpy()
        else:
            acts = lg.argmax(dim=-1).reshape(N).cpu().numpy()

        obs, _, _, _ = w.step(acts.tolist())
        prev = acts.astype(np.int64)

    net.communication.store_attn = False
    final = np.array(w.get_agents_xy(), dtype=np.int32)
    solved = bool(np.all(final == goals))
    return (np.stack(attn_steps), np.stack(pos_steps), goals,
            (grid != 0).astype(np.uint8), solved)


def pick_focal(attn):
    """选"通信最活跃"的体：活跃出边数在整段里累计最多。"""
    deg = (attn > 0).sum(axis=-1)            # (T,N)
    return int(deg.sum(axis=0).argmax())


def pick_times(attn, focal, fpos, goals, k=3):
    """挑 k 个"有东西可看"的时刻。

    两条约束都是被实际产出的图逼出来的：
    - 跳过焦点体抵达目标后的静止尾段，否则 on_target=nothing 下它原地守位，
      后两张快照会几乎一模一样。
    - 在候选里贪心选邻居集合差异最大的，确保三张图展示的是**不同**的聚焦对象。
    """
    T = attn.shape[0]
    at_goal = np.all(fpos == goals[None, focal], axis=-1)
    moving = np.nonzero(~at_goal)[0]
    hi = int(moving[-1]) + 1 if len(moving) else T
    deg = (attn[:, focal] > 0).sum(axis=-1)
    cand = [t for t in range(hi) if deg[t] >= 2] or [t for t in range(hi) if deg[t] >= 1]
    if not cand:
        return list(range(min(k, T)))
    chosen = [cand[0]]
    sets = {t: set(np.nonzero(attn[t, focal] > 0)[0].tolist()) for t in cand}
    while len(chosen) < min(k, len(cand)):
        best = max((t for t in cand if t not in chosen),
                   key=lambda t: min(len(sets[t] ^ sets[c]) for c in chosen))
        chosen.append(best)
    return sorted(int(t) for t in chosen)


class RawSvg:
    """直接写死坐标的 SVG 片段（pogema 的 Circle/Rectangle 会对 y 取负，
    文本/色标这类自绘元素反而更方便用最终坐标，故单独走这个类）。"""

    def __init__(self, markup: str):
        self.markup = markup

    def render(self):
        return self.markup


def _ramp_color(w: float, vmax: float) -> str:
    """权重 → sequential 蓝色 ramp（magnitude 用单色相、浅→深，不用彩虹）。"""
    t = 0.0 if vmax <= 0 else min(max(w / vmax, 0.0), 1.0)
    pos = t * (len(BLUE_RAMP) - 1)
    i = int(pos)
    if i >= len(BLUE_RAMP) - 1:
        return BLUE_RAMP[-1]
    f = pos - i
    c1 = BLUE_RAMP[i].lstrip("#"); c2 = BLUE_RAMP[i + 1].lstrip("#")
    mix = [round(int(c1[k:k+2], 16) * (1 - f) + int(c2[k:k+2], 16) * f) for k in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(mix)


def render_pogema_svg(occ, pos, goals, focal, attn, times, comm_radius, obs_radius, vmax,
                      title, out_svg, out_png=None):
    """用 pogema 自己的 SVG 原语画地图，多时刻并排。

    与早先 matplotlib 版的区别（按用户要求）：
      - 地图/障碍/智能体/目标环全部走 pogema 的 Drawing / RectangleHref / Circle，
        几何与样式常数（r=35, scale=100, rx=15, draw_start=100）与 pogema 动画一致。
      - **注意力权重编码在智能体自身的填充色上**，不再用连线颜色；图上没有连线。
      - 通信半径以虚线圆标注在焦点体周围。
    障碍色改成中性灰（pogema 原生 #84A1AE 是蓝灰，会与蓝色权重 ramp 混淆）。
    字号单位是 SVG 用户单位，面板宽约 (C-1)*100+200 ≈ 2000，故字号取 50~110 才够看。
    """
    st = AnimationStyle()
    st.obstacle_color = OBSTACLE
    scale, r0, ds = st.scale_size, st.r, st.draw_start

    # ★ 裁掉 pogema padding：w.global_grid 按 obs_radius 四周补边（21×21 的图变成 31×31），
    #   外圈整整 4 圈既无障碍也无智能体，直接画出来就是大片空白。
    #   包围盒取 障碍 ∪ 所有时刻的智能体 ∪ 焦点体目标，再留 1 格余量。
    ys, xs = np.nonzero(occ)
    pr = np.concatenate([pos[list(times)][..., 0].ravel(), [goals[focal, 0]], ys])
    pc = np.concatenate([pos[list(times)][..., 1].ravel(), [goals[focal, 1]], xs])
    r_lo, r_hi = max(int(pr.min()) - 1, 0), min(int(pr.max()) + 1, occ.shape[0] - 1)
    c_lo, c_hi = max(int(pc.min()) - 1, 0), min(int(pc.max()) + 1, occ.shape[1] - 1)
    R, C = r_hi - r_lo + 1, c_hi - c_lo + 1

    panel_w = (C - 1) * scale + 2 * ds
    panel_h = (R - 1) * scale + 2 * ds
    gap, title_h, legend_h = int(0.45 * scale), 250, 430
    n = len(times)
    W_tot = n * panel_w + (n - 1) * gap
    H_tot = panel_h + title_h + legend_h

    d = Drawing(height=H_tot, width=W_tot, svg_settings=st)
    d.background_color = SURFACE
    FONT = "DejaVu Sans, sans-serif"

    def txt(x, y, t, size, fill=INK, anchor="middle", weight="normal"):
        return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
                f'font-weight="{weight}" fill="{fill}" font-family="{FONT}">{t}</text>')

    def cxcy(row, col, ox):
        """格 (row,col) → pogema 坐标系的 (cx, cy)；cy 越大越靠上（Circle 内部会取负）。
        row/col 是**未裁剪**的原始坐标，这里统一减去包围盒原点。"""
        rr, cc = row - r_lo, col - c_lo
        return ds + cc * scale + ox, ds + legend_h + (R - rr - 1) * scale

    for k, t in enumerate(times):
        ox = k * (panel_w + gap)
        P, A = pos[t], attn[t, focal]

        # 通信范围：**逐格**画出真实候选集，而不是一个光滑圆。
        # 判据是 torch.norm(L2) ≤ comm_radius 的离散圆盘（r=4 时 48 格）；画成光滑圆
        # 会让 (3,3)=4.243、(4,1)=4.123 这类**不在**候选集里的格子看起来像在里面。
        # 另叠观测 FOV 的方框：FOV 是 L∞ 方块(11×11)，与通信圆盘**形状不同**，
        # 不画出来读者容易把圆当成 FOV。
        fr, fc = int(P[focal, 0]), int(P[focal, 1])
        cells = []
        for dr in range(-int(comm_radius) - 1, int(comm_radius) + 2):
            for dc in range(-int(comm_radius) - 1, int(comm_radius) + 2):
                if 0 < dr * dr + dc * dc <= comm_radius * comm_radius:
                    rr, cc = fr + dr, fc + dc
                    if r_lo <= rr <= r_hi and c_lo <= cc <= c_hi:
                        cx, cy = cxcy(rr, cc, ox)
                        cells.append(f'<rect x="{cx - scale/2}" y="{-cy - scale/2}" '
                                     f'width="{scale}" height="{scale}" fill="{COMM_TINT}"/>')
        d.add_element(RawSvg("".join(cells)))

        for rr in range(r_lo, r_hi + 1):
            for cc in range(c_lo, c_hi + 1):
                if occ[rr, cc]:
                    cx, cy = cxcy(rr, cc, ox)
                    d.add_element(RectangleHref(x=cx - r0, y=cy - r0, height=r0 * 2))

        ox0, oy0 = cxcy(fr - obs_radius, fc - obs_radius, ox)
        side = (2 * obs_radius + 1) * scale
        d.add_element(RawSvg(
            f'<rect x="{ox0 - scale/2}" y="{-oy0 - scale/2}" width="{side}" height="{side}" '
            f'fill="none" stroke="{BASELINE}" stroke-width="7" stroke-dasharray="30 22" '
            f'rx="12"/>'))

        gcx, gcy = cxcy(goals[focal, 0], goals[focal, 1], ox)
        d.add_element(Circle(cx=gcx, cy=gcy, r=r0, fill="none",
                             stroke=FOCAL_RED, stroke_width=st.stroke_width))

        for i in range(len(P)):
            cx, cy = cxcy(P[i, 0], P[i, 1], ox)
            if i == focal:
                fill = FOCAL_RED
            elif A[i] > 0:
                fill = _ramp_color(float(A[i]), vmax)
            else:
                fill = INACTIVE
            d.add_element(Circle(cx=cx, cy=cy, r=r0, fill=fill,
                                 stroke=SURFACE, stroke_width=6))
            # 权重标数：颜色之外的第二通道，灰度/色盲下仍可读。
            # ★ 必须画在圆**外**：画在圆心会被 r0=35 的圆裁掉（"0.59" 只剩 "5"），
            #   且深色端的填充上读不出深色字。两遍绘制做白描边光晕，压到障碍上也清楚。
            if i != focal and A[i] > 0.12 * vmax:
                lx, ly_ = cx + r0 * 0.95, -cy - r0 * 0.8
                lbl = f"{A[i]:.2f}"
                d.add_element(RawSvg(
                    f'<text x="{lx}" y="{ly_}" font-size="50" text-anchor="start" '
                    f'font-weight="700" fill="none" stroke="{SURFACE}" stroke-width="12" '
                    f'stroke-linejoin="round" font-family="{FONT}">{lbl}</text>'
                    + txt(lx, ly_, lbl, 50, INK, anchor="start", weight="700")))

        d.add_element(RawSvg(txt(ox + panel_w / 2, -(panel_h + legend_h) - 34,
                                 f"t = {t}   |   active neighbours: {int((A > 0).sum())}",
                                 80, INK)))

    d.add_element(RawSvg(txt(W_tot / 2, -H_tot + 104, title, 104, INK)))

    # ---- 色标 ----
    bw, bh = int(W_tot * 0.30), 68
    bx, by = (W_tot - bw) / 2, -(legend_h - 108)
    stops = "".join(f'<stop offset="{i/(len(BLUE_RAMP)-1):.3f}" stop-color="{c}"/>'
                    for i, c in enumerate(BLUE_RAMP))
    parts = [f'<defs><linearGradient id="wramp" x1="0" x2="1">{stops}</linearGradient></defs>',
             f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="url(#wramp)" rx="10"/>',
             txt(bx + bw / 2, by - 30, "attention weight  (encoded as the agent's fill colour)",
                 62, INK_2)]
    parts += [txt(bx + bw * f, by + bh + 68, f"{f*vmax:.2f}", 54, INK_MUTED)
              for f in (0.0, 0.25, 0.5, 0.75, 1.0)]

    # ---- 图例：焦点体 / 被关注 / 未被关注 / 目标点 / 通信半径 ----
    items = [
        ("focal agent",                 f'<circle cx="0" cy="0" r="{r0}" fill="{FOCAL_RED}" '
                                        f'stroke="{SURFACE}" stroke-width="6"/>'),
        ("attended neighbour",          f'<circle cx="0" cy="0" r="{r0}" fill="{BLUE_RAMP[9]}" '
                                        f'stroke="{SURFACE}" stroke-width="6"/>'),
        ("agent with no attention",     f'<circle cx="0" cy="0" r="{r0}" fill="{INACTIVE}" '
                                        f'stroke="{SURFACE}" stroke-width="6"/>'),
        ("focal agent's goal",          f'<circle cx="0" cy="0" r="{r0}" fill="none" '
                                        f'stroke="{FOCAL_RED}" stroke-width="{st.stroke_width}"/>'),
        (f"comm range (cells, d &#8804; {comm_radius:g})",
         f'<rect x="-{r0}" y="-{r0}" width="{r0*2}" height="{r0*2}" fill="{COMM_TINT}" '
         f'stroke="{BASELINE}" stroke-width="3"/>'),
        (f"observation FOV ({2*obs_radius+1}x{2*obs_radius+1})",
         f'<rect x="-{r0}" y="-{r0}" width="{r0*2}" height="{r0*2}" fill="none" '
         f'stroke="{BASELINE}" stroke-width="6" stroke-dasharray="14 10" rx="5"/>'),
    ]
    # 图例紧凑居中排一行：按文字长度估宽，避免铺满整幅造成大片空白
    ly = -(legend_h - 330)
    fs, chw, pad = 56, 29, 150          # chw ≈ 每字符宽度（DejaVu Sans @56）
    widths = [r0 * 2 + 30 + len(lb) * chw for lb, _ in items]
    total = sum(widths) + pad * (len(items) - 1)
    x = (W_tot - total) / 2
    for (label, swatch), w in zip(items, widths):
        parts.append(f'<g transform="translate({x + r0},{ly})">{swatch}</g>')
        parts.append(txt(x + r0 * 2 + 30, ly + 20, label, fs, INK_2, anchor="start"))
        x += w + pad
    d.add_element(RawSvg("\n".join(parts)))

    svg = d.render()
    open(out_svg, "w").write(svg)
    print(f"-> {out_svg}")
    if out_png:
        import cairosvg
        # Drawing.render() 会把 SVG 归一化到最长边 512px，靠 scale 倍率不够用，
        # 直接指定最终像素宽，否则三联图上的字会糊成一团。
        cairosvg.svg2png(bytestring=svg.encode(), write_to=out_png,
                         output_width=2800, background_color=SURFACE)
        print(f"-> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_il/capbig_iter_20000.pt")
    ap.add_argument("--network-config", default="configs/network_big.yaml")
    ap.add_argument("--set-net", action="append", default=[], metavar="K=V",
                    help="覆盖网络配置项（与 eval_bench 同语义）。例："
                         "--set-net confine_sticky_to_fov=true 画**修复后**的通信图；"
                         "不传则按配置文件/默认值，即保持历史（泄漏）行为。")
    ap.add_argument("--family", default="mazes", choices=["random", "mazes"])
    ap.add_argument("--map-index", type=int, default=0, help="族内第几张 validation 图")
    ap.add_argument("--maps-file", default=None)
    ap.add_argument("--num-agents", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=128)
    # 解码口径显式写死（INVARIANT 17），但注意无 TTA，见模块 docstring
    ap.add_argument("--action-mode", default="sample", choices=["sample", "argmax"])
    ap.add_argument("--base-temp", type=float, default=1.0)
    ap.add_argument("--wait-bias", type=float, default=1.0)
    ap.add_argument("--focal", type=int, default=-1, help="-1=自动选通信最活跃的体")
    ap.add_argument("--num-snapshots", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-prefix", required=True)
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
    net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["net"].items()})
    net.to(device).eval()
    comm_radius = float(getattr(net.communication, "comm_radius", 4))
    topk_cap = int(getattr(net.communication, "comm_topk_cap", 6))

    maps = load_maps(args.family, 128, args.maps_file)
    name, map_str = maps[args.map_index]
    print(f"[viz] {args.ckpt} | {args.family}/{name} | N={args.num_agents} seed={args.seed} "
          f"| comm_radius={comm_radius} | TTA=off(机制展示用) "
          f"| confine_sticky_to_fov={bool(nc.get('confine_sticky_to_fov', False))}", flush=True)

    w, obs = make_env(map_str, name, args.num_agents, args.max_steps, "nothing", args.seed)
    attn, pos, goals, occ, solved = rollout_with_attn(
        net, w, obs, args.num_agents, args.max_steps, device, use_c2g,
        float(nc.get("cost2go_clip", 10.0)), args.base_temp, args.wait_bias, args.action_mode)
    T, N = attn.shape[0], attn.shape[1]
    focal = args.focal if args.focal >= 0 else pick_focal(attn)
    times = pick_times(attn, focal, pos[:, focal], goals, args.num_snapshots)
    # 色标上界按实际权重定：行归一化到 1 后单边权重通常只到 0.5~0.6，
    # 钉死 [0,1] 会让整张图挤在 ramp 浅色端、看不出深浅差异。
    nzw = attn[attn > 0]
    # 用 98 分位：偶发的"唯一邻居 → 权重 1.0"会把色标撑到 1，使绝大多数格子
    # 挤在 ramp 最浅端、看不出深浅。超过 vmax 的值由 colorbar 的 extend 标出。
    vmax = float(np.ceil(np.percentile(nzw, 98) * 20) / 20) if nzw.size else 1.0
    vmax = max(vmax, 0.2)
    dmat = np.linalg.norm(pos[:, :, None, :].astype(np.float32)
                          - pos[:, None, :, :].astype(np.float32), axis=-1)
    nz_mask = attn > 0
    out_mask = nz_mask & (dmat > comm_radius + 1e-6)
    n_nz, n_out = int(nz_mask.sum()), int(out_mask.sum())
    far = float(dmat[out_mask].max()) if n_out else 0.0
    deg = (attn > 0).sum(axis=-1).astype(float)
    print(f"[viz] T={T} N={N} solved={solved} | focal=agent{focal} | snapshots t={times}")
    print(f"[viz] 平均活跃度数={deg.mean():.2f} | 行权重和(非零行)={attn.sum(-1)[deg>0].mean():.3f}")
    print(f"[viz] vmax={vmax} | 超 comm_radius 的活跃边 {n_out}/{n_nz} "
          f"({100*n_out/max(n_nz,1):.1f}%)，最远 {far:.2f}")

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    np.savez_compressed(args.out_prefix + "_attn.npz", attn=attn.astype(np.float16),
                        pos=pos, goals=goals, occ=occ, focal=focal, times=np.array(times),
                        map_name=name, family=args.family, num_agents=args.num_agents,
                        seed=args.seed, solved=solved)
    print(f"-> {args.out_prefix}_attn.npz")

    # ---------------- 图 1：空间视图（pogema 渲染）----------------
    render_pogema_svg(
        occ, pos, goals, focal, attn, times, comm_radius,
        int(nc.get('obs_radius', 5)), vmax,
        f"Communication attention of agent {focal}  |  {args.family}/{name}  |  N={N}",
        args.out_prefix + "_spatial.svg", args.out_prefix + "_spatial.png")

    # ---------------- 图 2：N×N 矩阵 + 焦点体的邻居×时间 ----------------
    t_mid = times[len(times) // 2]
    fig = plt.figure(figsize=(11.6, 5.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.45], height_ratios=[3.1, 1],
                          hspace=0.12, wspace=0.22)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)

    im1 = ax1.imshow(attn[t_mid], cmap=CMAP, vmin=0, vmax=vmax, interpolation="nearest")
    ax1.set_xlabel("attended agent (key)", fontsize=9, color=INK_2)
    ax1.set_ylabel("attending agent (query)", fontsize=9, color=INK_2)
    ax1.set_title(f"Full attention matrix,  t = {t_mid}", fontsize=9.5, color=INK, pad=6)
    # 焦点行的标记：不画横贯数据区的横线（会被读成数据），也不放轴外文字
    # —— 轴外左侧会压到 y 轴标签，右侧会压到右图的刻度标签，两边都试过。
    # 改为把该行的**刻度标签**本身染红加粗（刻度区是预留空间，不与任何元素冲突），
    # 再在数据区左缘放一个小三角。
    ticks = sorted({int(v) for v in ax1.get_yticks() if 0 <= v < N} | {focal})
    ax1.set_yticks(ticks)
    for lbl, tv in zip(ax1.get_yticklabels(), ticks):
        if tv == focal:
            lbl.set_color(FOCAL_RED); lbl.set_fontweight("bold"); lbl.set_fontsize(8.5)
    ax1.plot([-0.35], [focal], marker=">", color=FOCAL_RED, ms=6, clip_on=False, zorder=6)
    ax1.tick_params(labelsize=7.5, colors=INK_MUTED, length=0)
    for sp in ax1.spines.values():
        sp.set_color(GRIDLINE)

    # 只保留整段里被关注过的邻居，按累计权重排序（N 行里绝大多数恒为 0）
    series = attn[:, focal, :]                       # (T,N)
    keep = np.nonzero(series.sum(axis=0) > 0)[0]
    # ★ 按**首次被关注的时刻**排（并列时累计权重大的在前）。
    #   原先按累计权重排 —— 而热图的结构是时间上的，两者不对应，看起来毫无规律。
    #   按首次接触排会呈现一条阶梯，谁先进入通信范围一目了然。
    first = np.array([int(np.nonzero(series[:, j] > 0)[0][0]) for j in keep])
    keep = keep[np.lexsort((-series[:, keep].sum(axis=0), first))]

    # 构型冻结时刻：所有体到达目标后 on_target=nothing 要求守位，位置不再变化
    # ⇒ 邻居集与注意力自然恒定。不标出来，这段平直尾巴会被当成 bug（实际被误读过）。
    moved = [t for t in range(1, len(pos)) if not np.array_equal(pos[t], pos[t - 1])]
    t_frozen = (max(moved) + 1) if moved else None
    im2 = ax2.imshow(series[:, keep].T, cmap=CMAP, vmin=0, vmax=vmax, aspect="auto",
                     interpolation="nearest")
    ax2.set_yticks(range(len(keep)))
    ax2.set_yticklabels([f"agent {j}" for j in keep], fontsize=7.5, color=INK_2)
    # ★ 这 len(keep) 行是**整段的并集**，不是同时通信的数量 —— 不写清楚会被读成
    #   「同时跟这么多体通信」，与 comm_topk_cap 矛盾。下方曲线给出逐步真实度数。
    ax2.set_title(f"Attention of agent {focal} over time — {len(keep)} distinct partners "
                  f"across the episode (union, not concurrent; rows ordered by first contact)",
                  fontsize=8.5, color=INK, pad=6)
    ax2.tick_params(axis="x", labelbottom=False, length=0)
    ax2.tick_params(axis="y", length=0)
    for sp in ax2.spines.values():
        sp.set_color(GRIDLINE)
    cb2 = fig.colorbar(im2, ax=[ax2, ax3], fraction=0.030, pad=0.015, extend="max")
    cb2.set_label("Attention weight", fontsize=8.5, color=INK_2)
    cb2.ax.tick_params(labelsize=7.5, colors=INK_MUTED, length=0)
    cb2.outline.set_edgecolor(GRIDLINE)

    # 逐步真实度数：直观证明任一时刻 ≤ comm_topk_cap
    deg_f = (series > 0).sum(axis=1)
    ax3.fill_between(np.arange(len(deg_f)), deg_f, step="mid", color=BLUE_RAMP[3], lw=0)
    ax3.step(np.arange(len(deg_f)), deg_f, where="mid", color=BLUE_RAMP[8], lw=1.2)
    ax3.axhline(topk_cap, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)))
    if t_frozen is not None and t_frozen < len(deg_f) - 1:
        for ax_ in (ax2, ax3):
            ax_.axvline(t_frozen, color=FOCAL_RED, lw=1.1, ls=(0, (3, 2)), alpha=0.85)
        ax2.annotate(f"t={t_frozen}: all agents parked on goals\n"
                     f"(positions frozen → attention constant)",
                     xy=(t_frozen, ax2.get_ylim()[1]), xytext=(6, 4),
                     textcoords="offset points", ha="left", va="top",
                     fontsize=7, color=FOCAL_RED)
    ax3.annotate(f"comm_topk_cap = {topk_cap}", (len(deg_f) * 0.99, topk_cap),
                 xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                 fontsize=7.5, color=INK_MUTED)
    ax3.set_ylim(0, topk_cap + 1.4)
    ax3.set_yticks(range(0, topk_cap + 1, 2))
    ax3.set_xlabel("timestep", fontsize=9, color=INK_2)
    ax3.set_ylabel("active\nneighbors", fontsize=8, color=INK_2)
    ax3.tick_params(labelsize=7.5, colors=INK_MUTED, length=0)
    for sp in ax3.spines.values():
        sp.set_color(GRIDLINE)

    fig.savefig(args.out_prefix + "_matrix.png", dpi=200, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {args.out_prefix}_matrix.png")


if __name__ == "__main__":
    main()
