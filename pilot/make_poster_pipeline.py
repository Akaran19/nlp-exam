"""Methods pipeline for the A0 poster, drawn at poster scale (72.7 x 18.6 cm) from the real pilot outputs
of explore_exploit.py (ee_step_metrics.csv, ee_measures.csv)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).parent
plt.rcParams.update({"font.family": "Arial", "font.size": 26, "svg.fonttype": "none"})

PID = 1  # example problem: typical step counts for both groups
W, H = 72.7 / 2.54, 18.6 / 2.54
HUMAN_BAND, LLM_BAND = "#FFF4E3", "#EEF0FB"
HUMAN_C, LLM_C, INK, MUTED = "#B86E0A", "#3F51B5", "#262626", "#5A5A5A"
KIND_C = {"exploit": "#9AA5B1", "explore": "#E8590C", "return": "#7B2CBF", "start": "#9AA5B1"}

sm = pd.read_csv(HERE / "ee_step_metrics.csv")
ms = pd.read_csv(HERE / "ee_measures.csv")
TAU = np.percentile(sm.d_prev.dropna(), 75)

fig = plt.figure(figsize=(W, H), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def rbox(x, y, w, h, fc, ec="none", lw=3, r=0.15, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=38, lw=4, color=INK, zorder=6))


C0, C1, C2, C3, C4, C5 = (0.1, 3.9), (4.55, 2.85), (7.95, 2.6), (11.1, 5.9), (17.6, 4.55), (22.75, 5.85)
HEAD_Y, SUB_Y = H - 0.38, H - 0.95
heads = [(C0, "Data", "same 50 problems"), (C1, "1  Split", "into steps"),
         (C2, "2  Embed", "each step"), (C3, "3  Distance", f"to the previous step (problem {PID})"),
         (C4, "4  Paths", "explore, exploit or loop"), (C5, "5  Compare", "paired by problem, n = 50")]
for (x, w), h, s in heads:
    ax.text(x + 0.05, HEAD_Y, h, fontsize=36, weight="bold", color=INK, va="center")
    ax.text(x + 0.05, SUB_Y, s, fontsize=25, color=MUTED, va="center")

LANES = {"human": (3.55, 2.25, HUMAN_BAND, HUMAN_C, "Humans", "human_stage1"),
         "model": (1.0, 2.25, LLM_BAND, LLM_C, "LLM", "QwQ-32B-Preview")}
LANE_END = C4[0] + C4[1] + 0.05
for src, (y0, h, band, c, name, sub) in LANES.items():
    g = sm[(sm.problem_id == PID) & (sm.source == src)].sort_values("step_idx")
    m = ms[(ms.problem_id == PID) & (ms.source == src)].iloc[0]
    rbox(0.0, y0, LANE_END, h, band, r=0.2, z=0)
    yc = y0 + h / 2
    # data
    rbox(C0[0] + 0.15, y0 + 0.25, C0[1] - 0.3, h - 0.5, "white", ec=c, lw=4, r=0.12, z=2)
    ax.text(C0[0] + 0.45, yc + 0.35, name, fontsize=36, weight="bold", color=c, va="center", zorder=3)
    ax.text(C0[0] + 0.45, yc - 0.35, sub, fontsize=26, color=INK, va="center", zorder=3)
    # split: grey squares, wrapped
    per_row, size, gap = 13, 0.155, 0.045
    for i in range(len(g)):
        r_, k_ = divmod(i, per_row)
        ax.add_patch(Rectangle((C1[0] + 0.05 + k_ * (size + gap), y0 + h - 0.42 - r_ * (size + gap) - size), size,
                               size, fc="#BDBDBD", ec="none", zorder=3))
    ax.text(C1[0] + 0.05, y0 + 0.35, f"{len(g)} steps", fontsize=25, color=MUTED, va="center")
    # distance trace
    a = fig.add_axes([(C3[0] + 0.55) / W, (y0 + 0.32) / H, (C3[1] - 0.7) / W, (h - 0.55) / H])
    x = g.start_frac.to_numpy()
    a.axvspan(0.8, 1.0, color="#DADADA", alpha=0.6, lw=0)
    a.plot(x[1:], g.d_prev.to_numpy()[1:], color=c, lw=2.5, zorder=2)
    for kind in ("explore", "return"):
        s = g[g.kind == kind]
        a.scatter(s.start_frac, s.d_prev, s=150, color=KIND_C[kind], zorder=3, edgecolor="white", lw=1.5)
    a.axhline(TAU, color=INK, ls="--", lw=2)
    a.set_xlim(0, 1)
    a.set_ylim(0, 1.05)
    a.set_xticks([])
    a.set_yticks([0, 1])
    a.tick_params(labelsize=20)
    a.spines[["top", "right"]].set_visible(False)
    a.set_facecolor("none")
    # paths: one bar per segment, width proportional to steps, coloured tick = how the segment started
    segs, cur, start_kind = [], 1, "start"
    for kd in g.kind.tolist()[1:]:
        if kd in ("explore", "return"):
            segs.append((cur, start_kind))
            cur, start_kind = 1, kd
        else:
            cur += 1
    segs.append((cur, start_kind))
    bx, bw = C4[0] + 0.05, C4[1] - 0.25
    unit = bw / len(g)
    xx = bx
    for n, sk in segs:
        ax.add_patch(Rectangle((xx, yc + 0.15), n * unit - 0.03, 0.42, fc="#C9CED6", ec="none", zorder=3))
        if sk != "start":
            ax.add_patch(Rectangle((xx, yc + 0.08), 0.06, 0.56, fc=KIND_C[sk], ec="none", zorder=4))
        xx += n * unit
    ax.text(bx, y0 + 0.62, f"{m.n_paths} paths · {m.n_return} loop{'' if m.n_return == 1 else 's'}", fontsize=26, weight="bold", color=INK,
            va="center")
    ax.text(bx, y0 + 0.25, f"deepest path: {m.depth_max} steps", fontsize=24, color=MUTED, va="center")
    for x0, x1 in [(C0[0] + C0[1] - 0.12, C1[0] - 0.02), (C1[0] + C1[1] - 0.15, C2[0] - 0.02),
                   (C2[0] + C2[1] + 0.02, C3[0] - 0.02), (C3[0] + C3[1] - 0.12, C4[0] - 0.02)]:
        arrow(x0, x1, yc)

# embed box spanning both lanes
rbox(C2[0], 1.15, C2[1], 4.5, "white", ec=INK, lw=3, r=0.15, z=2)
ax.text(C2[0] + 0.25, 5.25, "sentence\nembedding\n(MiniLM)", fontsize=26, color=INK, va="top", linespacing=1.2,
        zorder=3)
rng = np.random.default_rng(1)
for j, v in enumerate(rng.uniform(0.15, 1, 8)):
    ax.add_patch(Rectangle((C2[0] + 0.3, 3.35 - j * 0.24), 0.32, 0.2, fc=plt.cm.Blues(v), ec="none", zorder=3))
ax.text(C2[0] + 0.8, 2.45, "step →\nvector", fontsize=24, color=MUTED, va="center", linespacing=1.2, zorder=3)
ax.text(C2[0] + 0.25, 1.5, "or: Vendi gain", fontsize=23, color=MUTED, style="italic", va="center", zorder=3)

# compare: two paired plots with the real pilot numbers
w = ms.pivot(index="problem_id", columns="source")
for k, (col, title) in enumerate([("explore_rate", "explore rate"), ("return_rate", "loop rate")]):
    a = fig.add_axes([(C5[0] + 0.95 + k * 2.75) / W, 1.45 / H, 1.85 / W, 3.6 / H])
    hv, qv = w[col].human.to_numpy(), w[col].model.to_numpy()
    for h_, q_ in zip(hv, qv):
        a.plot([0, 1], [h_, q_], color="#9E9E9E", lw=1.2, alpha=0.55)
    a.scatter(np.zeros(len(hv)), hv, s=45, color=HUMAN_C, zorder=3)
    a.scatter(np.ones(len(qv)), qv, s=45, color=LLM_C, zorder=3)
    for xpos, v in ((0, np.median(hv)), (1, np.median(qv))):
        a.plot([xpos - 0.22, xpos + 0.22], [v, v], color="black", lw=4, zorder=4)
    a.set_xticks([0, 1], ["Human", "LLM"], fontsize=24)
    a.set_xlim(-0.4, 1.4)
    a.tick_params(axis="y", labelsize=20)
    d = qv - hv
    a.set_title(title, fontsize=25, pad=10)
    ax.text(C5[0] + 0.95 + k * 2.75 + 1.85 / 2, 0.8, f"LLM higher: {int((d > 0).sum())}/50", fontsize=22,
            color=MUTED, ha="center", va="center")
    a.spines[["top", "right"]].set_visible(False)
arrow(LANE_END + 0.02, C5[0] + 0.3, 3.35)

# key line
ky = 0.38
xx = 0.1
for kd, txt in [("explore", "explore: far from the previous step"),
                ("return", "loop: far jump back to an earlier step")]:
    ax.scatter([xx + 0.12], [ky], s=260, color=KIND_C[kd], zorder=3)
    t = ax.text(xx + 0.35, ky, txt, fontsize=24, color=INK, va="center")
    xx += 0.6 + len(txt) * 0.168
ax.plot([xx, xx + 0.55], [ky, ky], color=INK, ls="--", lw=2.5)
ax.text(xx + 0.7, ky, "threshold (75th percentile)", fontsize=24, color=INK, va="center")
xx += 0.95 + 27 * 0.168
ax.add_patch(Rectangle((xx, ky - 0.13), 0.4, 0.26, fc="#DADADA", ec="none"))
ax.text(xx + 0.55, ky, "last 20% of solution: miracle-step zone", fontsize=24, color=INK, va="center")

for ext in ("png", "svg"):
    fig.savefig(HERE / f"poster_pipeline.{ext}", dpi=200, facecolor="white")
