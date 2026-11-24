"""Methods pipeline for the A0 poster, drawn at poster scale (72.7 x 18.6 cm) from the real pilot outputs
of explore_exploit.py (ee_step_metrics.csv, ee_measures.csv)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).parent
plt.rcParams.update({"font.family": "Arial", "font.size": 26, "svg.fonttype": "none"})

PID = 1  # example problem for steps 3-4 (typical step counts; its human solution is short in words)
W, H = 72.7 / 2.54, 18.6 / 2.54
HUMAN_BAND, LLM_BAND = "#FFF4E3", "#EEF0FB"
HUMAN_C, LLM_C, INK, MUTED = "#B86E0A", "#3F51B5", "#262626", "#5A5A5A"
KIND_C = {"exploit": "#9AA5B1", "explore": "#E8590C", "return": "#7B2CBF", "start": "#9AA5B1"}
NAVY, ORANGE, PURPLE = "#002546", "#F68B1F", "#662D91"  # poster colours
PATH_C = ["#4A90D9", "#F6A04D", "#8E5BB8", "#5DB37E", "#E06C75", "#2BB3A3", "#C9A227", "#7F8FA6"]  # one colour per path

sm = pd.read_csv(HERE / "ee_step_metrics.csv")
E = np.load(HERE / "step_embeddings.npy")  # same row order as ee_steps.csv / ee_step_metrics.csv
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
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=38, lw=4, color=NAVY, zorder=6))


C0, C1, C2, C3, C4, C5 = (0.1, 3.9), (4.55, 2.85), (7.95, 2.6), (11.1, 5.9), (17.6, 4.55), (22.75, 5.85)
HEAD_Y, SUB_Y = H - 0.38, H - 0.95
heads = [(C0, None, "Data", "same 50 problems"), (C1, 1, "Split", "average per solution"),
         (C2, 2, "Embed", "text → numbers"), (C3, 3, "Distance", f"to the previous step (problem {PID})"),
         (C4, 4, "Paths", "one colour per path"), (C5, 5, "Compare", "paired by problem, n = 50")]
for (x, w), num, h, s in heads:
    tx = x + 0.05
    if num is not None:  # navy number badge, like the poster's card badges
        rbox(tx, HEAD_Y - 0.28, 0.56, 0.56, NAVY, r=0.08, z=3)
        ax.text(tx + 0.28, HEAD_Y, str(num), fontsize=28, weight="bold", color="white", ha="center", va="center", zorder=4)
        tx += 0.72
    ax.text(tx, HEAD_Y, h, fontsize=36, weight="bold", color=NAVY, va="center")
    ax.text(x + 0.05, SUB_Y, s, fontsize=25, color=MUTED, va="center")
avg = (sm.assign(words=sm.text.str.split().str.len())
       .groupby(["source", "problem_id"]).agg(steps=("text", "size"), words=("words", "sum"))
       .groupby("source").mean())

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
    # split: one square per step, average number of steps per solution, in the lane's colour
    n_avg, w_avg = int(round(avg.loc[src, "steps"])), avg.loc[src, "words"]
    per_row, size, gap = 17, 0.125, 0.035
    for i in range(n_avg):
        r_, k_ = divmod(i, per_row)
        ax.add_patch(Rectangle((C1[0] + 0.05 + k_ * (size + gap), y0 + h - 0.35 - r_ * (size + gap) - size), size,
                               size, fc=c, alpha=0.55 if i % 2 else 0.8, ec="none", zorder=3))
    ax.text(C1[0] + 0.05, y0 + 0.66, f"{w_avg:,.0f} words", fontsize=26, weight="bold", color=INK, va="center")
    ax.text(C1[0] + 0.05, y0 + 0.3, f"{n_avg} steps", fontsize=24, color=MUTED, va="center")
    # distance trace
    a = fig.add_axes([(C3[0] + 1.0) / W, (y0 + 0.32) / H, (C3[1] - 1.15) / W, (h - 0.55) / H])
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
    a.set_ylabel("cosine\ndistance", fontsize=20, linespacing=1.1, labelpad=6)
    a.spines[["top", "right"]].set_visible(False)
    a.set_facecolor("none")
    # paths: each step coloured by its path. An explore step starts a new path (new colour); a loop step
    # goes back to the path of the earlier step it is closest to (that path's colour again).
    rows = g.index.to_numpy()
    X = E[rows]
    path_of, n_paths = [], 0
    for t, kd in enumerate(g.kind.tolist()):
        if t == 0:
            path_of.append(0)
        elif kd == "explore":
            n_paths += 1
            path_of.append(n_paths)
        elif kd == "return":
            k = int(np.argmax(X[:t - 1] @ X[t]))
            path_of.append(path_of[k])
        else:
            path_of.append(path_of[-1])
    bx, bw = C4[0] + 0.05, C4[1] - 0.25
    unit = bw / len(g)
    for t, (pth, kd) in enumerate(zip(path_of, g.kind)):
        ax.add_patch(Rectangle((bx + t * unit, yc + 0.15), unit, 0.42, fc=PATH_C[pth % len(PATH_C)], ec="none", zorder=3))
        if kd in ("explore", "return"):
            ax.add_patch(Rectangle((bx + t * unit, yc + 0.06), 0.07, 0.6, fc=KIND_C[kd], ec="white", lw=1, zorder=4))
    ax.text(bx, y0 + 0.62, f"{m.n_paths} paths · {m.n_return} loop{'' if m.n_return == 1 else 's'}", fontsize=26, weight="bold", color=INK,
            va="center")
    ax.text(bx, y0 + 0.25, f"deepest path: {m.depth_max} steps", fontsize=24, color=MUTED, va="center")
    for x0, x1 in [(C0[0] + C0[1] - 0.12, C1[0] - 0.02), (C1[0] + C1[1] - 0.15, C2[0] - 0.02),
                   (C2[0] + C2[1] + 0.02, C3[0] - 0.02), (C3[0] + C3[1] - 0.12, C4[0] - 0.02)]:
        arrow(x0, x1, yc)

# embed box spanning both lanes: one step of text becomes a list of numbers
rbox(C2[0], 1.15, C2[1], 4.5, "white", ec=NAVY, lw=3, r=0.15, z=2)
cx2 = C2[0] + C2[1] / 2
rbox(C2[0] + 0.3, 4.55, C2[1] - 0.6, 0.85, "#FDE6CC", r=0.1, z=3)
ax.text(cx2, 4.97, "one step\nof text", fontsize=23, style="italic", color=INK, ha="center", va="center",
        linespacing=1.1, zorder=4)
ax.add_patch(FancyArrowPatch((cx2, 4.5), (cx2, 3.95), arrowstyle="-|>", mutation_scale=32, lw=3.5, color=NAVY, zorder=6))
ax.text(cx2 + 0.18, 4.22, "MiniLM", fontsize=21, weight="bold", color=PURPLE, va="center", zorder=4)
rbox(cx2 - 0.5, 2.3, 1.0, 1.55, "#E5ECF4", ec=NAVY, lw=2.5, r=0.08, z=3)
ax.text(cx2, 3.07, "0.12\n−0.40\n…\n0.07", fontsize=23, color=NAVY, ha="center", va="center", linespacing=1.1, zorder=4)
ax.text(cx2, 1.72, "384 numbers:\nsimilar steps\nare close", fontsize=21, color=MUTED, ha="center", va="center",
        linespacing=1.15, zorder=4)

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
        a.plot([xpos - 0.22, xpos + 0.22], [v, v], color=NAVY, lw=4.5, zorder=4)
    a.set_xticks([0, 1], ["Human", "LLM"], fontsize=24)
    a.set_xlim(-0.4, 1.4)
    a.tick_params(axis="y", labelsize=20)
    d = qv - hv
    a.set_title(title, fontsize=25, pad=10, color=NAVY, weight="bold")
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
