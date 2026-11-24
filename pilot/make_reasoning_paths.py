"""Reasoning as pathfinding (after alexwang.ai/posts/reasoning-as-pathfinding), drawn from the pilot's real steps.

Each solution is a path from the problem (left) to its final step (right):
  x = position in the solution (share of characters written so far)
  y = how far the step is from the correct answer: cosine distance between the step's MiniLM embedding and the
      embedding of the dataset's expected answer (the problem itself is the first point)
Dots mark the pilot's explore (green) and loop (magenta) steps. A path ending in a wrong answer gets an x.
(A literal 2-D projection of the embeddings was tried first and was an unreadable tangle.)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from minilm import MiniLM
from pipeline import HAND_CHECK, load

HERE = Path(__file__).parent
PIDS = [1, 13, 18, 28, 36]  # 28: QwQ's answer is wrong
plt.rcParams.update({"font.family": "Arial", "font.size": 13, "svg.fonttype": "none"})
HUMAN_C, LLM_C, NAVY, MUTED = "#B86E0A", "#3F51B5", "#002546", "#6B7280"
EXPLORE_C, LOOP_C = "#2A9D5C", "#D6337F"

sm = pd.read_csv(HERE / "ee_step_metrics.csv")
E = np.load(HERE / "step_embeddings.npy")
df = load()
probs = df.drop_duplicates("problem_id").set_index("problem_id")
enc = MiniLM()


def frame(pid):
    """Return per-solution (x, y, kinds): position in solution vs. distance from the correct answer."""
    p = enc.encode([probs.problem[pid]])[0]
    a = enc.encode([str(probs.expected_answer[pid])])[0]
    out = {}
    for src in ("human", "model"):
        g = sm[(sm.problem_id == pid) & (sm.source == src)].sort_values("step_idx")
        x = (g.start_frac + g.n_chars / g.n_chars.sum()).to_numpy()  # end of each step
        y = 1 - E[g.index] @ a
        out[src] = (np.r_[0, x], np.r_[1 - p @ a, y], ["start"] + g.kind.tolist())
    return out


def smooth(x, y, n=10):
    """Gentle curve through the points (x always increases)."""
    xx = np.linspace(0, 1, 400)
    return xx, PchipInterpolator(x, y)(xx)


frames = {pid: frame(pid) for pid in PIDS}
wrong = {pid for pid, (ok, _) in HAND_CHECK.items() if not ok}

fig = plt.figure(figsize=(15, 9.2), facecolor="white")
gs = fig.add_gridspec(2, 5, height_ratios=[1.5, 1], hspace=0.38, wspace=0.18, left=0.06, right=0.98, top=0.86, bottom=0.06)
fig.text(0.04, 0.965, "Reasoning as pathfinding: 5 problems, human vs. LLM", fontsize=20, weight="bold", color=NAVY)
fig.text(0.04, 0.93, "Each line is one solution, from the problem (left) to its last step (right). Height = how far a step is from the "
         "correct answer in embedding space; lower is closer.", fontsize=13, color=MUTED)


def draw(ax, pids, big=False):
    ax.axhspan(-0.02, 0.15, xmin=0.93, xmax=1, color="#E6ECF4", zorder=0)
    for pid in pids:
        for src, col in (("human", HUMAN_C), ("model", LLM_C)):
            x, y, kinds = frames[pid][src]
            sx, sy = smooth(x, y)
            ax.plot(sx, sy, color=col, lw=1.8 if big else 1.6, alpha=0.8, zorder=2)
            for kd, c in (("explore", EXPLORE_C), ("return", LOOP_C)):
                idx = [i for i, k in enumerate(kinds) if k == kd]
                ax.scatter(x[idx], y[idx], s=22, color=c, zorder=3, edgecolor="white", lw=0.6)
            if src == "model" and pid in wrong:
                ax.scatter(x[-1], y[-1], marker="x", s=110, color="#C0392B", lw=3, zorder=4)
            else:
                ax.scatter(x[-1], y[-1], s=40, color=col, zorder=4, edgecolor="white", lw=0.8)
            ax.scatter([0], [y[0]], s=70 if big else 50, color="black", zorder=5)
    ax.set_xlim(-0.03, 1.07)
    ax.set_ylim(-0.02, max(1.05, max(frames[q][s][1].max() for q in pids for s in ("human", "model")) + 0.06))
    ax.set_xticks([0, 0.5, 1], ["problem", "halfway", "end"])
    ax.set_yticks([0, 0.5, 1])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11, colors=MUTED)
    if big:
        ax.text(1.065, 0.065, "correct\nanswer", ha="right", va="center", fontsize=11, color=NAVY, style="italic")


ax0 = fig.add_subplot(gs[0, :])
draw(ax0, PIDS, big=True)
ax0.set_ylabel("distance from the correct answer", fontsize=13, color=MUTED)
handles = [plt.Line2D([], [], color=HUMAN_C, lw=2.5, label="human (stage 1)"),
           plt.Line2D([], [], color=LLM_C, lw=2.5, label="LLM (QwQ-32B)"),
           plt.Line2D([], [], ls="", marker="o", color=EXPLORE_C, label="explore step"),
           plt.Line2D([], [], ls="", marker="o", color=LOOP_C, label="loop step"),
           plt.Line2D([], [], ls="", marker="x", color="#C0392B", mew=3, label="wrong final answer")]
ax0.legend(handles=handles, loc="lower left", frameon=False, fontsize=12, ncol=5, bbox_to_anchor=(0, 1.0))

for k, pid in enumerate(PIDS):
    ax = fig.add_subplot(gs[1, k])
    draw(ax, [pid])
    h, m = len(frames[pid]["human"][0]) - 1, len(frames[pid]["model"][0]) - 1
    ax.text(0, 1.02, f"steps: {h} human, {m} LLM" + (" · LLM wrong" if pid in wrong else ""),
            transform=ax.transAxes, fontsize=10.5, color="#C0392B" if pid in wrong else MUTED, va="bottom")
    ax.set_title(f"Problem {pid}", fontsize=13, color=NAVY, weight="bold", loc="left", pad=20)

for ext in ("png", "svg"):
    fig.savefig(HERE / f"reasoning_paths.{ext}", dpi=200, facecolor="white")
print("saved")
