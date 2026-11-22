"""Figure 1: the pilot pipeline in one picture, drawn from the real pilot outputs (labels.csv, measures.csv)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})

HUMAN_BAND, LLM_BAND, VALID_BAND = "#FFF4E3", "#EEF0FB", "#F4F4F4"
HUMAN_C, LLM_C = "#C77A12", "#3F51B5"
STRAT_COLORS = ["#4A90D9", "#E39B2B", "#6B2C91", "#7CC242", "#C9485B", "#2BB3A3"]  # last two only if needed
NEUTRAL = "#D0D0D0"
H, Q = "human_stage1", "qwq_32b_preview"
STAGES = ["human_stage1", "human_stage2", "human_stage3", "human_stage4"]

labels = pd.read_csv(HERE / "labels.csv")
ms = pd.read_csv(HERE / "measures.csv")


# ---------------------------------------------------------------- pick the example problem
def pick_problem():
    """Both solutions 6-14 non-neutral steps; prefer few distinct labels (<= 4 colours) and few total steps."""
    w = ms[ms.solution_type.isin([H, Q])].pivot(index="problem_id", columns="solution_type")
    ok = w[(w.n_nonneutral[H].between(6, 14)) & (w.n_nonneutral[Q].between(6, 14))].index
    cands = []
    for pid in ok:
        sub = labels[(labels.problem_id == pid) & labels.solution_type.isin([H, Q])]
        n_lab = sub.loc[sub.label != "neutral", "label"].nunique()
        cands.append((n_lab > 4, sub.groupby("solution_type").size().max(), pid))
    return sorted(cands)[0][2] if cands else None


PID = pick_problem()
seqs = {st: labels[(labels.problem_id == PID) & (labels.solution_type == st)].sort_values("step_idx").label.tolist()
        for st in (H, Q)}
used = [l for l in dict.fromkeys(seqs[H] + seqs[Q]) if l != "neutral"]
COLOR = {l: STRAT_COLORS[i] for i, l in enumerate(used)} | {"neutral": NEUTRAL}

fig = plt.figure(figsize=(18, 7.5), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 18)
ax.set_ylim(0, 7.5)
ax.axis("off")


def box(x, y, w, h, fc, ec="none", r=0.08, lw=1.0, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(x0, x1, y, color="#555555"):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=16, lw=1.6, color=color, zorder=5))


def squares(seq, x0, ytop, width, colored=True, max_per_row=None, ticks=False):
    """Row(s) of squares, wrapping; returns the y of the lowest row."""
    n = len(seq)
    per_row = max_per_row or n
    size = min(0.22, width / per_row * 0.86)
    gap = size * 0.16
    y = ytop
    for i, lab in enumerate(seq):
        r, c = divmod(i, per_row)
        x, y = x0 + c * (size + gap), ytop - r * (size + gap * 2.2)
        ax.add_patch(Rectangle((x, y - size), size, size, fc=COLOR[lab] if colored else "#B5B5B5",
                               ec="white", lw=0.6, zorder=3))
        if ticks and i > 0 and lab != seq[i - 1]:
            xt = x - gap / 2
            ax.plot([xt, xt], [y - size - 0.07, y + 0.07], color="black", lw=2.0, zorder=4)
    return y - size


# ---------------------------------------------------------------- title + bands
ax.text(0.25, 7.18, "Do LLMs explore more than humans when solving olympiad problems?", fontsize=17, weight="bold",
        va="center")
ax.text(0.25, 6.80, "Explore = abandon the current strategy and try another. Pilot pipeline: same problems, "
        "two kinds of solver, one labelling scheme.", fontsize=12, color="#444444", va="center")

LANES = {H: (4.30, 2.05, HUMAN_BAND, HUMAN_C, "Humans"), Q: (2.10, 2.05, LLM_BAND, LLM_C, "LLM")}
for st, (y0, h, fc, ec, _) in LANES.items():
    box(0.15, y0, 13.55, h, fc, r=0.12, z=0)

# column headers
COLS = [(0.25, 2.75, "0  Data"), (3.35, 3.0, "①  Split into paragraphs"), (6.75, 3.05, "②  Label each paragraph"),
        (10.2, 3.35, "③  Merge into episodes & count"), (14.05, 3.8, "④  Compare, paired by problem")]
for x, w, t in COLS:
    ax.text(x, 6.48, t, fontsize=13, weight="bold", va="center")

# ---------------------------------------------------------------- column 0: data
for st, (y0, h, fc, ec, name) in LANES.items():
    box(0.35, y0 + 0.35, 2.55, h - 0.7, "white", ec=ec, lw=1.6, r=0.1, z=2)
    ax.text(1.62, y0 + h / 2 + 0.28, name, ha="center", va="center", fontsize=14, weight="bold", color=ec, zorder=3)
    ax.text(1.62, y0 + h / 2 - 0.12, st, ha="center", va="center", fontsize=11, family="DejaVu Sans Mono", zorder=3)
    ax.text(1.62, y0 + h / 2 - 0.47, "50 solutions" if st == H else "50 solutions (QwQ-32B)", ha="center",
            va="center", fontsize=10.5, color="#555555", zorder=3)
ax.text(1.62, 6.17, "Same 50 olympiad problems\nNemotron-Math-HumanReasoning", ha="center", va="center", fontsize=10.5,
        color="#333333")

# ---------------------------------------------------------------- columns 1-3: the real example
for st, (y0, h, fc, ec, name) in LANES.items():
    seq = seqs[st]
    per_row = 18
    top = y0 + h - 0.42
    bottom = squares(seq, 3.4, top, 2.9, colored=False, max_per_row=per_row)
    ax.text(3.4, y0 + 0.25, f"{len(seq)} paragraphs", fontsize=11, color="#333333", va="center")
    squares(seq, 6.8, top, 2.9, colored=True, max_per_row=per_row)
    nn = [l for l in seq if l != "neutral"]
    ax.text(6.8, y0 + 0.25, f"{len(nn)} strategy-labelled, {len(seq) - len(nn)} neutral", fontsize=11,
            color="#333333", va="center")
    squares(nn, 10.3, y0 + h / 2 + 0.45, 3.2, colored=True, ticks=True)
    eps = [l for i, l in enumerate(nn) if i == 0 or l != nn[i - 1]]
    ax.text(10.3, y0 + h / 2 - 0.15, "neutral dropped; ticks = strategy change", fontsize=10.5, color="#555555",
            va="center")
    ax.text(10.3, y0 + 0.42, f"switches: {len(eps) - 1}  ·  strategies: {len(set(eps))}", fontsize=13,
            weight="bold", va="center")
    for xa in (2.95, 6.4, 9.95):
        arrow(xa, xa + 0.37, y0 + h / 2)
    arrow(13.62, 14.0, y0 + h / 2)

# legend for strategy colours (under the column 2 header)
for i, l in enumerate(used + ["neutral"]):
    xx, yy = 6.8 + (i % 3) * 2.3, 6.2 - (i // 3) * 0.24
    ax.add_patch(Rectangle((xx, yy - 0.075), 0.15, 0.15, fc=COLOR[l], ec="none", zorder=3))
    name = l.replace("_", " ")
    ax.text(xx + 0.2, yy, name if len(name) <= 26 else name[:25] + "…", fontsize=8.5, va="center", zorder=3)

# ---------------------------------------------------------------- column 4: paired dot plots (real numbers)
w = ms[ms.solution_type.isin([H, Q])].pivot(index="problem_id", columns="solution_type")
rng = np.random.default_rng(0)
for k, (col, title) in enumerate([("n_switches", "strategy switches"), ("n_strategies", "distinct strategies")]):
    a = fig.add_axes([(14.25 + k * 1.95) / 18, 2.45 / 7.5, 1.5 / 18, 3.55 / 7.5])
    hv, qv = w[col][H].to_numpy(float), w[col][Q].to_numpy(float)
    jit = rng.uniform(-0.12, 0.12, (2, len(hv))) * (0 if col == "n_switches" else 1)
    for i in range(len(hv)):
        a.plot([0, 1], [hv[i] + jit[0, i] * 0.5, qv[i] + jit[1, i] * 0.5], color="#9E9E9E", lw=0.6, alpha=0.6, zorder=1)
    a.scatter(np.zeros_like(hv) + jit[0] * 0.3, hv + jit[0, :] * 0.5, s=12, color=HUMAN_C, zorder=2)
    a.scatter(np.ones_like(qv) + jit[1] * 0.3, qv + jit[1, :] * 0.5, s=12, color=LLM_C, zorder=2)
    a.plot([-0.25, 0.25], [np.median(hv)] * 2, color="black", lw=2.2, zorder=3)
    a.plot([0.75, 1.25], [np.median(qv)] * 2, color="black", lw=2.2, zorder=3)
    a.set_xticks([0, 1], ["Human", "LLM"])
    a.set_xlim(-0.45, 1.45)
    a.set_title(title, fontsize=11)
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=10)
    d = qv - hv
    a.text(0.5, -0.17, f"LLM > human in {int((d > 0).sum())}/50", transform=a.transAxes, ha="center", fontsize=10)
ax.text(14.1, 2.02, "one grey line per problem; black bar = median", fontsize=10, color="#555555", va="center")

# ---------------------------------------------------------------- bottom band: validity check (real numbers)
box(0.15, 0.12, 17.7, 1.72, VALID_BAND, r=0.12, z=0)
ax.text(0.35, 1.5, "Validity check: stages 1–4 are the same human solutions, re-edited", fontsize=13, weight="bold",
        va="center")
ax.text(0.35, 1.0, "Stage 2 adds phase headers (“But wait…”, “Let's try another approach”), stage 3 adds a special-cases\n"
        "section, stage 4 reorders steps. Wording-based measures should jump; strategy-based measures should not.",
        fontsize=10.5, color="#333333", va="center")
med = ms[ms.solution_type.isin(STAGES)].groupby("solution_type")[["marker_rate", "n_strategies", "n_switches"]].median()
qmed = ms[ms.solution_type == Q][["marker_rate", "n_strategies", "n_switches"]].median()
for k, (col, title) in enumerate([("marker_rate", "marker words / 1k chars"), ("n_strategies", "distinct strategies"),
                                  ("n_switches", "strategy switches")]):
    a = fig.add_axes([(9.6 + k * 2.85) / 18, 0.33 / 7.5, 2.15 / 18, 1.25 / 7.5])
    a.plot([1, 2, 3, 4], med.loc[STAGES, col], "-o", color=HUMAN_C, lw=2, ms=5, label="human (median)")
    a.axhline(qmed[col], color=LLM_C, ls="--", lw=1.4, label="LLM (median)")
    a.set_xticks([1, 2, 3, 4], ["st.1", "st.2", "st.3", "st.4"], fontsize=9.5)
    a.tick_params(labelsize=9.5)
    a.set_title(title, fontsize=10.5, pad=3)
    a.spines[["top", "right"]].set_visible(False)
    lo, hi = min(med[col].min(), qmed[col]), max(med[col].max(), qmed[col])
    a.set_ylim(max(0, lo - 0.2 * (hi - lo + 1e-9)), hi + 0.25 * (hi - lo + 1e-9))
    if k == 0:
        a.legend(fontsize=8.5, frameon=False, loc="upper left", handlelength=1.6)

ax.text(17.85, 6.98, f"Example in ①–③: problem_id {PID} (real labels). Panels ④ and validity: pilot medians, n = 50 problems.",
        fontsize=9.5, color="#666666", ha="right", va="center")

for ext in ("png", "svg", "pdf"):
    fig.savefig(HERE / f"figure1_pipeline.{ext}", dpi=300, facecolor="white")
print("example problem:", PID, {k: len(v) for k, v in seqs.items()}, "labels used:", used)
