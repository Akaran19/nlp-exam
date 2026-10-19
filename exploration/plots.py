import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .stats import METRICS
from .trajectory import solution_matrix

COLORS = {"human": "#2a78d6", "model": "#eb6834"}
TITLES = {"explore_share": "Exploration share of marker words",
          "step_distance": "Mean distance between consecutive steps",
          "novelty": "Mean novelty of steps"}


def set_style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": 110})


def metric_boxplots(df, rng):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, m in zip(axes, METRICS):
        data = [df.loc[df["source"] == s, m].dropna() for s in ["human", "model"]]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": "black", "linewidth": 1.5})
        for patch, s in zip(bp["boxes"], ["human", "model"]):
            patch.set_facecolor(COLORS[s]); patch.set_alpha(0.35); patch.set_edgecolor(COLORS[s])
        for i, (d, s) in enumerate(zip(data, ["human", "model"]), start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=10, color=COLORS[s], alpha=0.6, zorder=3)
        ax.set_xticks([1, 2], [f"Human (n={len(data[0])})", f"Model (n={len(data[1])})"])
        ax.set_title(TITLES[m], fontsize=10)
    fig.suptitle("Exploration measures: human-written vs. QwQ-32B-Preview solutions", fontsize=12)
    fig.tight_layout()
    return fig


def example_trajectory(df, steps, X):
    example = df.loc[df["n_steps"] >= 8, "problem_id"].value_counts().index[0]
    fig, ax = plt.subplots(figsize=(9, 4))
    labelled = set()
    for _, row in df[df["problem_id"] == example].iterrows():
        Xs = solution_matrix(X, steps, row.name)
        if Xs.shape[0] < 3:
            continue
        d = 1 - np.diag(cosine_similarity(Xs), k=1)
        x = np.linspace(0, 1, len(d))
        is_model = row["source"] == "model"
        label = {"human": "Human (4 solutions)", "model": "Model (QwQ-32B-Preview)"}[row["source"]]
        ax.plot(x, d, color=COLORS[row["source"]], linewidth=2.5 if is_model else 1.5,
                alpha=1 if is_model else 0.6, label=None if label in labelled else label)
        labelled.add(label)
    ax.set_xlabel("Position in solution (0 = start, 1 = end)")
    ax.set_ylabel("Cosine distance to previous step")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Problem {example}: " + df.loc[df["problem_id"] == example, "problem"].iloc[0][:80] + "…",
                 fontsize=10)
    ax.legend(fontsize=9, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.38), frameon=False)
    fig.tight_layout()
    return fig
