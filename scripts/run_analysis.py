"""Run the full analysis and write tables and figures to results/."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exploration import plots
from exploration.data import load_solutions
from exploration.markers import add_marker_features
from exploration.stats import order_check, paired_comparison
from exploration.steps import add_steps, explode_steps
from exploration.trajectory import add_trajectory_metrics, fit_tfidf


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    plots.set_style()

    df = add_marker_features(add_steps(load_solutions()))
    steps = explode_steps(df)
    _, X = fit_tfidf(steps["text"])
    print(f"{X.shape[0]} steps x {X.shape[1]} vocabulary terms")
    df = add_trajectory_metrics(df, steps, X, rng)

    results, order = paired_comparison(df), order_check(df)
    print(results, order, sep="\n\n")
    results.to_csv(out / "paired_comparison.csv")
    order.to_csv(out / "order_check.csv")
    df.drop(columns=["steps"]).to_csv(out / "solution_metrics.csv", index=False)

    plots.metric_boxplots(df, rng).savefig(out / "metric_boxplots.png", dpi=150)
    plots.example_trajectory(df, steps, X).savefig(out / "example_trajectory.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
