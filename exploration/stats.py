import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["explore_share", "step_distance", "novelty"]


def paired_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Per problem: mean of the 4 human solutions vs. the model solution (Wilcoxon signed-rank)."""
    paired = df.groupby(["problem_id", "source"])[METRICS + ["n_words"]].mean().unstack("source")
    results = []
    for m in METRICS + ["n_words"]:
        sub = paired[m].dropna()
        _, p = wilcoxon(sub["human"], sub["model"])
        results.append({"metric": m, "problems": len(sub),
                        "human (median)": sub["human"].median(), "model (median)": sub["model"].median(),
                        "human > model (share of problems)": (sub["human"] > sub["model"]).mean(),
                        "Wilcoxon p": p})
    return pd.DataFrame(results).set_index("metric").round(3)


def order_check(df: pd.DataFrame) -> pd.DataFrame:
    """Real vs. shuffled step distance."""
    order = df.groupby("source")[["step_distance", "step_distance_shuffled"]].mean().round(3)
    order["real - shuffled"] = (order["step_distance"] - order["step_distance_shuffled"]).round(3)
    return order
