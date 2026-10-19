import pandas as pd

MIN_WORDS = 15


def split_steps(text: str, min_words: int = MIN_WORDS) -> list[str]:
    """Join lines until a chunk has at least `min_words` words."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    steps, buffer = [], ""
    for line in lines:
        buffer = f"{buffer} {line}".strip()
        if len(buffer.split()) >= min_words:
            steps.append(buffer)
            buffer = ""
    if buffer:                       # leftover short text joins the last step
        if steps:
            steps[-1] += " " + buffer
        else:
            steps.append(buffer)
    return steps


def add_steps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["steps"] = df["solution"].apply(split_steps)
    df["n_steps"] = df["steps"].str.len()
    return df


def explode_steps(df: pd.DataFrame) -> pd.DataFrame:
    """One row per step; `sol_id` is the index of the solution in `df`."""
    steps = (df[["problem_id", "solution_type", "source", "steps"]]
             .explode("steps")
             .rename(columns={"steps": "text"}))
    steps["sol_id"] = steps.index
    steps["pos"] = steps.groupby(level=0).cumcount()
    return steps.reset_index(drop=True)
