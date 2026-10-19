import numpy as np
import pandas as pd

DATASET = "nvidia/Nemotron-Math-HumanReasoning"


def load_solutions() -> pd.DataFrame:
    from datasets import load_dataset
    return prepare(load_dataset(DATASET, split="train").to_pandas())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["source"] = np.where(df["solution_type"].str.startswith("human"), "human", "model")
    df["problem_id"] = df.groupby("problem").ngroup()
    df["n_words"] = df["solution"].str.split().str.len()
    return df
