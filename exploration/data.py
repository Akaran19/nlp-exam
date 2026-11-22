import numpy as np
import pandas as pd

DATASET = "nvidia/Nemotron-Math-HumanReasoning"

# The four human "stages" are successive edits of ONE write-up per problem (~90-97% text overlap),
# with later stages adding templated headers ("Let's restate the problem...", "$ $" spacers).
# Stage 1 is the rawest human text, so it is the only human solution kept.
HUMAN_STAGE = "human_stage1"
MODEL = "qwq_32b_preview"


def load_solutions() -> pd.DataFrame:
    from datasets import load_dataset
    return prepare(load_dataset(DATASET, split="train").to_pandas())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["solution_type"].isin([HUMAN_STAGE, MODEL])].reset_index(drop=True)
    df["source"] = np.where(df["solution_type"] == HUMAN_STAGE, "human", "model")
    df["problem_id"] = df.groupby("problem").ngroup()
    df["n_words"] = df["solution"].str.split().str.len()
    return df
