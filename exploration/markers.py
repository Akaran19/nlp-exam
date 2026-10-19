import re

import pandas as pd

EXPLORE = [r"alternatively", r"another (?:approach|way|idea|method)", r"tr(?:y|ying|ied)", r"maybe",
           r"perhaps", r"what if", r"instead", r"wait", r"hmm+", r"guess", r"wonder", r"not sure",
           r"doesn'?t work", r"suppose"]
EXPLOIT = [r"therefore", r"thus", r"hence", r"it follows", r"substitut\w*", r"simplif\w*",
           r"which gives", r"we get", r"we obtain", r"plug\w* in", r"comput\w*", r"calculat\w*",
           r"conclude\w*"]

explore_re = re.compile(r"\b(?:" + "|".join(EXPLORE) + r")\b", re.IGNORECASE)
exploit_re = re.compile(r"\b(?:" + "|".join(EXPLOIT) + r")\b", re.IGNORECASE)


def add_marker_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_explore"] = df["solution"].str.count(explore_re)
    df["n_exploit"] = df["solution"].str.count(exploit_re)
    df["explore_per_1k"] = 1000 * df["n_explore"] / df["n_words"]
    df["exploit_per_1k"] = 1000 * df["n_exploit"] / df["n_words"]
    df["explore_share"] = df["n_explore"] / (df["n_explore"] + df["n_exploit"])
    return df
