import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

N_SHUFFLES = 20


def clean(text: str) -> str:
    text = re.sub(r"\\[a-zA-Z]+", " ", text)       # drop LaTeX commands
    return text.lower()


def fit_tfidf(step_texts):
    vectorizer = TfidfVectorizer(preprocessor=clean, stop_words="english", min_df=2,
                                 token_pattern=r"(?u)\b[a-z]{2,}\b")
    return vectorizer, vectorizer.fit_transform(step_texts)


def trajectory_metrics(Xs):
    """Mean consecutive-step distance and mean novelty for one solution."""
    n = Xs.shape[0]
    if n < 3:
        return np.nan, np.nan
    S = cosine_similarity(Xs)
    step_distance = 1 - np.diag(S, k=1)                       # step i vs. step i+1
    novelty = [1 - S[i, :i].max() for i in range(1, n)]       # step i vs. all earlier steps
    return step_distance.mean(), np.mean(novelty)


def solution_matrix(X, steps, sol_id):
    """TF-IDF rows of one solution, without steps that have no vocabulary words."""
    Xs = X[steps.index[steps["sol_id"] == sol_id].to_numpy()]
    return Xs[Xs.getnnz(axis=1) > 0]


def add_trajectory_metrics(df, steps, X, rng, n_shuffles=N_SHUFFLES):
    rows = []
    for sol_id, g in steps.groupby("sol_id"):
        Xs = X[g.index.to_numpy()]
        Xs = Xs[Xs.getnnz(axis=1) > 0]
        dist, nov = trajectory_metrics(Xs)
        shuffled = [trajectory_metrics(Xs[rng.permutation(Xs.shape[0])])[0] for _ in range(n_shuffles)]
        rows.append({"sol_id": sol_id, "step_distance": dist, "novelty": nov,
                     "step_distance_shuffled": np.nanmean(shuffled) if Xs.shape[0] >= 3 else np.nan})
    return df.join(pd.DataFrame(rows).set_index("sol_id"))
