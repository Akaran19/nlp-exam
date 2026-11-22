"""Pilot: explore/exploit on human vs. LLM olympiad solutions (Nemotron-Math-HumanReasoning).

Run from this folder:  python pipeline.py all      (or one stage: data split codebook label measures analyse check robust)
Every LLM call is cached in cache/ (keyed by model + prompt), so reruns cost nothing and give identical labels.

LLM access: no API key on this machine, so calls go through the local Claude Code CLI in print mode
(`claude -p`, no tools, no settings, custom system prompt). The CLI does not expose temperature, so
"temperature 0" from the plan is NOT enforced; determinism comes from the cache instead.
"""
import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score

# ----------------------------------------------------------------------------------------------------------------
# Models and prompts (quoted verbatim in the methods)
# ----------------------------------------------------------------------------------------------------------------
CODEBOOK_MODEL = "sonnet"   # Claude Sonnet (alias resolved by the CLI; the resolved id is saved per row as label_model)
LABEL_MODEL = "sonnet"
SECOND_LABEL_MODEL = "haiku"  # Step 6: independent second run with a different model, same prompt
WORKERS = 16
SEED = 42

SYSTEM_PROMPT = "You are a careful annotator of mathematical reasoning. Reply with a single JSON object and nothing else."

CODEBOOK_PROMPT = """Below is an olympiad-level math problem and five written solutions to it (in random order; some may be \
partial edits of each other).

List the 3 to 6 distinct mathematical STRATEGIES that appear across these solutions. A strategy is a line of attack, \
e.g. "substitute x=0", "parity argument", "try small cases", "induction on n", "AM-GM", "construct counterexample". \
Keep names short and specific to this problem. Do not list restating the problem, routine algebra, checking an answer \
or stating the final answer as strategies (those are covered by a separate fixed label "neutral").

Return JSON: {{"strategies": [{{"label": "<short_snake_case_name>", "description": "<one sentence>"}}, ...]}}

PROBLEM:
{problem}

{solutions}"""

NEUTRAL_DESCRIPTION = ("restating the problem, routine arithmetic/algebra that continues the current line, "
                       "checking/verifying an answer, stating the final answer")

LABEL_PROMPT = """You are labelling one step of a written solution to a competition math problem.

PROBLEM:
{problem}

CODEBOOK (strategies for this problem):
{codebook}
- neutral: {neutral}

Instructions:
- Give exactly one label from the codebook (or "neutral") for the CURRENT STEP.
- Label the strategy the step is using, not one it merely mentions.
- A step that continues the current strategy with routine computation gets that strategy, not "neutral".
- Use "neutral" only for restating, verifying, or stating the answer.

{context}CURRENT STEP:
{step}

Return JSON: {{"label": "<one label from the codebook or neutral>", "reason": "<one line>"}}"""

# Step 4 markers. "but wait" is listed before "wait" so each occurrence is counted once.
MARKERS = [r"but wait", r"wait", r"alternatively", r"let me try", r"another approach", r"let['’]s try", r"hmm+"]
MARKER_RE = re.compile(r"\b(?:" + "|".join(MARKERS) + r")\b", re.IGNORECASE)

HERE = Path(__file__).parent
CACHE = HERE / "cache"
TYPES = ["human_stage1", "human_stage2", "human_stage3", "human_stage4", "qwq_32b_preview"]
MIN_CHARS = 80


# ----------------------------------------------------------------------------------------------------------------
# LLM wrapper
# ----------------------------------------------------------------------------------------------------------------
def llm(prompt: str, model: str) -> dict:
    CACHE.mkdir(exist_ok=True)
    key = hashlib.sha1(f"{model}\n{SYSTEM_PROMPT}\n{prompt}".encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    cmd = ["claude", "-p", "--model", model, "--tools", "", "--strict-mcp-config", "--setting-sources", "",
           "--no-session-persistence", "--system-prompt", SYSTEM_PROMPT, "--output-format", "json"]
    for attempt in range(3):
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600)
        try:
            out = json.loads(r.stdout)
            if out.get("is_error"):
                raise ValueError(out.get("result"))
            text = out["result"]
            parsed = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            rec = {"model": next(iter(out["modelUsage"])), "parsed": parsed, "raw": text}
            path.write_text(json.dumps(rec))
            return rec
        except Exception as e:  # noqa: BLE001 - retry on any CLI/JSON failure, then give up loudly
            err = f"{e} | stdout={r.stdout[:300]} | stderr={r.stderr[:300]}"
    return {"model": model, "parsed": None, "raw": f"FAILED: {err}"}


def llm_many(prompts, model):
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(lambda p: llm(p, model), prompts))


# ----------------------------------------------------------------------------------------------------------------
# Step 0: data
# ----------------------------------------------------------------------------------------------------------------
def load() -> pd.DataFrame:
    raw = HERE / "raw.jsonl"
    if not raw.exists():
        from datasets import load_dataset
        load_dataset("nvidia/Nemotron-Math-HumanReasoning", split="train").to_pandas().to_json(
            raw, orient="records", lines=True)
    df = pd.read_json(raw, lines=True)
    df["problem_id"] = pd.factorize(df["problem"])[0]  # order of first appearance
    return df


def describe(df):
    print("rows per solution_type:\n", df.solution_type.value_counts().sort_index().to_string())
    print("\nsolution length (characters):")
    print(df.assign(n=df.solution.str.len()).groupby("solution_type").n.agg(["median", "min", "max"]).to_string())


# ----------------------------------------------------------------------------------------------------------------
# Step 1: split into steps
# ----------------------------------------------------------------------------------------------------------------
DISPLAY_ENVS = r"(?:equation|align|gather|aligned|eqnarray|multline)\*?"
DISPLAY_SEG = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]|\\begin\{(" + DISPLAY_ENVS + r")\}.*?\\end\{\1\}", re.S)


def _unclosed(s: str) -> bool:
    return (s.count("$$") % 2 == 1 or s.count(r"\[") > s.count(r"\]")
            or len(re.findall(r"\\begin\{" + DISPLAY_ENVS + r"\}", s)) > len(re.findall(r"\\end\{" + DISPLAY_ENVS + r"\}", s)))


def blocks(text: str) -> list[str]:
    """Split at line breaks, but never inside a display-maths block that spans several lines.

    The plan said "split at blank lines", but human solutions contain NO blank lines (paragraphs are separated by a
    single newline), while QwQ separates paragraphs by a blank line. Splitting at any line break treats both the
    same way."""
    out, buf = [], None
    for line in text.split("\n"):
        if buf is not None:
            buf.append(line)
            if not _unclosed("\n".join(buf)):
                out.append("\n".join(buf).strip())
                buf = None
        elif line.strip():
            if _unclosed(line):
                buf = [line]
            else:
                out.append(line.strip())
    if buf:
        out.append("\n".join(buf).strip())
    return out


def is_display_only(p: str) -> bool:
    return re.sub(r"[\s.,;:]", "", DISPLAY_SEG.sub("", p)) == ""


def glue_short(parts: list[str], min_chars=MIN_CHARS) -> list[str]:
    out, carry = [], ""
    for p in parts:
        p = f"{carry}\n{p}" if carry else p
        carry = ""
        if len(p) < min_chars:
            carry = p
        else:
            out.append(p)
    if carry:
        if out:
            out[-1] += "\n" + carry
        else:
            out.append(carry)
    return out


def split_paragraphs(text: str) -> list[str]:
    paras = []
    for b in blocks(text):
        if is_display_only(b) and paras:
            paras[-1] += "\n" + b
        else:
            paras.append(b)
    return glue_short(paras)


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\$(\[])")


def split_sentences(text: str) -> list[str]:
    """Step 7 alternative: sentence-level steps. Maths (inline and display) is masked so it is never cut;
    a display-only block stays attached to the previous sentence. No 80-character rule."""
    sents = []
    for b in blocks(text):
        if is_display_only(b) and sents:
            sents[-1] += "\n" + b
            continue
        masks = []

        def keep(m):
            masks.append(m.group(0))
            return f"\x00{len(masks) - 1}\x00"
        masked = re.sub(r"\$[^$]+\$|\\\(.*?\\\)", keep, DISPLAY_SEG.sub(keep, b))
        for s in _SENT.split(masked):
            s = re.sub(r"\x00(\d+)\x00", lambda m: masks[int(m.group(1))], s).strip()
            if s:
                sents.append(s)
    return sents


def make_steps(df, splitter=split_paragraphs) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        for i, t in enumerate(splitter(r.solution)):
            rows.append({"problem_id": r.problem_id, "solution_type": r.solution_type, "step_idx": i,
                         "n_chars": len(t), "text": t})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# Step 2: codebook
# ----------------------------------------------------------------------------------------------------------------
def build_codebook(df) -> dict:
    rng = np.random.default_rng(SEED)
    prompts, pids = [], []
    for pid, g in df.groupby("problem_id"):
        g = g.iloc[rng.permutation(len(g))]  # hide which solution is human / model
        sols = "\n\n".join(f"SOLUTION {i + 1}:\n{s}" for i, s in enumerate(g.solution))
        prompts.append(CODEBOOK_PROMPT.format(problem=g.problem.iloc[0], solutions=sols))
        pids.append(int(pid))
    res = llm_many(prompts, CODEBOOK_MODEL)
    cb = {}
    for pid, r in zip(pids, res):
        strategies = (r["parsed"] or {}).get("strategies", [])
        cb[pid] = {"strategies": strategies, "neutral": NEUTRAL_DESCRIPTION, "model": r["model"]}
    return cb


def codebook_text(entry) -> str:
    return "\n".join(f"- {s['label']}: {s['description']}" for s in entry["strategies"])


# ----------------------------------------------------------------------------------------------------------------
# Step 3: label steps
# ----------------------------------------------------------------------------------------------------------------
def label_prompts(steps, df, cb, context=True):
    problems = df.drop_duplicates("problem_id").set_index("problem_id").problem
    prompts = []
    for (pid, st), g in steps.groupby(["problem_id", "solution_type"], sort=False):
        texts = g.text.tolist()
        for i, t in enumerate(texts):
            ctx = ""
            if context:
                prev = texts[max(0, i - 2):i]
                ctx = "".join(f"PREVIOUS STEP ({len(prev) - j} back, context only):\n{p}\n\n"
                              for j, p in enumerate(prev))
            prompts.append(((pid, st, g.step_idx.iloc[i]), LABEL_PROMPT.format(
                problem=problems[pid], codebook=codebook_text(cb[pid]), neutral=NEUTRAL_DESCRIPTION,
                context=ctx, step=t)))
    return prompts


def label(steps, df, cb, model=LABEL_MODEL, context=True) -> pd.DataFrame:
    keyed = label_prompts(steps, df, cb, context)
    res = llm_many([p for _, p in keyed], model)
    valid = {pid: {s["label"] for s in e["strategies"]} | {"neutral"} for pid, e in cb.items()}
    out = []
    for (key, _), r in zip(keyed, res):
        p = r["parsed"] or {}
        lab = str(p.get("label", "")).strip()
        if lab not in valid[key[0]]:
            lab = f"INVALID:{lab}" if p else "FAILED"
        out.append({"problem_id": key[0], "solution_type": key[1], "step_idx": key[2], "label": lab,
                    "reason": p.get("reason", r["raw"][:200]), "label_model": r["model"]})
    return steps.merge(pd.DataFrame(out), on=["problem_id", "solution_type", "step_idx"])


# ----------------------------------------------------------------------------------------------------------------
# Step 4: measures
# ----------------------------------------------------------------------------------------------------------------
_ENCODER = None


def vendi(texts) -> float:
    global _ENCODER
    if len(texts) == 0:
        return np.nan
    if _ENCODER is None:
        from minilm import MiniLM
        _ENCODER = MiniLM()
    E = _ENCODER.encode(texts)
    lam = np.linalg.eigvalsh(E @ E.T / len(texts))
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


def episode_measures(g: pd.DataFrame, with_vendi=True) -> dict:
    g = g.sort_values("step_idx")
    starts = g.n_chars.cumsum() - g.n_chars
    total = g.n_chars.sum()
    nn = g[g.label != "neutral"]
    labs, ep_start = [], []
    for lab, s in zip(nn.label, starts[nn.index]):
        if not labs or lab != labs[-1]:
            labs.append(lab)
            ep_start.append(s)
    m = {"n_steps": len(g), "n_chars": int(total), "n_nonneutral": len(nn), "n_episodes": len(labs),
         "n_switches": max(len(labs) - 1, 0), "n_strategies": len(set(labs)),
         "first_switch_pos": ep_start[1] / total if len(labs) > 1 else np.nan,
         "episodes": " | ".join(labs)}
    if with_vendi:
        m["vendi"] = vendi(nn.text.tolist())
    return m


def last_boxed(s: str):
    i = s.rfind(r"\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i + 7:j - 1]


def norm_ans(a: str) -> str:
    a = re.sub(r"\\left|\\right|\\!|\\,|\\;|\$|\s|\\text\{[^}]*\}|\\displaystyle", "", str(a))
    return a.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac").rstrip(".")


def auto_correct(sol: str, expected: str) -> bool:
    b = last_boxed(sol)
    return b is not None and norm_ans(b) == norm_ans(expected)


# Hand check of every QwQ solution that failed the automatic check (filled in after reading them; see report).
# value: (correct?, note)
HAND_CHECK: dict[int, tuple[bool, str]] = {
    1: (True, '2*sqrt(3)/3 = sqrt(4/3)'),
    3: (True, 'same answer, different formatting'),
    4: (True, 'same answer, different formatting'),
    6: (True, 'same answer, different formatting'),
    9: (True, 'same answer, different formatting'),
    12: (True, 'same answer, different formatting'),
    13: (True, 'same answer, different formatting'),
    14: (True, 'same answer, different formatting'),
    15: (True, 'four lines normal to pi through incentre/excentres (expected gives coordinates)'),
    16: (True, 'same answer, different formatting'),
    17: (True, 'same answer, different formatting'),
    18: (True, '3 - 4*2^(-2^n) = 3 - 1/2^(2^n - 2)'),
    19: (True, 'same answer, different formatting'),
    20: (True, 'same answer, different formatting'),
    24: (True, 'same answer, different formatting'),
    27: (True, 'same answer, different formatting'),
    30: (True, 'same answer, different formatting'),
    33: (True, 'same answer, different formatting'),
    35: (True, 'same answer, different formatting'),
    36: (True, 'same answer, different formatting'),
    40: (True, 'same answer, different formatting'),
    45: (True, 'same answer, different formatting'),
    47: (True, 'same answer, different formatting'),
    48: (True, 'same answer, different formatting'),
    8: (False, "misses the constant solution f = 1"),
    10: (True, "same parametrisation up to naming/sign (k absorbs sign); gcd/parity conditions not stated"),
    11: (True, "does NOT match expected_answer, but expected_answer is wrong: (0, y, 2^y) works for every y"),
    28: (False, "claims all pairs except both odd > 1; 2x2 provably fails (checked by hand), answer is 'at least one odd'"),
    41: (False, "gives only 1125, expected 1125*m^6 for all m (incomplete)"),
    46: (False, "lists extra (spurious) solution pairs"),
}


def measures(labels, df) -> pd.DataFrame:
    rows = []
    for (pid, st), g in labels.groupby(["problem_id", "solution_type"]):
        sol = df[(df.problem_id == pid) & (df.solution_type == st)].iloc[0]
        m = {"problem_id": pid, "solution_type": st, **episode_measures(g)}
        m["marker_count"] = len(MARKER_RE.findall(sol.solution))
        m["marker_rate"] = 1000 * m["marker_count"] / len(sol.solution)
        if st == "qwq_32b_preview":
            m["correct_auto"] = auto_correct(sol.solution, sol.expected_answer)
            m["correct"] = HAND_CHECK[pid][0] if pid in HAND_CHECK else (m["correct_auto"] or np.nan)
        rows.append(m)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# Step 5: analyses
# ----------------------------------------------------------------------------------------------------------------
def paired(ms, a, b, cols):
    wide = ms.pivot(index="problem_id", columns="solution_type", values=cols)
    rows = []
    for c in cols:
        d = (wide[c][b] - wide[c][a]).dropna()
        p = wilcoxon(d).pvalue if (d != 0).sum() > 0 else np.nan
        rows.append({"measure": c, "comparison": f"{b} - {a}", "n_pairs": len(d),
                     f"median_{a}": wide[c][a].median(), f"median_{b}": wide[c][b].median(),
                     "median_diff": d.median(), "b_greater": int((d > 0).sum()), "equal": int((d == 0).sum()),
                     "b_less": int((d < 0).sum()), "wilcoxon_p": p})
    return pd.DataFrame(rows)


def analyse(ms) -> dict:
    main_cols = ["n_switches", "n_strategies", "vendi"]
    out = {"main": paired(ms, "human_stage1", "qwq_32b_preview", main_cols),
           "robust_stage4": paired(ms, "human_stage4", "qwq_32b_preview", main_cols)}
    val_cols = ["marker_rate", "n_strategies", "vendi", "n_switches", "first_switch_pos"]
    out["stage_medians"] = ms[ms.solution_type.str.startswith("human")].groupby("solution_type")[val_cols].median()
    out["stage_change"] = pd.concat([paired(ms, "human_stage1", s, val_cols) for s in ["human_stage2", "human_stage4"]])
    q = ms[ms.solution_type == "qwq_32b_preview"]
    out["correctness"] = q.groupby("correct").n_strategies.agg(["count", "median", "mean"])
    return out


# ----------------------------------------------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------------------------------------------
def pilot_problems():
    return sorted(np.random.default_rng(SEED).choice(50, 8, replace=False).tolist())


def main(stage):
    pd.set_option("display.width", 200, "display.max_columns", 20)
    df = load()
    if stage in ("data", "all"):
        describe(df)
    if stage in ("split", "all"):
        steps = make_steps(df)
        steps.to_csv(HERE / "steps.csv", index=False)
        print("\nsteps per solution:\n", steps.groupby(["solution_type", "problem_id"]).size()
              .groupby("solution_type").describe()[["min", "25%", "50%", "75%", "max"]].to_string())
    if stage in ("data", "split"):
        return
    steps = pd.read_csv(HERE / "steps.csv")
    if stage in ("codebook", "all"):
        cb = build_codebook(df)
        json.dump(cb, open(HERE / "codebook.json", "w"), indent=1, ensure_ascii=False)
        print("strategies per problem:", pd.Series([len(v["strategies"]) for v in cb.values()]).value_counts().to_dict())
    if stage == "codebook":
        return
    cb = {int(k): v for k, v in json.load(open(HERE / "codebook.json")).items()}
    if stage in ("label", "all"):
        lab = label(steps, df, cb)
        lab.to_csv(HERE / "labels.csv", index=False)
        print("labels:", lab.label.str.startswith(("INVALID", "FAILED")).sum(), "invalid/failed of", len(lab))
    if stage in ("measures", "all"):
        ms = measures(pd.read_csv(HERE / "labels.csv"), df)
        ms.to_csv(HERE / "measures.csv", index=False)
    if stage in ("analyse", "all"):
        res = analyse(pd.read_csv(HERE / "measures.csv"))
        for k, v in res.items():
            print(f"\n== {k} ==\n{v.round(3).to_string()}")
            v.to_csv(HERE / f"analysis_{k}.csv")
    if stage in ("check", "all"):
        check_labeller(df, steps, cb)
    if stage in ("robust", "all"):
        robust_split(df, cb)


def check_labeller(df, steps, cb):
    pids = pilot_problems()
    print("pilot problems (seed 42):", pids)
    sub = steps[steps.problem_id.isin(pids) & steps.solution_type.isin(["human_stage1", "qwq_32b_preview"])]
    sheet = sub.assign(codebook=sub.problem_id.map(lambda p: codebook_text(cb[p]) + "\n- neutral: " + NEUTRAL_DESCRIPTION),
                       my_label="", my_note="")
    sheet.to_csv(HERE / "hand_label_sheet.csv", index=False)
    a = pd.read_csv(HERE / "labels.csv").merge(sub[["problem_id", "solution_type", "step_idx"]])
    b = label(sub, df, cb, model=SECOND_LABEL_MODEL)
    b.to_csv(HERE / "labels_second_run.csv", index=False)
    m = a.merge(b, on=["problem_id", "solution_type", "step_idx"], suffixes=("_1", "_2"))
    rows = []
    for name, g in [("all", m), ("human_stage1", m[m.solution_type == "human_stage1"]),
                    ("qwq", m[m.solution_type == "qwq_32b_preview"])]:
        rows.append({"group": name, "n_steps": len(g), "agreement": (g.label_1 == g.label_2).mean(),
                     "kappa": cohen_kappa_score(g.label_1, g.label_2),
                     "neutral_share_run1": (g.label_1 == "neutral").mean(),
                     "neutral_share_run2": (g.label_2 == "neutral").mean()})
    k = pd.DataFrame(rows)
    k.to_csv(HERE / "analysis_kappa.csv", index=False)
    print(k.round(3).to_string())
    dis = m[m.label_1 != m.label_2].sample(min(5, (m.label_1 != m.label_2).sum()), random_state=SEED)
    dis[["problem_id", "solution_type", "step_idx", "text_1", "label_1", "reason_1", "label_2", "reason_2"]].to_csv(
        HERE / "analysis_disagreements.csv", index=False)
    for r in dis.itertuples():
        print(f"\n--- problem {r.problem_id} {r.solution_type} step {r.step_idx}\n{r.text_1[:400]}\n"
              f"  run1 ({LABEL_MODEL}): {r.label_1} - {r.reason_1}\n  run2 ({SECOND_LABEL_MODEL}): {r.label_2} - {r.reason_2}")


def robust_split(df, cb):
    pids = pilot_problems()
    sub = df[df.problem_id.isin(pids) & df.solution_type.isin(["human_stage1", "qwq_32b_preview"])]
    s_steps = make_steps(sub, split_sentences)
    s_lab = label(s_steps, df, cb)
    s_lab.to_csv(HERE / "labels_sentences.csv", index=False)
    para = pd.read_csv(HERE / "labels.csv")
    rows = []
    for (pid, st), g in s_lab.groupby(["problem_id", "solution_type"]):
        ms = episode_measures(g, with_vendi=False)
        mp = episode_measures(para[(para.problem_id == pid) & (para.solution_type == st)], with_vendi=False)
        rows.append({"problem_id": pid, "solution_type": st, "steps_para": mp["n_steps"], "steps_sent": ms["n_steps"],
                     "switches_para": mp["n_switches"], "switches_sent": ms["n_switches"],
                     "strategies_para": mp["n_strategies"], "strategies_sent": ms["n_strategies"]})
    r = pd.DataFrame(rows)
    r.to_csv(HERE / "analysis_split_robustness.csv", index=False)
    print(r.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
