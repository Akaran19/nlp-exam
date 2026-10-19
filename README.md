# nlp-exam
Repo for E26 NLP exam

## Exploration vs. exploitation in written mathematical reasoning

Compares 200 human-written and 50 model-written (QwQ-32B-Preview) solutions to 50 competition math problems from
[Nemotron-Math-HumanReasoning](https://huggingface.co/datasets/nvidia/Nemotron-Math-HumanReasoning) (CC BY-NC 4.0).

**Research question:** do human solutions show more *exploration* (trying and switching ideas) than model solutions,
or more *exploitation* (refining one line of attack)?

**Method:** split solutions into steps (>= 15 words) -> (1) exploration/exploitation marker words,
(2) TF-IDF step vectors -> consecutive-step distance and novelty, with a shuffled-order baseline ->
paired Wilcoxon test per problem.

### Layout
```
nbs/nemotron_exploration.ipynb   narrative notebook: dataset, RQ, analysis, assumptions
exploration/                     the same pipeline as a package
  data.py steps.py markers.py trajectory.py stats.py plots.py
scripts/run_analysis.py          runs everything, writes tables and figures to results/
```

### Usage
```bash
pip install -r requirements.txt
python scripts/run_analysis.py      # needs internet on first run to download the dataset
jupyter notebook nbs/nemotron_exploration.ipynb
```

Planned next: sentence embeddings instead of TF-IDF, and validation against a hand-labelled sample of steps.
