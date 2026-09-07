# Zero-Shot Requirements Classification — Replication

An independent, single-configuration replication of:

> Alhoshan, W., Ferrari, A., & Zhao, L. (2023).
> *Zero-shot learning for requirements classification: An exploratory study.*
> **Information and Software Technology, 159**, 107202.
> https://doi.org/10.1016/j.infsof.2023.107202

## What this does

Implements the paper's embedding-based zero-shot classifier. Each requirement
and each candidate class **label** are embedded with the same sentence encoder;
the class whose label embedding has the highest cosine similarity to the
requirement embedding is assigned.

```
class(r) = argmax_c  cos( E(r), E(label_c) )
cos(u,v) = (u · v) / (||u|| · ||v||)
```

No training and no labelled examples are used at inference time — that is the
paper's central claim, and it holds.

## Dataset

The original **PROMISE_exp** NFR dataset (Cleland-Huang et al., 2007), taken
from the [NoRBERT repository](https://github.com/tobhey/NoRBERT) that the
paper's own notebook points to.

- 625 requirements
- 12 classes: `F` plus 11 NFR subtypes
- 255 functional / 370 non-functional

## Results

| Task | n | Paper | This replication |
|---|---|---|---|
| Functional vs non-functional | 625 | F1 ≈ 0.66 | **macro-F1 = 0.514** |
| Top-4 NFR (US/SE/O/PE) | 249 | F1 0.72–0.80 | **macro-F1 = 0.521** |

Per-class results for the top-4 task:

| Class | precision | recall | F1 | support |
|---|---|---|---|---|
| Operational | 0.352 | 0.500 | 0.413 | 62 |
| Performance | 0.667 | 0.444 | 0.533 | 54 |
| **Security** | 0.747 | 0.848 | **0.794** | 66 |
| Usability | 0.400 | 0.299 | 0.342 | 67 |

## Why these numbers are lower than the paper's

This is the main finding, and it is not an implementation bug:

1. The paper ran **360+ experiments** across 4 language models and 19 label
   configurations and reports the **best** result. This runs **one** simple
   configuration.
2. The paper itself finds that label-selection strategy materially affects
   performance; expert-curated labels beat raw ones. This uses raw,
   single-phrase labels.
3. The encoder here was chosen for general availability, not tuned for the task.

So reaching the published figure depends on deliberate label engineering and a
broad configuration search. **Removing the need for labelled data does not
remove the need for domain expertise — it changes what kind of expertise is
required.**

Security is by far the strongest class (F1 0.794) because security vocabulary
is explicit in requirements text; usability is weakest (0.342) because the
concept is usually implicit.

## Reproducing

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 zsl_replication.py
```

Random seed is fixed at 42. Raw output is in [`results/run_log.txt`](results/run_log.txt).

## What was not reproduced

- The full 360-experiment grid across all 4 models and 19 label configurations
- The paper's model-specific variants (e.g. BERT4RE), which need dedicated weights
- The paper's third task (security classification on the second dataset)
- ANOVA / Friedman significance tests, which are meaningless for a single configuration

## Layout

```
zsl_replication.py     replication script
data/promise_nfr.csv   PROMISE_exp dataset
results/run_log.txt    raw output of the run above
requirements.txt       dependencies
```

## References

1. Alhoshan, W., Ferrari, A., & Zhao, L. (2023). Zero-shot learning for requirements classification: An exploratory study. *Information and Software Technology, 159*, 107202.
2. Binkhonain, M., & Zhao, L. (2019). A review of machine learning algorithms for identification and classification of non-functional requirements. *Expert Systems with Applications: X, 1*, 100001.
3. Cleland-Huang, J., Settimi, R., Zou, X., & Solc, P. (2007). Automated classification of non-functional requirements. *Requirements Engineering, 12*(2), 103–120.
4. Hey, T., Keim, J., Koziolek, A., & Tichy, W. F. (2020). NoRBERT: Transfer learning for requirements classification. *Proc. IEEE RE*, 169–179.
5. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proc. EMNLP*.

## License

MIT
