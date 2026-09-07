"""
Replication of the zero-shot requirements classification approach from:

  Alhoshan, W., Ferrari, A., & Zhao, L. (2023).
  "Zero-shot learning for requirements classification: An exploratory study."
  Information and Software Technology, 159, 107202.
  https://doi.org/10.1016/j.infsof.2023.107202

Method
------
This implements the paper's embedding-based ZSL family: each requirement and
each candidate class LABEL are embedded with the same sentence encoder, then
the class whose label embedding has the highest cosine similarity to the
requirement embedding is assigned. No training and no labelled examples are
used at inference time, which is the paper's central claim.

    class(r) = argmax_c  cos( E(r), E(label_c) )
    cos(u,v) = (u . v) / (||u|| * ||v||)

Dataset
-------
The original PROMISE_exp NFR dataset (Cleland-Huang et al., 2007), obtained
from the NoRBERT repository that the paper's own notebook points to:
https://github.com/tobhey/NoRBERT
625 requirements, 12 classes (F + 11 NFR subtypes).

Note on scope
-------------
This is a single-configuration replication. The original paper ran 360+
experiments across 4 language models and 19 label configurations and reports
the best result. Numbers produced here are therefore expected to be lower;
the gap is itself the finding.
"""
import csv
import random

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score

random.seed(42)
np.random.seed(42)

CLASS_NAMES = {
    "F": "functional requirement",
    "A": "availability",
    "FT": "fault tolerance",
    "L": "legal",
    "LF": "look and feel",
    "MN": "maintainability",
    "O": "operational",
    "PE": "performance",
    "PO": "portability",
    "SC": "scalability",
    "SE": "security",
    "US": "usability",
}

TOP4_NFR = ["US", "SE", "O", "PE"]  # matches Table 13 in the paper

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH = "data/promise_nfr.csv"


def load_promise(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows.append({"text": r["RequirementText"].strip(),
                         "label": r["class"].strip()})
    return rows


def zsl_predict(model, texts, candidate_labels):
    label_emb = model.encode([CLASS_NAMES[c] for c in candidate_labels],
                             normalize_embeddings=True)
    text_emb = model.encode(texts, normalize_embeddings=True,
                            show_progress_bar=False)
    sims = text_emb @ label_emb.T
    return [candidate_labels[i] for i in sims.argmax(axis=1)]


def task_binary(model, rows):
    print("=" * 70)
    print("Task FR/NFR (binary)")
    print("=" * 70)
    texts = [r["text"] for r in rows]
    gold = ["F" if r["label"] == "F" else "NFR" for r in rows]
    labels = {"F": "functional requirement",
              "NFR": "non-functional requirement, quality constraint"}
    lbl = model.encode(list(labels.values()), normalize_embeddings=True)
    txt = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    pred = [list(labels)[i] for i in (txt @ lbl.T).argmax(axis=1)]
    f1 = f1_score(gold, pred, average="macro")
    print(f"n={len(rows)}  macro-F1 = {f1:.3f}   (paper reports ~0.66)")
    print(classification_report(gold, pred, digits=3))
    return f1


def task_top4(model, rows):
    print("=" * 70)
    print("Task NFR, top-4 classes (US / SE / O / PE)")
    print("=" * 70)
    sub = [r for r in rows if r["label"] in TOP4_NFR]
    gold = [r["label"] for r in sub]
    pred = zsl_predict(model, [r["text"] for r in sub], TOP4_NFR)
    f1 = f1_score(gold, pred, average="macro")
    print(f"n={len(sub)}  macro-F1 = {f1:.3f}   (paper reports 0.72-0.80)")
    print(classification_report(gold, pred, digits=3))
    return f1


if __name__ == "__main__":
    print(f"Loading sentence encoder: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    rows = load_promise(DATA_PATH)
    print(f"Loaded {len(rows)} requirements from {DATA_PATH}\n")

    binary_f1 = task_binary(model, rows)
    top4_f1 = task_top4(model, rows)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"PROMISE FR/NFR binary macro-F1 : {binary_f1:.3f}")
    print(f"PROMISE top-4 NFR macro-F1     : {top4_f1:.3f}")
