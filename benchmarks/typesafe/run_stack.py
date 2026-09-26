# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "polars>=1.0",
#   "numpy",
#   "scikit-learn>=1.6",
#   "torch",
#   "sentence-transformers>=3.0",
# ]
# ///
"""E3 (PROTOCOL.md): logistic regression on Jev's zero-shot probabilities, alone and
stacked with bge embeddings, at every label budget. Runs on CrowdCent Cloud.

Inputs next to this file: <task>__train_jev.parquet and <task>__test_jev.parquet
(Jev probabilities from run_followup.py jevfeat and the ablation's jev_zeroshot cell).
Writes out/abl/<task>__{jev_lr,stack_lr}__n<n>__s<seed>.{json,parquet}.
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics import write_run
from tasks import SIZES, TASKS, load_task, seeds_for, subsample

OUT = Path("out/abl")
EMBEDDER = "BAAI/bge-base-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def logreg(y):
    per_class = np.unique(y, return_counts=True)[1].min()
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=[0.01, 0.1, 1, 10, 100],
            cv=int(min(5, per_class)),
            scoring="neg_log_loss",
            max_iter=3000,
        ),
    )


def jev_logp(path, texts, classes):
    p = pl.read_parquet(path)
    lookup = dict(
        zip(
            p["text"],
            np.log(np.clip(p.select([f"p_{c}" for c in classes]).to_numpy(), 1e-6, 1)),
        )
    )
    return np.stack([lookup[t] for t in texts])


def main():
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(EMBEDDER, device=DEVICE)
    for name, task in TASKS.items():
        classes = task.classes
        train, test = load_task(name)
        emb = dict(
            zip(
                train["text"],
                embedder.encode(
                    train["text"].to_list(), batch_size=128, normalize_embeddings=True
                ),
            )
        )
        test_emb = embedder.encode(
            test["text"].to_list(), batch_size=128, normalize_embeddings=True
        )
        train_jev = jev_logp(
            f"{name}__train_jev.parquet", train["text"].to_list(), classes
        )
        jev_row = dict(zip(train["text"], train_jev))
        test_jev = jev_logp(
            f"{name}__test_jev.parquet", test["text"].to_list(), classes
        )
        for n in SIZES:
            for seed in seeds_for(n):
                sub = subsample(train, n, seed)
                y = sub["label"].to_numpy()
                J = np.stack([jev_row[t] for t in sub["text"]])
                E = np.stack([emb[t] for t in sub["text"]])
                for method, X, X_test in [
                    ("jev_lr", J, test_jev),
                    ("stack_lr", np.hstack([E, J]), np.hstack([test_emb, test_jev])),
                ]:
                    model = logreg(y).fit(X, y)
                    proba = np.zeros((len(test), len(classes)))
                    for j, c in enumerate(model.classes_):
                        proba[:, classes.index(c)] = model.predict_proba(X_test)[:, j]
                    write_run(OUT, name, method, n, seed, test, proba, classes)
    print("done", flush=True)


main()
