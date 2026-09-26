# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "polars>=1.0",
#   "numpy",
#   "scikit-learn>=1.6",
#   "torch",
#   "transformers>=4.44",
#   "sentence-transformers>=3.0",
#   "sentencepiece",
#   "protobuf",
# ]
# ///
"""Classic baselines on the same splits and labeled subsets as run_ablation.py.
Runs on CrowdCent Cloud (gpu_s).

    crowdcent cloud run <project> --entrypoint run_local.py --envelope gpu_s

Every method gets the same tuning courtesy as GEPA: hyperparameters are chosen
inside the labeled budget only (CV for logistic regression, a stratified 20%
validation split for RoBERTa's learning rate). Writes out/abl/<cell>.{json,parquet}.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline

from metrics import write_run
from tasks import SIZES, TASKS, load_task, seeds_for, split_val, subsample

OUT = Path("out/abl")
GPU_USD_PER_HOUR = 1.10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDER = "BAAI/bge-base-en-v1.5"
NLI = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
FINETUNE = "FacebookAI/roberta-base"
FT_SIZES = [64, 256, 1024, 4000]
FT_LRS = [1e-5, 3e-5, 5e-5]
MIN_STEPS = 150


def gpu_usd(seconds):
    return seconds / 3600 * GPU_USD_PER_HOUR


def logreg(y):
    per_class = np.unique(y, return_counts=True)[1].min()
    return LogisticRegressionCV(
        Cs=[0.1, 1, 10, 100],
        cv=int(min(5, per_class)),
        scoring="neg_log_loss",
        max_iter=3000,
    )


def _align(proba, model_classes, classes):
    """Reorder columns to `classes`; classes missing from training get zero."""
    out = np.zeros((len(proba), len(classes)))
    for j, c in enumerate(model_classes):
        out[:, classes.index(c)] = proba[:, j]
    return out


def run_tfidf(name, train, test, classes):
    for n in SIZES:
        for seed in seeds_for(n):
            sub = subsample(train, n, seed)
            t = time.time()
            model = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                logreg(sub["label"].to_numpy()),
            ).fit(sub["text"].to_list(), sub["label"].to_list())
            fit_secs = time.time() - t
            t = time.time()
            proba = _align(
                model.predict_proba(test["text"].to_list()), model.classes_, classes
            )
            secs = time.time() - t
            write_run(
                OUT,
                name,
                "tfidf_lr",
                n,
                seed,
                test,
                proba,
                classes,
                seconds=secs,
                usd_per_1k=gpu_usd(secs) / len(test) * 1000,
                tune_seconds=fit_secs,
                tune_usd=gpu_usd(fit_secs),
            )


def run_embed(name, train, test, classes, embedder):
    t = time.time()
    test_emb = embedder.encode(
        test["text"].to_list(), batch_size=128, normalize_embeddings=True
    )
    secs = time.time() - t
    emb = dict(
        zip(
            train["text"].to_list(),
            embedder.encode(
                train["text"].to_list(), batch_size=128, normalize_embeddings=True
            ),
        )
    )
    for n in SIZES:
        for seed in seeds_for(n):
            sub = subsample(train, n, seed)
            X = np.stack([emb[x] for x in sub["text"]])
            t = time.time()
            clf = logreg(sub["label"].to_numpy()).fit(X, sub["label"].to_numpy())
            fit_secs = time.time() - t
            proba = _align(clf.predict_proba(test_emb), clf.classes_, classes)
            write_run(
                OUT,
                name,
                "embed_lr",
                n,
                seed,
                test,
                proba,
                classes,
                seconds=secs,
                usd_per_1k=gpu_usd(secs) / len(test) * 1000,
                tune_seconds=fit_secs,
                tune_usd=gpu_usd(fit_secs),
            )


def run_nli(name, task, test, classes, nli):
    labels = [task.descriptions[c] for c in classes]
    t = time.time()
    out = nli(
        test["text"].to_list(),
        candidate_labels=labels,
        hypothesis_template="This text is {}.",
        multi_label=False,
        batch_size=32,
    )
    secs = time.time() - t
    proba = np.array(
        [[dict(zip(o["labels"], o["scores"]))[lab] for lab in labels] for o in out]
    )
    write_run(
        OUT,
        name,
        "zeroshot_nli",
        0,
        0,
        test,
        proba,
        classes,
        seconds=secs,
        usd_per_1k=gpu_usd(secs) / len(test) * 1000,
        tune_seconds=0.0,
        tune_usd=0.0,
    )


def _batches(tok, texts, bs):
    for i in range(0, len(texts), bs):
        yield tok(
            texts[i : i + bs],
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)


def _train(tok, texts, y, lr, seed, n_classes):
    from transformers import AutoModelForSequenceClassification

    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        FINETUNE, num_labels=n_classes
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    bs = 16
    per_epoch = math.ceil(len(texts) / bs)
    epochs = max(3, math.ceil(MIN_STEPS / per_epoch))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * per_epoch, pct_start=0.1
    )
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(texts))
        for i in range(0, len(texts), bs):
            b = perm[i : i + bs]
            enc = tok(
                [texts[j] for j in b],
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
            model(**enc, labels=y[b].to(DEVICE)).loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
    model.eval()
    return model


def _predict(model, tok, texts):
    probs = []
    with torch.no_grad():
        for enc in _batches(tok, texts, 64):
            probs.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
    return np.vstack(probs)


def run_finetune(name, train, test, classes):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(FINETUNE)
    label_id = {c: i for i, c in enumerate(classes)}
    for n in FT_SIZES:
        for seed in seeds_for(n):
            labeled = subsample(train, n, seed)
            fit, val = split_val(labeled, 0.2, seed)
            y_fit = torch.tensor([label_id[c] for c in fit["label"]])
            y_val = np.array([label_id[c] for c in val["label"]])
            t = time.time()
            best = None
            for lr in FT_LRS:
                model = _train(
                    tok, fit["text"].to_list(), y_fit, lr, seed, len(classes)
                )
                p = np.clip(_predict(model, tok, val["text"].to_list()), 1e-6, 1)
                val_loss = -np.log(p[np.arange(len(y_val)), y_val]).mean()
                if best is None or val_loss < best[0]:
                    best = (val_loss, lr, model)
                del model
                torch.cuda.empty_cache()
            fit_secs = time.time() - t
            t = time.time()
            proba = _predict(best[2], tok, test["text"].to_list())
            secs = time.time() - t
            write_run(
                OUT,
                name,
                "roberta_ft",
                n,
                seed,
                test,
                proba,
                classes,
                seconds=secs,
                usd_per_1k=gpu_usd(secs) / len(test) * 1000,
                tune_seconds=fit_secs,
                tune_usd=gpu_usd(fit_secs),
                lr=best[1],
            )
            del best
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--methods", default="tfidf,embed,nli,finetune")
    args, _ = ap.parse_known_args()
    methods = args.methods.split(",")

    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

    print("device", DEVICE, flush=True)
    embedder = (
        SentenceTransformer(EMBEDDER, device=DEVICE) if "embed" in methods else None
    )
    nli = (
        pipeline(
            "zero-shot-classification", model=NLI, device=0 if DEVICE == "cuda" else -1
        )
        if "nli" in methods
        else None
    )
    for name in args.tasks.split(","):
        task = TASKS[name]
        train, test = load_task(name)
        classes = task.classes
        if "tfidf" in methods:
            run_tfidf(name, train, test, classes)
        if "embed" in methods:
            run_embed(name, train, test, classes, embedder)
        if "nli" in methods:
            run_nli(name, task, test, classes, nli)
        if "finetune" in methods:
            run_finetune(name, train, test, classes)
    print("done", flush=True)


main()
