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
"""Classic baselines on the same splits as run_api.py. Runs on CrowdCent Cloud (gpu_s).

    crowdcent cloud run <project> --entrypoint run_local.py --envelope gpu_s

Writes out/local/results.jsonl (one JSON row per task x method x n_labels).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from metrics import save_preds, score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tasks import TASKS, load_task

OUT = Path("out/local")
GPU_USD_PER_HOUR = 1.10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDER = "BAAI/bge-base-en-v1.5"
NLI = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
FINETUNE = "FacebookAI/roberta-base"


def subsample(train, n, seed):
    if n >= len(train):
        return train
    idx, _ = train_test_split(
        np.arange(len(train)), train_size=n, stratify=train["label"], random_state=seed
    )
    return train[np.sort(idx)]


def emit(rows, task, method, family, n, seeds, results, seconds, n_test):
    metrics = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    row = {
        "task": task,
        "method": method,
        "family": family,
        "n_labels": n,
        "n_seeds": seeds,
        "n_test": n_test,
        **metrics,
        "seconds": seconds,
        "usd_per_1k": seconds / 3600 * GPU_USD_PER_HOUR / n_test * 1000,
    }
    print(json.dumps(row), flush=True)
    rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def seeds_for(n):
    return [0, 1, 2] if n <= 256 else [0]


def run_tfidf(rows, name, train, test, classes, sizes):
    for n in sizes:
        results, secs = [], []
        for seed in seeds_for(n):
            sub = subsample(train, n, seed)
            model = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                LogisticRegression(max_iter=2000, C=10.0),
            ).fit(sub["text"].to_list(), sub["label"].to_list())
            t = time.time()
            proba = _align(
                model.predict_proba(test["text"].to_list()), model.classes_, classes
            )
            secs.append(time.time() - t)
            results.append(score(test["label"], proba, classes))
            if n == max(sizes):
                save_preds(OUT / f"preds-{name}-tfidf_lr.parquet", test, proba, classes)
        emit(
            rows,
            name,
            "tfidf_lr",
            "supervised",
            n,
            len(results),
            results,
            float(np.mean(secs)),
            len(test),
        )


def run_embed(rows, name, train, test, classes, sizes, embedder):
    t = time.time()
    test_emb = embedder.encode(
        test["text"].to_list(), batch_size=128, normalize_embeddings=True
    )
    embed_secs = time.time() - t
    train_emb = embedder.encode(
        train["text"].to_list(), batch_size=128, normalize_embeddings=True
    )
    for n in sizes:
        results = []
        for seed in seeds_for(n):
            idx = (
                np.arange(len(train))
                if n >= len(train)
                else np.sort(
                    train_test_split(
                        np.arange(len(train)),
                        train_size=n,
                        stratify=train["label"],
                        random_state=seed,
                    )[0]
                )
            )
            clf = LogisticRegression(max_iter=2000, C=10.0).fit(
                train_emb[idx], train["label"].to_numpy()[idx]
            )
            proba = _align(clf.predict_proba(test_emb), clf.classes_, classes)
            results.append(score(test["label"], proba, classes))
            if n == max(sizes):
                save_preds(OUT / f"preds-{name}-embed_lr.parquet", test, proba, classes)
        emit(
            rows,
            name,
            "embed_lr",
            "supervised",
            n,
            len(results),
            results,
            embed_secs,
            len(test),
        )


def run_nli(rows, name, task, test, classes, nli):
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
    save_preds(OUT / f"preds-{name}-zeroshot_nli.parquet", test, proba, classes)
    emit(
        rows,
        name,
        "zeroshot_nli",
        "zero-shot",
        0,
        1,
        [score(test["label"], proba, classes)],
        secs,
        len(test),
    )


def run_finetune(rows, name, train, test, classes, sizes, epochs=3):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(FINETUNE)
    label_id = {c: i for i, c in enumerate(classes)}
    for n in sizes:
        sub = subsample(train, n, 0)
        torch.manual_seed(0)
        model = AutoModelForSequenceClassification.from_pretrained(
            FINETUNE, num_labels=len(classes)
        ).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        texts, y = (
            sub["text"].to_list(),
            torch.tensor([label_id[c] for c in sub["label"]]),
        )
        bs = 16
        steps = max(1, epochs * ((len(texts) + bs - 1) // bs))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=2e-5, total_steps=steps, pct_start=0.1
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
                loss = model(**enc, labels=y[b].to(DEVICE)).loss
                loss.backward()
                opt.step()
                sched.step()
                opt.zero_grad()
        model.eval()
        t = time.time()
        probs = []
        with torch.no_grad():
            test_texts = test["text"].to_list()
            for i in range(0, len(test_texts), 64):
                enc = tok(
                    test_texts[i : i + 64],
                    truncation=True,
                    max_length=128,
                    padding=True,
                    return_tensors="pt",
                ).to(DEVICE)
                probs.append(
                    torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
                )
        secs = time.time() - t
        if n == max(sizes):
            save_preds(
                OUT / f"preds-{name}-roberta_ft.parquet",
                test,
                np.vstack(probs),
                classes,
            )
        emit(
            rows,
            name,
            "roberta_ft",
            "supervised",
            n,
            1,
            [score(test["label"], np.vstack(probs), classes)],
            secs,
            len(test),
        )
        del model, opt
        torch.cuda.empty_cache()


def _align(proba, model_classes, classes):
    """Reorder columns to `classes`; classes missing from training get zero."""
    out = np.zeros((len(proba), len(classes)))
    for j, c in enumerate(model_classes):
        out[:, classes.index(c)] = proba[:, j]
    return out


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

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    sizes = [16, 64, 256, 1024, 4000]
    for name in args.tasks.split(","):
        task = TASKS[name]
        train, test = load_task(name)
        classes = task.classes
        if "tfidf" in methods:
            run_tfidf(rows, name, train, test, classes, sizes)
        if "embed" in methods:
            run_embed(rows, name, train, test, classes, sizes, embedder)
        if "nli" in methods:
            run_nli(rows, name, task, test, classes, nli)
        if "finetune" in methods:
            run_finetune(rows, name, train, test, classes, [4000])
        (OUT / "results.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
    print("done", len(rows), flush=True)


main()
