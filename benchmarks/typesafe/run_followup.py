"""Pre-registered follow-ups (PROTOCOL.md): E1 GEPA scaling, E2 GridSearchCV over GEPA
settings, and the Jev training-set probabilities that E3 stacks on CrowdCent Cloud.

    uv run python benchmarks/typesafe/run_followup.py e1 --tasks onion sst2
    uv run python benchmarks/typesafe/run_followup.py e2 --sizes 256 64 1024
    uv run python benchmarks/typesafe/run_followup.py control
    uv run python benchmarks/typesafe/run_followup.py jevfeat

E1/E2/control write results/abl/<task>__<method>__n<n>__s<seed>.{json,parquet}; skips cells already done.
"""

import argparse
import time
import traceback
from pathlib import Path

import dspy
import numpy as np
from centimators.model_estimators import DSPyMator, DSPyOptimizer
from joblib import parallel_backend
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from metrics import save_preds, write_run
from run_ablation import OUT, estimator, metric_for, signature, sol_lm, usd
from run_api import make_lm
from tasks import SEEDS, TASKS, load_task, split_val, subsample

FEATURES = Path(__file__).parent / "results" / "features"


def e1_cell(task_name, n, seed, budget=None, method="jev_gepa"):
    task = TASKS[task_name]
    train, test = load_task(task_name)
    labeled = subsample(train, n, seed)
    n_val = min(len(labeled) // 2, 256)
    fit, val = split_val(labeled, n_val / len(labeled), seed)
    budget = budget or 300 + 2 * min(n, 1024)
    est = estimator(task, "jev")
    t0 = time.time()
    with dspy.track_usage() as tune_usage:
        est.fit(
            fit.select("text"),
            fit["label"],
            optimizer=dspy.GEPA(
                metric=metric_for(task),
                reflection_lm=sol_lm(),
                max_metric_calls=budget,
                num_threads=16,
                seed=seed,
            ),
            validation_data=(val.select("text"), val["label"]),
        )
    tune_seconds = time.time() - t0
    finish(
        est,
        task_name,
        method,
        n,
        seed,
        test,
        tune_usage,
        tune_seconds,
        metric_calls=budget,
        n_val=n_val,
    )


def f1_from_proba(est, X, y):
    return f1_score(
        y, np.asarray(est.classes_)[est.predict_proba(X).argmax(1)], average="macro"
    )


def e2_cell(task_name, seed, n=256):
    task = TASKS[task_name]
    train, test = load_task(task_name)
    labeled = subsample(train, n, seed)
    fold = len(labeled) * 2 // 3
    val_frac = min(0.5, 128 / fold)
    base = DSPyMator(
        program=dspy.Predict(signature(task)),
        target_names="label",
        feature_names=["text"],
        lm=make_lm("jev"),
        verbose=False,
        max_concurrent=16,
        optimizer=DSPyOptimizer(
            dspy.GEPA,
            metric=metric_for(task),
            reflection_lm=sol_lm(),
            max_metric_calls=500,
            reflection_minibatch_size=3,
            num_threads=8,
            seed=seed,
        ),
        validation_data=val_frac,
    )
    grid = {
        "optimizer__reflection_minibatch_size": [3, 8],
        "optimizer__max_metric_calls": [500, 1500],
    }
    search = GridSearchCV(
        base,
        grid,
        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        scoring=f1_from_proba,
        refit=True,
        n_jobs=4,
        error_score="raise",
    )
    t0 = time.time()
    with dspy.track_usage() as tune_usage, parallel_backend("threading"):
        search.fit(labeled.select("text"), labeled["label"].to_numpy())
    tune_seconds = time.time() - t0
    cv = {
        str(p): float(s)
        for p, s in zip(
            search.cv_results_["params"], search.cv_results_["mean_test_score"]
        )
    }
    finish(
        search.best_estimator_,
        task_name,
        "jev_gepacv",
        n,
        seed,
        test,
        tune_usage,
        tune_seconds,
        best_params=search.best_params_,
        cv_scores=cv,
        val_frac=val_frac,
    )


def finish(est, task_name, method, n, seed, test, tune_usage, tune_seconds, **extra):
    t0 = time.time()
    with dspy.track_usage() as usage:
        proba = est.predict_proba(test.select("text"))
    write_run(
        OUT,
        task_name,
        method,
        n,
        seed,
        test,
        proba,
        est.classes_,
        seconds=time.time() - t0,
        usd_per_1k=usd(usage) / len(test) * 1000,
        tune_seconds=tune_seconds,
        tune_usd=usd(tune_usage),
        instructions=est.program.signature.instructions,
        **extra,
    )


def jevfeat(task_name):
    """Jev zero-shot probabilities for every training-pool row (no labels involved)."""
    task = TASKS[task_name]
    train, _ = load_task(task_name)
    path = FEATURES / f"{task_name}__train_jev.parquet"
    if path.exists():
        return
    est = estimator(task, "jev")
    est.fit(train.select("text"), None)
    FEATURES.mkdir(parents=True, exist_ok=True)
    save_preds(path, train, est.predict_proba(train.select("text")), est.classes_)
    print("features", task_name, len(train), flush=True)


def guarded(fn, *args):
    try:
        fn(*args)
    except Exception:  # noqa: BLE001 - one bad cell must not stop the grid
        print("FAILED", fn.__name__, *args, flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["e1", "e2", "control", "jevfeat"])
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--sizes", nargs="*", type=int, default=[256, 64, 1024])
    args = ap.parse_args()

    def todo(task, method, n, seed):
        return not (OUT / f"{task}__{method}__n{n}__s{seed}.json").exists()

    if args.mode == "jevfeat":
        for task in args.tasks:
            guarded(jevfeat, task)
    elif args.mode == "e1":
        for task in args.tasks:
            for seed in SEEDS:
                for n in (1024, 4000):
                    if todo(task, "jev_gepa", n, seed):
                        guarded(e1_cell, task, n, seed)
    elif args.mode == "control":
        for seed in SEEDS:
            for task in args.tasks:
                if todo(task, "jev_gepa1500", 256, seed):
                    guarded(e1_cell, task, 256, seed, 1500, "jev_gepa1500")
    else:
        # Size-major so every task gets its 3 seeds at 256 before the curve extends.
        for n in args.sizes:
            for seed in SEEDS:
                for task in args.tasks:
                    if todo(task, "jev_gepacv", n, seed):
                        guarded(e2_cell, task, seed, n)
