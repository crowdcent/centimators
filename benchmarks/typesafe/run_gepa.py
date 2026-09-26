"""GEPA-optimize Jev's instructions through DSPyMator, then score on the shared test rows.

    uv run python benchmarks/typesafe/run_gepa.py [--tasks onion] [--sizes 64 256]

Half the labeled budget trains, half validates. GPT-6 Sol proposes instructions;
Jev is the student. Writes results/api-<task>-jev_gepa<n>.json and the tuned prompt.
"""

import argparse
import json
import os
import time
from pathlib import Path

import dspy
import numpy as np
from centimators.model_estimators import DSPyMator
from dspy.experimental import Choice

from metrics import save_preds, score
from run_api import PRICES, make_lm
from tasks import TASKS, load_task

OUT = Path(__file__).parent / "results"
SOL_PRICES = (2.0, 10.0)  # gpt-6-sol, USD per 1M tokens


def rich_signature(task):
    """Same task as run_api, typed as Choice so metrics can read the probabilities."""
    options = tuple((c, task.descriptions[c]) for c in task.classes)
    return dspy.Signature(
        {
            "text": (str, dspy.InputField()),
            "label": (Choice[options], dspy.OutputField(desc=task.question)),
        },
        task.instructions,
    )


def make_metric(task):
    classes = task.classes

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        probs = pred.label.probabilities or {}
        p_true = float(probs.get(gold.label, 0.0))
        brier = sum(
            (float(probs.get(c, 0.0)) - (c == gold.label)) ** 2 for c in classes
        )
        s = 1 - brier / 2
        top = max(probs, key=probs.get) if probs else "?"
        feedback = (
            f"Correct class: {gold.label} ({task.descriptions[gold.label]}). "
            f"Model put p={p_true:.2f} on it and picked {top}. Score {s:.2f}. "
            "The model is a non-generative classifier that reads only the task instructions, "
            "the text, and the class descriptions, and returns probabilities. Improve the "
            "instructions with concise decision criteria and boundary cases; no output-format rules."
        )
        return dspy.Prediction(score=s, feedback=feedback)

    return metric


def balanced(train, task, n):
    per = n // len(task.classes)
    return (
        train.group_by("label", maintain_order=True)
        .head(per)
        .sample(fraction=1.0, shuffle=True, seed=0)
    )


def run(task_name, n, budget):
    task = TASKS[task_name]
    train, test = load_task(task_name)
    labeled = balanced(train, task, n)
    budget = budget or 300 + 4 * len(labeled) // 2
    jev = make_lm("jev")
    sol = dspy.LM(
        "openai/gpt-6-sol",
        api_key=os.environ.get("OPENAI_API_KEY", "PROXY_OPENAI_KEY"),
        temperature=1.0,
        max_tokens=32000,
        cache=False,
    )
    est = DSPyMator(
        program=dspy.Predict(rich_signature(task)),
        target_names="label",
        feature_names=["text"],
        lm=jev,
        verbose=False,
        max_concurrent=16,
    )
    gepa = dspy.GEPA(
        metric=make_metric(task),
        reflection_lm=sol,
        max_metric_calls=budget,
        num_threads=16,
        track_stats=True,
        seed=0,
    )
    t = time.time()
    with dspy.track_usage() as usage:
        est.fit(
            labeled.select("text"),
            labeled["label"],
            optimizer=gepa,
            validation_data=0.5,
        )
    fit_seconds = time.time() - t
    tok = usage.get_total_tokens()
    jev_in = sum(v.get("prompt_tokens", 0) or 0 for k, v in tok.items() if "jev" in k)
    sol_in = sum(v.get("prompt_tokens", 0) or 0 for k, v in tok.items() if "sol" in k)
    sol_out = sum(
        v.get("completion_tokens", 0) or 0 for k, v in tok.items() if "sol" in k
    )
    tune_usd = (
        jev_in * PRICES["jev"][0] / 1e6
        + (sol_in * SOL_PRICES[0] + sol_out * SOL_PRICES[1]) / 1e6
    )

    t = time.time()
    with dspy.track_usage() as usage:
        proba = est.predict_proba(test.select("text"))
    seconds = time.time() - t
    infer_in = sum(
        v.get("prompt_tokens", 0) or 0 for v in usage.get_total_tokens().values()
    )

    instructions = est.program.signature.instructions
    method = f"jev_gepa{n}"
    result = {
        "task": task_name,
        "method": method,
        "family": "api",
        "n_labels": len(labeled),
        "n_test": len(test),
        **score(test["label"].to_numpy(), proba, est.classes_),
        "seconds": seconds,
        "usd_per_1k": infer_in * PRICES["jev"][0] / 1e6 / len(test) * 1000,
        "tune_seconds": fit_seconds,
        "tune_usd": tune_usd,
        "metric_calls": budget,
        "instructions": instructions,
        "seed_instructions": task.instructions,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"api-{task_name}-{method}.json").write_text(json.dumps(result, indent=2))
    save_preds(OUT / f"preds-{task_name}-{method}.parquet", test, proba, est.classes_)
    print(
        json.dumps({k: v for k, v in result.items() if k != "instructions"}), flush=True
    )
    print("INSTRUCTIONS", task_name, n, repr(instructions)[:600], flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--sizes", nargs="*", type=int, default=[64, 256])
    ap.add_argument("--budget", type=int, default=None)
    args = ap.parse_args()
    np.random.seed(0)
    for n in args.sizes:
        for task_name in args.tasks:
            run(task_name, n, args.budget)
