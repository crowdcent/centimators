"""API-backed methods, all through the same DSPyMator + signature; only `lm` changes.

    uv run python benchmarks/typesafe/run_api.py [--tasks onion sst2] [--methods jev_zeroshot]

Writes results/api-<task>-<method>.json.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Annotated, Literal

import dspy
import polars as pl
from centimators.model_estimators import DSPyMator
from dspy.experimental import Choice, TypeSafe

from metrics import save_preds, score
from tasks import TASKS, load_task

OUT = Path(__file__).parent / "results"
# USD per million tokens (input, output).
PRICES = {"jev": (0.042, 0.0), "gpt-5-mini": (0.25, 2.0), "gpt-6-luna": (0.10, 0.50)}


def signature_for(task):
    options = tuple((c, task.descriptions[c]) for c in task.classes)
    label_type = Annotated[Literal[tuple(task.classes)], Choice[options]]
    return dspy.Signature(
        {
            "text": (str, dspy.InputField()),
            "label": (label_type, dspy.OutputField(desc=task.question)),
        },
        task.instructions,
    )


def make_lm(kind):
    if kind == "jev":
        return TypeSafe(
            "jev-latest",
            api_key=os.environ.get("TYPESAFE_API_KEY", "PROXY_TYPESAFE_KEY"),
            cache=False,  # cached rows report no latency or tokens
        )
    if kind == "gpt-6-luna":
        return dspy.LM(
            "openai/gpt-6-luna",
            api_key=os.environ.get("OPENAI_API_KEY", "PROXY_OPENAI_KEY"),
            temperature=1.0,
            max_tokens=16000,
            reasoning_effort="none",
            cache=False,
        )
    return dspy.LM(
        "openai/gpt-5-mini",
        api_key=os.environ.get("OPENAI_API_KEY", "PROXY_OPENAI_KEY"),
        temperature=1.0,
        max_tokens=16000,
        reasoning_effort="minimal",
        cache=False,
    )


def balanced(train: pl.DataFrame, per_class: int) -> pl.DataFrame:
    return pl.concat(
        [g.head(per_class) for _, g in train.group_by("label", maintain_order=True)]
    ).sample(fraction=1.0, shuffle=True, seed=0)


METHODS = {
    # name: (lm kind, labeled demos per class or 0)
    "jev_zeroshot": ("jev", 0),
    "jev_fewshot": ("jev", 4),
    "gpt5mini_zeroshot": ("gpt-5-mini", 0),
    "gpt6luna_zeroshot": ("gpt-6-luna", 0),
}


def run(task_name, method):
    task = TASKS[task_name]
    kind, per_class = METHODS[method]
    train, test = load_task(task_name)
    est = DSPyMator(
        program=dspy.Predict(signature_for(task)),
        target_names="label",
        feature_names=["text"],
        lm=make_lm(kind),
        verbose=False,
        max_concurrent=16 if kind == "jev" else 32,
    )
    n_labels = 0
    if per_class:
        demos = balanced(train, per_class)
        n_labels = len(demos)
        est.fit(
            demos.select("text"),
            demos["label"],
            optimizer=dspy.LabeledFewShot(k=n_labels),
        )
    else:
        est.fit(test.select("text"), None)

    t = time.time()
    with dspy.track_usage() as usage:
        proba = est.predict_proba(test.select("text"))
    seconds = time.time() - t

    tokens_in = tokens_out = 0
    for model_usage in usage.get_total_tokens().values():
        tokens_in += model_usage.get("prompt_tokens", 0) or 0
        tokens_out += model_usage.get("completion_tokens", 0) or 0
    p_in, p_out = PRICES[kind]
    cost = (tokens_in * p_in + tokens_out * p_out) / 1e6

    result = {
        "task": task_name,
        "method": method,
        "family": "api",
        "n_labels": n_labels,
        "n_test": len(test),
        **score(test["label"].to_numpy(), proba, est.classes_),
        "seconds": seconds,
        "usd_per_1k": cost / len(test) * 1000,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"api-{task_name}-{method}.json").write_text(json.dumps(result, indent=2))
    save_preds(OUT / f"preds-{task_name}-{method}.parquet", test, proba, est.classes_)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    args = ap.parse_args()
    for task_name in args.tasks:
        for method in args.methods:
            run(task_name, method)
