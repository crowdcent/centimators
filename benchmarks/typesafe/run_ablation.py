"""Controlled ablation for the API models. Every cell uses the same signature,
the same test rows, and the same labeled subsets as the local baselines
(`tasks.subsample(train, n, seed)`).

    uv run python benchmarks/typesafe/run_ablation.py --lms jev --tasks onion sst2

Conditions, per LM (jev, luna):
  zeroshot   seed instructions, no labels
  solprompt  GPT-6 Sol rewrites the instructions from the task text alone (no labels):
             separates "a bigger model wrote a better prompt" from "labels helped"
  fewshot    all n labeled rows as demos
  gepa       GEPA on the n labels (stratified half search / half validation),
             GPT-6 Sol as reflection LM, Brier metric
Writes results/abl/<task>__<method>__n<n>__s<seed>.{json,parquet}; skips cells already done.
"""

import argparse
import os
import time
import traceback
from pathlib import Path

import dspy
from centimators.model_estimators import DSPyMator
from dspy.experimental import Choice

from metrics import write_run
from run_api import PRICES, make_lm
from tasks import SEEDS, TASKS, load_task, split_val, subsample

OUT = Path(__file__).parent / "results" / "abl"
SOL_PRICES = (2.0, 10.0)
LM_KIND = {"jev": "jev", "luna": "gpt-6-luna"}
SIZES = [16, 64, 256]


def signature(task, instructions=None):
    options = tuple((c, task.descriptions[c]) for c in task.classes)
    return dspy.Signature(
        {
            "text": (str, dspy.InputField()),
            "label": (Choice[options], dspy.OutputField(desc=task.question)),
        },
        instructions or task.instructions,
    )


def sol_lm():
    return dspy.LM(
        "openai/gpt-6-sol",
        api_key=os.environ.get("OPENAI_API_KEY", "PROXY_OPENAI_KEY"),
        temperature=1.0,
        max_tokens=32000,
        cache=False,
    )


def usd(usage):
    total = 0.0
    for model, tok in usage.get_total_tokens().items():
        tin = tok.get("prompt_tokens", 0) or 0
        tout = tok.get("completion_tokens", 0) or 0
        if "sol" in model:
            price = SOL_PRICES
        elif "luna" in model:
            price = PRICES["gpt-6-luna"]
        else:
            price = PRICES["jev"]
        total += (tin * price[0] + tout * price[1]) / 1e6
    return total


def metric_for(task):
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
            "The model reads only the task instructions, the text, and the class "
            "descriptions, and returns probabilities. Improve the instructions with "
            "concise decision criteria and boundary cases; no output-format rules."
        )
        return dspy.Prediction(score=s, feedback=feedback)

    return metric


def sol_instructions(task):
    classes = "\n".join(f"- {c}: {task.descriptions[c]}" for c in task.classes)
    prompt = (
        "Write instructions for a text classifier. It reads only these instructions, "
        "the input text, and the class list, and returns class probabilities.\n"
        f"Task: {task.instructions}\nClasses:\n{classes}\n"
        "Give concise decision criteria and boundary cases in under 200 words. "
        "Return only the instructions."
    )
    out = sol_lm()(prompt)
    return out[0] if isinstance(out, list) else out


def estimator(task, lm, instructions=None):
    return DSPyMator(
        program=dspy.Predict(signature(task, instructions)),
        target_names="label",
        feature_names=["text"],
        lm=make_lm(LM_KIND[lm]),
        verbose=False,
        max_concurrent=16 if lm == "jev" else 32,
    )


def cell(task_name, lm, cond, n, seed):
    task = TASKS[task_name]
    train, test = load_task(task_name)
    X_test = test.select("text")
    extra, t0 = {}, time.time()
    with dspy.track_usage() as tune_usage:
        if cond == "zeroshot":
            est = estimator(task, lm)
            est.fit(X_test, None)
        elif cond == "solprompt":
            instructions = sol_instructions(task)
            est = estimator(task, lm, instructions)
            est.fit(X_test, None)
        elif cond == "fewshot":
            labeled = subsample(train, n, seed)
            est = estimator(task, lm)
            est.fit(
                labeled.select("text"),
                labeled["label"],
                optimizer=dspy.LabeledFewShot(k=len(labeled)),
            )
        else:
            labeled = subsample(train, n, seed)
            fit, val = split_val(labeled, 0.5, seed)
            budget = 300 + 2 * n
            est = estimator(task, lm)
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
            extra["metric_calls"] = budget
    extra["tune_seconds"] = time.time() - t0
    extra["tune_usd"] = usd(tune_usage)
    t0 = time.time()
    with dspy.track_usage() as usage:
        proba = est.predict_proba(X_test)
    extra["seconds"] = time.time() - t0
    extra["usd_per_1k"] = usd(usage) / len(test) * 1000
    extra["instructions"] = est.program.signature.instructions
    write_run(
        OUT, task_name, f"{lm}_{cond}", n, seed, test, proba, est.classes_, **extra
    )


def plan(tasks, lms):
    for task in tasks:
        for lm in lms:
            yield task, lm, "zeroshot", 0, 0
            for seed in SEEDS:
                yield task, lm, "solprompt", 0, seed
            for n in SIZES:
                for seed in SEEDS:
                    yield task, lm, "gepa", n, seed
                    yield task, lm, "fewshot", n, seed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--lms", nargs="*", default=list(LM_KIND))
    ap.add_argument("--reverse", action="store_true", help="walk the grid backwards")
    args = ap.parse_args()
    grid = list(plan(args.tasks, args.lms))
    for task, lm, cond, n, seed in reversed(grid) if args.reverse else grid:
        if (OUT / f"{task}__{lm}_{cond}__n{n}__s{seed}.json").exists():
            continue
        try:
            cell(task, lm, cond, n, seed)
        except Exception:  # noqa: BLE001 - one bad cell must not stop the grid
            print("FAILED", task, lm, cond, n, seed, flush=True)
            traceback.print_exc()
