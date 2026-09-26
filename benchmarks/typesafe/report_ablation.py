"""Aggregate results/abl/ cells: seed mean and spread per (task, method, n), plus
paired bootstrap CIs for the head-to-head comparisons.

    uv run python benchmarks/typesafe/report_ablation.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import f1_score
from tasks import TASKS

ABL = Path(__file__).parent / "results" / "abl"
B = 2000


def cells() -> pl.DataFrame:
    rows = [json.loads(f.read_text()) for f in sorted(ABL.glob("*.json"))]
    keep = ["task", "method", "n_labels", "seed", "macro_f1", "log_loss", "ece"]
    keep += ["usd_per_1k", "tune_usd", "tune_seconds"]
    return pl.DataFrame([{k: r.get(k) for k in keep} for r in rows])


def table(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("task", "method", "n_labels")
        .agg(
            pl.len().alias("seeds"),
            pl.col("macro_f1").mean().alias("f1"),
            pl.col("macro_f1").std().alias("f1_sd"),
            pl.col("log_loss").mean(),
            pl.col("ece").mean(),
            pl.col("usd_per_1k").mean(),
            pl.col("tune_usd").mean(),
        )
        .sort("task", "method", "n_labels")
    )


def _preds(task, method, n):
    """Per-row predicted class for every seed of a cell, test order preserved."""
    classes = TASKS[task].classes
    out = []
    for f in sorted(ABL.glob(f"{task}__{method}__n{n}__s*.parquet")):
        p = pl.read_parquet(f)
        out.append(
            np.asarray(classes)[p.select([f"p_{c}" for c in classes]).to_numpy().argmax(1)]
        )
    y = pl.read_parquet(f)["label"].to_numpy() if out else None
    return y, out


def paired_delta(task, a, b):
    """Macro-F1(a) - macro-F1(b), seed-averaged, with a 95% paired bootstrap over test rows."""
    y, pa = _preds(task, *a)
    _, pb = _preds(task, *b)
    if not pa or not pb:
        return None
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(B, len(y)))

    def f1(y_, preds, ix):
        return np.mean([f1_score(y_[ix], p[ix], average="macro") for p in preds])

    full = f1(y, pa, slice(None)) - f1(y, pb, slice(None))
    boots = np.array([f1(y, pa, i) - f1(y, pb, i) for i in idx[:500]])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return full, lo, hi


if __name__ == "__main__":
    df = cells()
    t = table(df)
    t.write_csv(ABL.parent / "ablation_summary.csv")
    with pl.Config(tbl_rows=400, tbl_cols=12, fmt_float="mixed", float_precision=3):
        print(t)
