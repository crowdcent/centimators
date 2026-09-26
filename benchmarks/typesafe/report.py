"""Merge API + Cloud results into a table and a learning-curve chart.

uv run python benchmarks/typesafe/report.py results/ local_results.jsonl
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from metrics import score
from tasks import TASKS

LABELS = {
    "jev_zeroshot": "Jev, 0 labels",
    "jev_fewshot": "Jev + 4 examples/class",
    "gpt5mini_zeroshot": "gpt-5-mini, 0 labels",
    "gpt6luna_zeroshot": "GPT-6 Luna, 0 labels",
    "zeroshot_nli": "DeBERTa-v3-large zero-shot NLI",
    "tfidf_lr": "TF-IDF + logistic regression",
    "embed_lr": "bge-base embeddings + logistic regression",
    "roberta_ft": "RoBERTa-base fine-tuned",
    "jev_gepa": "Jev + GEPA-tuned prompt",
}
FLAT = {
    "jev_zeroshot": ("#e4572e", "-"),
    "jev_fewshot": ("#e4572e", "--"),
    "gpt6luna_zeroshot": ("#222222", "-"),
    "gpt5mini_zeroshot": ("#999999", "-"),
    "zeroshot_nli": ("#8a8a8a", ":"),
}
CURVES = {"tfidf_lr": "#4c78a8", "embed_lr": "#54a24b", "roberta_ft": "#b279a2"}
GEPA_COLOR = "#e4572e"
SHORT = {
    "tfidf_lr": "TF-IDF + LR",
    "embed_lr": "bge embeddings + LR",
    "jev_fewshot": "Jev + 4/class",
    "zeroshot_nli": "DeBERTa zero-shot NLI",
    "roberta_ft": "RoBERTa fine-tuned",
    "jev_gepa": "Jev + GEPA prompt",
}


def load(paths) -> pl.DataFrame:
    rows = []
    for p in map(Path, paths):
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        for f in files:
            text = f.read_text()
            rows += (
                [json.loads(line) for line in text.splitlines() if line.strip()]
                if f.suffix == ".jsonl"
                else [json.loads(text)]
            )
    cols = [
        "task",
        "method",
        "n_labels",
        "accuracy",
        "macro_f1",
        "auc",
        "log_loss",
        "seconds",
        "usd_per_1k",
    ]
    return pl.DataFrame([{c: r.get(c) for c in cols} for r in rows])


def chart(df: pl.DataFrame, path: Path, metric="macro_f1"):
    tasks = [t for t in TASKS if t in df["task"].unique().to_list()]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()
    for ax, task in zip(axes, tasks):
        d = df.filter(pl.col("task") == task)
        for m, color in CURVES.items():
            c = d.filter(pl.col("method") == m).sort("n_labels")
            if len(c):
                ax.plot(
                    c["n_labels"],
                    c[metric],
                    "o-",
                    color=color,
                    label=LABELS[m],
                    lw=2,
                    ms=4,
                )
        for m, (color, ls) in FLAT.items():
            c = d.filter(pl.col("method") == m)
            if len(c):
                ax.axhline(c[metric][0], color=color, ls=ls, lw=2, label=LABELS[m])
        g = d.filter(pl.col("method").str.starts_with("jev_gepa")).sort("n_labels")
        if len(g):
            ax.plot(
                g["n_labels"],
                g[metric],
                "*-",
                color=GEPA_COLOR,
                ms=13,
                lw=1.5,
                label=LABELS["jev_gepa"],
            )
        ax.set_xscale("log")
        ax.set_xticks([16, 64, 256, 1024, 4000], ["16", "64", "256", "1k", "4k"])
        ax.set_title(TASKS[task].title, fontsize=10)
        ax.set_xlabel("labeled training examples")
        ax.set_ylabel("macro-F1 on 500 held-out rows")
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=160)


def cost_chart(df: pl.DataFrame, path: Path):
    """Mean macro-F1 across tasks vs inference cost per 1k rows, one point per method."""
    best = (
        df.with_columns(
            pl.when(pl.col("method").str.starts_with("jev_gepa"))
            .then(pl.lit("jev_gepa"))
            .otherwise(pl.col("method"))
            .alias("method")
        )
        .filter(pl.col("n_labels") == pl.col("n_labels").max().over("task", "method"))
        .group_by("method")
        .agg(
            pl.col("macro_f1").mean(),
            pl.col("usd_per_1k").mean(),
            pl.col("n_labels").max(),
        )
    )
    colors = {**{m: c for m, (c, _) in FLAT.items()}, **CURVES, "jev_gepa": GEPA_COLOR}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for r in best.iter_rows(named=True):
        marker = "*" if r["method"] == "jev_gepa" else "o"
        ax.scatter(
            r["usd_per_1k"],
            r["macro_f1"],
            s=220 if marker == "*" else 70,
            marker=marker,
            color=colors[r["method"]],
            zorder=3,
        )
        tag = SHORT.get(r["method"], LABELS[r["method"]])
        if r["method"] not in FLAT:
            tag += f" ({r['n_labels']:,} labels)"
        ax.annotate(
            tag,
            (r["usd_per_1k"], r["macro_f1"]),
            xytext=(8, -3),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlim(best["usd_per_1k"].min() / 2, best["usd_per_1k"].max() * 30)
    ax.set_xlabel("inference cost, USD per 1,000 rows (log scale)")
    ax.set_ylabel("macro-F1, mean of 4 tasks")
    ax.grid(alpha=0.3)
    ax.set_title("Quality vs inference cost", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def rescore(df: pl.DataFrame, preds_dir: Path) -> pl.DataFrame:
    """Recompute quality metrics from saved per-row predictions (largest n per method)."""
    for f in preds_dir.glob("preds-*.parquet"):
        task, method = f.stem.removeprefix("preds-").split("-", 1)
        classes = TASKS[task].classes
        p = pl.read_parquet(f)
        m = score(
            p["label"].to_numpy(),
            p.select([f"p_{c}" for c in classes]).to_numpy(),
            classes,
        )
        sel = (pl.col("task") == task) & (pl.col("method") == method)
        n = df.filter(sel)["n_labels"].max()
        sel = sel & (pl.col("n_labels") == n)
        df = df.with_columns(
            [
                pl.when(sel).then(pl.lit(v)).otherwise(pl.col(k)).alias(k)
                for k, v in m.items()
            ]
        )
    return df


if __name__ == "__main__":
    df = rescore(load(sys.argv[1:]), Path(__file__).parent / "results")
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    df.sort("task", "method", "n_labels").write_csv(out / "summary.csv")
    chart(df, out / "learning_curves.png")
    cost_chart(df, out / "cost_vs_quality.png")
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=40):
        print(df.sort("task", "method", "n_labels"))
