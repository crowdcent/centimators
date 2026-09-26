"""Build the one-page results poster (self-contained HTML) from results/abl/.

uv run --with matplotlib python benchmarks/typesafe/poster.py out.html
"""

import base64
import html
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from report_ablation import ABL, cells, paired_delta, table
from tasks import TASKS

ORDER = ["fintweets", "onion", "agnews", "sst2"]
JEV, LUNA, ROB, EMB, TFIDF = "#e4572e", "#222222", "#7b4fa3", "#3b8f4c", "#8aa9cf"
LINES = [  # method, label, color, style
    ("roberta_ft", "RoBERTa-base fine-tuned", ROB, "-"),
    ("embed_lr", "bge embeddings + logistic regression", EMB, "-"),
    ("tfidf_lr", "TF-IDF + logistic regression", TFIDF, "-"),
    ("luna_fewshot", "GPT-6 Luna, labels as examples", LUNA, "--"),
    ("jev_gepa", "Jev + GEPA-tuned prompt", JEV, "-"),
    ("jev_lr", "Jev probabilities → logistic regression", "#f4a582", "-."),
    ("stack_lr", "embeddings + Jev probabilities → LR", "#b2182b", "-"),
]
FLATS = [
    ("jev_zeroshot", "Jev, no labels", JEV, ":"),
    ("luna_zeroshot", "GPT-6 Luna, no labels", LUNA, ":"),
]
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    }
)


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def curves(t: pl.DataFrame) -> str:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, task in zip(axes, ORDER):
        d = t.filter(pl.col("task") == task)
        for m, label, color, ls in LINES:
            c = d.filter(pl.col("method") == m, pl.col("n_labels") > 0).sort("n_labels")
            if not len(c):
                continue
            x, y = c["n_labels"].to_numpy(), c["f1"].to_numpy()
            sd = c["f1_sd"].fill_null(0).to_numpy()
            lw = 3 if m == "jev_gepa" else 1.8
            ax.plot(x, y, ls, color=color, lw=lw, marker="o", ms=4, label=label)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.12, lw=0)
        for m, label, color, ls in FLATS:
            c = d.filter(pl.col("method") == m)
            if len(c):
                ax.axhline(c["f1"][0], color=color, ls=ls, lw=1.6, label=label)
        cv = d.filter(pl.col("method") == "jev_gepacv")
        if len(cv):
            ax.scatter(
                256,
                cv["f1"][0],
                marker="*",
                s=180,
                facecolor="white",
                edgecolor=JEV,
                lw=1.5,
                zorder=4,
                label="Jev + GEPA, settings chosen by GridSearchCV",
            )
        ax.set_xscale("log")
        ax.set_xticks([16, 64, 256, 1024, 4000], ["16", "64", "256", "1k", "4k"])
        ax.set_title(TASKS[task].title, fontsize=10)
        ax.set_xlabel("labeled examples")
        ax.grid(alpha=0.25)
        lo = d.filter(pl.col("n_labels") >= 64)["f1"].min()
        ax.set_ylim(max(0.4, lo - 0.05), min(1.0, d["f1"].max() + 0.02))
    axes[0].set_ylabel("macro-F1 (500 test rows)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(
        h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.2)
    )
    return png(fig)


def decomposition() -> tuple[str, dict]:
    """Zero-shot -> Sol-written prompt (no labels) -> GEPA with 256 labels, with CIs."""
    stats = {}
    for task in ORDER:
        stats[task] = {
            "wording": paired_delta(task, ("jev_solprompt", 0), ("jev_zeroshot", 0)),
            "labels": paired_delta(task, ("jev_gepa", 256), ("jev_solprompt", 0)),
        }
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(ORDER))[::-1]
    for key, color, label, off in [
        ("wording", "#f2b8a2", "better wording only (Sol prompt, 0 labels)", 0.19),
        ("labels", JEV, "added by learning from 256 labels (GEPA)", -0.19),
    ]:
        for yi, task in zip(y, ORDER):
            v, lo, hi = stats[task][key]
            ax.barh(
                yi + off,
                v,
                color=color,
                height=0.36,
                label=label if yi == y[0] else None,
            )
            ax.errorbar(
                v, yi + off, xerr=[[v - lo], [hi - v]], color="#333", capsize=3, lw=1
            )
            ax.text(
                max(hi, 0) + 0.004, yi + off, f"{v:+.3f}", va="center", fontsize=8.5
            )
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_yticks(y, [TASKS[t].title for t in ORDER])
    ax.set_xlabel("macro-F1 change (95% paired bootstrap CI)")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.grid(axis="x", alpha=0.25)
    return png(fig), stats


def cost(t: pl.DataFrame) -> str:
    pts = [
        ("jev_zeroshot", 0, "Jev, no labels", JEV, "o", (7, -3)),
        ("jev_gepa", 256, "Jev + GEPA (256)", JEV, "*", (-18, 10)),
        ("luna_zeroshot", 0, "Luna, no labels", LUNA, "o", (7, -3)),
        ("luna_fewshot", 256, "Luna + 256 examples", LUNA, "s", (7, -3)),
        ("luna_fewshot", 16, "Luna + 16 examples", LUNA, "D", (7, -3)),
        ("roberta_ft", 256, "RoBERTa (256)", ROB, "o", (7, -3)),
        ("roberta_ft", 4000, "RoBERTa (4,000)", ROB, "s", (7, -3)),
        ("embed_lr", 256, "embeddings + LR (256)", EMB, "o", (7, -3)),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for m, n, label, color, marker, off in pts:
        d = t.filter(pl.col("method") == m, pl.col("n_labels") == n)
        if d["task"].n_unique() < len(ORDER):
            continue
        x, y = d["usd_per_1k"].mean(), d["f1"].mean()
        ax.scatter(
            x, y, s=260 if marker == "*" else 60, color=color, marker=marker, zorder=3
        )
        ax.annotate(
            label,
            (x, y),
            xytext=off,
            textcoords="offset points",
            fontsize=8.5,
            ha="right" if off[0] < 0 else "left",
        )
    ax.set_xscale("log")
    ax.set_xlabel("inference cost, USD per 1,000 rows (log)")
    ax.set_ylabel("macro-F1, mean of 4 tasks")
    ax.grid(alpha=0.25)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 8)
    return png(fig)


def hypotheses(t) -> str:
    """Pre-registered follow-ups (PROTOCOL.md), one row per test, 'pending' until the cells exist."""

    def fmt(d):
        if d is None:
            return "<td class=tie>pending</td>"
        v, lo, hi = d
        cls = "win" if lo > 0 else ("loss" if hi < 0 else "tie")
        return f"<td class={cls}>{v:+.3f} <span class=ci>[{lo:+.3f}, {hi:+.3f}]</span></td>"

    tests = [
        ("H1 GEPA 1k vs 256 labels", ("jev_gepa", 1024), ("jev_gepa", 256)),
        ("H1 GEPA 4k vs 256 labels", ("jev_gepa", 4000), ("jev_gepa", 256)),
        (
            "H2 GridSearchCV settings vs default (seed 0)",
            ("jev_gepacv", 256, 0),
            ("jev_gepa", 256, 0),
        ),
        ("H3 stack vs embeddings, 1k", ("stack_lr", 1024), ("embed_lr", 1024)),
        ("H3 stack vs embeddings, 4k", ("stack_lr", 4000), ("embed_lr", 4000)),
        ("H3 stack vs Jev + GEPA, 1k", ("stack_lr", 1024), ("jev_gepa", 1024)),
        ("H3 stack vs Jev + GEPA, 4k", ("stack_lr", 4000), ("jev_gepa", 4000)),
    ]
    head = "".join(f"<th>{TASKS[k].title.split(' (')[0]}</th>" for k in ORDER)
    rows = "".join(
        f"<tr><td>{name}</td>"
        + "".join(fmt(paired_delta(k, a, b)) for k in ORDER)
        + "</tr>"
        for name, a, b in tests
    )
    ll = []
    for n in (64, 256, 1024, 4000):
        cells = []
        for k in ORDER:
            raw = t.filter(pl.col("task") == k, pl.col("method") == "jev_zeroshot")[
                "log_loss"
            ]
            lr = t.filter(
                pl.col("task") == k,
                pl.col("method") == "jev_lr",
                pl.col("n_labels") == n,
            )["log_loss"]
            cells.append(
                f"<td>{raw[0]:.3f} → {lr[0]:.3f}</td>"
                if len(raw) and len(lr)
                else "<td class=tie>pending</td>"
            )
        ll.append(
            f"<tr><td>H3b log loss, raw Jev → Jev + LR ({n:,} labels)</td>{''.join(cells)}</tr>"
        )
    return f"<table><tr><th>Test (macro-F1 Δ, 95% CI)</th>{head}</tr>{rows}{''.join(ll)}</table>"


def cell_mean(t, task, m, n):
    d = t.filter(pl.col("task") == task, pl.col("method") == m, pl.col("n_labels") == n)
    return d["f1"][0] if len(d) else float("nan")


def median_prompt(task, n=256):
    rows = [
        json.loads(f.read_text()) for f in ABL.glob(f"{task}__jev_gepa__n{n}__s*.json")
    ]
    rows.sort(key=lambda r: r["macro_f1"])
    return rows[len(rows) // 2]


def page(t: pl.DataFrame, df: pl.DataFrame) -> str:
    img_curves = curves(t)
    img_dec, _ = decomposition()
    img_cost = cost(t)
    f = lambda task, m, n: cell_mean(t, task, m, n)
    vs256 = {
        task: paired_delta(task, ("jev_gepa", 256), ("roberta_ft", 256))
        for task in ORDER
    }
    vs4k = {
        task: paired_delta(task, ("jev_gepa", 256), ("roberta_ft", 4000))
        for task in ORDER
    }
    tune = df.filter(pl.col("method") == "jev_gepa", pl.col("n_labels") == 256)
    tune_usd, tune_min = tune["tune_usd"].mean(), tune["tune_seconds"].mean() / 60
    infer = tune["usd_per_1k"].mean()
    luna = df.filter(pl.col("method") == "luna_fewshot")
    luna16 = luna.filter(pl.col("n_labels") == 16)["usd_per_1k"].mean()
    luna256 = luna.filter(pl.col("n_labels") == 256)["usd_per_1k"].mean()
    ex = median_prompt("fintweets")

    def ci(d):
        v, lo, hi = d
        return f"{v:+.3f} <span class=ci>[{lo:+.3f}, {hi:+.3f}]</span>"

    def verdict(d):
        _, lo, hi = d
        return "win" if lo > 0 else ("loss" if hi < 0 else "tie")

    rows = "".join(
        f"<tr><td>{TASKS[k].title}</td><td>{f(k, 'jev_zeroshot', 0):.3f}</td>"
        f"<td><b>{f(k, 'jev_gepa', 256):.3f}</b></td><td>{f(k, 'roberta_ft', 256):.3f}</td>"
        f"<td class={verdict(vs256[k])}>{ci(vs256[k])}</td><td>{f(k, 'roberta_ft', 4000):.3f}</td>"
        f"<td class={verdict(vs4k[k])}>{ci(vs4k[k])}</td>"
        f"<td>{f(k, 'luna_fewshot', 16):.3f}</td></tr>"
        for k in ORDER
    )
    wins = sum(verdict(vs256[k]) != "loss" for k in ORDER)
    return f"""<!doctype html><html><head><meta charset=utf-8><style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#faf8f5;color:#1d1d1f}}
.wrap{{max-width:1180px;margin:0 auto;padding:28px 32px 48px}}
h1{{font-size:30px;margin:0 0 6px;letter-spacing:-.4px}} .sub{{font-size:16px;color:#555;margin:0 0 20px}}
.tiles{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:0 0 22px}}
.tile{{background:#fff;border:1px solid #e6e1da;border-radius:10px;padding:14px 16px}}
.big{{font-size:30px;font-weight:700;color:{JEV}}} .tile p{{margin:4px 0 0;font-size:13.5px;color:#444}}
section{{background:#fff;border:1px solid #e6e1da;border-radius:10px;padding:16px 20px;margin:0 0 16px}}
h2{{font-size:17px;margin:0 0 4px}} .lede{{margin:0 0 10px;color:#444;font-size:14px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} img{{width:100%}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}} th,td{{padding:6px 8px;border-bottom:1px solid #eee;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{font-size:12px;color:#666;font-weight:600}}
.ci{{color:#888;font-size:11.5px}} .win{{color:#1a7f37;font-weight:600}} .loss{{color:#b42318}} .tie{{color:#555}}
ul{{margin:6px 0 0 18px;padding:0;font-size:13.5px;line-height:1.5}} pre{{white-space:pre-wrap;font-size:12.5px;background:#f6f4f0;padding:10px 12px;border-radius:8px;margin:6px 0 0}}
.k{{font-size:12px;color:#777;text-transform:uppercase;letter-spacing:.5px;margin-top:8px}}
</style></head><body><div class=wrap>
<h1>A 256-example prompt tune makes Jev a match for fine-tuned RoBERTa</h1>
<p class=sub>DSPyMator wraps TypeSafe's Jev as a scikit-learn classifier; GEPA rewrites its prompt from a few hundred labels.
Same labeled rows, same 500 test rows, 3 seeds, every method tuned only on its own labels.</p>
<div class=tiles>
<div class=tile><div class=big>{wins} / 4</div><p>tasks where Jev + GEPA (256 labels) wins or ties RoBERTa fine-tuned on the same 256 labels</p></div>
<div class=tile><div class=big>{f("fintweets", "jev_gepa", 256):.2f} vs {f("fintweets", "roberta_ft", 4000):.2f}</div><p>financial tweets: Jev + 256 labels vs RoBERTa trained on 4,000 labels (macro-F1)</p></div>
<div class=tile><div class=big>${tune_usd:.2f} · {tune_min:.0f} min</div><p>to tune the prompt, then ${infer:.3f} per 1,000 predictions. No GPU.</p></div>
</div>
<section><h2>Learning curves: how much labeled data each method needs</h2>
<p class=lede>Lines are seed means, bands are ±1 sd across 3 labeled subsets. Dotted lines use no labels at all.
The red curve is Jev after GEPA tuned its prompt on that many labels.</p>
<img src="{img_curves}"></section>
<section><h2>Pre-registered follow-ups</h2>
<p class=lede>Hypotheses committed before these runs (benchmarks/typesafe/PROTOCOL.md). Green = CI above zero, red = below.</p>
{hypotheses(t)}</section>
<section><h2>Head to head at matched label budgets</h2>
<p class=lede>Macro-F1 differences with 95% paired bootstrap CIs over the test rows; green = CI above zero.</p>
<table><tr><th>Task</th><th>Jev, 0 labels</th><th>Jev + GEPA, 256</th><th>RoBERTa, 256</th><th>Δ vs RoBERTa 256</th><th>RoBERTa, 4,000</th><th>Δ vs RoBERTa 4,000</th><th>Luna + 16 examples</th></tr>{rows}</table></section>
<div class=two>
<section><h2>Is GEPA learning, or just rewording?</h2>
<p class=lede>Control: GPT-6 Sol (GEPA's reflection model) writes a prompt from the task text alone, no labels.
Whatever GEPA adds on top of that came from the labels.</p><img src="{img_dec}"></section>
<section><h2>Quality vs inference cost</h2>
<p class=lede>Mean over 4 tasks. API cost is measured tokens at list price; local models are GPU time at $1.10/h.
GPT-6 Luna with labels pasted into the prompt is the most accurate option, at ~{luna16 / infer:.0f}× (16 examples) to ~{luna256 / infer:.0f}× (256 examples) tuned Jev's cost per row, on every call.</p><img src="{img_cost}"></section>
</div>
<div class=two>
<section><h2>What GEPA actually learned (financial tweets, median seed)</h2>
<div class=k>Before</div><pre>{html.escape(TASKS["fintweets"].instructions)}</pre>
<div class=k>After, 256 labels (macro-F1 {f("fintweets", "jev_zeroshot", 0):.3f} → {ex["macro_f1"]:.3f})</div><pre>{html.escape(ex["instructions"])}</pre></section>
<section><h2>How it was tested</h2><ul>
<li><b>Tasks:</b> Onion vs HuffPost satire, SST-2, AG News (4 classes), financial tweet sentiment (3 classes). Public HF datasets, 500 seeded test rows each, never seen in training.</li>
<li><b>Labels:</b> 16 / 64 / 256 / 1k / 4k, stratified; every method gets the identical rows for a given size and seed (3 seeds below 4k).</li>
<li><b>Baselines, tuned fairly:</b> TF-IDF and bge-base embeddings + logistic regression with C chosen by CV; RoBERTa-base with learning rate picked on a 20% validation split of its own labels. Run on CrowdCent Cloud GPU.</li>
<li><b>Jev + GEPA:</b> <code>DSPyMator(lm=TypeSafe("jev-latest")).fit(X, y, optimizer=GEPA(...))</code>. Half the labels drive search, half validate. Metric: Brier score on Jev's class probabilities. Reflection model GPT-6 Sol.</li>
<li><b>Controls:</b> Jev and Luna both run zero-shot, with a Sol-written prompt, with labels as few-shot examples, and with GEPA.</li>
<li><b>Caveats:</b> SST-2 and AG News are old public sets that LLMs have likely seen. 500 test rows gives roughly ±0.02–0.04 CIs. RoBERTa with 4,000 labels still wins satire.</li>
</ul></section></div>
</div></body></html>"""


if __name__ == "__main__":
    df = cells()
    Path(sys.argv[1]).write_text(page(table(df), df))
