#!/usr/bin/env python3
"""
Generate feature exposure visualizations for documentation.

Creates before/after charts for:
- FeatureNeutralizer (neutralization.png)
- FeaturePenalizer (penalization.png)

Usage:
    uv run scripts/visualize_feature_exposure.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

os.environ["JAX_PLATFORMS"] = "cpu"

from centimators.feature_transformers import FeatureNeutralizer, FeaturePenalizer

# Centimators theme
COLORS = {
    "before": "#E57373",
    "after": "#62e4fb",
    "bg": "#181A2A",
    "text": "#ffffff",
    "muted": "#8892a0",
}

FEATURES = [
    "momentum",
    "volatility",
    "value",
    "size",
    "quality",
    "sentiment",
    "liquidity",
    "beta",
]
OUTPUT_DIR = "overrides/assets/images"


def compute_exposure(predictions: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Compute correlation between predictions and each feature."""
    pred = predictions - predictions.mean()
    pred = pred / np.linalg.norm(pred)
    feats = features - features.mean(axis=0)
    feats = feats / np.linalg.norm(feats, axis=0)
    return feats.T @ pred


def generate_data(n_samples: int = 2000, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic data with known feature exposures."""
    np.random.seed(seed)
    features = np.random.randn(n_samples, len(FEATURES))
    weights = np.array([0.8, -0.6, 0.5, -0.4, 0.3, -0.2, 0.15, -0.1])
    predictions = features @ weights + np.random.randn(n_samples) * 0.5
    predictions = (predictions - predictions.min()) / (
        predictions.max() - predictions.min()
    )

    return pl.DataFrame(
        {
            "era": ["era1"] * n_samples,
            "prediction": predictions,
            **{name: features[:, i] for i, name in enumerate(FEATURES)},
        }
    )


def create_chart(
    exp_before: np.ndarray,
    exp_after: np.ndarray,
    title: str,
    after_label: str,
    threshold: float | None,
    save_path: str,
):
    """Create a before/after bar chart."""
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    n = len(FEATURES)
    x = np.arange(n)
    w = 0.35

    ax.bar(x - w / 2, exp_before, w, color=COLORS["before"], label="Before", alpha=0.9)
    ax.bar(x + w / 2, exp_after, w, color=COLORS["after"], label=after_label, alpha=0.9)

    # Threshold lines (if applicable)
    if threshold:
        ax.axhline(threshold, color=COLORS["after"], lw=1.5, ls="--", alpha=0.6)
        ax.axhline(-threshold, color=COLORS["after"], lw=1.5, ls="--", alpha=0.6)
        ax.text(
            n - 0.5,
            threshold + 0.02,
            f"±{threshold} cap",
            ha="right",
            fontsize=9,
            color=COLORS["after"],
            alpha=0.8,
        )

    ax.axhline(0, color=COLORS["muted"], lw=0.5, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.tick_params(axis="y", colors=COLORS["muted"], length=0)
    ax.tick_params(axis="x", colors=COLORS["text"], length=0)
    ax.set_ylabel("Feature Exposure", color=COLORS["text"], fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, color=COLORS["text"])
    ax.set_title(title, color=COLORS["text"], fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=COLORS["text"])
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.7, 0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = generate_data()
    predictions = df["prediction"].to_numpy()
    features = df.select(FEATURES).to_numpy()
    era_series = df["era"]

    exp_before = compute_exposure(predictions, features)
    print(f"Original max |exposure|: {np.abs(exp_before).max():.2f}")

    # --- Neutralization chart ---
    proportion = 0.5
    neutralizer = FeatureNeutralizer(
        proportion=proportion, pred_name="prediction", feature_names=FEATURES
    )
    neut_df = neutralizer.fit_transform(
        df[["prediction"]], features=df.select(FEATURES), era_series=era_series
    )
    exp_neutralized = compute_exposure(neut_df.to_numpy().squeeze(), features)
    print(
        f"Neutralized ({proportion}) max |exposure|: {np.abs(exp_neutralized).max():.2f}"
    )

    create_chart(
        exp_before,
        exp_neutralized,
        title="Feature Neutralization",
        after_label=f"Neutralized ({int(proportion * 100)}%)",
        threshold=None,
        save_path=f"{OUTPUT_DIR}/neutralization.png",
    )

    # --- Penalization chart ---
    max_exp = 0.1
    penalizer = FeaturePenalizer(
        max_exposure=max_exp, pred_name="prediction", feature_names=FEATURES
    )
    pen_df = penalizer.fit_transform(
        df[["prediction"]], features=df.select(FEATURES), era_series=era_series
    )
    exp_penalized = compute_exposure(pen_df.to_numpy().squeeze(), features)
    print(f"Penalized ({max_exp}) max |exposure|: {np.abs(exp_penalized).max():.2f}")

    create_chart(
        exp_before,
        exp_penalized,
        title="Feature Penalization",
        after_label=f"Penalized (max={max_exp})",
        threshold=max_exp,
        save_path=f"{OUTPUT_DIR}/penalization.png",
    )


if __name__ == "__main__":
    main()
