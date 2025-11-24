"""Tests for neutralization and penalization transformers."""

import os

import numpy as np
import polars as pl
import pytest

os.environ["KERAS_BACKEND"] = "jax"

from centimators.feature_transformers import FeatureNeutralizer


def _make_test_data():
    """Create test data with eras, features, and predictions."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    # Generate features
    features = np.random.randn(n_samples, n_features)

    # Generate predictions with some correlation to features
    predictions = (
        features @ np.random.randn(n_features) + np.random.randn(n_samples) * 0.1
    )

    # Create eras (5 eras, 20 samples each)
    eras = np.repeat([f"era{i}" for i in range(5)], 20)

    df = pl.DataFrame(
        {
            **{f"feature{i}": features[:, i] for i in range(n_features)},
            "prediction": predictions,
            "era": eras,
        }
    )

    return df


def _compute_exposure(predictions: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Compute correlation (exposure) between predictions and each feature.

    Returns array of shape (n_features,) with correlations.
    """
    # Center
    pred_centered = predictions - predictions.mean()
    features_centered = features - features.mean(axis=0)

    # Normalize
    pred_norm = pred_centered / np.linalg.norm(pred_centered)
    features_norm = features_centered / np.linalg.norm(features_centered, axis=0)

    # Correlation
    return features_norm.T @ pred_norm


# ===== FeatureNeutralizer Tests =====


def test_neutralizer_basic():
    """Test basic neutralization functionality."""
    df = _make_test_data()

    neutralizer = FeatureNeutralizer(
        proportion=0.5,
        pred_name="prediction",
        feature_names=[f"feature{i}" for i in range(5)],
    )

    result = neutralizer.fit_transform(
        df[["prediction"]],
        features=df.select([f"feature{i}" for i in range(5)]),
        era_series=df["era"],
    )

    # Check output shape and column name
    assert len(result) == len(df)
    assert "prediction_neutralized_0.5" in result.columns


def test_neutralizer_reduces_exposure():
    """Test that neutralization actually reduces feature exposure."""
    df = _make_test_data()

    feature_cols = [f"feature{i}" for i in range(5)]

    # Compute initial exposure
    initial_predictions = df["prediction"].to_numpy()
    features = df.select(feature_cols).to_numpy()
    initial_exposures = _compute_exposure(initial_predictions, features)
    initial_max_exposure = np.abs(initial_exposures).max()

    # Neutralize with high proportion
    neutralizer = FeatureNeutralizer(
        proportion=1.0,
        pred_name="prediction",
        feature_names=feature_cols,
    )

    result = neutralizer.fit_transform(
        df[["prediction"]],
        features=df.select(feature_cols),
        era_series=df["era"],
    )

    # Compute neutralized exposure
    neutralized_predictions = result["prediction_neutralized_1.0"].to_numpy()
    neutralized_exposures = _compute_exposure(neutralized_predictions, features)
    neutralized_max_exposure = np.abs(neutralized_exposures).max()

    # Neutralization should reduce max exposure
    assert neutralized_max_exposure < initial_max_exposure


def test_neutralizer_multiple_proportions():
    """Test neutralization with multiple proportions."""
    df = _make_test_data()

    neutralizer = FeatureNeutralizer(
        proportion=[0.0, 0.5, 1.0],
        pred_name="prediction",
        feature_names=[f"feature{i}" for i in range(5)],
    )

    result = neutralizer.fit_transform(
        df[["prediction"]],
        features=df.select([f"feature{i}" for i in range(5)]),
        era_series=df["era"],
    )

    # Should have 3 output columns
    assert len(result.columns) == 3
    assert "prediction_neutralized_0.0" in result.columns
    assert "prediction_neutralized_0.5" in result.columns
    assert "prediction_neutralized_1.0" in result.columns


def test_neutralizer_with_suffix():
    """Test neutralization with custom suffix."""
    df = _make_test_data()

    neutralizer = FeatureNeutralizer(
        proportion=0.5,
        pred_name="prediction",
        feature_names=[f"feature{i}" for i in range(5)],
        suffix="custom",
    )

    result = neutralizer.fit_transform(
        df[["prediction"]],
        features=df.select([f"feature{i}" for i in range(5)]),
        era_series=df["era"],
    )

    assert "prediction_neutralized_0.5_custom" in result.columns


def test_neutralizer_no_era_series():
    """Test neutralization without era_series (single era)."""
    df = _make_test_data()

    neutralizer = FeatureNeutralizer(
        proportion=0.5,
        pred_name="prediction",
        feature_names=[f"feature{i}" for i in range(5)],
    )

    # Should work without era_series (treated as single era)
    result = neutralizer.fit_transform(
        df[["prediction"]],
        features=df.select([f"feature{i}" for i in range(5)]),
        era_series=None,
    )

    assert len(result) == len(df)
    assert "prediction_neutralized_0.5" in result.columns


def test_neutralizer_output_range():
    """Test that neutralized predictions are scaled to [0, 1]."""
    df = _make_test_data()

    neutralizer = FeatureNeutralizer(
        proportion=0.5,
        pred_name="prediction",
        feature_names=[f"feature{i}" for i in range(5)],
    )

    result = neutralizer.fit_transform(
        df[["prediction"]],
        features=df.select([f"feature{i}" for i in range(5)]),
        era_series=df["era"],
    )

    neutralized = result["prediction_neutralized_0.5"].to_numpy()

    # Check that values are in [0, 1]
    assert neutralized.min() >= 0.0
    assert neutralized.max() <= 1.0
    # Check that min and max are approximately 0 and 1 (scaled properly)
    assert pytest.approx(neutralized.min(), abs=1e-6) == 0.0
    assert pytest.approx(neutralized.max(), abs=1e-6) == 1.0
