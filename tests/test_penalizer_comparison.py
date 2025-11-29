"""
Compare feature penalization implementations across TensorFlow, PyTorch, and JAX.

This test verifies that all three implementations produce roughly equivalent results
when capping feature exposure.
"""

import numpy as np
import polars as pl
import pytest
from scipy import stats

# Skip entire module if JAX is not available
pytest.importorskip("jax")


# ============================================================================
# Reference Implementations
# ============================================================================


def _exposures_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute feature-prediction correlations using numpy."""
    x = x - x.mean(axis=0)
    x = x / np.linalg.norm(x, axis=0)
    y = y - y.mean(axis=0)
    y = y / np.linalg.norm(y, axis=0)
    return x.T @ y


def _gaussianize(values: np.ndarray) -> np.ndarray:
    """Gaussianize values via rank -> normalize -> inverse normal CDF."""
    ranks = stats.rankdata(values, method="ordinal")
    normalized = (ranks - 0.5) / len(values)
    return stats.norm.ppf(normalized)


def _min_max_scale(values: np.ndarray) -> np.ndarray:
    """Scale values to [0, 1]."""
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val - min_val < 1e-10:
        return np.full_like(values, 0.5)
    return (values - min_val) / (max_val - min_val)


# ============================================================================
# JAX Implementation (from our codebase)
# ============================================================================


def penalize_jax(
    prediction: np.ndarray,
    features: np.ndarray,
    max_exp: float,
    lr: float = 1e-3,
    max_iters: int = 100_000,
    tol: float = 1e-7,
) -> np.ndarray:
    """JAX implementation of feature penalization (fully JIT compiled)."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    feats = jnp.asarray(features - 0.5, dtype=jnp.float32)
    pred = jnp.asarray(prediction, dtype=jnp.float32)[:, None]
    n_features = feats.shape[1]

    def exposures(x, y):
        x = x - jnp.mean(x, axis=0)
        x = x / jnp.linalg.norm(x, axis=0)
        y = y - jnp.mean(y, axis=0)
        y = y / jnp.linalg.norm(y, axis=0)
        return x.T @ y

    target_exp = jnp.clip(exposures(feats, pred), -max_exp, max_exp)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-7

    def loss_fn(w):
        neutralized = pred - feats @ w
        exps = exposures(feats, neutralized)
        pos_excess = jax.nn.relu(jax.nn.relu(exps) - jax.nn.relu(target_exp))
        neg_excess = jax.nn.relu(jax.nn.relu(-exps) - jax.nn.relu(-target_exp))
        return jnp.sum(pos_excess + neg_excess)

    def cond_fn(state):
        w, m, u, t, loss = state
        return (loss >= tol) & (t < max_iters)

    def body_fn(state):
        w, m, u, t, _ = state
        loss, grads = jax.value_and_grad(loss_fn)(w)
        m_new = beta1 * m + (1 - beta1) * grads
        u_new = jnp.maximum(beta2 * u, jnp.abs(grads))
        m_hat = m_new / (1 - beta1 ** (t + 1))
        w_new = w - lr * m_hat / (u_new + eps)
        return w_new, m_new, u_new, t + 1, loss

    @jax.jit
    def optimize():
        init_state = (
            jnp.zeros((n_features, 1)),
            jnp.zeros((n_features, 1)),
            jnp.zeros((n_features, 1)),
            jnp.array(0),
            jnp.array(float("inf")),
        )
        w, m, u, t, loss = lax.while_loop(cond_fn, body_fn, init_state)
        return pred - feats @ w

    neutralized = optimize()
    return np.asarray(neutralized).squeeze()


# ============================================================================
# PyTorch Implementation (from forum post)
# ============================================================================


def penalize_pytorch(
    prediction: np.ndarray,
    features: np.ndarray,
    max_exp: float,
    lr: float = 1e-3,
    max_iters: int = 100_000,
    tol: float = 1e-7,
) -> np.ndarray:
    """PyTorch implementation of feature penalization."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from torch import nn

    def exposures(x, y):
        x = x - x.mean(dim=0)
        x = x / x.norm(dim=0)
        y = y - y.mean(dim=0)
        y = y / y.norm(dim=0)
        return torch.matmul(x.T, y)

    feats = torch.tensor(np.float32(features) - 0.5)
    pred = torch.tensor(np.float32(prediction))

    lin = nn.Linear(features.shape[1], 1, bias=False)
    lin.weight.data.fill_(0.0)
    model = nn.Sequential(lin)
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)

    start_exp = exposures(feats, pred[:, None])
    target_exp = torch.clamp(start_exp, -max_exp, max_exp)

    for _ in range(max_iters):
        optimizer.zero_grad()
        exps = exposures(feats, pred[:, None] - model(feats))
        loss = (
            F.relu(F.relu(exps) - F.relu(target_exp))
            + F.relu(F.relu(-exps) - F.relu(-target_exp))
        ).sum()
        if loss < tol:
            break
        loss.backward()
        optimizer.step()

    neutralized = pred[:, None] - model(feats)
    return neutralized.detach().numpy().squeeze()


# ============================================================================
# TensorFlow Implementation (from NumerBlox)
# ============================================================================


def penalize_tensorflow(
    prediction: np.ndarray,
    features: np.ndarray,
    max_exp: float,
    max_iters: int = 100_000,
    tol: float = 1e-7,
) -> np.ndarray:
    """TensorFlow implementation of feature penalization."""
    tf = pytest.importorskip("tensorflow")

    def exposures(x, y):
        x = x - tf.math.reduce_mean(x, axis=0)
        x = x / tf.norm(x, axis=0)
        y = y - tf.math.reduce_mean(y, axis=0)
        y = y / tf.norm(y, axis=0)
        return tf.matmul(x, y, transpose_a=True)

    feats = tf.convert_to_tensor(features - 0.5, dtype=tf.float32)
    pred = tf.convert_to_tensor(prediction, dtype=tf.float32)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(features.shape[1]),
            tf.keras.experimental.LinearModel(use_bias=False),
        ]
    )

    optimizer = tf.keras.optimizers.Adamax()
    start_exp = exposures(feats, pred[:, None])
    target_exp = tf.clip_by_value(start_exp, -max_exp, max_exp)

    for _ in range(max_iters):
        with tf.GradientTape() as tape:
            exps = exposures(feats, pred[:, None] - model(feats, training=True))
            loss = tf.reduce_sum(
                tf.nn.relu(tf.nn.relu(exps) - tf.nn.relu(target_exp))
                + tf.nn.relu(tf.nn.relu(-exps) - tf.nn.relu(-target_exp))
            )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if loss < tol:
            break

    neutralized = pred[:, None] - model(feats)
    return neutralized.numpy().squeeze()


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_data():
    """Create test data with features that have high exposure to predictions."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    # Generate features
    features = np.random.randn(n_samples, n_features)

    # Generate predictions with strong correlation to some features
    weights = np.array([0.5, -0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    predictions = features @ weights + np.random.randn(n_samples) * 0.1

    # Gaussianize predictions (as all implementations do internally)
    predictions = _gaussianize(predictions)

    return predictions, features


# ============================================================================
# Tests
# ============================================================================


def test_jax_reduces_exposure(test_data):
    """Test that JAX implementation actually reduces feature exposure."""
    predictions, features = test_data
    max_exp = 0.1

    # Initial exposures
    initial_exp = np.abs(_exposures_numpy(features - 0.5, predictions[:, None]))
    assert initial_exp.max() > max_exp, "Test data should have high initial exposure"

    # Penalize
    penalized = penalize_jax(predictions, features, max_exp)

    # Check exposures are reduced
    final_exp = np.abs(_exposures_numpy(features - 0.5, penalized[:, None]))
    assert (
        final_exp.max() <= max_exp + 0.01
    ), f"Max exposure {final_exp.max():.3f} exceeds {max_exp}"


def test_pytorch_reduces_exposure(test_data):
    """Test that PyTorch implementation actually reduces feature exposure."""
    pytest.importorskip("torch")
    predictions, features = test_data
    max_exp = 0.1

    # Penalize
    penalized = penalize_pytorch(predictions, features, max_exp)

    # Check exposures are reduced
    final_exp = np.abs(_exposures_numpy(features - 0.5, penalized[:, None]))
    assert (
        final_exp.max() <= max_exp + 0.01
    ), f"Max exposure {final_exp.max():.3f} exceeds {max_exp}"


def test_tensorflow_reduces_exposure(test_data):
    """Test that TensorFlow implementation actually reduces feature exposure."""
    pytest.importorskip("tensorflow")
    predictions, features = test_data
    max_exp = 0.1

    # Penalize
    penalized = penalize_tensorflow(predictions, features, max_exp)

    # Check exposures are reduced
    final_exp = np.abs(_exposures_numpy(features - 0.5, penalized[:, None]))
    assert (
        final_exp.max() <= max_exp + 0.01
    ), f"Max exposure {final_exp.max():.3f} exceeds {max_exp}"


def test_jax_vs_pytorch_equivalence(test_data):
    """Test that JAX and PyTorch implementations produce similar results."""
    pytest.importorskip("torch")
    predictions, features = test_data
    max_exp = 0.1

    jax_result = penalize_jax(predictions, features, max_exp)
    pytorch_result = penalize_pytorch(predictions, features, max_exp)

    # Both should produce valid [0,1] scaled results after min-max
    jax_scaled = _min_max_scale(jax_result / np.std(jax_result))
    pytorch_scaled = _min_max_scale(pytorch_result / np.std(pytorch_result))

    # Correlation should be very high (>0.95)
    correlation = np.corrcoef(jax_scaled, pytorch_scaled)[0, 1]
    assert (
        correlation > 0.95
    ), f"Correlation {correlation:.3f} too low between JAX and PyTorch"


def test_jax_vs_tensorflow_equivalence(test_data):
    """Test that JAX and TensorFlow implementations produce similar results."""
    pytest.importorskip("tensorflow")
    predictions, features = test_data
    max_exp = 0.1

    jax_result = penalize_jax(predictions, features, max_exp)
    tf_result = penalize_tensorflow(predictions, features, max_exp)

    # Both should produce valid [0,1] scaled results after min-max
    jax_scaled = _min_max_scale(jax_result / np.std(jax_result))
    tf_scaled = _min_max_scale(tf_result / np.std(tf_result))

    # Correlation should be very high (>0.95)
    correlation = np.corrcoef(jax_scaled, tf_scaled)[0, 1]
    assert (
        correlation > 0.95
    ), f"Correlation {correlation:.3f} too low between JAX and TensorFlow"


def test_feature_penalizer_class(test_data):
    """Test the FeaturePenalizer class from centimators."""
    from centimators.feature_transformers.penalization import FeaturePenalizer

    predictions, features = test_data
    n_features = features.shape[1]

    # Create DataFrame
    df = pl.DataFrame(
        {
            **{f"feature{i}": features[:, i] for i in range(n_features)},
            "prediction": predictions,
            "era": ["era1"] * 100 + ["era2"] * 100,
        }
    )

    penalizer = FeaturePenalizer(
        max_exposure=0.1,
        pred_name="prediction",
        feature_names=[f"feature{i}" for i in range(n_features)],
    )

    result = penalizer.fit_transform(
        df[["prediction"]],
        features=df.select([f"feature{i}" for i in range(n_features)]),
        era_series=df["era"],
    )

    # Check output
    assert len(result) == len(df)
    assert "prediction_penalized_0.1" in result.columns

    # Check values are in [0, 1]
    penalized = result["prediction_penalized_0.1"].to_numpy()
    assert penalized.min() >= 0.0
    assert penalized.max() <= 1.0


def test_feature_penalizer_multiple_exposures():
    """Test FeaturePenalizer with multiple max_exposure values."""
    from centimators.feature_transformers.penalization import FeaturePenalizer

    np.random.seed(42)
    n_samples = 100
    features = np.random.randn(n_samples, 5)
    predictions = features @ np.random.randn(5) + np.random.randn(n_samples) * 0.1

    df = pl.DataFrame(
        {
            **{f"f{i}": features[:, i] for i in range(5)},
            "pred": predictions,
            "era": ["era1"] * n_samples,
        }
    )

    penalizer = FeaturePenalizer(
        max_exposure=[0.05, 0.1, 0.2],
        pred_name="pred",
        feature_names=[f"f{i}" for i in range(5)],
    )

    result = penalizer.fit_transform(
        df[["pred"]],
        features=df.select([f"f{i}" for i in range(5)]),
        era_series=df["era"],
    )

    assert len(result.columns) == 3
    assert "pred_penalized_0.05" in result.columns
    assert "pred_penalized_0.1" in result.columns
    assert "pred_penalized_0.2" in result.columns


if __name__ == "__main__":
    import time

    # Quick manual test (without pytest dependency)
    np.random.seed(42)
    n_samples = 500
    n_features = 50

    features = np.random.randn(n_samples, n_features)
    weights = np.random.randn(n_features)
    weights[10:] = 0  # sparse weights
    predictions = features @ weights + np.random.randn(n_samples) * 0.1
    predictions = _gaussianize(predictions)

    max_exp = 0.1

    print(f"Data: {n_samples} samples, {n_features} features")
    print(
        "Initial max exposure:",
        np.abs(_exposures_numpy(features - 0.5, predictions[:, None])).max(),
    )

    print("\nTesting JAX (warmup)...")
    _ = penalize_jax(predictions, features, max_exp)

    print("Testing JAX (timed)...")
    start = time.perf_counter()
    jax_result = penalize_jax(predictions, features, max_exp)
    jax_time = time.perf_counter() - start
    jax_exp = np.abs(_exposures_numpy(features - 0.5, jax_result[:, None])).max()
    print(f"  Time: {jax_time * 1000:.0f}ms")
    print(f"  Max exposure after: {jax_exp:.4f}")

    try:
        import torch

        print("\nTesting PyTorch (warmup)...")
        import torch.nn.functional as F
        from torch import nn

        def _penalize_pytorch_standalone(
            prediction, features, max_exp, lr=1e-3, max_iters=100_000, tol=1e-7
        ):
            def exposures(x, y):
                x = x - x.mean(dim=0)
                x = x / x.norm(dim=0)
                y = y - y.mean(dim=0)
                y = y / y.norm(dim=0)
                return torch.matmul(x.T, y)

            feats = torch.tensor(np.float32(features) - 0.5)
            pred = torch.tensor(np.float32(prediction))

            lin = nn.Linear(features.shape[1], 1, bias=False)
            lin.weight.data.fill_(0.0)
            model = nn.Sequential(lin)
            optimizer = torch.optim.Adamax(model.parameters(), lr=lr)

            start_exp = exposures(feats, pred[:, None])
            target_exp = torch.clamp(start_exp, -max_exp, max_exp)

            for _ in range(max_iters):
                optimizer.zero_grad()
                exps = exposures(feats, pred[:, None] - model(feats))
                loss = (
                    F.relu(F.relu(exps) - F.relu(target_exp))
                    + F.relu(F.relu(-exps) - F.relu(-target_exp))
                ).sum()
                if loss < tol:
                    break
                loss.backward()
                optimizer.step()

            neutralized = pred[:, None] - model(feats)
            return neutralized.detach().numpy().squeeze()

        _ = _penalize_pytorch_standalone(predictions, features, max_exp)

        print("Testing PyTorch (timed)...")
        start = time.perf_counter()
        pytorch_result = _penalize_pytorch_standalone(predictions, features, max_exp)
        pytorch_time = time.perf_counter() - start
        pytorch_exp = np.abs(
            _exposures_numpy(features - 0.5, pytorch_result[:, None])
        ).max()
        print(f"  Time: {pytorch_time * 1000:.0f}ms")
        print(f"  Max exposure after: {pytorch_exp:.4f}")

        jax_scaled = _min_max_scale(jax_result / np.std(jax_result))
        pytorch_scaled = _min_max_scale(pytorch_result / np.std(pytorch_result))
        print(
            f"  Correlation with JAX: {np.corrcoef(jax_scaled, pytorch_scaled)[0, 1]:.4f}"
        )

        print(f"\n=== JAX is {pytorch_time / jax_time:.0f}x faster than PyTorch ===")
    except ImportError:
        print("\nPyTorch not installed, skipping...")

    try:
        import tensorflow as tf

        print("\nTesting TensorFlow...")

        def _penalize_tf_standalone(
            prediction, features, max_exp, max_iters=100_000, tol=1e-7
        ):
            def exposures(x, y):
                x = x - tf.math.reduce_mean(x, axis=0)
                x = x / tf.norm(x, axis=0)
                y = y - tf.math.reduce_mean(y, axis=0)
                y = y / tf.norm(y, axis=0)
                return tf.matmul(x, y, transpose_a=True)

            feats = tf.convert_to_tensor(features - 0.5, dtype=tf.float32)
            pred = tf.convert_to_tensor(prediction, dtype=tf.float32)

            model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Input(features.shape[1]),
                    tf.keras.experimental.LinearModel(use_bias=False),
                ]
            )

            optimizer = tf.keras.optimizers.Adamax()
            start_exp = exposures(feats, pred[:, None])
            target_exp = tf.clip_by_value(start_exp, -max_exp, max_exp)

            for _ in range(max_iters):
                with tf.GradientTape() as tape:
                    exps = exposures(feats, pred[:, None] - model(feats, training=True))
                    loss = tf.reduce_sum(
                        tf.nn.relu(tf.nn.relu(exps) - tf.nn.relu(target_exp))
                        + tf.nn.relu(tf.nn.relu(-exps) - tf.nn.relu(-target_exp))
                    )
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                if loss < tol:
                    break

            neutralized = pred[:, None] - model(feats)
            return neutralized.numpy().squeeze()

        tf_result = _penalize_tf_standalone(predictions, features, max_exp)
        tf_exp = np.abs(_exposures_numpy(features - 0.5, tf_result[:, None])).max()
        print(f"  Max exposure after: {tf_exp:.4f}")

        jax_scaled = _min_max_scale(jax_result / np.std(jax_result))
        tf_scaled = _min_max_scale(tf_result / np.std(tf_result))
        print(f"  Correlation with JAX: {np.corrcoef(jax_scaled, tf_scaled)[0, 1]:.4f}")
    except ImportError:
        print("\nTensorFlow not installed, skipping...")
