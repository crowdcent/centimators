import pytest
import keras.ops as K

from centimators.losses import SpearmanCorrelation, CombinedLoss


def _constant(array_like):
    # Helper to create backend tensor regardless of active backend
    return K.convert_to_tensor(array_like, dtype="float32")


def test_spearman_correlation_loss_negative():
    y_true = _constant([1.0, 2.0, 3.0, 4.0])
    y_pred = _constant([1.0, 2.0, 2.0, 4.0])

    loss_fn = SpearmanCorrelation(regularization_strength=1e-2)
    loss_val = loss_fn(y_true, y_pred)

    # Spearman correlation should be positive so negative correlation (loss) is negative
    assert K.convert_to_numpy(loss_val) < 0  # loss equals -corr


def test_combined_loss_positive():
    y_true = _constant([1.0, 2.0, 3.0, 4.0])
    y_pred = _constant([0.9, 2.1, 2.5, 4.2])

    loss_fn = CombinedLoss()
    loss_val = loss_fn(y_true, y_pred)

    # Manually compute expected combined loss to validate implementation.
    mse = K.mean(K.square(y_pred - y_true))
    spearman = SpearmanCorrelation()(y_true, y_pred)
    expected = loss_fn.mse_weight * mse + loss_fn.spearman_weight * spearman

    assert pytest.approx(K.convert_to_numpy(loss_val), rel=1e-5) == K.convert_to_numpy(expected) 