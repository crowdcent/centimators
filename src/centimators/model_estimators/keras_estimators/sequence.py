"""Sequence-based estimators for temporal/recurrent models."""

from dataclasses import dataclass, field
from typing import Any

from narwhals.typing import IntoFrame
from sklearn.base import RegressorMixin

try:
    from keras import ops, layers, models
except ImportError as e:
    raise ImportError(
        "Keras estimators require keras and jax (or another Keras-compatible backend). Install with:\n"
        "  uv add 'centimators[keras-jax]'\n"
        "or:\n"
        "  pip install 'centimators[keras-jax]'"
    ) from e

import numpy

from .base import BaseKerasEstimator, _ensure_numpy


@dataclass(kw_only=True)
class SequenceEstimator(BaseKerasEstimator):
    """Estimator for models that consume sequential data."""

    lag_windows: list[int]
    n_features_per_timestep: int

    def __post_init__(self):
        self.seq_length = len(self.lag_windows)

    def _reshape(self, X: IntoFrame, validation_data: tuple[Any, Any] | None = None):
        X = _ensure_numpy(X)
        X_reshaped = ops.reshape(
            X, (X.shape[0], self.seq_length, self.n_features_per_timestep)
        )

        if validation_data:
            X_val, y_val = validation_data
            X_val = _ensure_numpy(X_val)
            X_val_reshaped = ops.reshape(
                X_val,
                (X_val.shape[0], self.seq_length, self.n_features_per_timestep),
            )
            validation_data = X_val_reshaped, _ensure_numpy(y_val)

        return X_reshaped, validation_data

    def fit(
        self, X, y, validation_data: tuple[Any, Any] | None = None, **kwargs: Any
    ) -> "SequenceEstimator":
        X_reshaped, validation_data_reshaped = self._reshape(X, validation_data)
        super().fit(
            X_reshaped,
            y=_ensure_numpy(y),
            validation_data=validation_data_reshaped,
            **kwargs,
        )
        return self

    def predict(self, X, **kwargs: Any) -> numpy.ndarray:
        X_reshaped, _ = self._reshape(X)
        return super().predict(X_reshaped, **kwargs)


@dataclass(kw_only=True)
class LSTMRegressor(RegressorMixin, SequenceEstimator):
    """LSTM-based regressor for sequence prediction."""

    lstm_units: list[tuple[int, float, float]] = field(
        default_factory=lambda: [(64, 0.01, 0.01)]
    )
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    bidirectional: bool = False
    metrics: list[str] | None = field(default_factory=lambda: ["mse"])

    def build_model(self):
        if self._n_features_in_ is None:
            raise ValueError("Must call fit() before building the model")

        inputs = layers.Input(
            shape=(self.seq_length, self.n_features_per_timestep), name="sequence_input"
        )
        x = inputs

        for layer_num, (units, dropout, recurrent_dropout) in enumerate(
            self.lstm_units
        ):
            return_sequences = layer_num < len(self.lstm_units) - 1
            lstm_layer = layers.LSTM(
                units=units,
                activation="tanh",
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                name=f"lstm_{layer_num}",
            )
            if self.bidirectional:
                x = layers.Bidirectional(lstm_layer, name=f"bidirectional_{layer_num}")(
                    x
                )
            else:
                x = lstm_layer(x)
            if self.use_layer_norm:
                x = layers.LayerNormalization(name=f"layer_norm_{layer_num}")(x)
            if self.use_batch_norm:
                x = layers.BatchNormalization(name=f"batch_norm_{layer_num}")(x)

        outputs = layers.Dense(self.output_units, activation="linear", name="output")(x)
        self.model = models.Model(inputs=inputs, outputs=outputs, name="lstm_regressor")
        self.model.compile(
            optimizer=self.optimizer(learning_rate=self.learning_rate),
            loss=self.loss_function,
            metrics=self.metrics,
        )
        return self
