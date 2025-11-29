"""Dense feedforward neural network estimators."""

from dataclasses import dataclass, field
from typing import Any

from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler

from .base import BaseKerasEstimator
from keras import layers, models


@dataclass(kw_only=True)
class MLPRegressor(RegressorMixin, BaseKerasEstimator):
    """A minimal fully-connected multi-layer perceptron for tabular data."""

    hidden_units: tuple[int, ...] = (64, 64)
    activation: str = "relu"
    dropout_rate: float = 0.0
    metrics: list[str] | None = field(default_factory=lambda: ["mse"])
    target_scaler: Any = field(default_factory=StandardScaler)

    def build_model(self):
        inputs = layers.Input(shape=(self._n_features_in_,), name="features")
        x = inputs
        for units in self.hidden_units:
            x = layers.Dense(units, activation=self.activation)(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.output_units, activation="linear")(x)
        self.model = models.Model(inputs=inputs, outputs=outputs, name="mlp_regressor")

        self.model.compile(
            optimizer=self.optimizer(learning_rate=self.learning_rate),
            loss=self.loss_function,
            metrics=self.metrics,
        )
        return self
