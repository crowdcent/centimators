"""
Keras-based model estimators with scikit-learn compatible API.

Includes:
    - BaseKerasEstimator
    - SequenceEstimator
    - MLPRegressor
    - BottleneckEncoder
    - LSTMRegressor
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Type

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin

try:
    from keras import optimizers
    from keras import distribution
    from keras import ops
    from keras import layers, models
except ImportError as e:
    raise ImportError(
        "Keras estimators require keras and jax (or another Keras-compatible backend). Install with:\n"
        "  uv add 'centimators[keras-jax]'\n"
        "or:\n"
        "  pip install 'centimators[keras-jax]'"
    ) from e

import narwhals as nw
from narwhals.typing import IntoFrame
import numpy


def _ensure_numpy(data, allow_series: bool = False):
    """Convert data to numpy array, handling both numpy arrays and dataframes.

    Args:
        data: Input data (numpy array, dataframe, or series)
        allow_series: Whether to allow series inputs

    Returns:
        numpy.ndarray: Data converted to numpy array
    """
    if isinstance(data, numpy.ndarray):
        return data
    try:
        return nw.from_native(data, allow_series=allow_series).to_numpy()
    except Exception:
        return numpy.asarray(data)


@dataclass(kw_only=True)
class BaseKerasEstimator(TransformerMixin, BaseEstimator, ABC):
    """Meta-estimator for Keras models following the scikit-learn API."""

    output_units: int = 1
    optimizer: Type[optimizers.Optimizer] = optimizers.Adam
    learning_rate: float = 0.001
    loss_function: str = "mse"
    metrics: list[str] | None = None
    model: Any = None
    distribution_strategy: str | None = None

    @abstractmethod
    def build_model(self):
        pass

    def _setup_distribution_strategy(self) -> None:
        strategy = distribution.DataParallel()
        distribution.set_distribution(strategy)

    def fit(
        self,
        X,
        y,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: tuple[Any, Any] | None = None,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> "BaseKerasEstimator":
        self._n_features_in_ = X.shape[1]

        if self.distribution_strategy:
            self._setup_distribution_strategy()

        if not self.model:
            self.build_model()

        self.model.fit(
            _ensure_numpy(X),
            y=_ensure_numpy(y, allow_series=True),
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            **kwargs,
        )
        self._is_fitted = True
        return self

    def predict(self, X, batch_size: int = 512, **kwargs: Any) -> Any:
        if not self.model:
            raise ValueError("Model not built. Call `build_model` first.")
        return self.model.predict(X, batch_size=batch_size, **kwargs)

    def transform(self, X, **kwargs):
        return self.predict(X, **kwargs)

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "_is_fitted", False)


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
class MLPRegressor(RegressorMixin, BaseKerasEstimator):
    """A minimal fully-connected multi-layer perceptron for tabular data."""

    hidden_units: tuple[int, ...] = (64, 64)
    activation: str = "relu"
    dropout_rate: float = 0.0
    metrics: list[str] | None = field(default_factory=lambda: ["mse"])

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


@dataclass(kw_only=True)
class BottleneckEncoder(BaseKerasEstimator):
    """A bottleneck autoencoder that can learn latent representations and predict targets."""

    gaussian_noise: float = 0.035
    encoder_units: list[tuple[int, float]] = field(
        default_factory=lambda: [(1024, 0.1)]
    )
    latent_units: tuple[int, float] = (256, 0.1)
    ae_units: list[tuple[int, float]] = field(default_factory=lambda: [(96, 0.4)])
    activation: str = "swish"
    reconstruction_loss_weight: float = 1.0
    target_loss_weight: float = 1.0
    encoder: Any = None

    def build_model(self):
        if self._n_features_in_ is None:
            raise ValueError("Must call fit() before building the model")

        inputs = layers.Input(shape=(self._n_features_in_,), name="features")
        x0 = layers.BatchNormalization()(inputs)

        encoder = layers.GaussianNoise(self.gaussian_noise)(x0)
        for units, dropout in self.encoder_units:
            encoder = layers.Dense(units)(encoder)
            encoder = layers.BatchNormalization()(encoder)
            encoder = layers.Activation(self.activation)(encoder)
            encoder = layers.Dropout(dropout)(encoder)

        latent_units, latent_dropout = self.latent_units
        latent = layers.Dense(latent_units)(encoder)
        latent = layers.BatchNormalization()(latent)
        latent = layers.Activation(self.activation)(latent)
        latent_output = layers.Dropout(latent_dropout)(latent)

        self.encoder = models.Model(
            inputs=inputs, outputs=latent_output, name="encoder"
        )

        decoder = latent_output
        for units, dropout in reversed(self.encoder_units):
            decoder = layers.Dense(units)(decoder)
            decoder = layers.BatchNormalization()(decoder)
            decoder = layers.Activation(self.activation)(decoder)
            decoder = layers.Dropout(dropout)(decoder)

        reconstruction = layers.Dense(self._n_features_in_, name="reconstruction")(
            decoder
        )

        target_pred = reconstruction
        for units, dropout in self.ae_units:
            target_pred = layers.Dense(units)(target_pred)
            target_pred = layers.BatchNormalization()(target_pred)
            target_pred = layers.Activation(self.activation)(target_pred)
            target_pred = layers.Dropout(dropout)(target_pred)

        target_output = layers.Dense(
            self.output_units, activation="linear", name="target_prediction"
        )(target_pred)

        self.model = models.Model(
            inputs=inputs,
            outputs=[reconstruction, target_output],
            name="bottleneck_encoder",
        )

        self.model.compile(
            optimizer=self.optimizer(learning_rate=self.learning_rate),
            loss={"reconstruction": "mse", "target_prediction": self.loss_function},
            loss_weights={
                "reconstruction": self.reconstruction_loss_weight,
                "target_prediction": self.target_loss_weight,
            },
            metrics={"target_prediction": self.metrics or ["mse"]},
        )
        return self

    def fit(
        self,
        X,
        y,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: tuple[Any, Any] | None = None,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> "BottleneckEncoder":
        self._n_features_in_ = X.shape[1]

        if self.distribution_strategy:
            self._setup_distribution_strategy()

        if not self.model:
            self.build_model()

        X_np = _ensure_numpy(X)
        y_np = _ensure_numpy(y, allow_series=True)

        y_dict = {"reconstruction": X_np, "target_prediction": y_np}

        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_np = _ensure_numpy(X_val)
            y_val_np = _ensure_numpy(y_val, allow_series=True)
            validation_data = (
                X_val_np,
                {"reconstruction": X_val_np, "target_prediction": y_val_np},
            )

        self.model.fit(
            X_np,
            y_dict,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            **kwargs,
        )

        self._is_fitted = True
        return self

    def predict(self, X, batch_size: int = 512, **kwargs: Any) -> Any:
        if not self.model:
            raise ValueError("Model not built. Call 'fit' first.")
        X_np = _ensure_numpy(X)
        predictions = self.model.predict(X_np, batch_size=batch_size, **kwargs)
        return predictions[1] if isinstance(predictions, list) else predictions

    def transform(self, X, batch_size: int = 512, **kwargs: Any) -> Any:
        if not self.encoder:
            raise ValueError("Encoder not built. Call 'fit' first.")
        X_np = _ensure_numpy(X)
        return self.encoder.predict(X_np, batch_size=batch_size, **kwargs)

    def fit_transform(self, X, y, **kwargs) -> Any:
        return self.fit(X, y, **kwargs).transform(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        latent_dim = self.latent_units[0]
        return [f"latent_{i}" for i in range(latent_dim)]


@dataclass(kw_only=True)
class LSTMRegressor(RegressorMixin, SequenceEstimator):
    """LSTM-based regressor for time series prediction."""

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
