"""Autoencoder-based estimators for representation learning."""

from dataclasses import dataclass, field
from typing import Any

try:
    from keras import layers, models
except ImportError as e:
    raise ImportError(
        "Keras estimators require keras and jax (or another Keras-compatible backend). Install with:\n"
        "  uv add 'centimators[keras-jax]'\n"
        "or:\n"
        "  pip install 'centimators[keras-jax]'"
    ) from e

from .base import BaseKerasEstimator, _ensure_numpy


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
