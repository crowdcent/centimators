"""Base classes and utilities for Keras-based estimators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Type

from sklearn.base import BaseEstimator, TransformerMixin
import narwhals as nw
import numpy

try:
    from keras import optimizers
    from keras import distribution
except ImportError as e:
    raise ImportError(
        "Keras estimators require keras and jax (or another Keras-compatible backend). Install with:\n"
        "  uv add 'centimators[keras-jax]'\n"
        "or:\n"
        "  pip install 'centimators[keras-jax]'"
    ) from e


from centimators.narwhals_utils import _ensure_numpy


@dataclass(kw_only=True)
class BaseKerasEstimator(TransformerMixin, BaseEstimator, ABC):
    """Meta-estimator for Keras models following the scikit-learn API.

    Args:
        output_units (int, default=1): Number of output units in the final layer.
        optimizer (Type[optimizers.Optimizer], default=Adam): Keras optimizer class
            to use for training.
        learning_rate (float, default=0.001): Learning rate for the optimizer.
        loss_function (str, default="mse"): Loss function name passed to model.compile().
        metrics (list[str] | None, default=None): List of metric names to track
            during training.
        model (Any, default=None): The underlying Keras model (populated by build_model).
        distribution_strategy (str | None, default=None): If set, enables DataParallel
            distribution for multi-device training.
        target_scaler (sklearn transformer | None, default=None): Scaler for target
            values. Neural networks converge better when targets are normalized.
            Subclasses may override the default (e.g., regressors default to StandardScaler).
    """

    output_units: int = 1
    optimizer: Type[optimizers.Optimizer] = optimizers.Adam
    learning_rate: float = 0.001
    loss_function: str = "mse"
    metrics: list[str] | None = None
    model: Any = None
    distribution_strategy: str | None = None
    target_scaler: Any = None

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

        # Convert inputs to numpy
        X_np = _ensure_numpy(X)
        y_np = _ensure_numpy(y, allow_series=True)

        # Ensure y is 2D for scaler
        y_was_1d = y_np.ndim == 1
        if y_was_1d:
            y_np = y_np.reshape(-1, 1)

        # Scale targets for better neural network convergence
        if self.target_scaler:
            y_np = self.target_scaler.fit_transform(y_np).astype("float32")

            # Scale validation targets too
            if validation_data is not None:
                val_X, val_y = validation_data
                val_y_np = _ensure_numpy(val_y, allow_series=True)
                if val_y_np.ndim == 1:
                    val_y_np = val_y_np.reshape(-1, 1)
                val_y_scaled = self.target_scaler.transform(val_y_np).astype("float32")
                validation_data = (_ensure_numpy(val_X), val_y_scaled)

        if not self.model:
            self.build_model()

        self.model.fit(
            X_np,
            y=y_np,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            **kwargs,
        )
        self._is_fitted = True
        return self

    @nw.narwhalify
    def predict(self, X, batch_size: int = 512, **kwargs: Any) -> Any:
        if not self.model:
            raise ValueError("Model not built. Call `build_model` first.")

        predictions = self.model.predict(
            _ensure_numpy(X), batch_size=batch_size, **kwargs
        )

        # Inverse transform predictions back to original scale
        if self.target_scaler:
            predictions = self.target_scaler.inverse_transform(predictions)

        # Return numpy arrays for numpy input
        if isinstance(X, numpy.ndarray):
            return predictions

        # Return dataframe for dataframe input
        if predictions.ndim == 1:
            return nw.from_dict(
                {"prediction": predictions}, backend=nw.get_native_namespace(X)
            )
        elif predictions.shape[1] == 1:
            return nw.from_dict(
                {"prediction": predictions[:, 0]}, backend=nw.get_native_namespace(X)
            )
        else:
            cols = {
                f"prediction_{i}": predictions[:, i]
                for i in range(predictions.shape[1])
            }
            return nw.from_dict(cols, backend=nw.get_native_namespace(X))

    def transform(self, X, **kwargs):
        return self.predict(X, **kwargs)

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "_is_fitted", False)
