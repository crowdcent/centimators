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

    @nw.narwhalify
    def predict(self, X, batch_size: int = 512, **kwargs: Any) -> Any:
        if not self.model:
            raise ValueError("Model not built. Call `build_model` first.")

        predictions = self.model.predict(
            _ensure_numpy(X), batch_size=batch_size, **kwargs
        )

        # Return numpy arrays for sklearn compatibility if input is numpy
        if isinstance(X, numpy.ndarray):
            return predictions

        # Return dataframe if input was a dataframe
        if predictions.ndim == 1:
            return nw.from_dict(
                {"prediction": predictions}, backend=nw.get_native_namespace(X)
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
