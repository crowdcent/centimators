"""Centimators: essential data transformers and model estimators for ML competitions."""

import os
from importlib import import_module
from typing import Any

# Set default Keras backend to JAX (matches centimators[keras-jax] extra).
# Users can override by setting KERAS_BACKEND env var before import or calling
# set_keras_backend() early. This must happen before any Keras imports.
os.environ.setdefault("KERAS_BACKEND", "jax")

from centimators.feature_transformers import (  # noqa: E402
    RankTransformer,
    LagTransformer,
    MovingAverageTransformer,
    LogReturnTransformer,
    GroupStatsTransformer,
)

from centimators.config import set_keras_backend, get_keras_backend  # noqa: E402

__all__ = [
    # Model Estimators (resolved lazily via __getattr__)
    "BaseKerasEstimator",
    "SequenceEstimator",
    "MLPRegressor",
    "BottleneckEncoder",
    "LSTMRegressor",
    "DSPyMator",
    "KerasCortex",
    # Feature Transformers
    "RankTransformer",
    "LagTransformer",
    "MovingAverageTransformer",
    "LogReturnTransformer",
    "GroupStatsTransformer",
    "EmbeddingTransformer",
    "DimReducer",
    # Config helpers
    "set_keras_backend",
    "get_keras_backend",
]

_LAZY_IMPORTS = {
    # Keras estimators
    "BaseKerasEstimator": "centimators.model_estimators.keras_estimators.base",
    "SequenceEstimator": "centimators.model_estimators.keras_estimators.sequence",
    "MLPRegressor": "centimators.model_estimators.keras_estimators.dense",
    "BottleneckEncoder": "centimators.model_estimators.keras_estimators.autoencoder",
    "LSTMRegressor": "centimators.model_estimators.keras_estimators.sequence",
    # DSPy estimator
    "DSPyMator": "centimators.model_estimators.dspymator",
    # Meta-estimator
    "KerasCortex": "centimators.model_estimators.keras_cortex",
    # Feature transformers with optional dependencies
    "EmbeddingTransformer": "centimators.feature_transformers.embedding",
    "DimReducer": "centimators.feature_transformers.dimreduction",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'centimators' has no attribute {name!r}")
    module = import_module(module_path)
    attr = getattr(module, name)
    globals()[name] = attr  # cache
    return attr
