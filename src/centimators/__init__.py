"""Centimators: essential data transformers and model estimators for ML competitions."""

import os
from importlib import import_module
from typing import Any

# Default Keras backend to JAX (respect user override)
os.environ.setdefault("KERAS_BACKEND", "jax")

# Re-export feature transformers eagerly (they don't pull heavy backends)
from centimators.feature_transformers import (  # noqa: E402
    RankTransformer,
    LagTransformer,
    MovingAverageTransformer,
    LogReturnTransformer,
    GroupStatsTransformer,
)

# Re-export backend config helpers
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
    # Config helpers
    "set_keras_backend",
    "get_keras_backend",
]

_LAZY_IMPORTS = {
    # Keras estimators
    "BaseKerasEstimator": "centimators.model_estimators.keras_estimators",
    "SequenceEstimator": "centimators.model_estimators.keras_estimators",
    "MLPRegressor": "centimators.model_estimators.keras_estimators",
    "BottleneckEncoder": "centimators.model_estimators.keras_estimators",
    "LSTMRegressor": "centimators.model_estimators.keras_estimators",
    # DSPy estimator
    "DSPyMator": "centimators.model_estimators.dspymator",
    # Meta-estimator
    "KerasCortex": "centimators.model_estimators.keras_cortex",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'centimators' has no attribute {name!r}")
    module = import_module(module_path)
    attr = getattr(module, name)
    globals()[name] = attr  # cache
    return attr
