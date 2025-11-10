"""Model estimators package.

Provides lazy access to Keras-based estimators and DSPy-based estimators
without importing heavy backends unless needed.
"""

from importlib import import_module
from typing import Any

__all__ = [
    # Keras estimators
    "BaseKerasEstimator",
    "SequenceEstimator",
    "MLPRegressor",
    "BottleneckEncoder",
    "LSTMRegressor",
    # DSPy estimator
    "DSPyMator",
    # Meta-estimator
    "KerasCortex",
]

_LAZY_IMPORTS: dict[str, str] = {
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
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(
            f"module 'centimators.model_estimators' has no attribute {name!r}"
        )
    module = import_module(module_path)
    attr = getattr(module, name)
    globals()[name] = attr  # cache for future access
    return attr
