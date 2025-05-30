"""Centimators: essential data transformers and model estimators for ML competitions."""

import os

# Set JAX as the default backend for Keras if not already set
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

from .feature_transformers import (
    RankTransformer,
    LagTransformer,
    MovingAverageTransformer,
    LogReturnTransformer,
    GroupStatsTransformer,
)

from .model_estimators import (
    MLPRegressor,
    BottleneckEncoder,
    SequenceEstimator,
    LSTMRegressor,
)

from .losses import (
    SpearmanCorrelation,
    CombinedLoss,
)

from .keras_cortex import KerasCortex

from .config import set_keras_backend, get_keras_backend

__all__ = [
    # Feature transformers
    "RankTransformer",
    "LagTransformer",
    "MovingAverageTransformer",
    "LogReturnTransformer",
    "GroupStatsTransformer",
    # Model estimators
    "MLPRegressor",
    "BottleneckEncoder",
    "SequenceEstimator",
    "LSTMRegressor",
    # Keras cortex
    "KerasCortex",
    # Losses
    "SpearmanCorrelation",
    "CombinedLoss",
    # Configuration
    "set_keras_backend",
    "get_keras_backend",
]
