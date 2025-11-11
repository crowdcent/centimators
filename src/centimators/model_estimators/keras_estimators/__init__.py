"""
Keras-based model estimators with scikit-learn compatible API.

Organized by architectural family:
    - base: BaseKerasEstimator and shared utilities
    - dense: Simple feedforward networks (MLPRegressor)
    - autoencoder: Reconstruction-based architectures (BottleneckEncoder)
    - sequence: Sequence models for temporal data (SequenceEstimator, LSTMRegressor)
"""

from .base import BaseKerasEstimator
from .dense import MLPRegressor
from .autoencoder import BottleneckEncoder
from .sequence import SequenceEstimator, LSTMRegressor

__all__ = [
    "BaseKerasEstimator",
    "MLPRegressor",
    "BottleneckEncoder",
    "SequenceEstimator",
    "LSTMRegressor",
]
