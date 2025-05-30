"""Configuration utilities for centimators."""

import os
import warnings


def set_keras_backend(backend: str = "jax") -> None:
    """Set the Keras backend.
    
    Args:
        backend: The backend to use. Options are 'jax', 'tensorflow', or 'torch'.
                Defaults to 'jax'.
                
    Notes:
        This function must be called before importing any centimators modules
        that use Keras (model_estimators, keras_cortex).
    """
    valid_backends = {"jax", "tensorflow", "torch"}
    if backend not in valid_backends:
        raise ValueError(f"Invalid backend: {backend}. Choose from {valid_backends}")
    
    if "KERAS_BACKEND" in os.environ and os.environ["KERAS_BACKEND"] != backend:
        warnings.warn(
            f"KERAS_BACKEND is already set to '{os.environ['KERAS_BACKEND']}'. "
            f"Overriding to '{backend}'. This may cause issues if Keras has already been imported.",
            RuntimeWarning
        )
    
    os.environ["KERAS_BACKEND"] = backend


def get_keras_backend() -> str:
    """Get the current Keras backend.
    
    Returns:
        The current backend name ('jax', 'tensorflow', or 'torch').
    """
    return os.environ.get("KERAS_BACKEND", "jax") 