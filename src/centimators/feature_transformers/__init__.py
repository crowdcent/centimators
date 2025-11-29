"""
Feature transformers (in the scikit-learn sense) that integrate seamlessly with
pipelines. Using metadata routing, centimators' transformers specialize in
grouping features by a date or ticker series, and applying transformations to
each group independently.

This module provides a family of *stateless* feature/target transformers built on top of
narwhals. Each class follows the ``sklearn.base.
TransformerMixin`` interface which allows them to participate in
``sklearn.pipeline.Pipeline`` or ``ColumnTransformer`` objects without extra
boiler-plate.

All transformers are fully vectorised, backend-agnostic (pandas, polars, …)
and suitable for cross-validation, grid-search and other classic
machine-learning workflows.

Highlights:
    * **RankTransformer** – converts numeric features into their (0, 1]-normalised
    rank within a user-supplied grouping column (e.g. a date).
    * **LagTransformer** – creates shifted/lagged copies of features to expose
    temporal context for time-series models.
    * **MovingAverageTransformer** – rolling mean across arbitrary window sizes.
    * **LogReturnTransformer** – first-difference of the natural logarithm of a
    signal, a common way to compute returns.
    * **GroupStatsTransformer** – horizontally aggregates arbitrary sets of columns
    and exposes statistics such as mean, standard deviation, skew, kurtosis,
    range and coefficient of variation.
    * **EmbeddingTransformer** – embeds text and categorical features using DSPy's
    Embedder, supporting both hosted models and custom embedding functions.
    * **DimReducer** – reduces feature dimensionality using PCA, t-SNE, or UMAP
    for feature compression and visualization.
    * **FeatureNeutralizer** – neutralizes predictions by removing linear exposure
    to features, reducing feature correlation while preserving signal.
    * **FeaturePenalizer** – uses iterative optimization to cap feature exposure
    while preserving more of the original signal (requires JAX).
"""

from importlib import import_module
from typing import Any

__all__ = [
    # Core transformers (no optional deps)
    "RankTransformer",
    "LagTransformer",
    "MovingAverageTransformer",
    "LogReturnTransformer",
    "GroupStatsTransformer",
    "FeatureNeutralizer",
    # Optional deps
    "EmbeddingTransformer",  # requires dspy
    "DimReducer",  # umap is optional within
    "FeaturePenalizer",  # requires jax
]

_LAZY_IMPORTS: dict[str, str] = {
    # Core transformers
    "RankTransformer": "centimators.feature_transformers.ranking",
    "LagTransformer": "centimators.feature_transformers.time_series",
    "MovingAverageTransformer": "centimators.feature_transformers.time_series",
    "LogReturnTransformer": "centimators.feature_transformers.time_series",
    "GroupStatsTransformer": "centimators.feature_transformers.stats",
    "FeatureNeutralizer": "centimators.feature_transformers.neutralization",
    # Optional deps
    "EmbeddingTransformer": "centimators.feature_transformers.embedding",
    "DimReducer": "centimators.feature_transformers.dimreduction",
    "FeaturePenalizer": "centimators.feature_transformers.penalization",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(
            f"module 'centimators.feature_transformers' has no attribute {name!r}"
        )
    module = import_module(module_path)
    attr = getattr(module, name)
    globals()[name] = attr  # cache for future access
    return attr
