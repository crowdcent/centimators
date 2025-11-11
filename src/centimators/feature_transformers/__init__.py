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
"""

from .ranking import RankTransformer
from .time_series import LagTransformer, MovingAverageTransformer, LogReturnTransformer
from .stats import GroupStatsTransformer

__all__ = [
    "RankTransformer",
    "LagTransformer",
    "MovingAverageTransformer",
    "LogReturnTransformer",
    "GroupStatsTransformer",
]
