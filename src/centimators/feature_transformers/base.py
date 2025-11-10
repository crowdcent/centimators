"""Base classes and utilities for feature transformers."""

import narwhals as nw
from narwhals.typing import FrameT, IntoSeries
from sklearn.base import BaseEstimator, TransformerMixin


def _attach_group(X: FrameT, series: IntoSeries, default_name: str):
    """Attach *series* to *X* if supplied and return ``(X, col_name)``.

    When ``series`` is ``None`` a constant column named ``default_name`` is
    appended to ``X``.  This ensures downstream ``.over`` operations have a
    valid grouping column instead of referencing a non-existent one.
    """
    if series is not None:
        X = X.with_columns(series)
        return X, series.name

    X = X.with_columns(nw.lit(0).alias(default_name))
    return X, default_name


class _BaseFeatureTransformer(TransformerMixin, BaseEstimator):
    """Common plumbing for the feature transformers in this module.

    Stores *feature_names* (if given) and infers them during ``fit``.
    Implements a generic ``fit_transform`` that forwards any extra
    keyword arguments to ``transform`` – this means subclasses only
    need to implement ``transform`` and (optionally) override
    ``get_feature_names_out``.

    Attributes:
        feature_names (list[str] | None): Names of columns to transform.
    """

    def __init__(self, feature_names: list[str] | None = None):
        self.feature_names = feature_names

    def fit(self, X: FrameT, y=None):
        if self.feature_names is None:
            self.feature_names = X.columns

        self._is_fitted = True
        return self

    # Accept **kwargs so subclasses can expose arbitrary metadata
    # (e.g. *date_series* or *ticker_series*) without re-implementing
    # boiler-plate.
    def fit_transform(self, X: FrameT, y=None, **kwargs):
        return self.fit(X, y).transform(X, y, **kwargs)

    def __sklearn_is_fitted__(self) -> bool:
        """Return ``True`` when the transformer has been fitted."""
        return getattr(self, "_is_fitted", False)
