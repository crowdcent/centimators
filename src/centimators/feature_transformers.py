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
    * **RankTransformer** – converts numeric features into their (0, 1]-normalised
    rank within a user-supplied grouping column (e.g. a date).
    * **LagTransformer** – creates shifted/lagged copies of features to expose
    temporal context for time-series models.
    * **MovingAverageTransformer** – rolling mean across arbitrary window sizes.
    * **LogReturnTransformer** – first-difference of the natural logarithm of a
    signal, a common way to compute returns.
    * **GroupStatsTransformer** – horizontally aggregates arbitrary sets of columns
    and exposes statistics such as mean, standard deviation, skew, kurtosis,
    range and coefficient of variation.
"""

import warnings
import narwhals as nw
from narwhals.typing import FrameT, IntoSeries
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Callable

from .horizontal_utils import (
    std_horizontal,
    skew_horizontal,
    kurtosis_horizontal,
    range_horizontal,
    coefficient_of_variation_horizontal,
)


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


class RankTransformer(_BaseFeatureTransformer):
    """
    RankTransformer transforms features into their normalized rank within groups defined by a date series.

    Args:
        feature_names (list of str, optional): Names of columns to transform.
            If None, all columns of X are used.

    Examples:
        >>> import pandas as pd
        >>> from centimators.feature_transformers import RankTransformer
        >>> df = pd.DataFrame({
        ...     'date': ['2021-01-01', '2021-01-01', '2021-01-02'],
        ...     'feature1': [3, 1, 2],
        ...     'feature2': [30, 20, 10]
        ... })
        >>> transformer = RankTransformer(feature_names=['feature1', 'feature2'])
        >>> result = transformer.fit_transform(df[['feature1', 'feature2']], date_series=df['date'])
        >>> print(result)
           feature1_rank  feature2_rank
        0            0.5            0.5
        1            1.0            1.0
        2            1.0            1.0
    """

    def __init__(self, feature_names=None):
        super().__init__(feature_names)

    @nw.narwhalify(allow_series=True)
    def transform(self, X: FrameT, y=None, date_series: IntoSeries = None) -> FrameT:
        """Transforms features to their normalized rank.

        Args:
            X (FrameT): Input data frame.
            y (Any, optional): Ignored. Kept for compatibility.
            date_series (IntoSeries, optional): Series defining groups for ranking (e.g., dates).

        Returns:
            FrameT: Transformed data frame with ranked features.
        """
        X, date_col_name = _attach_group(X, date_series, "date")

        # compute absolute rank for each feature
        rank_columns: list[nw.Expr] = [
            nw.col(feature_name)
            .rank()
            .over(date_col_name)
            .alias(f"{feature_name}_rank_temp")
            for feature_name in self.feature_names
        ]

        # compute count for each feature
        count_columns: list[nw.Expr] = [
            nw.col(feature_name)
            .count()
            .over(date_col_name)
            .alias(f"{feature_name}_count")
            for feature_name in self.feature_names
        ]

        X = X.select([*rank_columns, *count_columns])

        # compute normalized rank for each feature
        final_columns: list[nw.Expr] = [
            (
                nw.col(f"{feature_name}_rank_temp") / nw.col(f"{feature_name}_count")
            ).alias(f"{feature_name}_rank")
            for feature_name in self.feature_names
        ]

        X = X.select(final_columns)

        return X

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Returns the output feature names.

        Args:
            input_features (list[str], optional): Ignored. Kept for compatibility.

        Returns:
            list[str]: List of transformed feature names.
        """
        return [f"{feature_name}_rank" for feature_name in self.feature_names]


class LagTransformer(_BaseFeatureTransformer):
    """
    LagTransformer shifts features by specified lag windows within groups defined by a ticker series.

    Args:
        windows (iterable of int): Lag periods to compute. Each feature will have
            shifted versions for each lag.
        feature_names (list of str, optional): Names of columns to transform.
            If None, all columns of X are used.

    Examples:
        >>> import pandas as pd
        >>> from centimators.feature_transformers import LagTransformer
        >>> df = pd.DataFrame({
        ...     'ticker': ['A', 'A', 'A', 'B', 'B'],
        ...     'price': [10, 11, 12, 20, 21]
        ... })
        >>> transformer = LagTransformer(windows=[1, 2], feature_names=['price'])
        >>> result = transformer.fit_transform(df[['price']], ticker_series=df['ticker'])
        >>> print(result)
           price_lag1  price_lag2
        0         NaN         NaN
        1        10.0         NaN
        2        11.0        10.0
        3         NaN         NaN
        4        20.0         NaN
    """

    def __init__(self, windows, feature_names=None):
        self.windows = sorted(windows, reverse=True)
        super().__init__(feature_names)

    @nw.narwhalify(allow_series=True)
    def transform(
        self,
        X: FrameT,
        y=None,
        ticker_series: IntoSeries = None,
    ) -> FrameT:
        """Applies lag transformation to the features.

        Args:
            X (FrameT): Input data frame.
            y (Any, optional): Ignored. Kept for compatibility.
            ticker_series (IntoSeries, optional): Series defining groups for lagging (e.g., tickers).

        Returns:
            FrameT: Transformed data frame with lagged features. Columns are ordered
                by lag (as in `self.windows`), then by feature (as in `self.feature_names`).
                For example, with `windows=[2,1]` and `feature_names=['A','B']`,
                the output columns will be `A_lag2, B_lag2, A_lag1, B_lag1`.
        """
        X, ticker_col_name = _attach_group(X, ticker_series, "ticker")

        lag_columns = [
            nw.col(feature_name)
            .shift(lag)
            .alias(f"{feature_name}_lag{lag}")
            .over(ticker_col_name)
            for lag in self.windows  # Iterate over lags first
            for feature_name in self.feature_names  # Then over feature names
        ]

        X = X.select(lag_columns)

        return X

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Returns the output feature names.

        Args:
            input_features (list[str], optional): Ignored. Kept for compatibility.

        Returns:
            list[str]: List of transformed feature names, ordered by lag, then by feature.
        """
        return [
            f"{feature_name}_lag{lag}"
            for lag in self.windows  # Iterate over lags first
            for feature_name in self.feature_names  # Then over feature names
        ]


class MovingAverageTransformer(_BaseFeatureTransformer):
    """
    MovingAverageTransformer computes the moving average of a feature over a specified window.

    Args:
        windows (list of int): The windows over which to compute the moving average.
        feature_names (list of str, optional): The names of the features to compute
            the moving average for.
    """

    def __init__(self, windows, feature_names=None):
        self.windows = windows
        super().__init__(feature_names)

    @nw.narwhalify(allow_series=True)
    def transform(self, X: FrameT, y=None, ticker_series: IntoSeries = None) -> FrameT:
        """Applies moving average transformation to the features.

        Args:
            X (FrameT): Input data frame.
            y (Any, optional): Ignored. Kept for compatibility.
            ticker_series (IntoSeries, optional): Series defining groups for moving average (e.g., tickers).

        Returns:
            FrameT: Transformed data frame with moving average features.
        """
        X, ticker_col_name = _attach_group(X, ticker_series, "ticker")

        ma_columns = [
            nw.col(feature_name)
            .rolling_mean(window_size=window)
            .over(ticker_col_name)
            .alias(f"{feature_name}_ma{window}")
            for feature_name in self.feature_names
            for window in self.windows
        ]

        X = X.select(ma_columns)

        return X

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Returns the output feature names.

        Args:
            input_features (list[str], optional): Ignored. Kept for compatibility.

        Returns:
            list[str]: List of transformed feature names.
        """
        return [
            f"{feature_name}_ma{window}"
            for feature_name in self.feature_names
            for window in self.windows
        ]


class LogReturnTransformer(_BaseFeatureTransformer):
    """
    LogReturnTransformer computes the log return of a feature.

    Args:
        feature_names (list of str, optional): Names of columns to transform.
            If None, all columns of X are used.
    """

    def __init__(self, feature_names=None):
        super().__init__(feature_names)

    @nw.narwhalify(allow_series=True)
    def transform(self, X: FrameT, y=None, ticker_series: IntoSeries = None) -> FrameT:
        """Applies log return transformation to the features.

        Args:
            X (FrameT): Input data frame.
            y (Any, optional): Ignored. Kept for compatibility.
            ticker_series (IntoSeries, optional): Series defining groups for log return (e.g., tickers).

        Returns:
            FrameT: Transformed data frame with log return features.
        """
        X, ticker_col_name = _attach_group(X, ticker_series, "ticker")

        log_return_columns = [
            nw.col(feature_name)
            .log()
            .diff()
            .over(ticker_col_name)
            .alias(f"{feature_name}_logreturn")
            for feature_name in self.feature_names
        ]

        X = X.select(log_return_columns)

        return X

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Returns the output feature names.

        Args:
            input_features (list[str], optional): Ignored. Kept for compatibility.

        Returns:
            list[str]: List of transformed feature names.
        """
        return [f"{feature_name}_logreturn" for feature_name in self.feature_names]


class GroupStatsTransformer(_BaseFeatureTransformer):
    """
    GroupStatsTransformer calculates statistical measures for defined feature groups.

    This transformer computes mean, standard deviation, and skewness for each
    group of features specified in the feature_group_mapping.

    Args:
        feature_group_mapping (dict): Dictionary mapping group names to lists of
            feature columns. Example: {'group1': ['feature1', 'feature2'],
            'group2': ['feature3', 'feature4']}
        stats (list of str, optional): List of statistics to compute for each group.
            If None, all statistics are computed. Valid options are 'mean', 'std',
            'skew', 'kurt', 'range', and 'cv'.

    Examples:
        >>> import pandas as pd
        >>> from centimators.feature_transformers import GroupStatsTransformer
        >>> df = pd.DataFrame({
        ...     'feature1': [1, 2, 3],
        ...     'feature2': [4, 5, 6],
        ...     'feature3': [7, 8, 9],
        ...     'feature4': [10, 11, 12]
        ... })
        >>> mapping = {'group1': ['feature1', 'feature2'], 'group2': ['feature3', 'feature4']}
        >>> transformer = GroupStatsTransformer(feature_group_mapping=mapping)
        >>> result = transformer.fit_transform(df)
        >>> print(result)
           group1_groupstats_mean  group1_groupstats_std  group1_groupstats_skew  group2_groupstats_mean  group2_groupstats_std  group2_groupstats_skew
        0                  2.5                 1.5                  0.0                  8.5                 1.5                  0.0
        1                  3.5                 1.5                  0.0                  9.5                 1.5                  0.0
        2                  4.5                 1.5                  0.0                 10.5                 1.5                  0.0
        >>> transformer_mean_only = GroupStatsTransformer(feature_group_mapping=mapping, stats=['mean'])
        >>> result_mean_only = transformer_mean_only.fit_transform(df)
        >>> print(result_mean_only)
           group1_groupstats_mean  group2_groupstats_mean
        0                  2.5                  8.5
        1                  3.5                  9.5
        2                  4.5                 10.5
    """

    def __init__(
        self,
        feature_group_mapping: dict,
        stats: list[str] = ["mean", "std", "skew", "kurt", "range", "cv"],
    ):
        super().__init__(feature_names=None)
        self.feature_group_mapping = feature_group_mapping
        self.groups = list(feature_group_mapping.keys())
        # Supported statistics
        valid_stats = ["mean", "std", "skew", "kurt", "range", "cv"]
        if not all(stat in valid_stats for stat in stats):
            raise ValueError(
                f"stats must be a list containing only {valid_stats}. Got {stats}"
            )
        self.stats = stats

    @nw.narwhalify(allow_series=True)
    def transform(self, X: FrameT, y=None) -> FrameT:
        """Calculates group statistics on the features.

        Args:
            X (FrameT): Input data frame.
            y (Any, optional): Ignored. Kept for compatibility.

        Returns:
            FrameT: Transformed data frame with group statistics features.
        """
        _expr_factories: dict[str, Callable[[list[str]], nw.Expr]] = {
            "mean": lambda cols: nw.mean_horizontal(*cols),
            "std": lambda cols: std_horizontal(*cols, ddof=1),
            "skew": lambda cols: skew_horizontal(*cols),
            "kurt": lambda cols: kurtosis_horizontal(*cols),
            "range": lambda cols: range_horizontal(*cols),
            "cv": lambda cols: coefficient_of_variation_horizontal(*cols),
        }

        _min_required_cols: dict[str, int] = {
            "mean": 1,
            "range": 1,
            "std": 2,  # ddof=1 ⇒ need at least 2 values for a finite result
            "cv": 2,  # depends on std
            "skew": 3,  # bias-corrected formula needs ≥3
            "kurt": 4,  # bias-corrected formula needs ≥4
        }

        stat_expressions: list[nw.Expr] = []

        for group, cols in self.feature_group_mapping.items():
            if not cols:
                raise ValueError(
                    f"No valid columns found for group '{group}' in the input frame."
                )

            n_cols = len(cols)

            for stat in self.stats:
                # Warn early if result is guaranteed to be NaN
                min_required = _min_required_cols[stat]
                if n_cols < min_required:
                    warnings.warn(
                        (
                            f"{self.__class__.__name__}: statistic '{stat}' for group "
                            f"'{group}' requires at least {min_required} feature column(s) "
                            f"but only {n_cols} provided – the resulting column will be NaN."
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )

                expr = _expr_factories[stat](cols).alias(f"{group}_groupstats_{stat}")
                stat_expressions.append(expr)

        return X.select(stat_expressions)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return feature names for all groups.

        Args:
            input_features (list[str], optional): Ignored. Kept for compatibility.

        Returns:
            list[str]: List of transformed feature names.
        """
        return [
            f"{group}_groupstats_{stat}" for group in self.groups for stat in self.stats
        ]
