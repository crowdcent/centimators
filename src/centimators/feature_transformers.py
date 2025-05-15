import warnings
import narwhals as nw
from narwhals.typing import FrameT, IntoSeries
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl
from typing import Callable


# Helper functions for horizontal statistics using narwhals expressions
def var_horizontal(*exprs: nw.Expr, ddof: int = 1) -> nw.Expr:
    """
    Computes the variance horizontally (row-wise) across a set of expressions.

    Parameters
    ----------
    *exprs : nw.Expr
        Narwhals expressions representing the columns to compute variance over.
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.

    Returns
    -------
    nw.Expr
        A Narwhals expression for the horizontal variance.
    """
    actual_exprs = list(exprs)
    n = len(actual_exprs)

    if not actual_exprs:
        return nw.lit(float("nan"), dtype=nw.Float64)

    mean_expr = nw.mean_horizontal(*actual_exprs)
    sum_sq_diff_expr = nw.sum_horizontal(
        *[(expr - mean_expr) ** 2 for expr in actual_exprs]
    )

    denominator = n - ddof
    if denominator <= 0:
        # Variance is undefined or NaN (e.g., single point with ddof=1)
        return nw.lit(float("nan"), dtype=nw.Float64)

    return sum_sq_diff_expr / nw.lit(denominator, dtype=nw.Float64)


def std_horizontal(*exprs: nw.Expr, ddof: int = 1) -> nw.Expr:
    """
    Computes the standard deviation horizontally (row-wise) across a set of expressions.

    Parameters
    ----------
    *exprs : nw.Expr
        Narwhals expressions representing the columns to compute standard deviation over.
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof.

    Returns
    -------
    nw.Expr
        A Narwhals expression for the horizontal standard deviation.
    """
    actual_exprs = list(exprs)
    if not actual_exprs:
        return nw.lit(float("nan"), dtype=nw.Float64)

    variance_expr = var_horizontal(*actual_exprs, ddof=ddof)
    # sqrt of NaN is NaN; sqrt of negative (float precision issues) also leads to NaN in backends
    return variance_expr**0.5


def skew_horizontal(*exprs: nw.Expr) -> nw.Expr:
    """
    Computes the skewness horizontally (row-wise) across a set of expressions.
    Uses a bias-corrected formula.

    Parameters
    ----------
    *exprs : nw.Expr
        Narwhals expressions representing the columns to compute skewness over.

    Returns
    -------
    nw.Expr
        A Narwhals expression for the horizontal skewness.
    """
    actual_exprs = list(exprs)
    n = len(actual_exprs)

    if n < 3:
        # Skewness with this specific correction factor is undefined for n < 3
        return nw.lit(float("nan"), dtype=nw.Float64)

    mean_expr = nw.mean_horizontal(*actual_exprs)
    # ddof=1 for sample standard deviation is standard in skewness calculations
    std_dev_expr = std_horizontal(*actual_exprs, ddof=1)

    # Calculate sum of ((expr - mean) / std_dev)**3
    # This relies on (0/0) -> NaN propagation if std_dev_expr is 0.
    # If std_dev_expr is 0, all (expr - mean_expr) must also be 0 for finite mean.
    # Then (0/0)**3 is NaN. Sum of NaNs is NaN. This is correct.
    standardized_cubed_deviations = [
        ((expr - mean_expr) / std_dev_expr) ** 3 for expr in actual_exprs
    ]
    sum_std_cubed = nw.sum_horizontal(*standardized_cubed_deviations)

    # Bias correction factor: n / ((n - 1) * (n - 2))
    correction_factor_val = n / ((n - 1) * (n - 2))

    # If std_dev_expr was 0, sum_std_cubed is NaN. NaN * factor is NaN.
    return sum_std_cubed * nw.lit(correction_factor_val, dtype=nw.Float64)


def kurtosis_horizontal(*exprs: nw.Expr) -> nw.Expr:
    """
    Computes the excess kurtosis (Fisher's g2) horizontally (row-wise)
    across a set of expressions. Uses a bias-corrected formula.

    Excess kurtosis indicates how much the tails of the distribution differ
    from the tails of a normal distribution. Positive values indicate heavier
    tails (leptokurtic), negative values indicate lighter tails (platykurtic).

    The formula for the sample excess kurtosis (G2) is used:
    G2 = { [n(n+1)] / [(n-1)(n-2)(n-3)] } * sum[ ( (x_i - mean) / std_sample )^4 ]
         - { [3(n-1)^2] / [(n-2)(n-3)] }
    This is undefined for n < 4.

    Parameters
    ----------
    *exprs : nw.Expr
        Narwhals expressions representing the columns to compute kurtosis over.

    Returns
    -------
    nw.Expr
        A Narwhals expression for the horizontal excess kurtosis.
    """
    actual_exprs = list(exprs)
    n = len(actual_exprs)

    if n < 4:
        # Kurtosis with this specific correction factor is undefined for n < 4
        return nw.lit(float("nan"), dtype=nw.Float64)

    mean_expr = nw.mean_horizontal(*actual_exprs)
    # ddof=1 for sample standard deviation is standard in this kurtosis formula
    std_dev_expr = std_horizontal(*actual_exprs, ddof=1)

    # Calculate sum of ((expr - mean) / std_dev)**4
    # If std_dev_expr is 0 (constant data), (0/0) -> NaN. Sum of NaNs is NaN. Correct.
    standardized_fourth_powers = [
        ((expr - mean_expr) / std_dev_expr) ** 4 for expr in actual_exprs
    ]
    sum_std_fourth = nw.sum_horizontal(*standardized_fourth_powers)

    # Bias correction terms
    term1_coeff_val = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    term2_val = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

    # If sum_std_fourth is NaN, the result will be NaN.
    return (sum_std_fourth * nw.lit(term1_coeff_val, dtype=nw.Float64)) - nw.lit(
        term2_val, dtype=nw.Float64
    )


def range_horizontal(*exprs: nw.Expr) -> nw.Expr:
    """
    Computes the range (max - min) horizontally (row-wise) across a set of expressions.

    Parameters
    ----------
    *exprs : nw.Expr
        Narwhals expressions representing the columns to compute range over.

    Returns
    -------
    nw.Expr
        A Narwhals expression for the horizontal range.
    """
    actual_exprs = list(exprs)

    if not actual_exprs:
        return nw.lit(float("nan"), dtype=nw.Float64)

    min_val = nw.min_horizontal(*actual_exprs)
    max_val = nw.max_horizontal(*actual_exprs)

    return max_val - min_val


def coefficient_of_variation_horizontal(*exprs: nw.Expr, ddof: int = 1) -> nw.Expr:
    """
    Computes the coefficient of variation (CV) horizontally (row-wise)
    across a set of expressions.

    CV = standard_deviation / mean

    Parameters
    ----------
    *exprs : nw.Expr
        Narwhals expressions representing the columns to compute CV over.
    ddof : int, default 1
        Delta Degrees of Freedom for the standard deviation calculation.

    Returns
    -------
    nw.Expr
        A Narwhals expression for the horizontal coefficient of variation.
        Returns NaN if mean is zero and std is zero.
        Returns Inf or -Inf if mean is zero and std is non-zero.
    """
    actual_exprs = list(exprs)

    if not actual_exprs:
        return nw.lit(float("nan"), dtype=nw.Float64)

    mean_expr = nw.mean_horizontal(*actual_exprs)
    std_expr = std_horizontal(*actual_exprs, ddof=ddof)

    # Division handles cases:
    # std/0 where std is non-zero -> inf
    # 0/0 -> NaN
    # std/mean
    return std_expr / mean_expr


def _attach_group(X: FrameT, series: IntoSeries, default_name: str):
    """Attach *series* to *X* if supplied and return *(X, col_name)*."""
    if series is not None:
        X = X.with_columns(series)
        return X, series.name
    # Series not provided – assume it already exists on the frame
    return X, default_name


class _BaseFeatureTransformer(TransformerMixin, BaseEstimator):
    """Common plumbing for the feature transformers in this module.

    * Stores *feature_names* (if given) and infers them during ``fit``.
    * Implements a generic ``fit_transform`` that forwards any extra
      keyword arguments to ``transform`` – this means subclasses only
      need to implement ``transform`` and (optionally) override
      ``get_feature_names_out``.
    """

    def __init__(self, feature_names: list[str] | None = None):
        self.feature_names = feature_names

    def fit(self, X: FrameT, y=None):
        if self.feature_names is None:
            self.feature_names = X.columns
        return self

    # Accept **kwargs so subclasses can expose arbitrary metadata
    # (e.g. *date_series* or *ticker_series*) without re-implementing
    # boiler-plate.
    def fit_transform(self, X: FrameT, y=None, **kwargs):
        return self.fit(X, y).transform(X, y, **kwargs)


class RankTransformer(_BaseFeatureTransformer):
    """
    RankTransformer transforms features into their normalized rank within groups defined by a date series.

    Parameters
    ----------
    feature_names : list of str, optional
        Names of columns to transform. If None, all columns of X are used.

    Examples
    --------
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
    0            0.5             0.5
    1            1.0             1.0
    2            1.0             1.0
    """

    def __init__(self, feature_names=None):
        super().__init__(feature_names)

    @nw.narwhalify(allow_series=True)
    def transform(self, X: FrameT, y=None, date_series: IntoSeries = None) -> FrameT:
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

    def get_feature_names_out(self, input_features=None):
        return [f"{feature_name}_rank" for feature_name in self.feature_names]


class LagTransformer(_BaseFeatureTransformer):
    """
    LagTransformer shifts features by specified lag windows within groups defined by a ticker series.

    Parameters
    ----------
    windows : iterable of int
        Lag periods to compute. Each feature will have shifted versions for each lag.
    feature_names : list of str, optional
        Names of columns to transform. If None, all columns of X are used.

    Examples
    --------
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
    ):
        X, ticker_col_name = _attach_group(X, ticker_series, "ticker")

        lag_columns = [
            nw.col(feature_name)
            .shift(lag)
            .alias(f"{feature_name}_lag{lag}")
            .over(ticker_col_name)
            for feature_name in self.feature_names
            for lag in self.windows
        ]

        X = X.select(lag_columns)

        return X

    def get_feature_names_out(self, input_features=None):
        return [
            f"{feature_name}_lag{lag}"
            for feature_name in self.feature_names
            for lag in self.windows
        ]


class MovingAverageTransformer(_BaseFeatureTransformer):
    """
    MovingAverageTransformer computes the moving average of a feature over a specified window.

    Parameters
    ----------
    windows : list of int
        The windows over which to compute the moving average.
    feature_names : list of str, optional
        The names of the features to compute the moving average for.
    """

    def __init__(self, windows, feature_names=None):
        self.windows = windows
        super().__init__(feature_names)

    @nw.narwhalify(allow_series=True)
    def transform(self, X: FrameT, y=None, ticker_series: IntoSeries = None):
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

    def get_feature_names_out(self, input_features=None):
        return [
            f"{feature_name}_ma{window}"
            for feature_name in self.feature_names
            for window in self.windows
        ]


class LogReturnTransformer(_BaseFeatureTransformer):
    """
    LogReturnTransformer computes the log return of a feature.
    TODO: Implement fully in Narwhals
    """

    def __init__(self, feature_names=None):
        super().__init__(feature_names)

    @nw.narwhalify(allow_series=True)
    def transform(self, X: FrameT, y=None, ticker_series: IntoSeries = None):
        X, ticker_col_name = _attach_group(X, ticker_series, "ticker")

        # WARNING: POLARS ONLY FOR NOW
        # LOG ON EXPR IS NOT IMPLEMENTED IN NARWHALS
        log_return_columns = [
            pl.col(feature_name)
            .log()
            .diff()
            .over(ticker_col_name)
            .alias(f"{feature_name}_logreturn")
            for feature_name in self.feature_names
        ]

        X = X.to_polars().select(log_return_columns)

        return X

    def get_feature_names_out(self, input_features=None):
        return [f"{feature_name}_logreturn" for feature_name in self.feature_names]


class GroupStatsTransformer(_BaseFeatureTransformer):
    """
    GroupStatsTransformer calculates statistical measures for defined feature groups.

    This transformer computes mean, standard deviation, and skewness for each
    group of features specified in the feature_group_mapping.

    Parameters
    ----------
    feature_group_mapping : dict, required
        Dictionary mapping group names to lists of feature columns.
        Example: {'group1': ['feature1', 'feature2'], 'group2': ['feature3', 'feature4']}
    stats : list of str, optional
        List of statistics to compute for each group. If None, all statistics are computed.
        Valid options are 'mean', 'std', 'skew', 'kurt', 'range', and 'cv'.

    Examples
    --------
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
        # 1️⃣  map each stat keyword to a function that returns a narwhals Expr
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
            "std": 2,   # ddof=1 ⇒ need at least 2 values for a finite result
            "cv": 2,    # depends on std
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

    def get_feature_names_out(self, input_features=None):
        """Return feature names for all groups."""
        return [
            f"{group}_groupstats_{stat}" for group in self.groups for stat in self.stats
        ]
