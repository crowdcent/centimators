import narwhals as nw
from narwhals.typing import FrameT, IntoSeries
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl


def _attach_group(X: FrameT, series: IntoSeries, default_name: str):
    """Attach *series* to *X* if supplied and return *(X, col_name)*.
    """
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
