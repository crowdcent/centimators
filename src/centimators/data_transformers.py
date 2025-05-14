import narwhals as nw
from narwhals.typing import FrameT, IntoSeries
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl


class RankTransformer(TransformerMixin, BaseEstimator):
    """
    RankTransformer transforms features into their normalized rank within groups defined by a date series.

    Parameters
    ----------
    feature_names : list of str, optional
        Names of columns to transform. If None, all columns of X are used.

    Examples
    --------
    >>> import pandas as pd
    >>> from centimators.data_transformers import RankTransformer
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
        self.feature_names = feature_names

    def fit(self, X, y=None):
        if self.feature_names is None:
            self.feature_names = X.columns
        return self

    @nw.narwhalify(allow_series=True)
    def transform(self, X: FrameT, y=None, date_series: IntoSeries = None) -> FrameT:
        date_col_name: str = date_series.name if date_series else "date"
        X = X.with_columns(date_series)

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

    def fit_transform(self, X: FrameT, y=None, date_series: IntoSeries = None):
        # This is only required if transform accepts metadata because
        # the fit_transform implementation in TransformerMixin doesn't pass metadata to transform.
        return self.fit(X, y).transform(X, y, date_series)

    def get_feature_names_out(self, input_features=None):
        return [f"{feature_name}_rank" for feature_name in self.feature_names]


class LagTransformer(TransformerMixin, BaseEstimator):
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
    >>> from centimators.data_transformers import LagTransformer
    >>> df = pd.DataFrame({
    ...     'ticker': ['A', 'A', 'A', 'B', 'B'],
    ...     'price': [10, 11, 12, 20, 21]
    ... })
    >>> transformer = LagTransformer(windows=[1, 2], feature_names=['price'])
    >>> result = transformer.fit_transform(df[['price']], ticker_series=df['ticker'])
    >>> print(result)
       price_lag_1  price_lag_2
    0         NaN         NaN
    1        10.0         NaN
    2        11.0        10.0
    3         NaN         NaN
    4        20.0         NaN
    """

    def __init__(self, windows, feature_names=None):
        self.windows = sorted(windows, reverse=True)
        self.feature_names = feature_names

    def fit(self, X: FrameT, y=None):
        if self.feature_names is None:
            self.feature_names = X.columns
        return self

    @nw.narwhalify(allow_series=True)
    def transform(
        self,
        X: FrameT,
        y=None,
        ticker_series: IntoSeries = None,
    ):
        X = X.with_columns(ticker_series)
        ticker_col_name: str = ticker_series.name if ticker_series else "ticker"

        lag_columns = [
            nw.col(feature_name)
            .shift(lag)
            .alias(f"{feature_name}_lag_{lag}")
            .over(ticker_col_name)
            for feature_name in self.feature_names
            for lag in self.windows
        ]

        X = X.select(lag_columns)

        return X

    def fit_transform(self, X: FrameT, y=None, ticker_series: IntoSeries = None):
        # This is only required if transform accepts metadata because
        # the fit_transform implementation in TransformerMixin doesn't pass metadata to transform.
        return self.fit(X, y).transform(X, y, ticker_series)

    def get_feature_names_out(self, input_features=None):
        return [
            f"{feature_name}_lag_{lag}"
            for feature_name in self.feature_names
            for lag in self.windows
        ]


class MovingAverageTransformer(TransformerMixin, BaseEstimator):
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
        self.feature_names = feature_names

    def fit(self, X: FrameT, y=None):
        if self.feature_names is None:
            self.feature_names = X.columns
        return self

    @nw.narwhalify(allow_series=True)
    def transform(self, X: FrameT, y=None, ticker_series: IntoSeries = None):
        X = X.with_columns(ticker_series)
        ticker_col_name: str = ticker_series.name if ticker_series else "ticker"

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

    def fit_transform(self, X: FrameT, y=None, ticker_series: IntoSeries = None):
        return self.fit(X, y).transform(X, y, ticker_series)

    def get_feature_names_out(self, input_features=None):
        return [
            f"{feature_name}_ma{window}"
            for feature_name in self.feature_names
            for window in self.windows
        ]


class LogReturnTransformer(TransformerMixin, BaseEstimator):
    """
    LogReturnTransformer computes the log return of a feature.
    TODO: Implement fully in Narwhals
    """

    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X: FrameT, y=None):
        if self.feature_names is None:
            self.feature_names = X.columns
        return self

    @nw.narwhalify(allow_series=True)
    def transform(self, X: FrameT, y=None, ticker_series: IntoSeries = None):
        X = X.with_columns(ticker_series)
        ticker_col_name: str = ticker_series.name if ticker_series else "ticker"

        # WARNING: POLARS ONLY FOR NOW
        # LOG ON EXPR IS NOT IMPLEMENTED IN NARWHALS
        log_return_columns = [
            pl.col(feature_name)
            .log()
            .diff()
            .over(ticker_col_name)
            .alias(f"{feature_name}_log_return")
            for feature_name in self.feature_names
        ]

        X = X.to_polars().select(log_return_columns)

        return X

    def fit_transform(self, X: FrameT, y=None, ticker_series: IntoSeries = None):
        return self.fit(X, y).transform(X, y, ticker_series)

    def get_feature_names_out(self, input_features=None):
        return [f"{feature_name}_log_return" for feature_name in self.feature_names]
