"""Time-series feature transformers for grouped temporal operations."""

import narwhals as nw
from narwhals.typing import FrameT, IntoSeries

from .base import _BaseFeatureTransformer, _attach_group


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
