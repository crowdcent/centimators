import narwhals as nw
from narwhals.typing import FrameT, IntoSeries
from sklearn.base import BaseEstimator, TransformerMixin


class RankTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y=None, date_series=None):
        return self

    @nw.narwhalify(allow_series=True)
    def transform(self, X: FrameT, date_series: IntoSeries) -> FrameT:
        date_col_name = date_series.name
        X = X.with_columns(date_series)

        # compute rank for each feature
        rank_columns: list[nw.Expr] = [
            nw.col(feature_name)
            .rank()
            .over(date_col_name)
            .alias(f"{feature_name}_rank_temp")
            for feature_name in self.feature_names
        ]

        count_columns: list[nw.Expr] = [
            nw.col(feature_name)
            .count()
            .over(date_col_name)
            .alias(f"{feature_name}_count")
            for feature_name in self.feature_names
        ]

        X = X.select([*rank_columns, *count_columns])

        final_columns: list[nw.Expr] = [
            (
                nw.col(f"{feature_name}_rank_temp") / nw.col(f"{feature_name}_count")
            ).alias(f"{feature_name}_rank")
            for feature_name in self.feature_names
        ]

        X = X.select(final_columns)

        return X


class LagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, windows, feature_names=None):
        self.windows = sorted(windows, reverse=True)
        self.feature_names = feature_names

    def fit(self, X: FrameT, y=None, ticker_series=None, date_series=None):
        if self.feature_names is None:
            self.feature_names = X.columns
        return self

    def transform(self, X: FrameT, y=None, ticker_series=None, date_series=None):
        pass
