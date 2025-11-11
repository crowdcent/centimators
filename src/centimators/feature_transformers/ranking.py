"""Ranking transformers for cross-sectional normalization."""

import narwhals as nw
from narwhals.typing import FrameT, IntoSeries

from .base import _BaseFeatureTransformer, _attach_group


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
