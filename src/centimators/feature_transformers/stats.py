"""Statistical transformers for horizontal aggregations."""

import warnings
from typing import Callable

import narwhals as nw
from narwhals.typing import FrameT

from centimators.narwhals_utils import (
    std_horizontal,
    skew_horizontal,
    kurtosis_horizontal,
    range_horizontal,
    coefficient_of_variation_horizontal,
)

from .base import _BaseFeatureTransformer


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
