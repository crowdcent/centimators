"""Neutralization transformers for reducing feature exposure."""

import warnings

import narwhals as nw
import numpy as np
from joblib import Parallel, delayed
from narwhals.typing import FrameT, IntoSeries

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


from ..narwhals_utils import _ensure_numpy
from .base import _BaseFeatureTransformer, _gaussianize, _min_max_scale


class FeatureNeutralizer(_BaseFeatureTransformer):
    """
    Classic feature neutralization by subtracting a linear model to reduce feature exposure.

    This transformer neutralizes predictions by removing their linear relationship with specified
    features. For each era, it:
    1. Gaussianizes the predictions (rank -> normalize -> inverse CDF)
    2. Fits a linear model: prediction ~ features
    3. Subtracts proportion * exposure from predictions
    4. Re-normalizes and scales to [0, 1]

    Args:
        proportion (float or list of float): How much to neutralize in range [0, 1].
            0 = no neutralization, 1 = full neutralization.
            If list, creates multiple output columns (one per proportion).
        pred_name (str or list of str): Name(s) of prediction column(s) to neutralize.
            Used for generating output column names.
        feature_names (list of str, optional): Names of feature columns to neutralize against.
            If None, all columns of X are used.
        suffix (str, optional): Suffix to append to output column names.
        n_jobs (int): Number of parallel jobs. 1 = sequential (default), -1 = all cores.
        verbose (bool): Show progress bar over eras. Default False.

    Examples:
        >>> import pandas as pd
        >>> from centimators.feature_transformers import FeatureNeutralizer
        >>> # Sample data with eras, features, and predictions
        >>> df = pd.DataFrame({
        ...     'era': ['era1', 'era1', 'era1', 'era2', 'era2', 'era2'],
        ...     'feature1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ...     'feature2': [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ...     'prediction': [0.7, 0.8, 0.9, 0.6, 0.7, 0.8]
        ... })
        >>> neutralizer = FeatureNeutralizer(
        ...     proportion=0.5,
        ...     pred_name='prediction',
        ...     feature_names=['feature1', 'feature2']
        ... )
        >>> # Predictions to neutralize (can be separate from features)
        >>> result = neutralizer.fit_transform(
        ...     df[['prediction']],
        ...     features=df[['feature1', 'feature2']],
        ...     era_series=df['era']
        ... )
    """

    def __init__(
        self,
        proportion: float | list[float] = 0.5,
        pred_name: str | list[str] = "prediction",
        feature_names: list[str] | None = None,
        suffix: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        # Normalize inputs to lists
        self.pred_names = [pred_name] if isinstance(pred_name, str) else pred_name
        self.proportions = [proportion] if isinstance(proportion, float) else proportion

        # Validate
        assert len(self.pred_names) == len(set(self.pred_names)), (
            "Duplicate pred_names found."
        )
        for prop in self.proportions:
            assert 0.0 <= prop <= 1.0, f"proportion should be in [0, 1]. Got {prop}."

        self.suffix = suffix
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Generate output column names
        self._output_names = [
            (
                f"{pname}_neutralized_{prop}_{suffix}"
                if suffix
                else f"{pname}_neutralized_{prop}"
            )
            for pname in self.pred_names
            for prop in self.proportions
        ]

        # Initialize with feature_names for the features to neutralize against
        super().__init__(feature_names)

    @nw.narwhalify(allow_series=True)
    def transform(
        self,
        X: FrameT,
        y=None,
        features: FrameT | None = None,
        era_series: IntoSeries | None = None,
    ) -> FrameT:
        """Neutralizes predictions against features.

        Args:
            X: Input predictions to neutralize (shape: n_samples x n_predictions).
            y: Ignored. Kept for sklearn compatibility.
            features: DataFrame with features for neutralization.
                If None, uses X as both predictions and features.
            era_series: Series with era labels for grouping.
                If None, treats all data as a single era.

        Returns:
            DataFrame with neutralized predictions, scaled to [0, 1].
        """
        # If features not provided, use X as features
        if features is None:
            features = X

        # Convert to numpy for numerical operations
        predictions = _ensure_numpy(X)
        feature_array = _ensure_numpy(features)

        # Ensure predictions is 2D
        if predictions.ndim == 1:
            assert len(self.pred_names) == 1, (
                "predictions is 1D but multiple pred_names given"
            )
            predictions = predictions.reshape(-1, 1)
        else:
            assert predictions.shape[1] == len(self.pred_names), (
                f"predictions has {predictions.shape[1]} cols but {len(self.pred_names)} pred_names"
            )

        # Convert era_series to numpy
        if era_series is not None:
            eras = _ensure_numpy(era_series, allow_series=True)
        else:
            warnings.warn(
                "era_series not provided. Treating all data as a single era. "
                "This is fine for live inference (1 era) but may be incorrect "
                "for training data with multiple eras.",
                UserWarning,
            )
            eras = np.array(["X"] * len(predictions))

        # Process each prediction column and proportion
        if self.n_jobs == 1:
            # Sequential
            results = [
                self._neutralize_by_era(
                    predictions[:, pred_idx], feature_array, eras, prop, self.verbose
                )
                for pred_idx in range(len(self.pred_names))
                for prop in self.proportions
            ]
        else:
            # Parallel via joblib (disable verbose)
            tasks = [
                delayed(self._neutralize_by_era)(
                    predictions[:, pred_idx], feature_array, eras, prop, False
                )
                for pred_idx in range(len(self.pred_names))
                for prop in self.proportions
            ]
            results = Parallel(n_jobs=self.n_jobs)(tasks)

        # Stack results and convert back to dataframe with native type
        result_array = np.column_stack(results)

        # Create dictionary for dataframe construction (works with both pandas and polars)
        result_dict = {
            col_name: result_array[:, i]
            for i, col_name in enumerate(self._output_names)
        }

        # Get the native namespace to create the appropriate dataframe type
        native_namespace = nw.get_native_namespace(X)
        result_df = nw.from_native(
            native_namespace.DataFrame(result_dict),
            eager_only=True,
        )

        return result_df

    def _neutralize_by_era(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        eras: np.ndarray,
        proportion: float,
        verbose: bool = False,
    ) -> np.ndarray:
        """Neutralize predictions era by era."""
        unique_eras = np.unique(eras)
        neutralized = np.zeros_like(predictions)

        era_iter = tqdm(unique_eras, desc=f"prop={proportion}", disable=not verbose)
        for era in era_iter:
            mask = eras == era
            era_pred = predictions[mask]
            era_features = features[mask]

            # Gaussianize then neutralize
            era_pred_norm = _gaussianize(era_pred)
            era_pred_neut = self._neutralize(era_pred_norm, era_features, proportion)
            neutralized[mask] = era_pred_neut

        # Scale all neutralized predictions to [0, 1]
        return _min_max_scale(neutralized)

    @staticmethod
    def _neutralize(
        predictions: np.ndarray, features: np.ndarray, proportion: float
    ) -> np.ndarray:
        """Neutralize predictions by removing linear exposure to features.

        Args:
            predictions: Gaussianized predictions (1D)
            features: Feature matrix (2D)
            proportion: How much to neutralize [0, 1]

        Returns:
            Neutralized predictions, standardized to mean=0, std=1
        """
        # Fit linear model: predictions = features @ coeffs
        # Use lstsq to solve: features @ coeffs = predictions
        coeffs, _, _, _ = np.linalg.lstsq(features, predictions, rcond=None)

        # Compute exposure: features @ coeffs
        exposure = features @ coeffs

        # Subtract proportion of exposure
        neutralized = predictions - proportion * exposure

        # Standardize
        return neutralized / np.std(neutralized)
