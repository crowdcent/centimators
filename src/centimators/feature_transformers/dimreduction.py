"""Dimensionality reduction transformers for feature compression."""

import narwhals as nw
from narwhals.typing import FrameT
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .base import _BaseFeatureTransformer


class DimReducer(_BaseFeatureTransformer):
    """
    DimReducer applies dimensionality reduction to features using PCA, t-SNE, or UMAP.

    This transformer reduces the dimensionality of input features by projecting them
    into a lower-dimensional space using one of three methods: Principal Component
    Analysis (PCA), t-distributed Stochastic Neighbor Embedding (t-SNE), or Uniform
    Manifold Approximation and Projection (UMAP).

    Args:
        method (str): The dimensionality reduction method to use. Options are:
            - 'pca': Principal Component Analysis (linear, preserves global structure)
            - 'tsne': t-SNE (non-linear, preserves local structure, visualization)
            - 'umap': UMAP (non-linear, preserves local + global structure)
            Default: 'pca'
        n_components (int): Number of dimensions in the reduced space. Default: 2
        feature_names (list[str] | None): Names of columns to reduce. If None,
            all columns are used.
        **reducer_kwargs: Additional keyword arguments passed to the underlying
            reducer (sklearn.decomposition.PCA, sklearn.manifold.TSNE, or umap.UMAP).

    Examples:
        >>> import polars as pl
        >>> from centimators.feature_transformers import DimReducer
        >>> df = pl.DataFrame({
        ...     'feature1': [1.0, 2.0, 3.0, 4.0],
        ...     'feature2': [4.0, 5.0, 6.0, 7.0],
        ...     'feature3': [7.0, 8.0, 9.0, 10.0],
        ... })
        >>>
        >>> # PCA reduction
        >>> reducer = DimReducer(method='pca', n_components=2)
        >>> reduced = reducer.fit_transform(df)
        >>> print(reduced.columns)  # ['dim_0', 'dim_1']
        >>>
        >>> # t-SNE for visualization
        >>> reducer = DimReducer(method='tsne', n_components=2, random_state=42)
        >>> reduced = reducer.fit_transform(df)
        >>>
        >>> # UMAP (requires umap-learn)
        >>> reducer = DimReducer(method='umap', n_components=2, random_state=42)
        >>> reduced = reducer.fit_transform(df)

    Notes:
        - PCA is deterministic and fast, suitable for preprocessing
        - t-SNE is stochastic and slower, primarily for visualization (does not support
          separate transform - uses fit_transform internally)
        - UMAP balances speed and quality, good for both preprocessing and visualization
        - UMAP requires the umap-learn package: `uv add 'centimators[all]'`
        - All methods work with any narwhals-compatible backend (pandas, polars, etc.)
    """

    def __init__(
        self,
        method: str = "pca",
        n_components: int = 2,
        feature_names: list[str] | None = None,
        **reducer_kwargs,
    ):
        super().__init__(feature_names=feature_names)

        valid_methods = ["pca", "tsne", "umap"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

        self.method = method
        self.n_components = n_components
        self.reducer_kwargs = reducer_kwargs
        self._reducer = None

    def fit(self, X: FrameT, y=None):
        """Fit the dimensionality reduction model.

        Args:
            X (FrameT): Input data frame.
            y: Ignored. Kept for compatibility.

        Returns:
            DimReducer: The fitted transformer.
        """
        super().fit(X, y)

        # Initialize the appropriate reducer
        if self.method == "pca":
            self._reducer = PCA(n_components=self.n_components, **self.reducer_kwargs)
        elif self.method == "tsne":
            self._reducer = TSNE(n_components=self.n_components, **self.reducer_kwargs)
        elif self.method == "umap":
            try:
                import umap
            except ImportError as e:
                raise ImportError(
                    "DimReducer with method='umap' requires umap-learn. Install with:\n"
                    "  uv add 'centimators[all]'\n"
                    "or:\n"
                    "  pip install 'centimators[all]'"
                ) from e
            self._reducer = umap.UMAP(
                n_components=self.n_components, **self.reducer_kwargs
            )

        # Fit the reducer on the selected features
        X_native = nw.from_native(X)
        X_subset = X_native.select(self.feature_names)
        X_numpy = X_subset.to_numpy()

        # For t-SNE, we skip fit since it doesn't support separate fit/transform
        if self.method != "tsne":
            self._reducer.fit(X_numpy)

        return self

    @nw.narwhalify(allow_series=True)
    def transform(self, X: FrameT, y=None) -> FrameT:
        """Transform features by reducing their dimensionality.

        Args:
            X (FrameT): Input data frame.
            y: Ignored. Kept for compatibility.

        Returns:
            FrameT: Transformed data frame with reduced dimensionality.
                Columns are named 'dim_0', 'dim_1', ..., 'dim_{n_components-1}'.
        """
        if self._reducer is None:
            raise ValueError("Transformer not fitted. Call fit() first.")

        # Extract features and convert to numpy
        X_subset = X.select(self.feature_names)
        X_numpy = X_subset.to_numpy()

        # Apply dimensionality reduction
        # Note: t-SNE doesn't support transform(), so we use fit_transform
        if self.method == "tsne":
            X_reduced = self._reducer.fit_transform(X_numpy)
        else:
            X_reduced = self._reducer.transform(X_numpy)

        # Create output column names
        output_cols = {f"dim_{i}": X_reduced[:, i] for i in range(self.n_components)}

        # Return as narwhals DataFrame with the same backend as input
        return nw.from_dict(output_cols, backend=nw.get_native_namespace(X))

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return the output feature names.

        Args:
            input_features (list[str], optional): Ignored. Kept for compatibility.

        Returns:
            list[str]: List of output feature names: ['dim_0', 'dim_1', ...].
        """
        return [f"dim_{i}" for i in range(self.n_components)]
