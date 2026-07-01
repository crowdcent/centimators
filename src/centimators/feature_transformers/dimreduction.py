"""Dimensionality reduction transformers for feature compression."""

import narwhals as nw
import numpy as np
from narwhals.typing import FrameT
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .base import _BaseFeatureTransformer


class DimReducer(_BaseFeatureTransformer):
    """Dimensionality reduction using PCA, t-SNE, or UMAP.

    Reduces ``feature_names`` columns into ``n_components`` output columns
    named ``{prefix}_{i}``.

    Args:
        method: ``"pca"``, ``"umap"``, or ``"tsne"``.
        n_components: Number of dimensions in the reduced space.
        feature_names: Columns to reduce.  If None, all columns are used.
        prefix: Output column name prefix.  Default ``"dim"`` produces
            ``dim_0, dim_1, ...``.
        **reducer_kwargs: Forwarded to the underlying reducer.
            Common: ``random_state=42``.

    Notes:
        To use GPU-accelerated cuML backends, swap the import before
        constructing DimReducer::

            import umap
            from cuml.manifold import UMAP as cuUMAP
            umap.UMAP = cuUMAP  # drop-in replacement

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        >>> reducer = DimReducer(method="pca", n_components=2)
        >>> reduced = reducer.fit_transform(df)
        >>> reduced.columns
        ['dim_0', 'dim_1']

        >>> reducer = DimReducer(
        ...     method="pca", n_components=2, prefix="emb_thesis"
        ... )
        >>> reducer.fit_transform(df).columns
        ['emb_thesis_0', 'emb_thesis_1']
    """

    _VALID_METHODS = ("pca", "tsne", "umap")

    def __init__(
        self,
        method: str = "pca",
        n_components: int = 2,
        feature_names: list[str] | None = None,
        prefix: str = "dim",
        **reducer_kwargs,
    ):
        super().__init__(feature_names=feature_names)
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"method must be one of {self._VALID_METHODS}, got {method!r}"
            )
        self.method = method
        self.n_components = n_components
        self.prefix = prefix
        self.reducer_kwargs = reducer_kwargs
        self._reducer = None

    def fit(self, X: FrameT, y=None):
        super().fit(X, y)

        if self.method == "pca":
            self._reducer = PCA(n_components=self.n_components, **self.reducer_kwargs)
        elif self.method == "tsne":
            self._reducer = TSNE(n_components=self.n_components, **self.reducer_kwargs)
        elif self.method == "umap":
            try:
                import umap
            except ImportError as e:
                raise ImportError(
                    "DimReducer with method='umap' requires umap-learn. "
                    "Install with: uv pip install 'centimators[all]'"
                ) from e
            self._reducer = umap.UMAP(
                n_components=self.n_components, **self.reducer_kwargs
            )

        X_native = nw.from_native(X)
        X_numpy = np.asarray(
            X_native.select(self.feature_names).to_numpy(), dtype=np.float32
        )

        if self.method != "tsne":
            self._reducer.fit(X_numpy)

        return self

    @nw.narwhalify(allow_series=True)
    def transform(self, X: FrameT, y=None) -> FrameT:
        if self._reducer is None:
            raise ValueError("Transformer not fitted. Call fit() first.")

        X_numpy = np.asarray(X.select(self.feature_names).to_numpy(), dtype=np.float32)

        if self.method == "tsne":
            X_reduced = self._reducer.fit_transform(X_numpy)
        else:
            X_reduced = self._reducer.transform(X_numpy)

        X_reduced = np.asarray(X_reduced)

        output_cols = {
            f"{self.prefix}_{i}": X_reduced[:, i] for i in range(self.n_components)
        }
        return nw.from_dict(output_cols, backend=nw.get_native_namespace(X))

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return [f"{self.prefix}_{i}" for i in range(self.n_components)]
