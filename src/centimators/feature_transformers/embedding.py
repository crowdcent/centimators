"""Embedding transformers for text and categorical features using DSPy."""

import narwhals as nw
from narwhals.typing import FrameT
import numpy as np

try:
    import dspy
except ImportError as e:
    raise ImportError(
        "EmbeddingTransformer requires dspy. Install with:\n"
        "  uv add 'centimators[dspy]'\n"
        "or:\n"
        "  pip install 'centimators[dspy]'"
    ) from e

from .base import _BaseFeatureTransformer


class EmbeddingTransformer(_BaseFeatureTransformer):
    """
    EmbeddingTransformer embeds text and categorical features using DSPy's Embedder.

    This transformer converts text or categorical columns into dense vector embeddings
    using either hosted embedding models (e.g., OpenAI) or custom embedding functions
    (e.g., local SentenceTransformers). The embeddings are expanded into multiple
    columns for sklearn compatibility.

    Args:
        model (str or Callable): The embedding model to use. Can be:
            - A string for hosted models (e.g., "openai/text-embedding-3-small")
            - A callable function (e.g., SentenceTransformer.encode)
        feature_names (list[str] | None): Names of columns to embed. If None,
            all columns are embedded.
        categorical_mapping (dict[str, str] | None): Optional mapping from categorical
            column names to text templates. For example:
            {"sector": "Company sector: {}"} will format the sector value as
            "Company sector: Technology" before embedding.
        batch_size (int): Batch size for embedding computation. Default: 200.
        caching (bool): Whether to cache embeddings (for hosted models). Default: True.
        **embedder_kwargs: Additional keyword arguments passed to dspy.Embedder.

    Examples:
        >>> import polars as pl
        >>> from centimators.feature_transformers import EmbeddingTransformer
        >>> from sentence_transformers import SentenceTransformer
        >>>
        >>> # Example 1: Using a local model
        >>> model = SentenceTransformer('all-MiniLM-L6-v2')
        >>> df = pl.DataFrame({
        ...     'text': ['AI company', 'Bank', 'Pharma firm'],
        ...     'sector': ['Technology', 'Finance', 'Healthcare']
        ... })
        >>>
        >>> transformer = EmbeddingTransformer(
        ...     model=model.encode,
        ...     feature_names=['text', 'sector'],
        ...     categorical_mapping={'sector': 'Company sector: {}'}
        ... )
        >>> embedded = transformer.fit_transform(df[['text', 'sector']])
        >>> print(embedded.columns)  # text_embed_0, text_embed_1, ..., sector_embed_0, ...
        >>>
        >>> # Example 2: Using a hosted model
        >>> transformer = EmbeddingTransformer(
        ...     model="openai/text-embedding-3-small",
        ...     feature_names=['text']
        ... )
        >>> embedded = transformer.fit_transform(df[['text']])

    Notes:
        - Null values are skipped and filled with zero vectors
        - Embedding dimension is inferred from the first batch
        - Output columns follow the pattern: `{feature_name}_embed_{dim_idx}`
        - Requires `centimators[dspy]` installation
    """

    def __init__(
        self,
        model,
        feature_names: list[str] | None = None,
        categorical_mapping: dict[str, str] | None = None,
        batch_size: int = 200,
        caching: bool = True,
        **embedder_kwargs,
    ):
        super().__init__(feature_names=feature_names)
        self.model = model
        self.categorical_mapping = categorical_mapping or {}
        self.batch_size = batch_size
        self.caching = caching
        self.embedder_kwargs = embedder_kwargs
        self._embedder = None
        self._embedding_dims = {}  # Track dimension per feature

    def fit(self, X: FrameT, y=None):
        """Fit the transformer and initialize the embedder.

        Args:
            X (FrameT): Input data frame.
            y: Ignored. Kept for compatibility.

        Returns:
            EmbeddingTransformer: The fitted transformer.
        """
        super().fit(X, y)

        # Initialize DSPy embedder
        self._embedder = dspy.Embedder(
            model=self.model,
            batch_size=self.batch_size,
            caching=self.caching,
            **self.embedder_kwargs,
        )

        return self

    @nw.narwhalify(allow_series=True)
    def transform(self, X: FrameT, y=None) -> FrameT:
        """Transform features by embedding them into dense vectors.

        Args:
            X (FrameT): Input data frame.
            y: Ignored. Kept for compatibility.

        Returns:
            FrameT: Transformed data frame with embedding columns expanded.
                Each input feature becomes multiple columns:
                {feature_name}_embed_0, {feature_name}_embed_1, etc.
        """
        if self._embedder is None:
            raise ValueError("Transformer not fitted. Call fit() first.")

        all_embedding_cols = []

        for feature_name in self.feature_names:
            # Extract column values
            col_values = X.select(nw.col(feature_name)).to_native()

            # Convert to list of strings
            if hasattr(col_values, "to_list"):
                values_list = col_values[feature_name].to_list()
            elif hasattr(col_values, "tolist"):
                values_list = col_values[feature_name].tolist()
            else:
                values_list = list(col_values[feature_name])

            # Apply categorical mapping if specified
            if feature_name in self.categorical_mapping:
                template = self.categorical_mapping[feature_name]
                values_list = [
                    template.format(val) if val is not None else None
                    for val in values_list
                ]
            else:
                # Convert to string
                values_list = [
                    str(val) if val is not None else None for val in values_list
                ]

            # Separate null and non-null indices
            non_null_indices = [
                i for i, val in enumerate(values_list) if val is not None
            ]
            non_null_values = [values_list[i] for i in non_null_indices]

            # Compute embeddings for non-null values
            if non_null_values:
                embeddings = self._embedder(non_null_values)
                embedding_dim = embeddings.shape[1]

                # Store dimension for this feature
                self._embedding_dims[feature_name] = embedding_dim

                # Create full embedding matrix with zeros for nulls
                full_embeddings = np.zeros(
                    (len(values_list), embedding_dim), dtype=np.float32
                )
                full_embeddings[non_null_indices] = embeddings
            else:
                # All nulls - can't infer dimension
                if feature_name not in self._embedding_dims:
                    raise ValueError(
                        f"Cannot determine embedding dimension for '{feature_name}' - "
                        f"all values are null. Ensure at least one non-null value exists."
                    )
                embedding_dim = self._embedding_dims[feature_name]
                full_embeddings = np.zeros(
                    (len(values_list), embedding_dim), dtype=np.float32
                )

            # Store embeddings for this feature
            for dim_idx in range(embedding_dim):
                col_name = f"{feature_name}_embed_{dim_idx}"
                all_embedding_cols.append(
                    (col_name, full_embeddings[:, dim_idx].tolist())
                )

        # Build a df from all embedding columns and return as the same backend as X
        if all_embedding_cols:
            columns_dict = {col_name: values for col_name, values in all_embedding_cols}
            return nw.from_dict(columns_dict, backend=nw.get_native_namespace(X))
        else:
            # Return empty frame with correct number of rows
            return nw.from_dict(
                {"_empty": [None] * len(X)}, backend=nw.get_native_namespace(X)
            )

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return the output feature names.

        Args:
            input_features (list[str], optional): Ignored. Kept for compatibility.

        Returns:
            list[str]: List of transformed feature names in the format
                {feature_name}_embed_{dim_idx}.

        Raises:
            ValueError: If called before transform() when dimensions are unknown.
        """
        output_names = []
        for feature_name in self.feature_names:
            if feature_name not in self._embedding_dims:
                raise ValueError(
                    f"Cannot determine output feature names for '{feature_name}' - "
                    f"call transform() first to infer embedding dimensions."
                )
            embedding_dim = self._embedding_dims[feature_name]
            for dim_idx in range(embedding_dim):
                output_names.append(f"{feature_name}_embed_{dim_idx}")
        return output_names
