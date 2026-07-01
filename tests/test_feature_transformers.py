import polars as pl
import pytest
import os
import numpy as np

os.environ["KERAS_BACKEND"] = "jax"
from centimators.feature_transformers import (
    RankTransformer,
    LagTransformer,
    MovingAverageTransformer,
    LogReturnTransformer,
    GroupStatsTransformer,
    DimReducer,
)

# EmbeddingTransformer import with optional dependency check
try:
    from centimators.feature_transformers import EmbeddingTransformer

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


def _make_simple_frame():
    return pl.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"],
            "ticker": ["A", "A", "B", "B"],
            "feature1": [10, 20, 30, 40],
            "feature2": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_rank_transformer():
    df = _make_simple_frame()
    tr = RankTransformer(feature_names=["feature1", "feature2"])
    ranked = tr.fit_transform(
        df.select(["feature1", "feature2"]), date_series=df["date"]
    )

    # Within each date the higher value should get rank 1.0, lower 0.5 because 2 rows.
    assert pytest.approx(ranked["feature1_rank"][0]) == 0.5
    assert pytest.approx(ranked["feature1_rank"][1]) == 1.0
    # Second date ranks again starting at 0.5 then 1.0
    assert pytest.approx(ranked["feature1_rank"][2]) == 0.5
    assert pytest.approx(ranked["feature1_rank"][3]) == 1.0


def test_lag_transformer():
    df = _make_simple_frame()
    lt = LagTransformer(windows=[1], feature_names=["feature1"])
    lagged = lt.fit_transform(df.select(["feature1"]), ticker_series=df["ticker"])

    # First row for each ticker should be null (None in polars) after lag of 1.
    assert lagged["feature1_lag1"][0] is None
    assert lagged["feature1_lag1"][2] is None  # first row for ticker B
    # Second row of ticker A should equal previous value 10
    assert lagged["feature1_lag1"][1] == 10


def test_moving_average_transformer():
    df = _make_simple_frame()
    ma_t = MovingAverageTransformer(windows=[2], feature_names=["feature1"])
    ma = ma_t.fit_transform(df.select(["feature1"]), ticker_series=df["ticker"])

    # Moving average with window 2 for ticker A second row: (10+20)/2 = 15
    assert pytest.approx(ma["feature1_ma2"][1]) == 15.0


def test_log_return_transformer():
    df = _make_simple_frame()
    lr_t = LogReturnTransformer(feature_names=["feature1"])
    lr = lr_t.fit_transform(df.select(["feature1"]), ticker_series=df["ticker"])

    # Log return of second row for ticker A: log(20) - log(10)
    import math

    expected = math.log(20) - math.log(10)
    assert pytest.approx(lr["feature1_logreturn"][1]) == expected
    # First row log return should be null
    assert lr["feature1_logreturn"][0] is None


def test_group_stats_transformer():
    df = _make_simple_frame()
    mapping = {"grp": ["feature1", "feature2"]}
    gst = GroupStatsTransformer(feature_group_mapping=mapping, stats=["mean", "range"])
    stats_df = gst.fit_transform(df)

    # Mean across two features row 0: (10 + 1)/2 = 5.5
    assert pytest.approx(stats_df["grp_groupstats_mean"][0]) == 5.5
    # Range across row 0: max-min = 10-1 = 9
    assert pytest.approx(stats_df["grp_groupstats_range"][0]) == 9


# EmbeddingTransformer tests (requires dspy)
@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
def test_embedding_transformer_custom_function():
    """Test EmbeddingTransformer with a simple custom embedding function."""

    def simple_embedder(texts):
        """Simple mock embedder that returns fixed-size embeddings."""
        # Return a simple embedding: length of text and character count as features
        embeddings = []
        for text in texts:
            embeddings.append([len(text), sum(ord(c) for c in text) % 100])
        return np.array(embeddings, dtype=np.float32)

    df = pl.DataFrame(
        {
            "text": ["hello", "world", "test"],
        }
    )

    transformer = EmbeddingTransformer(model=simple_embedder, feature_names=["text"])

    embedded = transformer.fit_transform(df[["text"]])

    # Should have 2 embedding dimensions (0 and 1)
    assert "text_embed_0" in embedded.columns
    assert "text_embed_1" in embedded.columns
    assert len(embedded.columns) == 2

    # Check that embeddings are correct
    assert embedded["text_embed_0"][0] == 5  # len('hello')
    assert embedded["text_embed_0"][1] == 5  # len('world')
    assert embedded["text_embed_0"][2] == 4  # len('test')


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
def test_embedding_transformer_categorical_mapping():
    """Test EmbeddingTransformer with categorical mapping."""

    def simple_embedder(texts):
        """Mock embedder."""
        embeddings = []
        for text in texts:
            embeddings.append([len(text)])
        return np.array(embeddings, dtype=np.float32)

    df = pl.DataFrame(
        {
            "sector": ["Tech", "Finance", "Healthcare"],
        }
    )

    transformer = EmbeddingTransformer(
        model=simple_embedder,
        feature_names=["sector"],
        categorical_mapping={"sector": "Company sector: {}"},
    )

    embedded = transformer.fit_transform(df[["sector"]])

    # Should have 1 embedding dimension
    assert "sector_embed_0" in embedded.columns

    # Check that the template was applied: "Company sector: Tech" has length 20
    assert embedded["sector_embed_0"][0] == 20  # len('Company sector: Tech')


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
def test_embedding_transformer_null_handling():
    """Test EmbeddingTransformer handles null values correctly."""

    def simple_embedder(texts):
        """Mock embedder."""
        embeddings = []
        for text in texts:
            embeddings.append([len(text), 1.0])
        return np.array(embeddings, dtype=np.float32)

    df = pl.DataFrame(
        {
            "text": ["hello", None, "world"],
        }
    )

    transformer = EmbeddingTransformer(model=simple_embedder, feature_names=["text"])

    embedded = transformer.fit_transform(df[["text"]])

    # Null value should be filled with zeros
    assert embedded["text_embed_0"][1] == 0.0
    assert embedded["text_embed_1"][1] == 0.0

    # Non-null values should be embedded normally
    assert embedded["text_embed_0"][0] == 5  # len('hello')
    assert embedded["text_embed_0"][2] == 5  # len('world')


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
def test_embedding_transformer_multiple_features():
    """Test EmbeddingTransformer with multiple features."""

    def simple_embedder(texts):
        """Mock embedder."""
        embeddings = []
        for text in texts:
            embeddings.append([len(text)])
        return np.array(embeddings, dtype=np.float32)

    df = pl.DataFrame(
        {
            "text1": ["hello", "world"],
            "text2": ["foo", "bar"],
        }
    )

    transformer = EmbeddingTransformer(
        model=simple_embedder, feature_names=["text1", "text2"]
    )

    embedded = transformer.fit_transform(df[["text1", "text2"]])

    # Should have embeddings for both features
    assert "text1_embed_0" in embedded.columns
    assert "text2_embed_0" in embedded.columns
    assert len(embedded.columns) == 2

    # Check values
    assert embedded["text1_embed_0"][0] == 5  # len('hello')
    assert embedded["text2_embed_0"][0] == 3  # len('foo')


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
def test_embedding_transformer_get_feature_names_out():
    """Test that get_feature_names_out returns correct column names."""

    def simple_embedder(texts):
        """Mock embedder with 3 dimensions."""
        embeddings = []
        for text in texts:
            embeddings.append([1.0, 2.0, 3.0])
        return np.array(embeddings, dtype=np.float32)

    df = pl.DataFrame(
        {
            "text": ["hello"],
        }
    )

    transformer = EmbeddingTransformer(model=simple_embedder, feature_names=["text"])

    transformer.fit_transform(df[["text"]])

    feature_names = transformer.get_feature_names_out()
    assert feature_names == ["text_embed_0", "text_embed_1", "text_embed_2"]


def _make_reduction_frame():
    rng = np.random.default_rng(0)
    return pl.DataFrame({f"f{i}": rng.standard_normal(20) for i in range(5)})


def test_dimreducer_default_prefix_backward_compat():
    """Default prefix must remain ``dim`` (locks 0.3.x column names)."""
    df = _make_reduction_frame()
    reducer = DimReducer(method="pca", n_components=2)
    reduced = reducer.fit_transform(df)
    assert reduced.columns == ["dim_0", "dim_1"]


def test_dimreducer_custom_prefix():
    df = _make_reduction_frame()
    reducer = DimReducer(method="pca", n_components=3, prefix="emb_thesis")
    reduced = reducer.fit_transform(df)
    assert reduced.columns == ["emb_thesis_0", "emb_thesis_1", "emb_thesis_2"]


def test_dimreducer_get_feature_names_out_matches_transform():
    df = _make_reduction_frame()
    for prefix in ("dim", "emb_thesis"):
        reducer = DimReducer(method="pca", n_components=2, prefix=prefix)
        reduced = reducer.fit_transform(df)
        assert reducer.get_feature_names_out() == reduced.columns


def test_dimreducer_invalid_method_raises_at_construction():
    """Invalid method must fail eagerly in __init__, not at fit."""
    with pytest.raises(ValueError, match="method must be one of"):
        DimReducer(method="not_a_method")


def test_dimreducer_pandas_backend():
    """Transformer is backend-agnostic (narwhals)."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {f"f{i}": np.random.default_rng(1).standard_normal(20) for i in range(4)}
    )
    reducer = DimReducer(method="pca", n_components=2, prefix="p")
    reduced = reducer.fit_transform(df)
    assert list(reduced.columns) == ["p_0", "p_1"]
    assert len(reduced) == 20


def test_dimreducer_tsne():
    df = _make_reduction_frame()
    reducer = DimReducer(method="tsne", n_components=2, perplexity=5.0, random_state=0)
    reduced = reducer.fit_transform(df)
    assert reduced.columns == ["dim_0", "dim_1"]
    assert len(reduced) == 20


def test_dimreducer_umap():
    pytest.importorskip("umap")
    df = _make_reduction_frame()
    reducer = DimReducer(method="umap", n_components=2, random_state=0)
    reduced = reducer.fit_transform(df)
    assert reduced.columns == ["dim_0", "dim_1"]
    assert len(reduced) == 20
