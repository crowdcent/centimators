import polars as pl
import pytest
import os

os.environ["KERAS_BACKEND"] = "jax"
from centimators.feature_transformers import (
    RankTransformer,
    LagTransformer,
    MovingAverageTransformer,
    LogReturnTransformer,
    GroupStatsTransformer,
)


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
    ranked = tr.fit_transform(df.select(["feature1", "feature2"]), date_series=df["date"])

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