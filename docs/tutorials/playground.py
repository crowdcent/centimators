import marimo

__generated_with = "0.13.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import polars as pl
    import numpy as np
    import altair as alt
    from datetime import datetime, timedelta

    from centimators.feature_transformers import (
        RankTransformer,
        LagTransformer,
        MovingAverageTransformer,
        LogReturnTransformer,
    )
    return (
        LagTransformer,
        LogReturnTransformer,
        MovingAverageTransformer,
        RankTransformer,
        alt,
        datetime,
        mo,
        np,
        pd,
        pl,
        timedelta,
    )


@app.cell(hide_code=True)
def _(datetime, np, pd, pl, timedelta):
    dates = [datetime.now() - timedelta(days=i) for i in range(90)]
    dates.reverse()
    tickers = [f'Ticker{i}' for i in range(1, 21)]
    data = {'ticker': [], 'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
    for _ticker in tickers:
        base_price = np.random.uniform(10, 1000)
        for date in dates:
            daily_return = np.random.normal(0.005, 0.03)
            close = base_price * (1 + daily_return)
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = close * (1 + np.random.normal(-0.005, 0.005))
            volume = int(np.random.lognormal(10, 1))
            data['ticker'].append(_ticker)
            data['date'].append(date)
            data['open'].append(round(open_price, 2))
            data['high'].append(round(high, 2))
            data['low'].append(round(low, 2))
            data['close'].append(round(close, 2))
            data['volume'].append(volume)
            base_price = close
    df_pandas = pd.DataFrame(data)
    df_polars = pl.DataFrame(data)
    return df_pandas, df_polars


@app.cell(hide_code=True)
def _(df_polars):
    df_polars.plot.line(x="date", y="close", color="ticker").properties(
        width=600, height=400, title="Mock Price Data Over Time Grouped by Tickers"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Create transformers""")
    return


@app.cell
def _(
    LagTransformer,
    LogReturnTransformer,
    MovingAverageTransformer,
    RankTransformer,
):
    ranker: RankTransformer = RankTransformer()
    ranker

    lag_windows = [0, 5, 10, 15]
    _lagger: LagTransformer = LagTransformer(windows=lag_windows)

    ma_windows = [5, 10, 20, 40]
    _ma_transformer = MovingAverageTransformer(windows=ma_windows)
    _log_return_transformer = LogReturnTransformer()

    _log_return_transformer, ranker, _lagger, _ma_transformer
    return lag_windows, ma_windows, ranker


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Use individually (dataframe agnostic)""")
    return


@app.cell
def _(df_pandas, df_polars, ranker):
    # Compare Pandas vs Polars performance
    import time

    # Test with Pandas
    start_time = time.time()
    result_pd = ranker.fit_transform(df_pandas, date_series=df_pandas["date"])
    pandas_time = time.time() - start_time

    # Test with Polars
    start_time = time.time()
    result_pl = ranker.fit_transform(df_polars, date_series=df_polars["date"])
    polars_time = time.time() - start_time

    print(f"Pandas execution time: {pandas_time:.4f} seconds")
    print(f"Polars execution time: {polars_time:.4f} seconds")
    print(f"Polars Speedup: {pandas_time/polars_time:.2f}x")

    # Verify results are equivalent
    pd_result = result_pd
    pl_result = result_pl.to_pandas()
    assert pd_result.equals(pl_result), "Results should be identical!"

    # Display sample of results
    result_pl.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Or chain them together in a pipeline""")
    return


@app.cell(hide_code=True)
def _():
    from sklearn import set_config
    from sklearn.pipeline import make_pipeline

    set_config(enable_metadata_routing=True)
    return (make_pipeline,)


@app.cell
def _(
    LagTransformer,
    LogReturnTransformer,
    MovingAverageTransformer,
    RankTransformer,
    lag_windows,
    ma_windows,
    make_pipeline,
):
    _lagger = LagTransformer(windows=lag_windows).set_transform_request(ticker_series=True)
    ranker_1 = RankTransformer().set_transform_request(date_series=True)
    _ma_transformer = MovingAverageTransformer(windows=ma_windows).set_transform_request(ticker_series=True)
    _log_return_transformer = LogReturnTransformer().set_transform_request(ticker_series=True)
    lagged_ranker = make_pipeline(_log_return_transformer, ranker_1, _lagger, _ma_transformer)
    lagged_ranker
    return (lagged_ranker,)


@app.cell
def _(df_polars, lagged_ranker):
    feature_names = ["open", "close", "volume"]
    transformed_df = lagged_ranker.fit_transform(
        df_polars[feature_names],
        date_series=df_polars["date"],
        ticker_series=df_polars["ticker"],
    )
    transformed_df.tail()
    return (transformed_df,)


@app.cell(hide_code=True)
def _(alt, df_polars, pl, transformed_df):
    # Visualization of the transformation into features
    chart_df = pl.concat([df_polars, transformed_df], how="horizontal")
    original_chart = chart_df.plot.line(x="date", y="close", color="ticker").properties(
        width=300, height=300, title="Input: Raw Stock Prices Over Time"
    )

    transformed_chart = chart_df.plot.line(
        x="date",
        y="close_logreturn_rank_lag0_ma20",
        color="ticker",
    ).properties(
        width=300, height=300, title="Pipeline Output: Normalized/Smoothed Features"
    )
    transformed_chart.encoding.y.scale = alt.Scale(domain=[0, 1])

    chart = original_chart | transformed_chart
    chart.interactive()
    return (chart_df,)


@app.cell(hide_code=True)
def _(alt, chart_df, lag_windows, ma_windows, pl):
    def create_feature_visualization(df, columns, title, width=300, height=300):
        melted_df = df.unpivot(index=['date'], on=columns, variable_name='variable', value_name='value')
        chart = melted_df.plot.line(x='date', y='value', color='variable').properties(width=width, height=height, title=title)
        chart.encoding.y.scale = alt.Scale(domain=[0, 1])
        return chart
    _ticker = 'Ticker1'
    filtered_df = chart_df.filter(pl.col('ticker') == _ticker)
    ma_columns = [f'close_logreturn_rank_lag0_ma{w}' for w in ma_windows]
    lag_columns = [f'close_logreturn_rank_lag{i}_ma5' for i in lag_windows]
    moving_average_chart = create_feature_visualization(filtered_df, ma_columns, f'Different Moving Average Windows for {_ticker}')
    lagged_chart = create_feature_visualization(filtered_df, lag_columns, f'Different Lag Periods for {_ticker}')
    (moving_average_chart | lagged_chart).interactive()
    return


if __name__ == "__main__":
    app.run()
