import marimo

__generated_with = "0.13.7"
app = marimo.App(width="full")

with app.setup:
    import pandas as pd
    import polars as pl
    import marimo as mo
    import altair as alt
    from centimators.data_transformers import RankTransformer
    import numpy as np
    from datetime import datetime, timedelta


@app.cell
def _():
    ranker = RankTransformer()
    ranker
    return


@app.cell
def make_data():
    # Generate dates for last 90 days
    end_date = datetime.now()
    dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(90)]
    dates.reverse()

    # Generate 5 tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    # Generate random OHLCV data
    np.random.seed(42)
    data = {
        'ticker': [],
        'date': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }

    for ticker in tickers:
        # Start with random base price between 100 and 1000
        base_price = np.random.uniform(100, 1000)
        for date in dates:
            # Generate daily price movements
            daily_return = np.random.normal(0.001, 0.02)  # Mean 0.1%, std 2%
            close = base_price * (1 + daily_return)
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = close * (1 + np.random.normal(0, 0.005))
            volume = int(np.random.lognormal(10, 1))

            data['ticker'].append(ticker)
            data['date'].append(date)
            data['open'].append(round(open_price, 2))
            data['high'].append(round(high, 2))
            data['low'].append(round(low, 2))
            data['close'].append(round(close, 2))
            data['volume'].append(volume)

            base_price = close  # Use today's close as tomorrow's base price

    pandas_df = pd.DataFrame(data)
    polars_df = pl.DataFrame(data)

    return pandas_df, polars_df


@app.cell
def _(polars_df):
    polars_df
    return


@app.cell
def transform_data(polars_df):
    ranker = RankTransformer()
    ranker.fit(polars_df)
    return ranker.transform(polars_df)


if __name__ == "__main__":
    app.run()
