# Installation & Quick Start

## Installation
=== "uv (Recommended)"

    ```bash
    uv add centimators
    ```

=== "pip"

    ```bash
    pip install centimators
    ```

## Quick Start

`centimators` transformers are dataframe-agnostic, powered by [narwhals](https://narwhals-dev.github.io/narwhals/).
You can use the same transformer (like `RankTransformer`) seamlessly with both Pandas and Polars DataFrames. This transformer calculates the normalized rank of features within each date group.

First, let's define some common data:
```python
import pandas as pd
import polars as pl
# Create sample OHLCV data for two stocks over four trading days
data = {
    'date': ['2021-01-01', '2021-01-01', '2021-01-02', '2021-01-02', 
             '2021-01-03', '2021-01-03', '2021-01-04', '2021-01-04'],
    'ticker': ['AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL', 'MSFT'],
    'open': [150.0, 280.0, 151.0, 282.0, 152.0, 283.0, 153.0, 284.0],    # Opening prices
    'high': [152.0, 282.0, 153.0, 284.0, 154.0, 285.0, 155.0, 286.0],    # Daily highs
    'low': [149.0, 278.0, 150.0, 280.0, 151.0, 281.0, 152.0, 282.0],     # Daily lows
    'close': [151.0, 281.0, 152.0, 283.0, 153.0, 284.0, 154.0, 285.0],   # Closing prices
    'volume': [1000000, 800000, 1200000, 900000, 1100000, 850000, 1050000, 820000]  # Trading volume
}

# Create both Pandas and Polars DataFrames
df_pd = pd.DataFrame(data)
df_pl = pl.DataFrame(data)

# Define the OHLCV features we want to transform
feature_cols = ['volume', 'close']
```

Now, let's use the transformer:
```python
from centimators.feature_transformers import RankTransformer

transformer = RankTransformer(feature_names=feature_cols)
result_pd = transformer.fit_transform(df_pd[feature_cols], date_series=df_pd['date'])
result_pl = transformer.fit_transform(df_pl[feature_cols], date_series=df_pl['date'])
```

Both `result_pd` (from Pandas) and `result_pl` (from Polars) will contain the same transformed data in their native DataFrame formats. You may find significant performance gains using Polars for certain operations. 