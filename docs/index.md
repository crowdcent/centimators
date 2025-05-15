<img src="assets/images/centimators_banner_transparent_thinner.png" alt="Centimators" width="100%" style="max-width: 800px;"/>

# Centimators: essential data transformers and model estimators for ML and data science competitions

`centimators` is an open-source python library built on scikit-learn, keras, and narwhals: designed for building and sharing dataframe-agnostic (pandas/polars), multi-framework (jax/tf/pytorch), sklearn-style (fit/transform/predict) transformers, meta-estimators, and machine learning models for data science competitions like Numerai, Kaggle, and the CrowdCent Challenge. 

`centimators` makes heavy use of advanced scikit-learn concepts such as metadata routing. Familiarity with these concepts is recommended for optimal use of the library. You can learn more about metadata routing in the [scikit-learn documentation](https://scikit-learn.org/stable/metadata_routing.html).

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
You can use the same transformer (like `RankTransformer`) seamlessly with both Pandas and Polars DataFrames (NOTE: currently, some transformers only support Polars).

First, let's define some common data:
```python
from centimators.feature_transformers import RankTransformer

# 1. Define your data
data = {
    'date': ['2021-01-01', '2021-01-01', '2021-01-02'],
    'feature1': [3, 1, 2],       # For 2021-01-01: 3 is 2nd, 1 is 1st
    'feature2': [30, 20, 10]      # For 2021-01-01: 30 is 2nd, 20 is 1st
}
feature_cols = ['feature1', 'feature2']
```

**2. With Pandas:**
```python
import pandas as pd

df_pd = pd.DataFrame(data)
transformer = RankTransformer(feature_names=feature_cols)
result_pd = transformer.fit_transform(df_pd[feature_cols], date_series=df_pd['date'])
```

**3. With Polars:**
```python
import polars as pl

df_pl = pl.DataFrame(data)
# The same transformer instance can be used, or a new one initialized
result_pl = transformer.fit_transform(df_pl[feature_cols], date_series=df_pl['date'])
```

**Expected Output:**

Both `result_pd` (from Pandas) and `result_pl` (from Polars) will contain the same transformed data.
For the example data, the output would be:
```
   feature1_rank  feature2_rank
0            1.0            1.0  # Corresponds to original (feature1=3, feature2=30) on 2021-01-01
1            0.5            0.5  # Corresponds to original (feature1=1, feature2=20) on 2021-01-01
2            1.0            1.0  # Corresponds to original (feature1=2, feature2=10) on 2021-01-02
```
This transformer calculates the normalized rank of features within each date group. By default, higher original values receive higher ranks (e.g., a rank of 1.0 is "higher" or later in sort order than 0.5 when scaled by group size).

## Pipeline with Metadata Routing

`centimators` transformers are designed to work seamlessly within scikit-learn Pipelines, leveraging its metadata routing capabilities. This allows you to pass data like date or ticker series through the pipeline to the specific transformers that need them, while also chaining together multiple transformers. This is useful for building more complex feature pipelines, but also allows for better cross-validation, hyperparameter tuning, and model selection. For example, if you add a Regressor at the end of the pipeline, you can imagine searching over various combinations of lags, moving average windows, and model hyperparameters during the training process.

Here's an example using `RankTransformer` and `LagTransformer`:

```python
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from centimators.feature_transformers import RankTransformer, LagTransformer

# 1. Enable metadata routing globally (once per session)
set_config(enable_metadata_routing=True)

# 2. Sample Data (Pandas OR Polars DataFrame)
data = {
    'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-03']),
    'ticker': ['A', 'B', 'A', 'B', 'A', 'B'],
    'price': [10, 20, 11, 21, 12, 22],
    'volume': [100, 200, 110, 210, 120, 220]
}
df = pd.DataFrame(data)

X = df[['price', 'volume']]
dates = df['date']
tickers = df['ticker']

# 3. Instantiate transformers and request metadata
# RankTransformer needs 'date_series'
rank_transformer = RankTransformer().set_transform_request(date_series=True)

# LagTransformer needs 'ticker_series'
lag_transformer = LagTransformer(windows=[0, 1, 2, 3, 4]).set_transform_request(ticker_series=True)

# 4. Create the pipeline
pipeline = make_pipeline(
    rank_transformer, 
    lag_transformer
)

# 5. Fit and transform
# The metadata (dates, tickers) is passed to fit_transform
transformed_data = pipeline.fit_transform(X, date_series=dates, ticker_series=tickers)
```

**Explanation:**

- `set_config(enable_metadata_routing=True)` turns on scikit-learn's metadata routing.
- `set_transform_request(metadata_name=True)` on each transformer tells the pipeline that this transformer expects `metadata_name` (e.g., `date_series`).
- When `pipeline.fit_transform(X, date_series=dates, ticker_series=tickers)` is called:
    - The `date_series` is automatically passed to `RankTransformer`.
    - The `ticker_series` is automatically passed to `LagTransformer`.
    - The output of `RankTransformer` (ranked features) becomes the input to `LagTransformer`.
- The `LagTransformer` then computes lagged values for the ranked features.

This allows for complex data transformations where different steps require different auxiliary information, all managed cleanly by the pipeline.
