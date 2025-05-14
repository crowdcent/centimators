<img src="/assets/centimators_banner_transparent_thinner.png" alt="Centimators" width="100%" style="max-width: 800px;"/>

# Centimators: essential data transformers and model estimators for ML and data science competitions

Centimators is an open-source python library built on scikit-learn, keras, and narwhals: designed for building and sharing dataframe-agnostic (pandas/polars), multi-framework (jax/tf/pytorch), sklearn-style (fit/transform/predict) transformers, meta-estimators, and machine learning models for data science competitions like Numerai, Kaggle, and the CrowdCent Challenge. 

Centimators makes heavy use of advanced scikit-learn concepts such as metadata routing. Familiarity with these concepts is recommended for optimal use of the library. You can learn more about metadata routing in the [scikit-learn documentation](https://scikit-learn.org/stable/metadata_routing.html).

## Installation

Recommended (using uv):
```bash
uv add centimators
```

Or, using pip:
```bash
pip install centimators
```

## Quick Start

Here's a quick example of how to use the `RankTransformer` with a pandas DataFrame:

```python
import pandas as pd
from centimators.data_transformers import RankTransformer

# Sample DataFrame
df = pd.DataFrame({
    'date': ['2021-01-01', '2021-01-01', '2021-01-02', '2021-01-02', '2021-01-03'],
    'feature1': [3, 1, 2, 5, 4],
    'feature2': [30, 20, 10, 50, 40]
})

# Initialize and fit the transformer
transformer = RankTransformer(feature_names=['feature1', 'feature2'])
result = transformer.fit_transform(df[['feature1', 'feature2']], date_series=df['date'])

print(result)
```

This will output:

```
   feature1_rank  feature2_rank
0            1.0             1.0
1            0.5             0.5
2            0.5             0.5
3            1.0             1.0
4            1.0             1.0
```

This transformer calculates the normalized rank of `feature1` and `feature2` for each date.
