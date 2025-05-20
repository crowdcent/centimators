# User Guide

## Model Estimators

Centimators ships with *Keras-backed* model estimators that implement the familiar scikit-learn API.  This means you can train state-of-the-art neural networks while still benefitting from the rich tooling ecosystem around scikit-learn – cross-validation, pipelines, grid-search and more.

### MLPRegressor

`centimators.model_estimators.MLPRegressor` is a minimal, fully-connected **multilayer perceptron** that works out-of-the-box for any tabular regression task.

```python
import numpy as np
import polars as pl
from centimators.model_estimators import MLPRegressor

# Dummy data: 1 000 samples × 20 features
grng = np.random.default_rng(seed=42)
X = pl.DataFrame(grng.standard_normal((1000, 20)))
y = pl.Series(grng.standard_normal(1000))

estimator = MLPRegressor(hidden_units=(128, 64), dropout_rate=0.1, epochs=10)
estimator.fit(X, y)

predictions = estimator.predict(X)
print(predictions[:5])
```

Because the estimator inherits from scikit-learn's `BaseEstimator`, you can seamlessly compose it with the feature transformers provided elsewhere in the library:

```python
from sklearn.pipeline import make_pipeline
from centimators.feature_transformers import RankTransformer

pipeline = make_pipeline(
    RankTransformer(feature_names=X.columns),
    MLPRegressor(hidden_units=(128, 64), epochs=10),
)

pipeline.fit(X, y)
```

### SequenceEstimator

If your dataset already contains *lagged* features arranged in a flattened matrix, `SequenceEstimator` takes care of reshaping that matrix into the 3-D tensor expected by recurrent or convolutional sequence models.  You only need to specify the `lag_windows` you used during feature engineering.

```python
from centimators.model_estimators import SequenceEstimator

seq_model = SequenceEstimator(
    lag_windows=[0, 1, 2, 3, 4],         # 5 time-steps
    n_features_per_timestep=10,          # original feature dimensionality
    output_units=1,
)
```

### Where to go next

For a complete reference of all public classes and their hyper-parameters, check the *API Reference ➜ Model Estimators* section in the navigation sidebar.