# Model Estimators

Centimators ships with *Keras-backed* model estimators that implement the familiar scikit-learn API.  This means you can train state-of-the-art neural networks while still benefitting from the rich tooling ecosystem around scikit-learn – cross-validation, pipelines, grid-search and more.

## Tabular Models

These models are designed for traditional tabular data where each row represents an independent observation.

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

estimator = MLPRegressor(
    hidden_units=(128, 64), 
    dropout_rate=0.1,
    activation="relu",
    learning_rate=0.001
)
estimator.fit(X, y, epochs=10)

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

### BottleneckEncoder

`centimators.model_estimators.BottleneckEncoder` implements a bottleneck autoencoder that can learn latent representations and predict targets simultaneously. This estimator:

1. Encodes input features to a lower-dimensional latent space
2. Decodes the latent representation back to reconstruct the input  
3. Uses an additional MLP branch to predict targets from the decoded features

The model can be used both as a regressor (via `predict`) and as a transformer (via `transform`) to get latent space representations for dimensionality reduction.

```python
from centimators.model_estimators import BottleneckEncoder

# Create bottleneck autoencoder
encoder = BottleneckEncoder(
    gaussian_noise=0.035,
    encoder_units=[(1024, 0.1)],  # [(units, dropout_rate), ...]
    latent_units=(256, 0.1),       # (units, dropout_rate)
    ae_units=[(96, 0.4)],          # prediction branch architecture
    activation="swish",
    reconstruction_loss_weight=1.0,
    target_loss_weight=1.0
)

# Fit the model (learns reconstruction + target prediction)
encoder.fit(X, y, epochs=10)

# Get target predictions
predictions = encoder.predict(X)

# Get latent space representations for dimensionality reduction
latent_features = encoder.transform(X)
print(f"Latent shape: {latent_features.shape}")  # (1000, 256)
```

## Sequence Models

These models are designed for sequential/time-series data where temporal dependencies matter.

### SequenceEstimator

`SequenceEstimator` is a base class that handles the reshaping of lagged features into the 3-D tensor format required by sequence models like LSTMs and CNNs. It's not meant to be used directly, but rather inherited from by specific sequence model implementations.

Key responsibilities:
- Reshapes flattened lag matrices into (batch, timesteps, features) tensors
- Manages sequence length and feature dimensionality
- Provides common sequence model functionality

### LSTMRegressor

`centimators.model_estimators.LSTMRegressor` provides a ready-to-use LSTM implementation for time series regression. It supports stacked LSTM layers, bidirectional processing, and various normalization strategies.

```python
from centimators.model_estimators import LSTMRegressor
from centimators.feature_transformers import LagTransformer

# Create lagged features
lag_transformer = LagTransformer(windows=[1, 2, 3, 4, 5])
X_lagged = lag_transformer.fit_transform(X, ticker_series=tickers)

# Create LSTM model
lstm = LSTMRegressor(
    lag_windows=[1, 2, 3, 4, 5],       # Must match lag transformer
    n_features_per_timestep=2,          # e.g., price and volume
    lstm_units=[
        (128, 0.2, 0.1),                # (units, dropout, recurrent_dropout)
        (64, 0.1, 0.1),                 # Second LSTM layer
    ],
    bidirectional=True,                 # Use bidirectional LSTMs
    use_layer_norm=True,                # Layer normalization after each LSTM
    use_batch_norm=False,               # Batch normalization (usually not both)
    learning_rate=0.001,
    output_units=1
)

# Fit the model
lstm.fit(X_lagged, y, epochs=50, batch_size=32)

# Make predictions
predictions = lstm.predict(X_lagged)
```


## Loss Functions

Centimators provides custom loss functions, alongside support for standard Keras losses.

### SpearmanCorrelation

`centimators.losses.SpearmanCorrelation` is a differentiable loss function that optimizes for rank correlation rather than absolute error. This is particularly useful for:
- Ranking tasks where relative ordering matters more than exact values
- Financial signals where direction and magnitude are more important than precise predictions
- Robust training in the presence of outliers

```python
from centimators.losses import SpearmanCorrelation
from centimators.model_estimators import MLPRegressor

# Optimize for rank correlation
model = MLPRegressor(
    hidden_units=(128, 64),
    loss_function=SpearmanCorrelation(regularization_strength=1e-3),
    metrics=["mae", "mse"]
)
```

### CombinedLoss

`centimators.losses.CombinedLoss` blends mean squared error with Spearman correlation, allowing you to optimize for both accurate predictions and correct ranking simultaneously:

```python
from centimators.losses import CombinedLoss
from centimators.model_estimators import LSTMRegressor

# Balance between MSE and rank correlation
combined_loss = CombinedLoss(
    mse_weight=2.0,           # Weight for mean squared error
    spearman_weight=1.0,      # Weight for Spearman correlation
    spearman_regularization=1e-3
)

lstm = LSTMRegressor(
    lag_windows=[1, 2, 3, 4, 5],
    n_features_per_timestep=2,
    lstm_units=[(128, 0.2, 0.1)],
    loss_function=combined_loss
)
```

### Standard Keras Losses

All estimators also support standard Keras loss functions:

```python
from keras.losses import Huber

# Using Huber loss for robust regression
model = MLPRegressor(
    hidden_units=(128, 64),
    loss_function=Huber(delta=1.0),  # Or just "huber"
    metrics=["mae", "mse"]
)
```