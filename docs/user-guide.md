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

### KerasCortex

`centimators.keras_cortex.KerasCortex` introduces a novel approach to model development by automating aspects of architecture search. It wraps a Keras-based estimator and leverages a Large Language Model (LLM) to recursively self-reflect on its own architecture. The LLM suggests improvements to the model's source code, which are then tested. This iterative process allows `KerasCortex` to refine its internal model over several cycles, potentially discovering more optimal architectures for the given data.

At its core, `KerasCortex` utilizes DSPy to manage the interaction with the LLM. We use DSPy's `ChainOfThought` to even enable access to the LLM's reasoning process as it improves its own architecture. One could even finetune the prompts or LLM weights directly to improve the quality of the suggestions in its own meta-loop. Access to tools like the keras documentation or arxiv papers could be added as well.

![KerasCortex Reasoning](assets/images/reasoning.png)

While `KerasCortex` is an advanced tool, its scikit-learn compatible API makes it surprisingly easy to integrate into existing workflows.

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # Or "tensorflow", "pytorch"

import polars as pl
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from centimators.model_estimators import MLPRegressor
from centimators.keras_cortex import KerasCortex

# Generate some dummy regression data
X, y = make_regression(
    n_samples=2000,
    n_features=20,
    noise=0.1,
    random_state=42,
)
X = pl.DataFrame(X)
y = pl.Series(y)

# Standard train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a base Keras estimator, can be anything with a `build_model` method
base_mlp = MLPRegressor(
    hidden_units=(64, 32),
    dropout_rate=0.1,
)

# Initialize KerasCortex
# Ensure your LM (e.g., OpenAI API key) is configured in your environment
cortex = KerasCortex(
    base_estimator=base_mlp,
    n_iterations=5,  # Number of self-reflection iterations
    lm="openai/gpt-4o-mini", # Or any other dspy.LM compatible model
    verbose=True
)

# Fit KerasCortex like any scikit-learn estimator
cortex.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val), # Crucial for self-reflection
    epochs=5, # Epochs for each iteration's model training
    batch_size=128,
)

# Make predictions with the best model found
predictions = cortex.predict(X_val)
print(predictions[:5])
```

`KerasCortex` requires validation data to evaluate the performance of different architectures. It uses this information to guide its self-improvement process. The `lm` parameter specifies the language model to be used for code generation, and `n_iterations` controls how many times the model attempts to refine itself.
