import os

# Ensure deterministic backend selection before importing Keras/centimators
os.environ.setdefault("KERAS_BACKEND", "jax")  # use JAX which is lightweight and default in this repo

import numpy as np
import pytest

from centimators.model_estimators import (
    MLPRegressor,
    LSTMRegressor,
    BottleneckEncoder,
)


@pytest.mark.parametrize(
    "n_samples,n_features,hidden_units",
    [
        (20, 4, (8,)),
        (10, 6, (16, 8)),
    ],
)
def test_mlp_regressor_fit_predict(n_samples, n_features, hidden_units):
    """MLPRegressor should fit and return predictions with correct shape."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype("float32")
    y = rng.standard_normal((n_samples, 1)).astype("float32")

    est = MLPRegressor(hidden_units=hidden_units, output_units=1, learning_rate=1e-3)

    # Train for a single epoch to keep the test fast
    est.fit(X, y, epochs=1, batch_size=4, verbose=0)

    preds = est.predict(X, batch_size=4, verbose=0)

    # Shape checks
    assert preds.shape == (n_samples, 1)

    # The estimator should report itself as fitted for sklearn utilities
    assert est.__sklearn_is_fitted__() is True


def test_lstm_regressor_sequence_handling():
    """LSTMRegressor (via SequenceEstimator) must reshape flat lag matrices correctly."""
    rng = np.random.default_rng(0)

    seq_length = 3
    n_features_per_step = 2
    n_samples = 15

    # Flattened design matrix: shape (n_samples, seq_length * n_features_per_step)
    X = rng.standard_normal((n_samples, seq_length * n_features_per_step)).astype("float32")
    y = rng.standard_normal((n_samples, 1)).astype("float32")

    est = LSTMRegressor(
        lstm_units=[(4, 0.0, 0.0)],  # small network for speed
        bidirectional=False,
        lag_windows=list(range(seq_length)),
        n_features_per_timestep=n_features_per_step,
        output_units=1,
        learning_rate=1e-3,
    )

    est.fit(X, y, epochs=1, batch_size=5, verbose=0)

    preds = est.predict(X, batch_size=5, verbose=0)

    # Predicted shape should match (n_samples, output_units)
    assert preds.shape == (n_samples, 1)
    assert est.__sklearn_is_fitted__() is True


def test_bottleneck_encoder_transform_and_predict():
    """BottleneckEncoder should provide both latent representations and target predictions."""
    rng = np.random.default_rng(123)

    n_samples, n_features = 12, 6
    X = rng.standard_normal((n_samples, n_features)).astype("float32")
    y = rng.standard_normal((n_samples, 1)).astype("float32")

    encoder = BottleneckEncoder(
        encoder_units=[(16, 0.0)],  # keep network tiny for speed
        latent_units=(4, 0.0),
        ae_units=[(8, 0.0)],
        output_units=1,
        reconstruction_loss_weight=0.5,
        target_loss_weight=0.5,
        learning_rate=1e-3,
    )

    encoder.fit(X, y, epochs=1, batch_size=4, verbose=0)

    preds = encoder.predict(X, batch_size=4, verbose=0)
    latent = encoder.transform(X, batch_size=4, verbose=0)

    # Basic shape assertions
    assert preds.shape == (n_samples, 1)
    assert latent.shape == (n_samples, 4)
    assert encoder.__sklearn_is_fitted__() is True


# Test different input types (pandas, polars, numpy)
def test_mlp_regressor_with_different_input_types():
    """MLPRegressor should handle pandas, polars, and numpy inputs."""
    import pandas as pd
    import polars as pl
    
    rng = np.random.default_rng(42)
    n_samples, n_features = 20, 4
    
    # Create data in different formats
    X_numpy = rng.standard_normal((n_samples, n_features)).astype("float32")
    y_numpy = rng.standard_normal((n_samples,)).astype("float32")
    
    X_pandas = pd.DataFrame(X_numpy, columns=[f"feat_{i}" for i in range(n_features)])
    y_pandas = pd.Series(y_numpy, name="target")
    
    X_polars = pl.DataFrame(X_numpy, schema=[f"feat_{i}" for i in range(n_features)])
    y_polars = pl.Series("target", y_numpy)
    
    # Test with each input type
    for X, y, input_type in [
        (X_numpy, y_numpy, "numpy"),
        (X_pandas, y_pandas, "pandas"),
        (X_polars, y_polars, "polars"),
    ]:
        est = MLPRegressor(hidden_units=(8,), output_units=1, learning_rate=1e-3)
        
        # Fit should work with all types
        est.fit(X, y, epochs=1, batch_size=4, verbose=0)
        
        # Predict should also work with all types
        preds = est.predict(X, batch_size=4, verbose=0)
        assert preds.shape == (n_samples, 1), f"Failed for {input_type} input"


def test_lstm_regressor_with_dataframes():
    """LSTMRegressor should handle dataframe inputs for both fit and predict."""
    import pandas as pd
    import polars as pl
    
    rng = np.random.default_rng(0)
    seq_length = 3
    n_features_per_step = 2
    n_samples = 10
    
    # Create flattened lag matrix
    X_numpy = rng.standard_normal((n_samples, seq_length * n_features_per_step)).astype("float32")
    y_numpy = rng.standard_normal((n_samples,)).astype("float32")
    
    # Convert to pandas
    col_names = [f"lag_{i}_feat_{j}" for i in range(seq_length) for j in range(n_features_per_step)]
    X_pandas = pd.DataFrame(X_numpy, columns=col_names)
    y_pandas = pd.Series(y_numpy, name="target")
    
    # Convert to polars
    X_polars = pl.DataFrame(X_numpy, schema=col_names)
    y_polars = pl.Series("target", y_numpy)
    
    for X, y, input_type in [
        (X_pandas, y_pandas, "pandas"),
        (X_polars, y_polars, "polars"),
    ]:
        est = LSTMRegressor(
            lstm_units=[(4, 0.0, 0.0)],
            lag_windows=list(range(seq_length)),
            n_features_per_timestep=n_features_per_step,
            output_units=1,
            learning_rate=1e-3,
        )
        
        # Both fit and predict should handle dataframes
        est.fit(X, y, epochs=1, batch_size=4, verbose=0)
        preds = est.predict(X, batch_size=4, verbose=0)
        
        assert preds.shape == (n_samples, 1), f"Failed for {input_type} input"


def test_sklearn_pipeline_compatibility():
    """Model estimators should work within sklearn pipelines."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    rng = np.random.default_rng(42)
    n_samples, n_features = 30, 5
    X = rng.standard_normal((n_samples, n_features)).astype("float32")
    y = rng.standard_normal((n_samples,)).astype("float32")
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(hidden_units=(10,), output_units=1))
    ])
    
    # Fit and predict through pipeline (epochs passed via fit params)
    pipeline.fit(X, y, model__epochs=1, model__verbose=0)
    preds = pipeline.predict(X)
    
    assert preds.shape == (n_samples, 1)
    
    # Test with pandas input to pipeline
    import pandas as pd
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    
    # This should also work
    preds_df = pipeline.predict(X_df)
    assert preds_df.shape == (n_samples, 1)


def test_validation_data_handling():
    """Test that validation data is properly converted for different input types."""
    import pandas as pd
    
    rng = np.random.default_rng(42)
    n_train, n_val, n_features = 20, 10, 4
    
    # Create training data
    X_train = rng.standard_normal((n_train, n_features)).astype("float32")
    y_train = rng.standard_normal((n_train,)).astype("float32")
    
    # Create validation data as pandas (mixed types scenario)
    X_val_np = rng.standard_normal((n_val, n_features)).astype("float32")
    y_val_np = rng.standard_normal((n_val,)).astype("float32")
    
    X_val_df = pd.DataFrame(X_val_np, columns=[f"feat_{i}" for i in range(n_features)])
    y_val_series = pd.Series(y_val_np, name="target")
    
    est = MLPRegressor(hidden_units=(8,), output_units=1)
    
    # Should handle mixed input types for train/validation
    est.fit(
        X_train, y_train,
        validation_data=(X_val_df, y_val_series),
        epochs=1,
        batch_size=4,
        verbose=0
    )
    
    # Predict on both numpy and dataframe
    preds_np = est.predict(X_train)
    preds_df = est.predict(X_val_df)
    
    assert preds_np.shape == (n_train, 1)
    assert preds_df.shape == (n_val, 1)


def test_bottleneck_encoder_fit_transform():
    """Test BottleneckEncoder's fit_transform method."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 15, 8
    
    X = rng.standard_normal((n_samples, n_features)).astype("float32")
    y = rng.standard_normal((n_samples,)).astype("float32")
    
    encoder = BottleneckEncoder(
        encoder_units=[(16, 0.0)],
        latent_units=(4, 0.0),
        ae_units=[(8, 0.0)],
        output_units=1,
        learning_rate=1e-3,
    )
    
    # fit_transform should return latent representations
    latent = encoder.fit_transform(X, y, epochs=1, batch_size=4, verbose=0)
    
    assert latent.shape == (n_samples, 4)
    assert encoder.__sklearn_is_fitted__() is True
    
    # get_feature_names_out should work
    feature_names = encoder.get_feature_names_out()
    assert len(feature_names) == 4
    assert all(name.startswith("latent_") for name in feature_names)


def test_error_handling():
    """Test proper error messages for common mistakes."""
    est = MLPRegressor()
    
    # Should raise error when predicting before fitting
    with pytest.raises(ValueError, match="Model not built"):
        est.predict(np.array([[1, 2, 3]]))
    
    # BottleneckEncoder specific errors
    encoder = BottleneckEncoder()
    with pytest.raises(ValueError, match="Encoder not built"):
        encoder.transform(np.array([[1, 2, 3]]))


def test_base_keras_estimator_validation_data_bug():
    """Test that BaseKerasEstimator properly converts validation data.
    
    This is a regression test for a potential bug where validation_data
    wasn't being converted to numpy in the base class.
    """
    import pandas as pd
    
    rng = np.random.default_rng(42)
    n_train, n_val, n_features = 20, 10, 4
    
    # Training data as numpy
    X_train = rng.standard_normal((n_train, n_features)).astype("float32")
    y_train = rng.standard_normal((n_train,)).astype("float32")
    
    # Validation data as pandas DataFrame/Series
    X_val = pd.DataFrame(
        rng.standard_normal((n_val, n_features)).astype("float32"),
        columns=[f"feat_{i}" for i in range(n_features)]
    )
    y_val = pd.Series(
        rng.standard_normal((n_val,)).astype("float32"),
        name="target"
    )
    
    # This should work without errors
    est = MLPRegressor(hidden_units=(8,), output_units=1)
    
    # The key test: BaseKerasEstimator.fit should handle DataFrame validation data
    # even though it doesn't explicitly convert validation_data
    try:
        est.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1,
            batch_size=4,
            verbose=0
        )
        # If we get here, Keras 3 is handling the DataFrame conversion
        print("Note: Keras 3 appears to handle DataFrame inputs natively")
    except Exception as e:
        # If this fails, there's a bug in BaseKerasEstimator
        pytest.fail(f"BaseKerasEstimator failed to handle DataFrame validation data: {e}")
