"""
Model estimator abstractions that combine Keras with the scikit-learn API.

This module exposes two estimator classes that conform to scikit-learn's
`BaseEstimator`/`TransformerMixin` contracts while delegating all heavy‐
lifting to **Keras**.  The goal is to let neural networks participate in
classic ML pipelines without boilerplate.

Highlights:
    * **Drop-in compatibility** – works with `sklearn.pipeline.Pipeline`,
      `GridSearchCV`, etc.
    * **Distribution strategies** – opt-in data-parallel training across
      multiple devices/GPUs.
    * **Sequence support** – :class:`SequenceEstimator` reshapes a flattened
      lag matrix into the 3-D tensor expected by recurrent or convolutional
      sequence layers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Type

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from keras import optimizers
from keras import distribution
from keras import ops
import narwhals as nw
from narwhals.typing import IntoFrame
from keras import layers, models
import numpy


def _ensure_numpy(data, allow_series: bool = False):
    """Convert data to numpy array, handling both numpy arrays and dataframes.

    Args:
        data: Input data (numpy array, dataframe, or series)
        allow_series: Whether to allow series inputs

    Returns:
        numpy.ndarray: Data converted to numpy array
    """
    # If already a numpy array, return as-is
    if isinstance(data, numpy.ndarray):
        return data

    # If it's a dataframe/series, use narwhals
    try:
        return nw.from_native(data, allow_series=allow_series).to_numpy()
    except Exception:
        # Fallback: try to convert directly to numpy
        return numpy.asarray(data)


@dataclass(kw_only=True)
class BaseKerasEstimator(TransformerMixin, BaseEstimator, ABC):
    """Meta-estimator for Keras models following the scikit-learn API.

    Args:
        output_units (int, default=1): Dimensionality of the model output.
            It is forwarded to :meth:`build_model` and can be used there when
            constructing the final layer.
        optimizer (Type[optimizers.Optimizer], default=keras.optimizers.Adam):
            Optimiser class **not instance**. The class is instantiated in
            :meth:`fit` with the requested ``learning_rate``.
        learning_rate (float, default=1e-3): Learning-rate passed to the
            optimiser constructor.
        loss_function (str or keras.losses.Loss, default="mse"): Loss
            forwarded to ``model.compile``.
        metrics (list[str] | None, default=None): List of metrics forwarded
            to ``model.compile``.
        model (keras.Model | None, default=None): Internal Keras model instance.
            If *None* it is lazily built on the first call to :meth:`fit`.
        distribution_strategy (str | None, default=None): Name of a Keras
            distribution strategy to activate before training. At the moment
            only ``"DataParallel"`` is recognised.

    Attributes:
        _n_features_in_ (int | None): Inferred number of features from the data
            passed to :meth:`fit`.

    Notes:
        Sub-classes **must** implement :meth:`build_model` which should return
        a compiled (or at least constructed) ``keras.Model`` instance.
    """

    output_units: int = 1
    optimizer: Type[optimizers.Optimizer] = optimizers.Adam
    learning_rate: float = 0.001
    loss_function: str = "mse"
    metrics: list[str] | None = None
    model: Any = None
    distribution_strategy: str | None = None

    @abstractmethod
    def build_model(self):
        pass

    def _setup_distribution_strategy(self) -> None:
        """Activate a distribution strategy for multi-device training.

        The current implementation always uses
        ``keras.distribution.DataParallel`` which mirrors the model on all
        available devices and splits the batch.  Support for additional
        strategies can be added later.
        """
        # TODO: allow for different distribution strategies
        strategy = distribution.DataParallel()
        distribution.set_distribution(strategy)

    def fit(
        self,
        X,
        y,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: tuple[Any, Any] | None = None,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> "BaseKerasEstimator":
        """Fit the underlying Keras model.

        The model is **lazily** built and compiled on the first call. All
        extra keyword arguments are forwarded to ``keras.Model.fit``.

        Args:
            X (array-like): Training data of shape (n_samples, n_features).
            y (array-like): Training targets of shape (n_samples,) or (n_samples, n_outputs).
            epochs (int, default=100): Number of training epochs.
            batch_size (int, default=32): Minibatch size.
            validation_data (tuple[Any, Any] | None, default=None): Optional
                validation split forwarded to Keras.
            callbacks (list[Any] | None, default=None): Optional list of callbacks.
            **kwargs: Additional keyword arguments forwarded to ``keras.Model.fit``.

        Returns:
            BaseKerasEstimator: Fitted estimator.
        """
        self._n_features_in_ = X.shape[1]

        if self.distribution_strategy:
            self._setup_distribution_strategy()

        if not self.model:
            self.build_model()

        self.model.fit(
            _ensure_numpy(X),
            y=_ensure_numpy(y, allow_series=True),
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            **kwargs,
        )
        self._is_fitted = True
        return self

    def predict(self, X, batch_size: int = 512, **kwargs: Any) -> Any:
        """Generate predictions with the trained model.

        Args:
            X (array-like): Input samples of shape (n_samples, n_features).
            batch_size (int, default=512): Batch size used for inference.
            **kwargs: Additional keyword arguments forwarded to ``keras.Model.predict``.

        Returns:
            Any: Model predictions of shape (n_samples, output_units)
                in the same order as *X*.
        """
        if not self.model:
            raise ValueError("Model not built. Call `build_model` first.")

        return self.model.predict(X, batch_size=batch_size, **kwargs)

    def transform(self, X, **kwargs):
        """Alias for :meth:`predict` to comply with scikit-learn pipelines."""
        return self.predict(X, **kwargs)

    def __sklearn_is_fitted__(self) -> bool:
        """Return ``True`` when the estimator has been fitted.

        scikit-learn relies on :func:`sklearn.utils.validation.check_is_fitted`
        to decide whether an estimator is ready for inference.
        """
        return getattr(self, "_is_fitted", False)


@dataclass(kw_only=True)
class SequenceEstimator(BaseKerasEstimator):
    """Estimator for models that consume sequential data.

    The class assumes that *X* is a **flattened** 2-D representation of a
    sequence built from multiple lagged views of the original signal.
    The shape transformation performed by :meth:`_reshape` is visualised
    below for a concrete example.

    Args:
        lag_windows (list[int]): Offsets (in number of timesteps) that have been
            concatenated to form the flattened design matrix.
        n_features_per_timestep (int): Number of *original* features per timestep
            **before** creating the lags.

    Attributes:
        seq_length (int): Inferred sequence length from lag_windows.
    """

    lag_windows: list[int]
    n_features_per_timestep: int

    def __post_init__(self):
        self.seq_length = len(self.lag_windows)

    def _reshape(self, X: IntoFrame, validation_data: tuple[Any, Any] | None = None):
        """Reshape a flattened lag matrix into a 3-D tensor.

        Args:
            X (IntoFrame): Design matrix containing the lagged features.
            validation_data (tuple[Any, Any] | None, default=None): Optional
                validation split; its *X* part will be reshaped in the same way.

        Returns:
            tuple[numpy.ndarray, tuple[Any, Any] | None]:
                A tuple containing the reshaped training data (numpy.ndarray with shape
                ``(n_samples, seq_length, n_features_per_timestep)``) and the
                (potentially reshaped) validation data.
        """
        X = _ensure_numpy(X)
        X_reshaped = ops.reshape(
            X, (X.shape[0], self.seq_length, self.n_features_per_timestep)
        )

        if validation_data:
            X_val, y_val = validation_data
            X_val = _ensure_numpy(X_val)
            X_val_reshaped = ops.reshape(
                X_val,
                (X_val.shape[0], self.seq_length, self.n_features_per_timestep),
            )
            validation_data = X_val_reshaped, _ensure_numpy(y_val)

        return X_reshaped, validation_data

    def fit(
        self, X, y, validation_data: tuple[Any, Any] | None = None, **kwargs: Any
    ) -> "SequenceEstimator":
        """Redefines :meth:`BaseKerasEstimator.fit`
        to include reshaping for sequence data.

        Args:
            X (array-like): Training data.
            y (array-like): Training targets.
            validation_data (tuple[Any, Any] | None, default=None): Optional
                validation split.
            **kwargs: Additional keyword arguments passed to the parent fit method.

        Returns:
            SequenceEstimator: Fitted estimator.
        """
        X_reshaped, validation_data_reshaped = self._reshape(X, validation_data)
        super().fit(
            X_reshaped,
            y=_ensure_numpy(y),
            validation_data=validation_data_reshaped,
            **kwargs,
        )
        return self

    def predict(self, X, **kwargs: Any) -> numpy.ndarray:
        """Redefines :meth:`BaseKerasEstimator.predict`
        to include reshaping for sequence data.

        Args:
            X (array-like): Input data.
            **kwargs: Additional keyword arguments passed to the parent predict method.

        Returns:
            numpy.ndarray: Predictions of shape (n_samples, output_units).
        """
        X_reshaped, _ = self._reshape(X)
        return super().predict(X_reshaped, **kwargs)


@dataclass(kw_only=True)
class MLPRegressor(RegressorMixin, BaseKerasEstimator):
    """A minimal fully-connected multi-layer perceptron for tabular data.

    The class follows the scikit-learn *estimator* interface while delegating
    the heavy lifting to Keras.  It is intended as a sensible baseline model
    that works *out of the box* with classic ML workflows such as pipelines or
    cross-validation.

    Args:
        hidden_units (tuple[int, ...], default=(64, 64)): Width (number of
            neurons) for each hidden layer.  The length of the tuple defines
            the depth of the network.
        activation (str, default="relu"): Activation function applied after
            each hidden ``Dense`` layer.
        dropout_rate (float, default=0.0): Optional dropout applied **after**
            each hidden layer.  Set to *0* to disable dropout entirely.
        output_units (int, default=1): Copied from :class:`BaseKerasEstimator`.
            Defines the dimensionality of the final layer.

    Attributes:
        _n_features_in_ (int | None): Inferred number of features from the data
            passed to :meth:`fit`.
    """

    hidden_units: tuple[int, ...] = (64, 64)
    activation: str = "relu"
    dropout_rate: float = 0.0
    metrics: list[str] | None = field(default_factory=lambda: ["mse"])

    def build_model(self):
        """Construct a simple MLP with the configured hyper-parameters."""
        inputs = layers.Input(shape=(self._n_features_in_,), name="features")
        x = inputs
        for units in self.hidden_units:
            x = layers.Dense(units, activation=self.activation)(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.output_units, activation="linear")(x)
        self.model = models.Model(inputs=inputs, outputs=outputs, name="mlp_regressor")

        self.model.compile(
            optimizer=self.optimizer(learning_rate=self.learning_rate),
            loss=self.loss_function,
            metrics=self.metrics,
        )

        return self


@dataclass(kw_only=True)
class BottleneckEncoder(BaseKerasEstimator):
    """A bottleneck autoencoder that can learn latent representations and predict targets.

    This estimator implements a bottleneck autoencoder architecture that:
    1. Encodes input features to a lower-dimensional latent space
    2. Decodes the latent representation back to reconstruct the input
    3. Uses an additional MLP branch to predict targets from the decoded features

    The model can be used both as a regressor (via predict) and as a transformer
    (via transform) to get latent space representations for dimensionality reduction.

    Args:
        gaussian_noise (float, default=0.035): Standard deviation of Gaussian noise
            applied to inputs for regularization.
        encoder_units (list[tuple[int, float]], default=[(1024, 0.1)]): List of
            (units, dropout_rate) tuples defining the encoder architecture.
        latent_units (tuple[int, float], default=(256, 0.1)): Tuple of
            (units, dropout_rate) for the latent bottleneck layer.
        ae_units (list[tuple[int, float]], default=[(96, 0.4)]): List of
            (units, dropout_rate) tuples for the autoencoder prediction branch.
        activation (str, default="swish"): Activation function used throughout the network.
        reconstruction_loss_weight (float, default=1.0): Weight for the reconstruction loss.
        target_loss_weight (float, default=1.0): Weight for the target prediction loss.

    Attributes:
        encoder (keras.Model): The encoder submodel for transforming inputs to latent space.
    """

    gaussian_noise: float = 0.035
    encoder_units: list[tuple[int, float]] = field(
        default_factory=lambda: [(1024, 0.1)]
    )
    latent_units: tuple[int, float] = (256, 0.1)
    ae_units: list[tuple[int, float]] = field(default_factory=lambda: [(96, 0.4)])
    activation: str = "swish"
    reconstruction_loss_weight: float = 1.0
    target_loss_weight: float = 1.0
    encoder: Any = None

    def build_model(self):
        """Construct the bottleneck autoencoder architecture."""
        if self._n_features_in_ is None:
            raise ValueError("Must call fit() before building the model")

        # Input layer
        inputs = layers.Input(shape=(self._n_features_in_,), name="features")
        x0 = layers.BatchNormalization()(inputs)

        # Encoder path
        encoder = layers.GaussianNoise(self.gaussian_noise)(x0)
        for units, dropout in self.encoder_units:
            encoder = layers.Dense(units)(encoder)
            encoder = layers.BatchNormalization()(encoder)
            encoder = layers.Activation(self.activation)(encoder)
            encoder = layers.Dropout(dropout)(encoder)

        # Latent bottleneck layer
        latent_units, latent_dropout = self.latent_units
        latent = layers.Dense(latent_units)(encoder)
        latent = layers.BatchNormalization()(latent)
        latent = layers.Activation(self.activation)(latent)
        latent_output = layers.Dropout(latent_dropout)(latent)

        # Create separate encoder model for transform method
        self.encoder = models.Model(
            inputs=inputs, outputs=latent_output, name="encoder"
        )

        # Decoder path (reverse of encoder)
        decoder = latent_output
        for units, dropout in reversed(self.encoder_units):
            decoder = layers.Dense(units)(decoder)
            decoder = layers.BatchNormalization()(decoder)
            decoder = layers.Activation(self.activation)(decoder)
            decoder = layers.Dropout(dropout)(decoder)

        # Reconstruction output
        reconstruction = layers.Dense(self._n_features_in_, name="reconstruction")(
            decoder
        )

        # Target prediction branch from decoded features
        target_pred = reconstruction
        for units, dropout in self.ae_units:
            target_pred = layers.Dense(units)(target_pred)
            target_pred = layers.BatchNormalization()(target_pred)
            target_pred = layers.Activation(self.activation)(target_pred)
            target_pred = layers.Dropout(dropout)(target_pred)

        target_output = layers.Dense(
            self.output_units, activation="linear", name="target_prediction"
        )(target_pred)

        # Create the full model with multiple outputs
        self.model = models.Model(
            inputs=inputs,
            outputs=[reconstruction, target_output],
            name="bottleneck_encoder",
        )

        # Compile with multiple losses
        self.model.compile(
            optimizer=self.optimizer(learning_rate=self.learning_rate),
            loss={"reconstruction": "mse", "target_prediction": self.loss_function},
            loss_weights={
                "reconstruction": self.reconstruction_loss_weight,
                "target_prediction": self.target_loss_weight,
            },
            metrics={"target_prediction": self.metrics or ["mse"]},
        )

        return self

    def fit(
        self,
        X,
        y,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: tuple[Any, Any] | None = None,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> "BottleneckEncoder":
        """Fit the bottleneck autoencoder.

        Args:
            X (array-like): Training data (features).
            y (array-like): Training targets.
            epochs (int, default=100): Number of training epochs.
            batch_size (int, default=32): Minibatch size.
            validation_data (tuple[Any, Any] | None, default=None): Optional
                validation split.
            callbacks (list[Any] | None, default=None): Optional callbacks.
            **kwargs: Additional arguments passed to keras.Model.fit.

        Returns:
            BottleneckEncoder: Fitted estimator.
        """
        # Store input dimension and build model
        self._n_features_in_ = X.shape[1]

        if self.distribution_strategy:
            self._setup_distribution_strategy()

        if not self.model:
            self.build_model()

        # Convert inputs to numpy arrays
        X_np = _ensure_numpy(X)
        y_np = _ensure_numpy(y, allow_series=True)

        # Create target dictionary for multiple outputs
        y_dict = {"reconstruction": X_np, "target_prediction": y_np}

        # Handle validation data
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_np = _ensure_numpy(X_val)
            y_val_np = _ensure_numpy(y_val, allow_series=True)
            validation_data = (
                X_val_np,
                {"reconstruction": X_val_np, "target_prediction": y_val_np},
            )

        # Train the model
        self.model.fit(
            X_np,
            y_dict,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            **kwargs,
        )

        self._is_fitted = True
        return self

    def predict(self, X, batch_size: int = 512, **kwargs: Any) -> Any:
        """Generate target predictions using the fitted model.

        Args:
            X (array-like): Input samples.
            batch_size (int, default=512): Batch size for prediction.
            **kwargs: Additional arguments passed to keras.Model.predict.

        Returns:
            array-like: Target predictions.
        """
        if not self.model:
            raise ValueError("Model not built. Call 'fit' first.")

        X_np = _ensure_numpy(X)
        predictions = self.model.predict(X_np, batch_size=batch_size, **kwargs)

        # Return only the target predictions (second output)
        return predictions[1] if isinstance(predictions, list) else predictions

    def transform(self, X, batch_size: int = 512, **kwargs: Any) -> Any:
        """Transform input data to latent space representation.

        Args:
            X (array-like): Input samples.
            batch_size (int, default=512): Batch size for transformation.
            **kwargs: Additional arguments passed to keras.Model.predict.

        Returns:
            array-like: Latent space representations.
        """
        if not self.encoder:
            raise ValueError("Encoder not built. Call 'fit' first.")

        X_np = _ensure_numpy(X)
        return self.encoder.predict(X_np, batch_size=batch_size, **kwargs)

    def fit_transform(self, X, y, **kwargs) -> Any:
        """Fit the model and return latent space representations.

        Args:
            X (array-like): Training data.
            y (array-like): Training targets.
            **kwargs: Additional arguments passed to fit.

        Returns:
            array-like: Latent space representations of X.
        """
        return self.fit(X, y, **kwargs).transform(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Generate feature names for the latent space output.

        Args:
            input_features (array-like, optional): Ignored. Present for API compatibility.

        Returns:
            list[str]: Feature names for latent dimensions.
        """
        latent_dim = self.latent_units[0]
        return [f"latent_{i}" for i in range(latent_dim)]


@dataclass(kw_only=True)
class LSTMRegressor(RegressorMixin, SequenceEstimator):
    """LSTM-based regressor for time series prediction.

    This estimator uses stacked LSTM layers to model sequential dependencies
    in time series data. It supports bidirectional processing and various
    normalization strategies.

    Args:
        lstm_units (list[tuple[int, float, float]], default=[(64, 0.01, 0.01)]):
            List of tuples defining LSTM layers. Each tuple contains:
            - units: Number of LSTM units
            - dropout_rate: Dropout rate applied to inputs
            - recurrent_dropout_rate: Dropout rate applied to recurrent connections
        use_batch_norm (bool, default=False): Whether to apply batch normalization
            after each LSTM layer.
        use_layer_norm (bool, default=False): Whether to apply layer normalization
            after each LSTM layer.
        bidirectional (bool, default=False): Whether to use bidirectional LSTM layers.
        lag_windows (list[int]): Inherited from SequenceEstimator.
        n_features_per_timestep (int): Inherited from SequenceEstimator.

    Attributes:
        _n_features_in_ (int | None): Inferred number of features from training data.
    """

    lstm_units: list[tuple[int, float, float]] = field(
        default_factory=lambda: [(64, 0.01, 0.01)]
    )
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    bidirectional: bool = False
    metrics: list[str] | None = field(default_factory=lambda: ["mse"])

    def build_model(self):
        """Construct the LSTM architecture."""
        if self._n_features_in_ is None:
            raise ValueError("Must call fit() before building the model")

        # Input layer expecting 3D tensor (batch, timesteps, features)
        inputs = layers.Input(
            shape=(self.seq_length, self.n_features_per_timestep), name="sequence_input"
        )

        x = inputs

        # Stack LSTM layers
        for layer_num, (units, dropout, recurrent_dropout) in enumerate(
            self.lstm_units
        ):
            return_sequences = layer_num < len(self.lstm_units) - 1

            lstm_layer = layers.LSTM(
                units=units,
                activation="tanh",
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                name=f"lstm_{layer_num}",
            )

            # Apply bidirectional wrapper if requested
            if self.bidirectional:
                x = layers.Bidirectional(lstm_layer, name=f"bidirectional_{layer_num}")(
                    x
                )
            else:
                x = lstm_layer(x)

            # Apply normalization layers if requested
            if self.use_layer_norm:
                x = layers.LayerNormalization(name=f"layer_norm_{layer_num}")(x)
            if self.use_batch_norm:
                x = layers.BatchNormalization(name=f"batch_norm_{layer_num}")(x)

        # Output layer
        outputs = layers.Dense(self.output_units, activation="linear", name="output")(x)

        # Create and compile model
        self.model = models.Model(inputs=inputs, outputs=outputs, name="lstm_regressor")

        self.model.compile(
            optimizer=self.optimizer(learning_rate=self.learning_rate),
            loss=self.loss_function,
            metrics=self.metrics,
        )

        return self
