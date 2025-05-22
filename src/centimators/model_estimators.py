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
            self.model.compile(
                optimizer=self.optimizer(learning_rate=self.learning_rate),
                loss=self.loss_function,
                metrics=self.metrics,
            )

        self.model.fit(
            nw.from_native(X).to_numpy(),
            y=nw.from_native(y, allow_series=True).to_numpy(),
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
        X = nw.from_native(X).to_numpy()
        X_reshaped = ops.reshape(
            X, (X.shape[0], self.seq_length, self.n_features_per_timestep)
        )

        if validation_data:
            X_val, y_val = validation_data
            X_val = nw.from_native(X_val).to_numpy()
            X_val_reshaped = ops.reshape(
                X_val,
                (X_val.shape[0], self.seq_length, self.n_features_per_timestep),
            )
            validation_data = X_val_reshaped, nw.from_native(y_val).to_numpy()

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
            y=nw.from_native(y).to_numpy(),
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
