"""Transformer-based sequence estimator."""

from dataclasses import dataclass, field
from typing import Any

from keras import initializers, layers, models, ops
from keras.saving import register_keras_serializable
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler

from .sequence import SequenceEstimator


@register_keras_serializable(package="centimators")
class PositionEmbedding(layers.Layer):
    """Learned positional embedding with fixed sequence length."""

    def __init__(
        self, sequence_length: int, initializer: str = "glorot_uniform", **kwargs
    ):
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        self.initializer = initializers.get(initializer)

    def build(self, input_shape):
        d_model = int(input_shape[-1])
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.sequence_length, d_model),
            initializer=self.initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # (seq_len, d_model) -> (1, seq_len, d_model) for broadcasting over batch
        return ops.expand_dims(self.position_embedding, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": initializers.serialize(self.initializer),
            }
        )
        return config


@register_keras_serializable(package="centimators")
class CrossAttention(layers.Layer):
    """Dual-axis attention: temporal attention + feature attention."""

    def __init__(
        self, key_dim: int = 32, num_heads: int = 4, dropout: float = 0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.key_dim = int(key_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.temporal_attention = layers.MultiHeadAttention(
            key_dim=self.key_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            attention_axes=(1,),
        )
        self.feature_attention = layers.MultiHeadAttention(
            key_dim=self.key_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            attention_axes=(2,),
        )

    def call(self, inputs):
        temporal_out = self.temporal_attention(inputs, inputs)
        feature_out = self.feature_attention(inputs, inputs)
        return temporal_out + feature_out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
            }
        )
        return config


@register_keras_serializable(package="centimators")
class AttentionPooling(layers.Layer):
    """Learned weighted pooling over the sequence dimension."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = layers.Dense(1)

    def call(self, inputs):
        # inputs: (batch, seq_len, d_model)
        logits = self.score(inputs)  # (batch, seq_len, 1)
        weights = ops.softmax(logits, axis=1)
        weighted = inputs * weights
        return ops.sum(weighted, axis=1)  # (batch, d_model)


@dataclass(kw_only=True)
class TransformerRegressor(RegressorMixin, SequenceEstimator):
    """Transformer encoder regressor for lagged sequence inputs.

    Stacks one or more encoder blocks (multi-head attention + feed-forward)
    over the 3-D tensor produced by :class:`SequenceEstimator`, then collapses
    the sequence dimension via pooling before a final MLP prediction head.

    Three attention modes are available:

    - ``"temporal"`` -- standard self-attention over timesteps (default).
    - ``"feature"`` -- iTransformer-style attention over the feature axis.
    - ``"cross"`` -- dual-axis attention (temporal + feature combined).

    Two pooling strategies collapse the sequence before the MLP head:

    - ``"attention"`` -- learned weighted pooling (:class:`AttentionPooling`).
    - ``"average"`` -- global average pooling.

    Parameters
    ----------
    d_model : int
        Internal embedding dimension (default: 32).
    num_heads : int
        Number of attention heads (default: 4).
    ff_dim : int
        Feed-forward hidden dimension per encoder block (default: 128).
    num_blocks : int
        Number of stacked encoder blocks (default: 1).
    dropout_rate : float
        Dropout applied in attention and feed-forward layers (default: 0.1).
    attention_type : str
        One of ``"temporal"``, ``"feature"``, or ``"cross"`` (default: ``"temporal"``).
    pooling_type : str
        One of ``"attention"`` or ``"average"`` (default: ``"attention"``).
    use_pre_norm : bool
        Apply LayerNorm before (True) or after (False) attention/FFN (default: True).
    mlp_units : tuple[int, ...]
        Hidden layer sizes for the prediction head (default: ``(64,)``).
    """

    d_model: int = 32
    num_heads: int = 4
    ff_dim: int = 128
    num_blocks: int = 1
    dropout_rate: float = 0.1
    attention_type: str = "temporal"
    pooling_type: str = "attention"
    use_pre_norm: bool = True
    mlp_units: tuple[int, ...] = (64,)
    metrics: list[str] | None = field(default_factory=lambda: ["mse"])
    target_scaler: Any = field(default_factory=StandardScaler)

    def _encoder_block(self, inputs):
        x = (
            layers.LayerNormalization(epsilon=1e-6)(inputs)
            if self.use_pre_norm
            else inputs
        )

        if self.attention_type == "cross":
            key_dim = max(1, self.d_model // self.num_heads)
            x = CrossAttention(
                key_dim=key_dim, num_heads=self.num_heads, dropout=self.dropout_rate
            )(x)
        elif self.attention_type == "temporal":
            x = layers.MultiHeadAttention(
                key_dim=max(1, self.d_model // self.num_heads),
                num_heads=self.num_heads,
                dropout=self.dropout_rate,
            )(x, x)
        elif self.attention_type == "feature":
            # iTransformer-style feature attention.
            feature_tokens = layers.Permute((2, 1))(x)
            feature_tokens = layers.MultiHeadAttention(
                key_dim=max(1, self.seq_length // self.num_heads),
                num_heads=self.num_heads,
                dropout=self.dropout_rate,
            )(feature_tokens, feature_tokens)
            x = layers.Permute((2, 1))(feature_tokens)
        else:
            raise ValueError(
                f"Unknown attention_type={self.attention_type!r}. "
                "Use one of {'cross', 'temporal', 'feature'}."
            )

        x = inputs + x
        ffn_input = (
            layers.LayerNormalization(epsilon=1e-6)(x) if self.use_pre_norm else x
        )

        ffn = layers.Dense(self.ff_dim, activation="relu")(ffn_input)
        ffn = layers.Dropout(self.dropout_rate)(ffn)
        ffn = layers.Dense(self.d_model)(ffn)
        ffn = layers.Dropout(self.dropout_rate)(ffn)
        return x + ffn

    def build_model(self):
        if self._n_features_in_ is None:
            raise ValueError("Must call fit() before building the model")

        inputs = layers.Input(
            shape=(self.seq_length, self.n_features_per_timestep),
            name="sequence_input",
        )

        x = layers.Dense(self.d_model)(inputs)
        x = x + PositionEmbedding(sequence_length=self.seq_length)(x)

        for _ in range(self.num_blocks):
            x = self._encoder_block(x)

        if self.use_pre_norm:
            x = layers.LayerNormalization(epsilon=1e-6)(x)

        if self.pooling_type == "attention":
            x = AttentionPooling()(x)
        elif self.pooling_type == "average":
            x = layers.GlobalAveragePooling1D()(x)
        else:
            raise ValueError(
                f"Unknown pooling_type={self.pooling_type!r}. Use one of {'attention', 'average'}."
            )

        for units in self.mlp_units:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(self.dropout_rate)(x)

        outputs = layers.Dense(self.output_units, activation="linear", name="output")(x)
        self.model = models.Model(
            inputs=inputs, outputs=outputs, name="transformer_regressor"
        )
        self.model.compile(
            optimizer=self.optimizer(learning_rate=self.learning_rate),
            loss=self.loss_function,
            metrics=self.metrics,
        )
        return self
