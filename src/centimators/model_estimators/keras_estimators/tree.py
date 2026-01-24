"""Tree-based neural network estimators.

Implements differentiable decision trees and forests using stochastic routing.
Based on Neural Decision Forests approach where routing probabilities are learned
through backpropagation. See https://keras.io/examples/structured_data/deep_neural_decision_forests/
for original implementation and more details.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler

from .base import BaseKerasEstimator
from keras import layers, models, ops as K, regularizers, callbacks, initializers
from keras.saving import register_keras_serializable


class TemperatureAnnealing(callbacks.Callback):
    """Anneal tree routing temperature from soft to sharp over training.

    Starts with high temperature (soft routing, samples flow through many paths)
    and linearly decreases to low temperature (sharp routing, more tree-like).
    This can theoretically help training converge to better solutions.

    Args:
        ndf (NeuralDecisionForestRegressor): The forest instance whose trees will be annealed.
        start (float, default=2.0): Starting temperature (soft routing).
        end (float, default=0.5): Ending temperature (sharp routing).
        epochs (int, default=50): Total epochs over which to anneal. Should match fit() epochs.

    Examples:
        >>> ndf = NeuralDecisionForestRegressor(temperature=2.0)
        >>> annealer = TemperatureAnnealing(ndf, start=2.0, end=0.5, epochs=50)
        >>> ndf.fit(X, y, epochs=50, callbacks=[annealer])
    """

    def __init__(self, ndf, start: float = 2.0, end: float = 0.5, epochs: int = 50):
        super().__init__()
        self.ndf = ndf
        self.start = start
        self.end = end
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        t = self.start - (self.start - self.end) * ((epoch + 1) / self.epochs)
        for tree in self.ndf.trees:
            tree.temperature.assign(t)


@register_keras_serializable(package="centimators")
class NeuralDecisionTree(models.Model):
    """A differentiable decision tree with stochastic routing.

    This implements a single decision tree where routing decisions are learned
    through gradient descent. At each internal node, the model learns a probability
    distribution over left/right routing decisions. The final prediction is a
    weighted combination of leaf node outputs based on the routing probabilities.

    Args:
        depth (int): Depth of the tree. A tree of depth d has 2^d leaf nodes.
        num_features (int): Number of input features.
        used_features_rate (float): Fraction of features to randomly select and use
            for this tree (0 to 1). Provides feature bagging similar to random forests.
        output_units (int, default=1): Number of output units (targets to predict).
        l2_decision (float, default=1e-4): L2 regularization strength for routing
            decision layer. Lower values allow sharper routing decisions.
        l2_leaf (float, default=1e-3): L2 regularization strength for leaf output weights.
        temperature (float, default=0.5): Temperature for sigmoid sharpness.
            Lower = sharper routing (more tree-like), higher = softer routing
            (more like weighted average of leaves).
        rng (np.random.Generator | None, default=None): Random number generator
            for reproducible feature mask sampling.

    Attributes:
        num_leaves (int): Number of leaf nodes = 2^depth
        used_features_mask (Tensor): Binary mask indicating which features this tree uses
        pi (Tensor): Learned output values for each leaf node, shape (num_leaves, output_units)
        decision_fn (Dense layer): Learns routing logits for all internal nodes

    Note:
        The tree traversal uses breadth-first order. At each level, routing probabilities
        are computed and multiplied to give the final probability of reaching each leaf.
    """

    def __init__(
        self,
        depth,
        num_features,
        used_features_rate,
        output_units=1,
        l2_decision=1e-4,
        l2_leaf=1e-3,
        temperature=0.5,
        rng=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Store config params for serialization
        self.depth = depth
        self.num_features = num_features
        self.used_features_rate = used_features_rate
        self.num_leaves = 2**depth
        self.output_units = output_units
        self.l2_decision = l2_decision
        self.l2_leaf = l2_leaf
        self._init_temperature = temperature  # Store initial value for get_config

        # Create a mask for the randomly selected features
        num_used_features = max(1, int(round(num_features * used_features_rate)))
        one_hot = np.eye(num_features)
        if rng is None:
            rng = np.random.default_rng()
        sampled_feature_indices = rng.choice(
            np.arange(num_features), num_used_features, replace=False
        )
        mask_value = one_hot[sampled_feature_indices].astype("float32")

        self.used_features_mask = self.add_weight(
            name="used_features_mask",
            shape=mask_value.shape,
            initializer=initializers.Constant(mask_value),
            trainable=False,
        )

        # Initialize the weights of the outputs in leaves
        self.pi = self.add_weight(
            initializer="random_normal",
            shape=[self.num_leaves, self.output_units],
            dtype="float32",
            trainable=True,
            regularizer=regularizers.l2(l2_leaf) if l2_leaf > 0 else None,
        )

        # Temperature for controlling sigmoid sharpness (non-trainable)
        self.temperature = self.add_weight(
            name="temperature",
            shape=(),
            initializer=initializers.Constant(temperature),
            trainable=False,
        )

        # Initialize the stochastic routing layer (outputs logits, not probabilities)
        self.decision_fn = layers.Dense(
            units=self.num_leaves,
            activation=None,  # Raw logits - sigmoid applied with temperature in call()
            name="decision",
            kernel_regularizer=regularizers.l2(l2_decision)
            if l2_decision > 0
            else None,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "num_features": self.num_features,
                "used_features_rate": self.used_features_rate,
                "output_units": self.output_units,
                "l2_decision": self.l2_decision,
                "l2_leaf": self.l2_leaf,
                "temperature": self._init_temperature,
            }
        )
        return config

    def call(self, features):
        batch_size = K.shape(features)[0]

        # Apply the feature mask to the input features
        features = K.matmul(
            features, K.transpose(self.used_features_mask)
        )  # [batch_size, num_used_features]

        # Compute routing logits and apply temperature-scaled sigmoid
        logits = self.decision_fn(features)  # [batch_size, num_leaves]
        decisions = K.sigmoid(logits / self.temperature)  # [batch_size, num_leaves]

        decisions = K.expand_dims(decisions, axis=2)  # [batch_size, num_leaves, 1]

        # Concatenate the routing probabilities with their complements
        decisions = layers.Concatenate(axis=2)(
            [decisions, 1 - decisions]
        )  # [batch_size, num_leaves, 2]

        mu = K.ones([batch_size, 1, 1])

        begin_idx = 1
        end_idx = 2
        # Traverse the tree in breadth-first order
        for level in range(self.depth):
            mu = K.reshape(mu, [batch_size, -1, 1])  # [batch_size, 2 ** level, 1]
            mu = K.tile(mu, (1, 1, 2))  # [batch_size, 2 ** level, 2]
            level_decisions = decisions[
                :, begin_idx:end_idx, :
            ]  # [batch_size, 2 ** level, 2]
            mu = mu * level_decisions  # [batch_size, 2**level, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = K.reshape(mu, [batch_size, self.num_leaves])  # [batch_size, num_leaves]
        outputs = K.matmul(mu, self.pi)  # [batch_size, output_units]
        return outputs


@dataclass(kw_only=True)
class NeuralDecisionForestRegressor(RegressorMixin, BaseKerasEstimator):
    """Neural Decision Forest regressor with differentiable tree ensembles.

    A Neural Decision Forest is an ensemble of differentiable decision trees
    trained end-to-end via gradient descent. Each tree uses stochastic routing
    where internal nodes learn probability distributions over routing decisions.
    The forest combines predictions by averaging over all trees.

    This architecture provides:

    - Interpretable tree-like structure with learned routing
    - Feature bagging via used_features_rate (like random forests)
    - End-to-end differentiable training
    - Ensemble averaging for improved generalization
    - Temperature-controlled routing sharpness
    - Input noise, per-tree noise, and tree dropout for ensemble diversity

    Args:
        num_trees (int, default=25): Number of decision trees in the forest ensemble.
        depth (int, default=4): Depth of each tree. Each tree will have 2^depth leaf nodes.
            Deeper trees have more capacity but harder gradient flow.
        used_features_rate (float, default=0.5): Fraction of features each tree randomly
            selects (0 to 1). Provides feature bagging. Lower values increase diversity.
        l2_decision (float, default=1e-4): L2 regularization for routing decision layers.
            Lower values allow sharper routing decisions.
        l2_leaf (float, default=1e-3): L2 regularization for leaf output weights.
            Can be stronger than l2_decision since leaves are regression weights.
        temperature (float, default=0.5): Temperature for sigmoid sharpness in routing.
            Lower values (0.3-0.5) give sharper, more tree-like routing. Higher values
            (1-3) give softer routing where samples flow through multiple paths.
        input_noise_std (float, default=0.0): Gaussian noise std applied to inputs
            before trunk. Makes trunk robust to input perturbations. Try 0.02-0.05.
        tree_noise_std (float, default=0.0): Gaussian noise std applied per-tree after
            trunk. Each tree sees a different noisy view, decorrelating the ensemble.
            Try 0.03-0.1.
        tree_dropout_rate (float, default=0.0): Dropout rate for tree outputs during
            training (0 to 1). Randomly drops tree contributions to decorrelate ensemble.
        trunk_units (list[int] | None, default=None): Hidden layer sizes for optional
            shared MLP trunk before trees. E.g. [64, 64] adds two Dense+ReLU layers.
            Trees then split on learned features instead of raw columns.
        random_state (int | None, default=None): Random seed for reproducible feature
            mask sampling across trees.
        output_units (int, default=1): Number of output targets to predict.
        optimizer (Type[keras.optimizers.Optimizer], default=Adam): Keras optimizer
            class to use for training.
        learning_rate (float, default=0.001): Learning rate for the optimizer.
        loss_function (str, default="mse"): Loss function for training.
        metrics (list[str] | None, default=None): List of metrics to track during training.
        distribution_strategy (str | None, default=None): Distribution strategy for
            multi-device training.

    Attributes:
        model (keras.Model): The compiled Keras model containing the ensemble of trees.
        trees (list[NeuralDecisionTree]): List of tree models in the ensemble.

    Examples:
        >>> from centimators.model_estimators import NeuralDecisionForestRegressor
        >>> import numpy as np
        >>> X = np.random.randn(100, 10).astype('float32')
        >>> y = np.random.randn(100, 1).astype('float32')
        >>> ndf = NeuralDecisionForestRegressor(num_trees=5, depth=4)
        >>> ndf.fit(X, y, epochs=10, verbose=0)
        >>> predictions = ndf.predict(X)

    Note:
        - Larger depth increases model capacity but may lead to overfitting
        - More trees generally improve performance but increase computation
        - Lower used_features_rate increases diversity but may hurt individual tree performance
        - Works well on tabular data where tree-based methods traditionally excel
        - Lower temperature (0.3-0.5) gives sharper, more tree-like routing

        The approach is based on Neural Decision Forests and related differentiable
        tree architectures that enable end-to-end learning of routing decisions.
    """

    num_trees: int = 25
    depth: int = 4
    used_features_rate: float = 0.5
    l2_decision: float = 1e-4
    l2_leaf: float = 1e-3
    temperature: float = 0.5
    input_noise_std: float = 0.0
    tree_noise_std: float = 0.0
    tree_dropout_rate: float = 0.0
    trunk_units: list[int] | None = None
    random_state: int | None = None
    metrics: list[str] | None = field(default_factory=lambda: ["mse"])
    target_scaler: Any = field(default_factory=StandardScaler)

    def __post_init__(self):
        self.trees: list[NeuralDecisionTree] = []

    def build_model(self):
        """Build the neural decision forest model.

        Creates an ensemble of NeuralDecisionTree models with shared input
        and averaged output. Each tree receives normalized input features
        via BatchNormalization. Optionally includes input noise (before trunk
        for robustness), per-tree noise (for diversity), tree dropout, and
        a shared MLP trunk.

        Returns:
            self: Returns self for method chaining.
        """
        if self.model is None:
            if self.distribution_strategy:
                self._setup_distribution_strategy()

            # Set up RNG for reproducibility
            rng = np.random.default_rng(self.random_state)

            # Input layer
            inputs = layers.Input(shape=(self._n_features_in_,))
            x = layers.BatchNormalization()(inputs)

            # Input noise before trunk (makes trunk robust to perturbations)
            if self.input_noise_std > 0:
                x = layers.GaussianNoise(self.input_noise_std)(x)

            # Optional shared trunk (MLP before trees)
            if self.trunk_units:
                for units in self.trunk_units:
                    x = layers.Dense(units, activation="relu")(x)

            # Determine feature count for trees (trunk output or raw features)
            tree_num_features = (
                self.trunk_units[-1] if self.trunk_units else self._n_features_in_
            )

            # Create ensemble of trees
            self.trees = []
            for _ in range(self.num_trees):
                tree = NeuralDecisionTree(
                    depth=self.depth,
                    num_features=tree_num_features,
                    used_features_rate=self.used_features_rate,
                    output_units=self.output_units,
                    l2_decision=self.l2_decision,
                    l2_leaf=self.l2_leaf,
                    temperature=self.temperature,
                    rng=rng,
                )
                self.trees.append(tree)

            # each tree gets its own noisy view for diversity
            tree_outputs = []
            for tree in self.trees:
                if self.tree_noise_std > 0:
                    noisy_x = layers.GaussianNoise(self.tree_noise_std)(x)
                    tree_outputs.append(tree(noisy_x))
                else:
                    tree_outputs.append(tree(x))

            if len(tree_outputs) > 1:
                stacked = K.stack(tree_outputs, axis=1)  # [batch, num_trees, out_units]
                if self.tree_dropout_rate > 0:
                    # Drop entire trees
                    stacked = layers.Dropout(
                        self.tree_dropout_rate,
                        noise_shape=(
                            None,
                            self.num_trees,
                            1,
                        ),  # broadcasts so whole tree is dropped
                    )(stacked)
                outputs = K.mean(stacked, axis=1)
            else:
                outputs = tree_outputs[0]

            self.model = models.Model(inputs=inputs, outputs=outputs)
            opt = self.optimizer(learning_rate=self.learning_rate)
            self.model.compile(
                optimizer=opt, loss=self.loss_function, metrics=self.metrics
            )
        return self
