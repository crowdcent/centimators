"""Tree-based neural network estimators.

Implements differentiable decision trees and forests using stochastic routing.
Based on Neural Decision Forests approach where routing probabilities are learned
through backpropagation. See https://keras.io/examples/structured_data/deep_neural_decision_forests/
for original implementation and more details.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import RegressorMixin

from .base import BaseKerasEstimator
from keras import layers, models, ops as K


class NeuralDecisionTree(models.Model):
    """A differentiable decision tree with stochastic routing.

    This implements a single decision tree where routing decisions are learned
    through gradient descent. At each internal node, the model learns a probability
    distribution over left/right routing decisions. The final prediction is a
    weighted combination of leaf node outputs based on the routing probabilities.

    Parameters
    ----------
    depth : int
        Depth of the tree. A tree of depth d has 2^d leaf nodes.
    num_features : int
        Number of input features.
    used_features_rate : float
        Fraction of features to randomly select and use for this tree (0 to 1).
        Provides feature bagging similar to random forests.
    output_units : int, default=1
        Number of output units (targets to predict).

    Attributes
    ----------
    num_leaves : int
        Number of leaf nodes = 2^depth
    used_features_mask : Tensor
        Binary mask indicating which features this tree uses
    pi : Tensor
        Learned output values for each leaf node, shape (num_leaves, output_units)
    decision_fn : Dense layer
        Learns routing probabilities for all internal nodes

    Notes
    -----
    The tree traversal uses breadth-first order. At each level, routing probabilities
    are computed and multiplied to give the final probability of reaching each leaf.
    """

    def __init__(self, depth, num_features, used_features_rate, output_units=1):
        super().__init__()
        self.depth = depth
        self.num_leaves = 2**depth
        self.output_units = output_units

        # Create a mask for the randomly selected features
        num_used_features = int(num_features * used_features_rate)
        one_hot = np.eye(num_features)
        sampled_feature_indices = np.random.choice(
            np.arange(num_features), num_used_features, replace=False
        )
        self.used_features_mask = K.convert_to_tensor(
            one_hot[sampled_feature_indices], dtype="float32"
        )

        # Initialize the weights of the outputs in leaves
        self.pi = self.add_weight(
            initializer="random_normal",
            shape=[self.num_leaves, self.output_units],
            dtype="float32",
            trainable=True,
        )

        # Initialize the stochastic routing layer
        self.decision_fn = layers.Dense(
            units=self.num_leaves, activation="sigmoid", name="decision"
        )

    def call(self, features):
        batch_size = K.shape(features)[0]

        # Apply the feature mask to the input features
        features = K.matmul(
            features, K.transpose(self.used_features_mask)
        )  # [batch_size, num_used_features]

        # Compute the routing probabilities
        decisions = K.expand_dims(
            self.decision_fn(features), axis=2
        )  # [batch_size, num_leaves, 1]

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

    Parameters
    ----------
    num_trees : int, default=10
        Number of decision trees in the forest ensemble.
    depth : int, default=5
        Depth of each tree. Each tree will have 2^depth leaf nodes.
    used_features_rate : float, default=0.5
        Fraction of features each tree randomly selects (0 to 1).
        Provides feature bagging. Lower values increase diversity.
    output_units : int, default=1
        Number of output targets to predict.
    optimizer : Type[keras.optimizers.Optimizer], default=Adam
        Keras optimizer class to use for training.
    learning_rate : float, default=0.001
        Learning rate for the optimizer.
    loss_function : str, default="mse"
        Loss function for training.
    metrics : list[str] | None, default=None
        List of metrics to track during training.
    distribution_strategy : str | None, default=None
        Distribution strategy for multi-device training.

    Attributes
    ----------
    model : keras.Model
        The compiled Keras model containing the ensemble of trees.

    Examples
    --------
    >>> from centimators.model_estimators import NeuralDecisionForestRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 10).astype('float32')
    >>> y = np.random.randn(100, 1).astype('float32')
    >>> ndf = NeuralDecisionForestRegressor(num_trees=5, depth=4)
    >>> ndf.fit(X, y, epochs=10, verbose=0)
    >>> predictions = ndf.predict(X)

    Notes
    -----
    - Larger depth increases model capacity but may lead to overfitting
    - More trees generally improve performance but increase computation
    - Lower used_features_rate increases diversity but may hurt individual tree performance
    - Works well on tabular data where tree-based methods traditionally excel

    References
    ----------
    The approach is based on Neural Decision Forests and related differentiable
    tree architectures that enable end-to-end learning of routing decisions.
    """

    num_trees: int = 10
    depth: int = 5
    used_features_rate: float = 0.5
    metrics: list[str] | None = field(default_factory=lambda: ["mse"])

    def build_model(self):
        """Build the neural decision forest model.

        Creates an ensemble of NeuralDecisionTree models with shared input
        and averaged output. Each tree receives normalized input features
        via BatchNormalization.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        if self.model is None:
            if self.distribution_strategy:
                self._setup_distribution_strategy()

            # Input layer
            inputs = layers.Input(shape=(self._n_features_in_,))
            features = layers.BatchNormalization()(inputs)

            # Create ensemble of trees
            trees = []
            for _ in range(self.num_trees):
                tree = NeuralDecisionTree(
                    depth=self.depth,
                    num_features=self._n_features_in_,
                    used_features_rate=self.used_features_rate,
                    output_units=self.output_units,
                )
                trees.append(tree)

            # Aggregate predictions from all trees
            tree_outputs = [tree(features) for tree in trees]
            if len(tree_outputs) > 1:
                outputs = K.mean(K.stack(tree_outputs, axis=0), axis=0)
            else:
                outputs = tree_outputs[0]

            self.model = models.Model(inputs=inputs, outputs=outputs)
            opt = self.optimizer(learning_rate=self.learning_rate)
            self.model.compile(
                optimizer=opt, loss=self.loss_function, metrics=self.metrics
            )
        return self
