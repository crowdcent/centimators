"""
Custom loss functions for neural network training.

This module provides specialized loss functions that extend beyond standard
metrics. The main focus is on rank-based losses that better capture relative
ordering patterns in predictions, which can be particularly useful for
financial or ranking tasks.

Highlights:
    * **SpearmanCorrelation** – Differentiable approximation of Spearman's rank
      correlation coefficient that can be used as a loss function.
    * **CombinedLoss** – Weighted combination of MSE and Spearman correlation
      losses for balancing absolute accuracy with rank preservation.
"""

import keras.ops as K
from keras.losses import Loss
from keras.config import epsilon

class SpearmanCorrelation(Loss):
    """Differentiable Spearman rank correlation loss.
    
    This loss function computes a soft approximation of Spearman's rank
    correlation coefficient between predictions and targets. Unlike the
    standard non-differentiable rank correlation, this implementation uses
    sigmoid-based soft rankings that allow gradient flow during backpropagation.
    
    The loss is computed as the negative correlation (to minimize during training)
    between the soft ranks of predictions and targets.
    
    Args:
        regularization_strength (float, default=1e-3): Temperature parameter for
            the sigmoid function used in soft ranking. Smaller values create
            sharper (more discrete) rankings, while larger values create smoother
            approximations. Typically ranges from 1e-4 to 1e-1.
        name (str, default="spearman_correlation"): Name of the loss function.
        **kwargs: Additional keyword arguments passed to the base Loss class.
    
    Examples:
        >>> import keras
        >>> loss_fn = SpearmanCorrelation(regularization_strength=0.01)
        >>> model = keras.Sequential([...])
        >>> model.compile(optimizer='adam', loss=loss_fn)
    """
    
    def __init__(
        self, regularization_strength=1e-3, name="spearman_correlation", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.regularization_strength = regularization_strength

    def call(self, y_true, y_pred):
        """Compute the Spearman correlation loss.
        
        Args:
            y_true: Ground truth values of shape (batch_size,) or (batch_size, 1).
            y_pred: Predicted values of shape (batch_size,) or (batch_size, 1).
            
        Returns:
            Scalar loss value (negative correlation).
        """
        # Reshape inputs to ensure 2D
        y_true = K.reshape(y_true, (-1, 1))
        y_pred = K.reshape(y_pred, (-1, 1))

        # Calculate soft ranks for both true and predicted values
        true_ranks = self._soft_rank(y_true)
        pred_ranks = self._soft_rank(y_pred)

        # Calculate correlation between ranks
        return -self._correlation(true_ranks, pred_ranks)

    def _soft_rank(self, x):
        """Compute differentiable soft ranks using sigmoid approximation.
        
        Args:
            x: Input tensor of shape (batch_size, 1).
            
        Returns:
            Soft ranks tensor of shape (batch_size, 1).
        """
        # Create pairwise differences matrix
        x_expanded1 = K.expand_dims(x, 1)
        x_expanded2 = K.expand_dims(x, 0)
        diff = x_expanded1 - x_expanded2

        # Apply soft step function
        soft_step = K.sigmoid(diff / self.regularization_strength)

        # Sum over rows to get ranks
        ranks = K.sum(soft_step, axis=1)
        return ranks

    def _correlation(self, x, y):
        """Compute Pearson correlation between two tensors.
        
        Args:
            x: First tensor of shape (batch_size, 1).
            y: Second tensor of shape (batch_size, 1).
            
        Returns:
            Scalar correlation value in range [-1, 1].
        """
        # Mean center
        x_centered = x - K.mean(x)
        y_centered = y - K.mean(y)

        # Calculate correlation
        numerator = K.sum(x_centered * y_centered)
        denominator = K.sqrt(
            K.sum(K.square(x_centered)) * K.sum(K.square(y_centered)) + epsilon()
        )

        return numerator / denominator


class CombinedLoss(Loss):
    """Weighted combination of MSE and Spearman correlation losses.
    
    This loss function combines mean squared error (for absolute accuracy)
    with Spearman correlation loss (for rank preservation). This can be
    particularly useful when both the exact values and their relative
    ordering are important.
    
    Args:
        mse_weight (float, default=2.0): Weight applied to the MSE component.
            Higher values prioritize absolute accuracy.
        spearman_weight (float, default=1.0): Weight applied to the Spearman
            correlation component. Higher values prioritize rank preservation.
        spearman_regularization (float, default=1e-3): Regularization strength
            passed to the SpearmanCorrelation loss.
        name (str, default="combined_loss"): Name of the loss function.
        **kwargs: Additional keyword arguments passed to the base Loss class.
        
    Examples:
        >>> # Prioritize ranking accuracy over absolute values
        >>> loss_fn = CombinedLoss(mse_weight=0.5, spearman_weight=2.0)
        >>> model.compile(optimizer='adam', loss=loss_fn)
    """
    
    def __init__(
        self,
        mse_weight=2.0,
        spearman_weight=1.0,
        spearman_regularization=1e-3,
        name="combined_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.mse_weight = mse_weight
        self.spearman_weight = spearman_weight
        self.spearman_loss = SpearmanCorrelation(
            regularization_strength=spearman_regularization
        )

    def call(self, y_true, y_pred):
        """Compute the combined loss.
        
        Args:
            y_true: Ground truth values of shape (batch_size,) or (batch_size, 1).
            y_pred: Predicted values of shape (batch_size,) or (batch_size, 1).
            
        Returns:
            Scalar loss value (weighted sum of MSE and negative Spearman correlation).
        """
        mse = K.mean(K.square(y_pred - y_true))
        spearman = self.spearman_loss(y_true, y_pred)

        return self.mse_weight * mse + self.spearman_weight * spearman
