"""Feature penalization transformers using iterative optimization (requires JAX)."""

import warnings

import narwhals as nw
import numpy as np
from joblib import Parallel, delayed
from narwhals.typing import FrameT, IntoSeries

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except ImportError as e:
    raise ImportError(
        "FeaturePenalizer requires JAX. Install with:\n"
        "  uv add 'centimators[keras-jax]'\n"
        "or:\n"
        "  pip install 'centimators[keras-jax]'"
    ) from e

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


from ..narwhals_utils import _ensure_numpy
from .base import _BaseFeatureTransformer, _gaussianize, _min_max_scale


class FeaturePenalizer(_BaseFeatureTransformer):
    """
    Feature penalization using iterative optimization to cap feature exposure.

    Unlike FeatureNeutralizer which subtracts a fixed proportion of linear exposure,
    this transformer uses gradient descent to find the minimal adjustment that caps
    all feature exposures below a threshold. This preserves more of the original
    signal while ensuring no single feature dominates.

    For each era, it:
    1. Gaussianizes the predictions (rank -> normalize -> inverse CDF)
    2. Trains a linear model to subtract from predictions such that
       |exposure to any feature| <= max_exposure
    3. Re-normalizes and scales to [0, 1]

    Args:
        max_exposure (float or list of float): Maximum allowed feature exposure in [0, 1].
            Lower = more aggressive penalization. If list, creates multiple outputs.
        pred_name (str or list of str): Name(s) of prediction column(s) to penalize.
        feature_names (list of str, optional): Names of feature columns.
        suffix (str, optional): Suffix to append to output column names.
        lr (float): Learning rate for Adamax optimizer. Default 1e-3.
        max_iters (int): Maximum optimization iterations per era.
        tol (float): Early stopping tolerance for loss.
        n_jobs (int): Number of parallel jobs. 1 = sequential, -1 = all cores.
        verbose (bool): Show progress bar over eras. Default False.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from centimators.feature_transformers import FeaturePenalizer
        >>> df = pd.DataFrame({
        ...     'era': ['era1'] * 50 + ['era2'] * 50,
        ...     'feature1': np.random.randn(100),
        ...     'feature2': np.random.randn(100),
        ...     'prediction': np.random.randn(100)
        ... })
        >>> penalizer = FeaturePenalizer(
        ...     max_exposure=0.1,
        ...     pred_name='prediction',
        ...     feature_names=['feature1', 'feature2']
        ... )
        >>> result = penalizer.fit_transform(
        ...     df[['prediction']],
        ...     features=df[['feature1', 'feature2']],
        ...     era_series=df['era']
        ... )
    """

    def __init__(
        self,
        max_exposure: float | list[float] = 0.1,
        pred_name: str | list[str] = "prediction",
        feature_names: list[str] | None = None,
        suffix: str | None = None,
        lr: float = 1e-3,
        max_iters: int = 100_000,
        tol: float = 1e-7,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        # Normalize inputs to lists
        self.pred_names = [pred_name] if isinstance(pred_name, str) else pred_name
        self.max_exposures = (
            [max_exposure] if isinstance(max_exposure, float) else max_exposure
        )

        # Validate
        assert len(self.pred_names) == len(set(self.pred_names)), (
            "Duplicate pred_names found."
        )
        for exp in self.max_exposures:
            assert 0.0 <= exp <= 1.0, f"max_exposure should be in [0, 1]. Got {exp}."

        self.suffix = suffix
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Generate output column names
        self._output_names = [
            (
                f"{pname}_penalized_{exp}_{suffix}"
                if suffix
                else f"{pname}_penalized_{exp}"
            )
            for pname in self.pred_names
            for exp in self.max_exposures
        ]

        super().__init__(feature_names)

    @nw.narwhalify(allow_series=True)
    def transform(
        self,
        X: FrameT,
        y=None,
        features: FrameT | None = None,
        era_series: IntoSeries | None = None,
    ) -> FrameT:
        """Penalize predictions to cap feature exposure.

        Args:
            X: Input predictions to penalize (shape: n_samples x n_predictions).
            y: Ignored. Kept for sklearn compatibility.
            features: DataFrame with features for penalization.
            era_series: Series with era labels for grouping.

        Returns:
            DataFrame with penalized predictions, scaled to [0, 1].
        """
        if features is None:
            features = X

        predictions = _ensure_numpy(X)
        feature_array = _ensure_numpy(features)

        if predictions.ndim == 1:
            assert len(self.pred_names) == 1
            predictions = predictions.reshape(-1, 1)
        else:
            assert predictions.shape[1] == len(self.pred_names)

        if era_series is not None:
            eras = _ensure_numpy(era_series, allow_series=True)
        else:
            warnings.warn(
                "era_series not provided. Treating all data as a single era. "
                "This is fine for live inference (1 era) but may be incorrect "
                "for training data with multiple eras.",
                UserWarning,
            )
            eras = np.array(["X"] * len(predictions))

        # Process each prediction column and max_exposure
        if self.n_jobs == 1:
            results = [
                self._penalize_by_era(
                    predictions[:, pred_idx], feature_array, eras, max_exp, self.verbose
                )
                for pred_idx in range(len(self.pred_names))
                for max_exp in self.max_exposures
            ]
        else:
            # Disable verbose in parallel mode
            tasks = [
                delayed(self._penalize_by_era)(
                    predictions[:, pred_idx], feature_array, eras, max_exp, False
                )
                for pred_idx in range(len(self.pred_names))
                for max_exp in self.max_exposures
            ]
            results = Parallel(n_jobs=self.n_jobs)(tasks)

        result_array = np.column_stack(results)
        result_dict = {
            col_name: result_array[:, i]
            for i, col_name in enumerate(self._output_names)
        }

        native_namespace = nw.get_native_namespace(X)
        return nw.from_native(
            native_namespace.DataFrame(result_dict),
            eager_only=True,
        )

    def _penalize_by_era(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        eras: np.ndarray,
        max_exposure: float,
        verbose: bool = False,
    ) -> np.ndarray:
        """Penalize predictions era by era."""
        unique_eras = np.unique(eras)
        penalized = np.zeros_like(predictions)

        era_iter = tqdm(
            unique_eras, desc=f"max_exp={max_exposure}", disable=not verbose
        )
        for era in era_iter:
            mask = eras == era
            era_pred = predictions[mask]
            era_features = features[mask]

            # Gaussianize then penalize
            era_pred_norm = _gaussianize(era_pred)
            era_pred_pen = self._reduce_exposure(
                era_pred_norm, era_features, max_exposure
            )
            # Standardize within era
            era_pred_pen = era_pred_pen / np.std(era_pred_pen)
            penalized[mask] = era_pred_pen

        return _min_max_scale(penalized)

    def _reduce_exposure(
        self,
        prediction: np.ndarray,
        features: np.ndarray,
        max_exp: float,
    ) -> np.ndarray:
        """
        Learn a linear adjustment to predictions that caps feature exposure.

        Uses Adamax optimization with full JIT compilation via lax.while_loop
        to find weights such that:
            neutralized = prediction - features @ weights
        has |exposure to any feature| <= max_exp.
        """
        feats = jnp.asarray(features - 0.5, dtype=jnp.float32)
        pred = jnp.asarray(prediction, dtype=jnp.float32)[:, None]
        n_features = feats.shape[1]

        # Target: clamp current exposures to [-max_exp, max_exp]
        target_exp = jnp.clip(self._exposures(feats, pred), -max_exp, max_exp)

        # Adamax hyperparameters
        beta1, beta2 = 0.9, 0.999
        eps = 1e-7
        lr = self.lr
        tol = self.tol
        max_iters = self.max_iters

        def loss_fn(w):
            neutralized = pred - feats @ w
            exps = self._exposures(feats, neutralized)
            pos_excess = jax.nn.relu(jax.nn.relu(exps) - jax.nn.relu(target_exp))
            neg_excess = jax.nn.relu(jax.nn.relu(-exps) - jax.nn.relu(-target_exp))
            return jnp.sum(pos_excess + neg_excess)

        def cond_fn(state):
            w, m, u, t, loss = state
            return (loss >= tol) & (t < max_iters)

        def body_fn(state):
            w, m, u, t, _ = state
            loss, grads = jax.value_and_grad(loss_fn)(w)
            m_new = beta1 * m + (1 - beta1) * grads
            u_new = jnp.maximum(beta2 * u, jnp.abs(grads))
            m_hat = m_new / (1 - beta1 ** (t + 1))
            w_new = w - lr * m_hat / (u_new + eps)
            return w_new, m_new, u_new, t + 1, loss

        @jax.jit
        def optimize():
            init_state = (
                jnp.zeros((n_features, 1)),  # weights
                jnp.zeros((n_features, 1)),  # m (first moment)
                jnp.zeros((n_features, 1)),  # u (infinity norm)
                jnp.array(0),  # t (iteration)
                jnp.array(float("inf")),  # loss
            )
            w, m, u, t, loss = lax.while_loop(cond_fn, body_fn, init_state)
            return pred - feats @ w

        neutralized = optimize()
        return np.asarray(neutralized).squeeze()

    @staticmethod
    def _exposures(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Correlation between features (x) and predictions (y)."""
        x = x - jnp.mean(x, axis=0)
        x = x / jnp.linalg.norm(x, axis=0)
        y = y - jnp.mean(y, axis=0)
        y = y / jnp.linalg.norm(y, axis=0)
        return x.T @ y
