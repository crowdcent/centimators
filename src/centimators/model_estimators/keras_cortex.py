"""
Keras Cortex: A self-improving Keras estimator wrapper using DSPy to self-reflect
and improve its architecture.

This module provides KerasCortex, a scikit-learn compatible meta-estimator.
"""

import inspect
import types

from sklearn.base import BaseEstimator, RegressorMixin, clone

try:
    import dspy
    from dspy import InputField, OutputField, Signature, Module, ChainOfThought
except ImportError as e:
    raise ImportError(
        "KerasCortex requires dspy. Install with:\n"
        "  uv add 'centimators[dspy]'\n"
        "or:\n"
        "  pip install 'centimators[dspy]'"
    ) from e

try:
    # Make Keras APIs available to LLM-generated code via exec
    from keras import (  # noqa: F401
        layers,
        models,
        regularizers,
        optimizers,
    )
except ImportError as e:
    raise ImportError(
        "KerasCortex requires keras and jax (or another Keras-compatible backend). Install with:\n"
        "  uv add 'centimators[keras-jax]'\n"
        "or:\n"
        "  pip install 'centimators[keras-jax]'"
    ) from e

from .keras_estimators import MLPRegressor


class KerasCodeRefinements(Signature):
    """Suggest modifications to build_model code to improve performance."""

    current_keras_code = InputField(desc="Source code of build_model method.")
    performance_log = InputField(desc="History of (code, metric) pairs.")
    optimization_goal = InputField(desc="Objective, e.g., 'improve validation scores'.")
    suggested_keras_code_modification = OutputField(
        desc=(
            "Modified build_model method body as code. No code fences. "
            "You must start with only 'def build_model(self):'"
        )
    )


class Think(Module):
    """DSPy Module for suggesting Keras model code modifications."""

    def __init__(self, verbose=False):
        super().__init__()
        self.suggest_code = ChainOfThought(KerasCodeRefinements)
        self.verbose = verbose

    def forward(self, current_keras_code, performance_log, optimization_goal):
        prediction = self.suggest_code(
            current_keras_code=current_keras_code,
            performance_log=performance_log,
            optimization_goal=optimization_goal,
        )
        if self.verbose:
            print(f"Reasoning: \n{prediction.reasoning}")
            print(f"Suggested code: \n{prediction.suggested_keras_code_modification}")
        return prediction.suggested_keras_code_modification


class KerasCortex(RegressorMixin, BaseEstimator):
    """A scikit-learn meta-estimator that iteratively refines a Keras model."""

    def __init__(
        self,
        base_estimator=None,
        n_iterations=5,
        lm="openai/gpt-4o-mini",
        verbose=False,
    ):
        if base_estimator is None:
            base_estimator = MLPRegressor()
        self.base_estimator = base_estimator
        self.n_iterations = n_iterations
        self.lm = dspy.LM(lm)
        dspy.configure(lm=self.lm)
        self.verbose = verbose

    def think_loop(
        self, base_estimator, X, y, validation_data, n_iterations=5, **kwargs
    ) -> tuple[BaseEstimator, list[tuple[str, float]]]:
        baseline_model = clone(base_estimator)
        baseline_model.fit(X, y, **kwargs)

        X_val, y_val = validation_data
        best_metric = baseline_model.score(X_val, y_val)
        current_code = inspect.getsource(type(baseline_model).build_model)
        performance_log = [(current_code, best_metric)]

        best_model = baseline_model
        suggestion = current_code

        think = Think(verbose=self.verbose)
        for i in range(n_iterations):
            print(f"\n--- Iteration {i + 1} ---")
            try:
                suggestion = think.forward(
                    current_keras_code=suggestion,
                    performance_log=performance_log,
                    optimization_goal="improve validation metrics (R2)",
                )
                namespace = {}
                exec(suggestion, globals(), namespace)
                build_model_fn = namespace["build_model"]

                new_model = clone(base_estimator)
                new_model.build_model = types.MethodType(build_model_fn, new_model)
                new_model.fit(X, y, **kwargs)
                metric = new_model.score(X_val, y_val)

                performance_log.append((suggestion, metric))
                if metric > best_metric:
                    print(
                        f"Improvement! New validation score: {metric:.4f} > {best_metric:.4f}"
                    )
                    best_metric = metric
                    best_model = new_model
                else:
                    print(
                        f"No improvement ({metric:.4f} <= {best_metric:.4f}), keeping best code."
                    )
            except Exception as e:
                print("Error during optimization iteration:", e)
                break

        return best_model, performance_log

    def fit(self, X, y, validation_data=None, **kwargs):
        self.best_model_, self.performance_log_ = self.think_loop(
            base_estimator=self.base_estimator,
            X=X,
            y=y,
            validation_data=validation_data,
            n_iterations=self.n_iterations,
            **kwargs,
        )
        return self

    def predict(self, X):
        if not hasattr(self, "best_model_"):
            raise ValueError("Estimator not fitted. Call 'fit' first.")
        return self.best_model_.predict(X)
