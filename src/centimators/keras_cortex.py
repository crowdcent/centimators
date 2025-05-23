"""
Keras Cortex: A self-improving Keras estimator wrapper using DSPy to self-reflect and improve its architecture.

This module provides KerasCortex, a scikit-learn compatible meta-estimator.
KerasCortex wraps a base Keras estimator (which must have a `build_model`
method) and iteratively refines the `build_model` method's implementation using
a Large Language Model (LLM) through the DSPy library. The goal is to
autonomously improve the model's architecture and performance based on
validation scores.

Highlights:
    * **KerasCortex**: Meta-estimator that wraps Keras models and uses an LLM
      to iteratively suggest improvements to the `build_model` method.
    * **Think**: A DSPy module that orchestrates the LLM interaction to generate
      Keras code modifications.
    * **KerasCodeRefinements**: A DSPy signature defining the LLM's task for
      suggesting code changes.

!!! Warning

    This module is a work in progress. It is not yet ready for production use.
"""

import inspect
import dspy
from dspy import InputField, OutputField, Signature, Module, ChainOfThought
from centimators.model_estimators import MLPRegressor
from keras import (  # noqa: F401
    layers,
    models,
    regularizers,
    optimizers,
)  # unused, but needed to be made available in global namespace for LLM-generated code
from sklearn.base import BaseEstimator, RegressorMixin, clone
import types


class KerasCodeRefinements(Signature):
    """Suggest modifications to build_model code to improve performance. Consider the history of attempted code. Use Keras 3, there is no tensorflow or tf.keras. Don't use code fences."""

    current_keras_code = InputField(desc="Source code of build_model method.")
    performance_log = InputField(desc="History of (code, metric) pairs.")
    optimization_goal = InputField(desc="Objective, e.g., 'improve validation scores'.")
    suggested_keras_code_modification = OutputField(
        desc="Modified build_model method body as code. No code fences. You must start with only 'def build_model(self):'"
    )


class Think(Module):
    """DSPy Module for suggesting Keras model code modifications.

    This module uses a `ChainOfThought` DSPy program with the
    `KerasCodeRefinements` signature to prompt an LLM for improvements to a
    Keras model's `build_model` method.

    Args:
        verbose (bool, default=False): If True, prints the LLM's reasoning
            and suggested code during the `forward` call.

    TODO: Add Keras docs, arXiv access, optimize prompts, pass errors back to LLM, etc.
    """

    def __init__(self, verbose=False):
        super().__init__()
        self.suggest_code = ChainOfThought(KerasCodeRefinements)
        self.verbose = verbose

    def forward(
        self,
        current_keras_code,
        performance_log,
        optimization_goal,
    ):
        """Generates a Keras code modification suggestion using an LLM.

        Args:
            current_keras_code (str): The source code of the current
                `build_model` method.
            performance_log (list[tuple[str, float]]): A list of (code, metric)
                tuples representing the history of attempted `build_model` code
                and their corresponding validation scores.
            optimization_goal (str): The objective for the LLM, e.g.,
                'improve validation scores'.

        Returns:
            str: The LLM's suggested `build_model` method body as a string of code.
        """
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
    """A scikit-learn meta-estimator that iteratively refines a Keras model.

    `KerasCortex` wraps a base Keras estimator (which must expose a `build_model`
    method) and uses an LLM via DSPy to suggest modifications to this
    `build_model` method. It iteratively attempts these suggestions, evaluates
    their performance on validation data, and keeps the best-performing model
    architecture.

    Args:
        base_estimator (BaseEstimator, optional): An instance of a Keras-based
            estimator that has a `build_model` method. Defaults to `MLPRegressor()`.
        n_iterations (int, default=5): The number of iterations to run the
            refinement loop.
        lm (str, default="openai/gpt-4o-mini"): The language model to use for
            code generation, specified as a string recognized by `dspy.LM`.
        verbose (bool, default=False): If True, prints detailed information during
            the refinement process, including LLM reasoning and code suggestions.

    Attributes:
        best_model_ (BaseEstimator): The best Keras estimator found during the
            refinement process, after fitting.
        performance_log_ (list[tuple[str, float]]): A log of (code, metric) pairs
            from the refinement process, after fitting.
    """

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
        """Iteratively refine and retrain a Keras-based estimator.

        This method forms the core of `KerasCortex`. It takes an initial Keras
        estimator, trains it to get a baseline, and then enters a loop:
        1.  The current `build_model` code is sent to the `Think` module.
        2.  The `Think` module (using an LLM) suggests a modification to the code.
        3.  A new model is created with the modified `build_model` method.
        4.  The new model is trained and evaluated on validation data.
        5.  If the new model performs better, its code becomes the current best.
        This loop repeats for `n_iterations`.

        Args:
            base_estimator (BaseEstimator): An instance of a Keras-based estimator
                with `fit`, `predict`, and `build_model` methods.
            X (array-like): Training data (features).
            y (array-like): Training data (targets).
            validation_data (tuple[array-like, array-like]): Data (X_val, y_val) for
                evaluating model performance during refinement.
            n_iterations (int, default=5): The number of refinement iterations.
            **kwargs: Additional keyword arguments passed to the `fit` method of the
                Keras estimator during each iteration (e.g., `epochs`, `batch_size`).

        Returns:
            - best_model: The Keras estimator instance with the best-performing
                `build_model` method found.
            - performance_log: A list of (code_string, validation_metric)
                tuples, recording each attempted `build_model` code and its score.
        """
        # Initial baseline: clone the provided estimator and fit
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

                # clone from original base_estimator to avoid state pollution
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
        """Fit the KerasCortex estimator.

        This method initiates the `think_loop` to find the best model architecture
        and then fits this best model. The primary purpose of `fit` is to expose
        a scikit-learn compatible API.

        Args:
            X (array-like): Training data (features).
            y (array-like): Training data (targets).
            validation_data (tuple[array-like, array-like], optional): Data for
                evaluating model performance during the refinement loop. If None,
                KerasCortex cannot effectively optimize the model architecture.
            **kwargs: Additional keyword arguments passed to the `fit` method of the
                base Keras estimator during the `think_loop` (e.g., `epochs`, `batch_size`).

        Returns:
            KerasCortex: The fitted estimator instance.
        """
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
        """Generate predictions using the best model found by KerasCortex.

        Args:
            X (array-like): Input data (features) for which to make predictions.

        Returns:
            array-like: Predictions from the `best_model_`.

        Raises:
            ValueError: If the estimator has not been fitted (i.e., `fit` has not
                been called).
        """
        if not hasattr(self, "best_model_"):
            raise ValueError("Estimator not fitted. Call 'fit' first.")
        return self.best_model_.predict(X)
