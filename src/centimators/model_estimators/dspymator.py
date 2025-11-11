"""
DSPyMator: A scikit-learn compatible wrapper for DSPy modules.
"""

from dataclasses import dataclass
from typing import Any
import asyncio
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import narwhals as nw
import numpy
import dspy

from centimators.narwhals_utils import _ensure_numpy


@dataclass(kw_only=True)
class DSPyMator(TransformerMixin, BaseEstimator):
    """DSPyMator is a scikit-learn compatible wrapper for DSPy modules.

    Integrates DSPy programs (e.g., ChainOfThought, Predict) into the centimators
    ecosystem, enabling LLM-based predictions that work seamlessly with sklearn
    pipelines, cross-validation, and other ML tooling. DSPyMator turns a
    DSPy `Module` (e.g., `ChainOfThought`, `Predict`) and optimizer (e.g., `GEPA`,
    `BootstrapFewShot`, `MIPROv2`) into a standard scikit-learn
    estimator/transformer that operates on tabular rows.

    The estimator is dataframe-agnostic through narwhals, accepting Polars,
    Pandas, or numpy arrays. Input features are automatically mapped to the DSPy
    program's signature fields based on `feature_names` or column names.

    Execution Modes:
        By default, uses asynchronous execution (`use_async=True`) with bounded
        concurrency for efficient batch processing. Set `use_async=False` for
        synchronous execution. Async mode automatically handles nested event loops
        (e.g., in Jupyter notebooks) when `nest_asyncio` is installed. Current async
        support with asyncio means that concurrent requests are best handled for API
        requests, rather than for fine-tuning of local models' weights.

    Output Methods:
        - `predict(X)`: Returns target predictions in the same format as input.
          If input is numpy array, returns numpy array. If input is dataframe,
          returns dataframe with target column(s). For single targets, returns
          1D array or single-column dataframe. For multiple targets, returns
          2D array or multi-column dataframe.
        - `transform(X)`: Returns all output fields from the DSPy program
          (including reasoning, intermediate steps, etc.) as a dataframe in the
          same backend as the input. Use this to access full program outputs.

    Progress Tracking:
        When `verbose=True`, displays progress bars using tqdm. Requires `tqdm`
        for sync mode and `tqdm.asyncio` for async mode. Falls back gracefully
        if tqdm is not installed.

    Optimization:
        DSPyMator supports automatic prompt optimization via any DSPy optimizer.
        Pass a configured optimizer instance (e.g., `dspy.GEPA`, `dspy.BootstrapFewShot`,
        `dspy.MIPROv2`, etc.) to `fit()` to optimize prompts during training.

        Different optimizers have different requirements:

        - **Few-shot optimizers** (e.g., `BootstrapFewShot`, `LabeledFewShot`):
          Only need `trainset`. Pass `validation_data=None`.

        - **Instruction optimizers** (e.g., `GEPA`, `MIPROv2`, `COPRO`):
          Need both `trainset` and `valset`. Provide validation data via `validation_data`.

        - **Finetuning optimizers** (e.g., `BootstrapFinetune`):
          May have specific requirements. Consult optimizer documentation.

        To use optimization:

        1. Create an optimizer instance:

        ```python
        # Example: GEPA for instruction optimization
        gepa = dspy.GEPA(metric=my_metric, auto='light')

        # Example: BootstrapFewShot for few-shot learning
        bootstrap = dspy.BootstrapFewShot()

        # Example: MIPROv2 for instruction optimization
        mipro = dspy.MIPROv2(metric=my_metric)
        ```

        2. Pass the optimizer to fit():

        ```python
        # With validation split (for optimizers that need valset)
        estimator.fit(X_train, y_train, optimizer=gepa, validation_data=0.2)

        # With explicit validation set
        estimator.fit(X_train, y_train, optimizer=gepa, validation_data=(X_val, y_val))

        # Without validation (for optimizers that only need trainset)
        estimator.fit(X_train, y_train, optimizer=bootstrap, validation_data=None)

        # To use trainset as valset, pass it explicitly
        estimator.fit(X_train, y_train, optimizer=gepa, validation_data=(X_train, y_train))
        ```

        After optimization, the original program is stored in `original_program_`
        and optimizer results are available in `optimizer_results_` for inspection
        (if the optimizer provides detailed results).

        For more details on optimizers, see: https://dspy.ai/learn/optimization/optimizers/

    Parameters:
        program: DSPy module (e.g., dspy.ChainOfThought, dspy.Predict) with a signature
            defining input and output fields. The signature must be accessible
            via `.predict.signature` or `.signature`.
        target_names: Field name(s) from the program's output signature to use
            as predictions. Can be a single string or list of strings. These
            fields are extracted and returned by `predict()`.
        feature_names: Column names mapping input data to signature input fields.
            If None, inferred from dataframe columns or uses signature field names
            for numpy arrays. Must match the number of input fields in the signature.
        lm: Language model identifier (e.g., "openai/gpt-4", "anthropic/claude-3").
            Defaults to "openai/gpt-5-nano".
        temperature: Sampling temperature for the language model. Defaults to 1.0.
        max_tokens: Maximum tokens in model responses. Defaults to 16000.
        use_async: Whether to use asynchronous execution for batch predictions.
            Defaults to True. Set to False for synchronous execution.
        max_concurrent: Maximum number of concurrent async requests when
            `use_async=True`. Defaults to 50.
        verbose: Whether to display progress bars during prediction. Defaults to True.
            Requires `tqdm` package for sync mode or `tqdm.asyncio` for async mode.

    Examples:
        Basic usage with a ChainOfThought or Predict program:

        ```python
        import dspy
        from centimators.model_estimators import DSPyMator

        # Create a DSPy program (e.g., Predict, ChainOfThought, etc.)
        program = dspy.Predict("text -> sentiment")

        # Create estimator
        estimator = DSPyMator(
            program=program,
            target_names="sentiment"
        )

        X_train = pl.DataFrame({
            "text": ["I love this product!", "This is terrible.", "It's okay."]
        })
        y_train = pl.Series(["positive", "negative", "neutral"])

        # Fit and predict (get only target predictions)
        estimator.fit(X_train, y_train)  # y_train can be None
        predictions = estimator.predict(X_test)  # returns same type as X_test

        # Get all outputs (including reasoning and other intermediate steps of the program)
        full_outputs = estimator.transform(X_test)  # always returns dataframe

        # With optimization:
        import dspy

        gepa = dspy.GEPA(metric=my_metric, auto='light')
        estimator.fit(X_train, y_train, optimizer=gepa, validation_data=0.2)
        ```
    """

    program: dspy.Module
    target_names: str | list[str]
    feature_names: list[str] | None = None
    lm: str = "openai/gpt-5-nano"
    temperature: float = 1.0
    max_tokens: int = 16000
    use_async: bool = True
    max_concurrent: int = 50
    verbose: bool = True

    def _get_signature(self):
        """Extract signature from the DSPy program.

        ChainOfThought stores signature in .predict.signature, while other
        modules may expose it directly as .signature.
        """
        if hasattr(self.program, "predict") and hasattr(
            self.program.predict, "signature"
        ):
            return self.program.predict.signature
        elif hasattr(self.program, "signature"):
            return self.program.signature
        else:
            raise ValueError(
                f"Cannot extract signature from program of type {type(self.program)}. "
                "Expected a DSPy module with .predict.signature or .signature attribute."
            )

    def __post_init__(self):
        self.signature_ = self._get_signature()
        if isinstance(self.target_names, str):
            self._target_names = [self.target_names]
        else:
            self._target_names = list(self.target_names)

        if not self._target_names:
            raise ValueError("target_names must contain at least one field.")

    @nw.narwhalify
    def fit(
        self,
        X,
        y,
        optimizer: Any | None = None,
        validation_data: "tuple[Any, Any] | float | None" = None,
        **kwargs,
    ):
        """Fit the DSPyMator estimator.

        Parameters:
            X: Training data (dataframe or numpy array).
            y: Target values (can be None for unsupervised tasks).
            optimizer: Optional DSPy optimizer instance (e.g., dspy.GEPA, dspy.BootstrapFewShot,
                dspy.MIPROv2). When provided, enables prompt optimization or finetuning during fit.
            validation_data: Validation data for optimizers that require it.
                - If tuple: Use as (X_val, y_val) directly.
                - If float (0-1): Fraction of training data to use for validation.
                - If None: No validation set (for optimizers that only need trainset).
                  To use trainset as valset, pass `(X, y)` explicitly, although some
                  optimizers may automatically use the trainset as valset if None is passed.

        Returns:
            self: The fitted estimator.

        Examples:
            Basic fitting without optimization:

            ```python
            estimator = DSPyMator(program=program, target_names='label')
            estimator.fit(X_train, y_train)
            ```

            With optimizer using auto-split validation:

            ```python
            gepa_optimizer = dspy.GEPA(metric=my_metric, auto='light', ..., **kwargs)
            estimator.fit(X_train, y_train, optimizer=gepa_optimizer, validation_data=0.2)
            ```
        """
        self.lm_ = dspy.LM(
            self.lm, temperature=self.temperature, max_tokens=self.max_tokens
        )

        self.input_fields_ = list(self.signature_.input_fields.keys())

        if self.feature_names is None:
            if isinstance(X, numpy.ndarray):
                self.feature_names = self.input_fields_
            else:
                self.feature_names = list(X.columns)

        if len(self.feature_names) != len(self.input_fields_):
            raise ValueError(
                f"Number of feature_names ({len(self.feature_names)}) must match "
                f"number of input_fields ({len(self.input_fields_)})"
            )

        # Optimization if requested
        if optimizer is not None:
            # Handle validation_data parameter
            if isinstance(validation_data, float):
                # Convert to numpy for sklearn compatibility
                X_train, X_val, y_train, y_val = train_test_split(
                    _ensure_numpy(X),
                    _ensure_numpy(y, allow_series=True),
                    test_size=validation_data,
                    random_state=42,
                )
            elif validation_data is None:
                # No validation set (for optimizers that only need trainset)
                X_train, y_train = X, y
                X_val, y_val = None, None
            else:
                # Use provided validation set
                X_train, y_train = X, y
                X_val, y_val = validation_data

            # Store original program before optimization
            self.original_program_ = self.program

            # Convert data to DSPy Examples
            train_examples = self._convert_to_examples(X_train, y_train)
            val_examples = (
                self._convert_to_examples(X_val, y_val) if X_val is not None else None
            )

            # Run optimizer compilation
            compile_kwargs = {
                "trainset": train_examples,
                **({"valset": val_examples} if val_examples is not None else {}),
                **kwargs,
            }

            with dspy.context(lm=self.lm_):
                optimized_program = optimizer.compile(self.program, **compile_kwargs)

            # Update program with optimized version
            self.program = optimized_program

            # Refresh signature and input_fields after optimization
            self.signature_ = self._get_signature()
            self.input_fields_ = list(self.signature_.input_fields.keys())

            # Store optimizer results for inspection
            if hasattr(optimized_program, "detailed_results"):
                self.optimizer_results_ = optimized_program.detailed_results

        self._is_fitted = True
        return self

    @nw.narwhalify
    def _convert_to_examples(self, X, y):
        """Convert X, y data to DSPy Example objects.

        Parameters:
            X: Input features (dataframe or numpy array). Can be None.
            y: Target values. Can be None.

        Returns:
            List of dspy.Example objects with inputs marked, or None if X is None.
        """
        if X is None:
            return None

        examples = []

        # Build input kwargs for each row
        if isinstance(X, numpy.ndarray):
            input_kwargs_list = [
                {inp: val for inp, val in zip(self.input_fields_, row)} for row in X
            ]
        else:
            input_kwargs_list = [
                {
                    inp: row[col]
                    for inp, col in zip(self.input_fields_, self.feature_names)
                }
                for row in X.iter_rows(named=True)
            ]

        # Add targets and create examples
        for kwargs, label in zip(input_kwargs_list, y):
            for i, target_name in enumerate(self._target_names):
                kwargs[target_name] = label[i] if len(self._target_names) > 1 else label
            examples.append(dspy.Example(**kwargs).with_inputs(*self.input_fields_))

        return examples

    @nw.narwhalify
    def _iter_input_kwargs(self, X):
        if isinstance(X, numpy.ndarray):
            for row in X:
                yield {inp: val for inp, val in zip(self.input_fields_, row)}
        else:
            for row in X.iter_rows(named=True):
                yield {
                    inp: row[col]
                    for inp, col in zip(self.input_fields_, self.feature_names)
                }

    def _predict_raw_sync(self, X):
        """Synchronously predict all samples with optional progress bar."""
        input_kwargs = list(self._iter_input_kwargs(X))

        if self.verbose:
            try:
                from tqdm import tqdm

                return [
                    self.program(**kwargs)
                    for kwargs in tqdm(input_kwargs, desc="DSPyMator predicting")
                ]
            except ImportError:
                warnings.warn(
                    "tqdm not installed; progress bar unavailable. Install tqdm for progress tracking.",
                    stacklevel=2,
                )
                return [self.program(**kwargs) for kwargs in input_kwargs]
        else:
            return [self.program(**kwargs) for kwargs in input_kwargs]

    async def _predict_raw_async(self, X):
        """Asynchronously predict all samples with bounded concurrency and optional progress bar."""
        input_kwargs = list(self._iter_input_kwargs(X))
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_one(kwargs):
            async with semaphore:
                return await self.program.acall(**kwargs)

        tasks = [run_one(kwargs) for kwargs in input_kwargs]

        if self.verbose:
            try:
                from tqdm.asyncio import tqdm as tqdm_asyncio

                return await tqdm_asyncio.gather(*tasks, desc="DSPyMator predicting")
            except ImportError:
                warnings.warn(
                    "tqdm not installed; progress bar unavailable. Install tqdm for progress tracking.",
                    stacklevel=2,
                )
                return await asyncio.gather(*tasks)
        else:
            return await asyncio.gather(*tasks)

    def _predict_raw(self, X):
        """Route to sync or async prediction based on use_async flag."""
        with dspy.context(lm=self.lm_):
            if not self.use_async:
                return self._predict_raw_sync(X)

            try:
                asyncio.get_running_loop()
                # Already in an event loop, use nest_asyncio to enable nested loops
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                    return asyncio.run(self._predict_raw_async(X))
                except ImportError:
                    warnings.warn(
                        "nest_asyncio not installed; falling back to synchronous. "
                        "Install nest_asyncio to use async mode in notebooks/event loops.",
                        stacklevel=2,
                    )
                    return self._predict_raw_sync(X)
            except RuntimeError:
                # No event loop, safe to use asyncio.run
                return asyncio.run(self._predict_raw_async(X))

    @nw.narwhalify
    def predict(self, X):
        if not hasattr(self, "_is_fitted"):
            raise ValueError("Classifier not fitted. Call fit() first.")
        preds = self._predict_raw(X)
        fields = self._target_names

        if not preds:
            if len(fields) == 1:
                empty = numpy.array([], dtype=object)
            else:
                empty = numpy.empty((0, len(fields)), dtype=object)

            # Return numpy for numpy input, dataframe for dataframe input
            if isinstance(X, numpy.ndarray):
                return empty
            col_names = fields if len(fields) > 1 else [fields[0]]
            return nw.from_dict(
                {name: [] for name in col_names}, backend=nw.get_native_namespace(X)
            )

        labels = numpy.array(
            [[getattr(pred, field) for field in fields] for pred in preds]
        )
        predictions = labels.squeeze(axis=1) if len(fields) == 1 else labels

        # Return numpy for numpy input, dataframe for dataframe input
        if isinstance(X, numpy.ndarray):
            return predictions

        if len(fields) == 1:
            return nw.from_dict(
                {fields[0]: predictions}, backend=nw.get_native_namespace(X)
            )
        else:
            cols = {fields[i]: predictions[:, i] for i in range(len(fields))}
            return nw.from_dict(cols, backend=nw.get_native_namespace(X))

    def _get_output_fields(self):
        """Get all output fields for transform."""
        signature = self._get_signature()
        output_fields = signature.output_fields.keys()

        return output_fields

    @nw.narwhalify
    def transform(self, X, y=None):
        if not hasattr(self, "_is_fitted"):
            raise ValueError("Classifier not fitted. Call fit() first.")

        output_fields = self._get_output_fields()
        preds = self._predict_raw(X)

        # Build a dictionary of columns
        data = {
            field: [getattr(pred, field, None) for pred in preds]
            for field in output_fields
        }

        # Create a dataframe in the same backend as input X
        return nw.from_dict(data, backend=nw.get_native_namespace(X))

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y).transform(X, y, **kwargs)

    def get_feature_names_out(self, input_features=None):
        return self._get_output_fields()

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted", False)
