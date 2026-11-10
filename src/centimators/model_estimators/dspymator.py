"""
DSPyMator: A scikit-learn compatible wrapper for DSPy modules.
"""

from dataclasses import dataclass
from typing import Any, get_args
import asyncio
import copy
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import ClassifierTags, RegressorTags
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
import narwhals as nw
import numpy
import dspy
from dspy.adapters.decision import record_evidence
from dspy.adapters.types.decision import Choice, Noul, Score, decision_type

from centimators.narwhals_utils import _ensure_numpy


@dataclass(kw_only=True)
class DSPyMator(TransformerMixin, BaseEstimator):
    """DSPyMator is a scikit-learn compatible wrapper for DSPy modules.

    Integrates DSPy programs (e.g., `Predict`, `ChainOfThought`) into the
    centimators ecosystem so LLM predictions work with sklearn pipelines,
    cross-validation, hyperparameter search, calibration and stacking. The
    estimator is dataframe-agnostic through narwhals, accepting Polars, Pandas,
    or numpy arrays. Input columns map to the program's input fields by
    `feature_names`, column names, or position.

    Decision outputs:
        When the target is a DSPy decision output, DSPyMator behaves like a
        native sklearn classifier or regressor, whichever backend answers it
        (Jev via `dspy.LM("typesafe:jev-latest")`, a hosted LLM, or a local
        model served through `dspy.LM`):

        - `Noul` or `bool` targets: binary classifier, `classes_ = [False, True]`.
        - `Choice` or `Literal[...]` targets: multiclass classifier over the
          declared options.
        - `Score` targets: regressor on the continuous expected level.

        Classifiers expose `predict_proba` and `classes_`, so `log_loss`,
        `roc_auc`, `CalibratedClassifierCV` and stacking work unchanged.
        Probabilities are the backend's raw distribution, before thresholds or
        weights. On Jev they are measured; on generative LMs the model reports
        them.

    Optimization:
        Pass any DSPy optimizer instance as `optimizer` (e.g., `dspy.GEPA`,
        `dspy.BootstrapFewShot`, `dspy.experimental.ReAnchor`). `fit` compiles
        a copy of the program with a copy of the optimizer and stores the
        result in `program_`, so the parameters are never modified and the
        estimator clones cleanly for `GridSearchCV` and `cross_val_score`.

        Some optimizers (e.g., `GEPA`, `MIPROv2`) want a validation set. Pass
        `validation_data` to `fit` as `(X_val, y_val)` or as a fraction of
        the training data to hold out.

    Execution:
        Predictions run asynchronously with bounded concurrency by default
        (`use_async=True`). Inside a running event loop (e.g., Jupyter), async
        mode needs `nest_asyncio`; otherwise it falls back to sync.

    Parameters:
        program: DSPy module with a signature defining input and output fields.
        target_names: Output field name(s) returned by `predict()`.
        feature_names: Column names mapped, in order, to the signature's input
            fields. If None, dataframe columns are used; numpy columns map to
            the input fields by position.
        lm: A `dspy.LM`, a model string (e.g., `"openai/gpt-6-luna"`,
            `"typesafe:jev-latest"`, `"ollama_chat/qwen3"`), or None to use
            the LM set with `dspy.configure`.
        temperature: Sampling temperature, used only when `lm` is a string.
        max_tokens: Maximum response tokens, used only when `lm` is a string.
        optimizer: Optional DSPy optimizer instance run during `fit`.
        use_async: Whether to predict asynchronously. Defaults to True.
        max_concurrent: Maximum concurrent requests when `use_async=True`.
        verbose: Whether to show progress bars (requires `tqdm`).

    Examples:
        ```python
        import dspy
        from typing import Literal
        from sklearn.model_selection import cross_val_score
        from centimators.model_estimators import DSPyMator


        class Sentiment(dspy.Signature):
            \"\"\"Classify the sentiment of a financial headline.\"\"\"

            headline: str = dspy.InputField()
            sentiment: Literal["bearish", "neutral", "bullish"] = dspy.OutputField(
                desc="Market sentiment of the headline."
            )


        clf = DSPyMator(
            program=dspy.Predict(Sentiment),
            target_names="sentiment",
            lm="typesafe:jev-latest",
        )
        cross_val_score(clf, X, y, cv=5, scoring="neg_log_loss")
        ```
    """

    program: dspy.Module
    target_names: str | list[str]
    feature_names: list[str] | None = None
    lm: str | dspy.LM | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    optimizer: Any = None
    use_async: bool = True
    max_concurrent: int = 50
    verbose: bool = True

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _signature_of(program):
        """Signature of the program's first predictor (ChainOfThought nests it)."""
        if hasattr(program, "predict") and hasattr(program.predict, "signature"):
            return program.predict.signature
        if hasattr(program, "signature"):
            return program.signature
        raise ValueError(
            f"Cannot extract signature from program of type {type(program)}. "
            "Expected a DSPy module with .predict.signature or .signature attribute."
        )

    def _targets(self):
        targets = (
            [self.target_names]
            if isinstance(self.target_names, str)
            else list(self.target_names)
        )
        if not targets:
            raise ValueError("target_names must contain at least one field.")
        return targets

    def _decision_kind(self):
        """Decision type of the single target field, or None."""
        targets = self._targets()
        if len(targets) != 1:
            return None
        field = self._signature_of(self.program).output_fields.get(targets[0])
        if field is None:
            raise ValueError(f"Target {targets[0]!r} is not an output field.")
        return decision_type(field)

    @staticmethod
    def _classes_for(kind):
        if issubclass(kind, Noul):
            return numpy.array([False, True])
        values = get_args(kind.model_fields["value"].annotation)
        return numpy.array(sorted(values, key=str), dtype=object)

    def _resolve_lm(self):
        if self.lm is None or isinstance(self.lm, dspy.BaseLM):
            return self.lm
        kwargs = {
            k: v
            for k, v in (
                ("temperature", self.temperature),
                ("max_tokens", self.max_tokens),
            )
            if v is not None
        }
        return dspy.LM(self.lm, **kwargs)

    def _lm_context(self):
        return dspy.context(lm=self.lm_) if self.lm_ is not None else dspy.context()

    def _has_decision_target(self):
        return self._decision_kind() is not None

    def _is_classifier(self):
        kind = self._decision_kind()
        return kind is not None and not issubclass(kind, Score)

    # ------------------------------------------------------------- sklearn API

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        kind = self._decision_kind()
        if kind is None:
            return tags
        if issubclass(kind, Score):
            tags.estimator_type = "regressor"
            tags.regressor_tags = RegressorTags()
        else:
            tags.estimator_type = "classifier"
            tags.classifier_tags = ClassifierTags(multi_class=issubclass(kind, Choice))
        return tags

    def __sklearn_clone__(self):
        # DSPy optimizers expose get_params() without sklearn's deep argument and
        # have no set_params, so sklearn's default clone cannot rebuild them.
        params = self.get_params(deep=False)
        params["optimizer"] = copy.copy(params["optimizer"])
        return type(self)(**params)

    def __sklearn_is_fitted__(self):
        return hasattr(self, "program_")

    @nw.narwhalify
    def fit(self, X, y=None, validation_data: "tuple[Any, Any] | float | None" = None):
        """Fit the estimator, running `optimizer` on a copy of `program` if set.

        Parameters:
            X: Training data (dataframe or numpy array).
            y: Target values. Optional when no optimizer is set.
            validation_data: Validation data for optimizers that use it, as
                `(X_val, y_val)` or a float fraction of `X` to hold out.

        Returns:
            self: The fitted estimator.
        """
        signature = self._signature_of(self.program)
        targets = self._targets()
        missing = [t for t in targets if t not in signature.output_fields]
        if missing:
            raise ValueError(
                f"target_names {missing} are not output fields of the program."
            )

        self.lm_ = self._resolve_lm()
        self.input_fields_ = list(signature.input_fields)
        if self.feature_names is not None:
            columns = list(self.feature_names)
        elif isinstance(X, numpy.ndarray):
            columns = self.input_fields_
        else:
            columns = list(X.columns)
        if len(columns) != len(self.input_fields_):
            raise ValueError(
                f"Number of feature_names ({len(columns)}) must match "
                f"number of input_fields ({len(self.input_fields_)})"
            )
        self.feature_names_in_ = numpy.array(columns, dtype=object)
        self.n_features_in_ = len(columns)

        kind = self._decision_kind()
        if kind is not None and not issubclass(kind, Score):
            self.classes_ = self._classes_for(kind)

        program = self.program.deepcopy()
        if kind is not None:
            _request_evidence(program, targets[0], kind)
        if self.optimizer is not None:
            if y is None:
                raise ValueError("An optimizer needs y to build its trainset.")
            if isinstance(validation_data, float):
                X_train, X_val, y_train, y_val = train_test_split(
                    _ensure_numpy(X),
                    _ensure_numpy(y, allow_series=True),
                    test_size=validation_data,
                    random_state=42,
                )
            elif validation_data is None:
                X_train, y_train, X_val, y_val = X, y, None, None
            else:
                X_train, y_train = X, y
                X_val, y_val = validation_data

            compile_kwargs = {"trainset": self._to_examples(X_train, y_train)}
            if X_val is not None:
                compile_kwargs["valset"] = self._to_examples(X_val, y_val)
            optimizer = copy.copy(self.optimizer)
            with self._lm_context():
                program = optimizer.compile(program, **compile_kwargs)
            self.optimizer_results_ = getattr(program, "detailed_results", None)

        self.program_ = program
        return self

    @nw.narwhalify
    def predict(self, X):
        """Target predictions as a numpy array.

        One target returns a 1-D array (bool or option labels for classifiers,
        floats for `Score` regressors); several targets return a 2-D array.
        """
        check_is_fitted(self)
        targets = self._targets()
        preds, _ = self._predict_raw(X)
        rows = [[_value(getattr(p, t)) for t in targets] for p in preds]
        if len(targets) == 1:
            if hasattr(self, "classes_"):
                dtype = self.classes_.dtype
            else:
                dtype = float if self._decision_kind() is not None else object
            return numpy.array([r[0] for r in rows], dtype=dtype)
        return numpy.array(rows, dtype=object).reshape(len(rows), len(targets))

    @available_if(_is_classifier)
    @nw.narwhalify
    def predict_proba(self, X):
        """Class probabilities, columns ordered as `classes_`.

        Available when the single target is a `Noul`/`bool` or
        `Choice`/`Literal` decision output.
        """
        check_is_fitted(self)
        target = self._targets()[0]
        _, evidence = self._predict_raw(X)
        proba = numpy.empty((len(evidence), len(self.classes_)))
        for i, row in enumerate(evidence):
            answer = row.get(target)
            if answer is None:
                raise ValueError(f"No decision evidence recorded for {target!r}.")
            if "noul" in answer:
                p = float(answer["noul"])
                proba[i] = (1.0 - p, p)
            else:
                dist = {str(k): float(v) for k, v in answer["probabilities"].items()}
                proba[i] = [dist[str(c)] for c in self.classes_]
        return proba / proba.sum(axis=1, keepdims=True)

    @available_if(_has_decision_target)
    def score(self, X, y, sample_weight=None):
        """Accuracy for classifiers, R^2 for `Score` regressors."""
        kind = self._decision_kind()
        metric = r2_score if issubclass(kind, Score) else accuracy_score
        return metric(
            _ensure_numpy(y, allow_series=True),
            self.predict(X),
            sample_weight=sample_weight,
        )

    @nw.narwhalify
    def transform(self, X, y=None):
        """All output fields of the program as a dataframe (decisions as values)."""
        check_is_fitted(self)
        fields = list(self._signature_of(self.program_).output_fields)
        preds, _ = self._predict_raw(X)
        data = {f: [_value(getattr(p, f, None)) for p in preds] for f in fields}
        if isinstance(X, numpy.ndarray):
            return numpy.array([data[f] for f in fields], dtype=object).T
        return nw.from_dict(data, backend=nw.get_native_namespace(X))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return numpy.array(
            list(self._signature_of(self.program_).output_fields), dtype=object
        )

    # -------------------------------------------------------------- execution

    @nw.narwhalify
    def _to_examples(self, X, y):
        targets = self._targets()
        examples = []
        for kwargs, label in zip(
            self._iter_inputs(X), _ensure_numpy(y, allow_series=True)
        ):
            labels = label if len(targets) > 1 else [label]
            kwargs.update({t: _native(v) for t, v in zip(targets, labels)})
            examples.append(dspy.Example(**kwargs).with_inputs(*self.input_fields_))
        return examples

    @nw.narwhalify
    def _iter_inputs(self, X):
        if isinstance(X, numpy.ndarray):
            for row in X:
                yield {f: _native(v) for f, v in zip(self.input_fields_, row)}
        else:
            columns = list(self.feature_names_in_)
            for row in X.iter_rows(named=True):
                yield {f: row[c] for f, c in zip(self.input_fields_, columns)}

    def _call_sync(self, kwargs):
        with record_evidence() as log:
            pred = self.program_(**kwargs)
        return pred, _last_evidence(log)

    async def _call_async(self, kwargs, semaphore):
        async with semaphore:
            with record_evidence() as log:
                pred = await self.program_.acall(**kwargs)
            return pred, _last_evidence(log)

    def _run_sync(self, inputs):
        if self.verbose:
            try:
                from tqdm import tqdm

                inputs = tqdm(inputs, desc="DSPyMator predicting")
            except ImportError:
                warnings.warn(
                    "tqdm not installed; progress bar unavailable.", stacklevel=3
                )
        return [self._call_sync(kwargs) for kwargs in inputs]

    async def _run_async(self, inputs):
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._call_async(kwargs, semaphore) for kwargs in inputs]
        if self.verbose:
            try:
                from tqdm.asyncio import tqdm as tqdm_asyncio

                return await tqdm_asyncio.gather(*tasks, desc="DSPyMator predicting")
            except ImportError:
                warnings.warn(
                    "tqdm not installed; progress bar unavailable.", stacklevel=3
                )
        return await asyncio.gather(*tasks)

    def _predict_raw(self, X):
        """Run the fitted program on every row: (predictions, evidence per row)."""
        inputs = list(self._iter_inputs(X))
        with self._lm_context():
            if not self.use_async:
                results = self._run_sync(inputs)
            else:
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    results = asyncio.run(self._run_async(inputs))
                else:
                    try:
                        import nest_asyncio

                        nest_asyncio.apply()
                        results = asyncio.run(self._run_async(inputs))
                    except ImportError:
                        warnings.warn(
                            "nest_asyncio not installed; falling back to synchronous. "
                            "Install nest_asyncio to use async mode in notebooks/event loops.",
                            stacklevel=2,
                        )
                        results = self._run_sync(inputs)
        return [p for p, _ in results], [e for _, e in results]


def _request_evidence(program, target, kind):
    """Decode `target` from probability evidence on every backend.

    System One backends always answer decision outputs with a distribution;
    generative LMs only do so for outputs configured in `Predict.fields`.
    """
    if issubclass(kind, Noul):
        default = {"threshold": 0.5}
    elif issubclass(kind, Score):
        default = {"cuts": [i + 0.5 for i in range(len(kind.criteria()) - 1)]}
    else:
        default = {"weights": dict.fromkeys(kind.criteria(), 1.0)}
    for _, predictor in program.named_predictors():
        if (
            target in predictor.signature.output_fields
            and target not in predictor.fields
        ):
            predictor.fields[target] = copy.deepcopy(default)


def _value(output):
    """Plain value of a program output (decision objects unwrap to `.value`)."""
    return output.value if isinstance(output, (Noul, Choice, Score)) else output


def _native(value):
    """numpy scalars to Python scalars, so DSPy types validate strictly."""
    return value.item() if isinstance(value, numpy.generic) else value


def _last_evidence(log):
    """Final evidence per output field (the last predictor to answer it wins)."""
    return {name: answer for _, name, answer in log}
