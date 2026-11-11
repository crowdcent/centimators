"""DSPyMator tests against a deterministic fake decision engine (no network)."""

from typing import Literal

import numpy as np
import polars as pl
import pytest

dspy = pytest.importorskip("dspy")

from dspy.experimental import ReAnchor, Score  # noqa: E402
from dspy.lm15 import Message, Response, Usage  # noqa: E402
from sklearn.base import clone, is_classifier, is_regressor  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score  # noqa: E402

from centimators.model_estimators import DSPyMator  # noqa: E402

BULLISH = ("beat", "record", "surge", "upgrade")
BEARISH = ("miss", "cut", "plunge", "downgrade")


class FakeJudge:
    """Scores headlines by keyword, the way a System One backend answers."""

    supports_decision_requests = True

    def __init__(self, sharpness=0.9):
        self.sharpness = sharpness
        self.calls = 0

    def _lean(self, text):
        text = text.lower()
        return sum(w in text for w in BULLISH) - sum(w in text for w in BEARISH)

    def complete(self, request):
        self.calls += 1
        text = request.messages[0].parts[0].value["inputs"]["headline"]
        lean = self._lean(text)
        hi, lo = self.sharpness, 1 - self.sharpness
        answers = {}
        for name, prop in request.config.response_format["schema"][
            "properties"
        ].items():
            if prop.get("type") == "boolean":
                answers[name] = {
                    "type": "noul",
                    "noul": hi if lean > 0 else lo,
                    "confidence": 0.9,
                }
            elif prop.get("type") == "integer":
                keys = [str(b["const"]) for b in prop["anyOf"]]
                probs = (
                    [0.1, 0.2, 0.7]
                    if lean > 0
                    else [0.7, 0.2, 0.1]
                    if lean < 0
                    else [0.2, 0.6, 0.2]
                )
                answers[name] = {
                    "type": "score",
                    "probabilities": dict(zip(keys, probs)),
                    "confidence": 0.9,
                }
            else:
                keys = prop["enum"]
                side = "bullish" if lean > 0 else "bearish" if lean < 0 else "neutral"
                rest = (1 - hi) / (len(keys) - 1)
                probs = {k: hi if k == side else rest for k in keys}
                answers[name] = {
                    "type": "choice",
                    "probabilities": probs,
                    "confidence": 0.9,
                }
        return Response(
            id="fake",
            model=request.model,
            message=Message.assistant("{}"),
            finish_reason="stop",
            usage=Usage(),
            provider_data={"typesafe": {"answers": answers}},
        )


class IsBullish(dspy.Signature):
    """Is this financial headline bullish for the stock?"""

    headline: str = dspy.InputField()
    bullish: bool = dspy.OutputField(desc="Is the headline bullish?")


class Sentiment(dspy.Signature):
    """Classify the market sentiment of a financial headline."""

    headline: str = dspy.InputField()
    sentiment: Literal["bearish", "neutral", "bullish"] = dspy.OutputField(
        desc="Market sentiment."
    )


ImpactLevel = Score["Down", "Flat", "Up"]


class Impact(dspy.Signature):
    """How strongly does this headline move the stock up?"""

    headline: str = dspy.InputField()
    impact: ImpactLevel = dspy.OutputField(desc="Price impact.")


HEADLINES = [
    "Acme posts record quarter, shares surge",
    "Analysts upgrade Acme after earnings beat",
    "Acme beats estimates on cloud growth",
    "Record bookings lift Acme",
    "Acme misses revenue estimates",
    "Acme cuts full-year guidance",
    "Shares plunge after downgrade",
    "Acme plunges on guidance cut",
    "Acme holds annual meeting",
    "Acme names new CFO",
]
SENTIMENT = ["bullish"] * 4 + ["bearish"] * 4 + ["neutral"] * 2


class AsyncFakeJudge:
    supports_decision_requests = True

    def __init__(self, judge):
        self.judge = judge

    async def complete(self, request):
        return self.judge.complete(request)


def make_lm(engine=None):
    engine = engine or FakeJudge()
    return dspy.LM(
        "fake/judge", engine=engine, async_engine=AsyncFakeJudge(engine), cache=False
    )


@pytest.fixture
def X():
    return pl.DataFrame({"headline": HEADLINES})


def binary_clf(**kwargs):
    return DSPyMator(
        program=dspy.Predict(IsBullish),
        target_names="bullish",
        lm=make_lm(),
        verbose=False,
        **kwargs,
    )


def test_binary_classifier_contract(X):
    clf = binary_clf().fit(X)
    assert is_classifier(clf)
    np.testing.assert_array_equal(clf.classes_, [False, True])
    pred = clf.predict(X)
    assert pred.dtype == bool and pred.shape == (10,)
    np.testing.assert_array_equal(pred, [True] * 4 + [False] * 6)
    proba = clf.predict_proba(X)
    assert proba.shape == (10, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_allclose(proba[0], [0.1, 0.9])


def test_multiclass_classifier_contract(X):
    clf = DSPyMator(
        program=dspy.Predict(Sentiment),
        target_names="sentiment",
        lm=make_lm(),
        verbose=False,
    ).fit(X)
    assert is_classifier(clf)
    assert list(clf.classes_) == ["bearish", "bullish", "neutral"]
    np.testing.assert_array_equal(clf.predict(X), SENTIMENT)
    proba = clf.predict_proba(X)
    assert proba.shape == (10, 3)
    assert clf.classes_[proba[0].argmax()] == "bullish"
    assert clf.score(X, SENTIMENT) == 1.0


def test_score_target_is_regressor(X):
    clf = DSPyMator(
        program=dspy.Predict(Impact), target_names="impact", lm=make_lm(), verbose=False
    ).fit(X)
    assert is_regressor(clf)
    pred = clf.predict(X)
    assert pred.dtype == float
    assert pred[0] == pytest.approx(1.6) and pred[4] == pytest.approx(0.4)
    assert not hasattr(clf, "predict_proba")


def test_sync_and_async_agree(X):
    a = binary_clf(use_async=True).fit(X).predict_proba(X)
    b = binary_clf(use_async=False).fit(X).predict_proba(X)
    np.testing.assert_allclose(a, b)


def test_numpy_input_maps_by_position():
    X = np.array(HEADLINES, dtype=object).reshape(-1, 1)
    clf = binary_clf().fit(X)
    assert list(clf.feature_names_in_) == ["headline"]
    np.testing.assert_array_equal(clf.predict(X)[:5], [True] * 4 + [False])


def test_feature_names_map_columns():
    X = pl.DataFrame({"title": HEADLINES})
    clf = binary_clf(feature_names=["title"]).fit(X)
    assert clf.predict(X)[0]


def test_fit_does_not_modify_params(X):
    program = dspy.Predict(IsBullish)
    optimizer = ReAnchor(
        lambda ex, pred, trace=None: float(pred.bullish == ex.bullish),
        require_cache=False,
    )
    clf = DSPyMator(
        program=program,
        target_names="bullish",
        lm=make_lm(),
        optimizer=optimizer,
        verbose=False,
    )
    y = np.array([True] * 4 + [False] * 6)
    clf.fit(X, y)
    assert clf.program is program and clf.optimizer is optimizer
    assert clf.program_ is not program
    assert clf.feature_names is None
    assert "bullish" not in program.fields


def test_fit_requests_evidence_for_generative_lms(X):
    clf = DSPyMator(
        program=dspy.Predict(Sentiment),
        target_names="sentiment",
        lm=make_lm(),
        verbose=False,
    ).fit(X)
    assert clf.program_.fields["sentiment"] == {
        "weights": {"bearish": 1.0, "neutral": 1.0, "bullish": 1.0}
    }
    cot = DSPyMator(
        program=dspy.ChainOfThought(IsBullish),
        target_names="bullish",
        lm=make_lm(),
        verbose=False,
    )
    assert cot.fit(X).program_.predict.fields["bullish"] == {"threshold": 0.5}


def test_clone_with_optimizer_instance(X):
    optimizer = ReAnchor(lambda ex, pred, trace=None: 1.0, require_cache=False)
    clf = binary_clf(optimizer=optimizer)
    copy = clone(clf)
    assert copy.optimizer is not optimizer and isinstance(copy.optimizer, ReAnchor)
    assert copy.program is clf.program
    assert not copy.__sklearn_is_fitted__()


def test_cross_val_score_log_loss(X):
    y = np.array([True] * 4 + [False] * 6)
    scores = cross_val_score(binary_clf(), X, y, cv=2, scoring="neg_log_loss")
    assert scores.shape == (2,) and np.all(scores < 0)
    auc = cross_val_score(binary_clf(), X, y, cv=2, scoring="roc_auc")
    np.testing.assert_allclose(auc, 1.0)


def test_grid_search_over_lms(X):
    y = np.array([True] * 4 + [False] * 6)
    sharp, blunt = make_lm(FakeJudge(0.95)), make_lm(FakeJudge(0.6))
    search = GridSearchCV(
        binary_clf(), {"lm": [blunt, sharp]}, cv=2, scoring="neg_log_loss"
    ).fit(X, y)
    assert search.best_params_["lm"] is sharp


def test_cross_val_predict_proba_and_calibration(X):
    y = np.array([True] * 4 + [False] * 6)
    oof = cross_val_predict(binary_clf(), X, y, cv=2, method="predict_proba")
    assert oof.shape == (10, 2)
    calibrated = CalibratedClassifierCV(binary_clf(), cv=2, method="sigmoid").fit(X, y)
    assert calibrated.predict_proba(X).shape == (10, 2)


def test_stacking_with_classic_model(X):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import StackingClassifier
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    y = np.array([True] * 4 + [False] * 6)
    text = X["headline"].to_numpy().reshape(-1, 1)
    bow = make_pipeline(
        ColumnTransformer([("words", CountVectorizer(), 0)]), LogisticRegression()
    )
    stack = StackingClassifier(
        [("llm", binary_clf()), ("bow", bow)],
        final_estimator=LogisticRegression(),
        cv=2,
    )
    assert stack.fit(text, y).predict_proba(text).shape == (10, 2)


def test_reanchor_runs_through_fit(X):
    y = np.array([True] * 4 + [False] * 6)
    optimizer = ReAnchor(
        lambda ex, pred, trace=None: float(pred.bullish == ex.bullish),
        require_cache=False,
    )
    clf = binary_clf(optimizer=optimizer).fit(X, y)
    assert clf.score(X, y) == 1.0


def test_stacking_with_optimizer(X):
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression

    y = np.array([True] * 4 + [False] * 6)
    optimizer = ReAnchor(
        lambda ex, pred, trace=None: float(pred.bullish == ex.bullish),
        require_cache=False,
    )
    stack = StackingClassifier(
        [("llm", binary_clf(optimizer=optimizer))],
        final_estimator=LogisticRegression(),
        cv=2,
    ).fit(X, y)
    assert stack.predict_proba(X).shape == (10, 2)
    assert stack.estimators_[0].optimizer is not optimizer


def test_transform_returns_values(X):
    clf = DSPyMator(
        program=dspy.Predict(Sentiment),
        target_names="sentiment",
        lm=make_lm(),
        verbose=False,
    ).fit(X)
    out = clf.transform(X)
    assert isinstance(out, pl.DataFrame)
    assert out["sentiment"].to_list() == SENTIMENT
    assert list(clf.get_feature_names_out()) == ["sentiment"]


def test_generative_target_is_plain_transformer():
    lm = dspy.utils.DummyLM([{"summary": "short"}] * 2)
    clf = DSPyMator(
        program=dspy.Predict("text -> summary"),
        target_names="summary",
        lm=lm,
        use_async=False,
        verbose=False,
    )
    X = pl.DataFrame({"text": ["a", "b"]})
    clf.fit(X)
    assert not is_classifier(clf) and not hasattr(clf, "classes_")
    np.testing.assert_array_equal(clf.predict(X), ["short", "short"])
    assert not hasattr(clf, "predict_proba") and not hasattr(clf, "score")


def test_unfitted_and_bad_target(X):
    with pytest.raises(ValueError, match="not fitted"):
        binary_clf().predict(X)
    with pytest.raises(ValueError, match="not output fields"):
        DSPyMator(
            program=dspy.Predict(IsBullish), target_names="nope", lm=make_lm()
        ).fit(X)
