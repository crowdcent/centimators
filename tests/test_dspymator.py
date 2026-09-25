from typing import Literal

import numpy as np
import pytest

dspy = pytest.importorskip("dspy")
pytest.importorskip("dspy.adapters.decision", reason="decision outputs need dspy>=3.4")

import polars as pl
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from centimators.model_estimators import DSPyMator


class FakeDecisionLM:
    """Offline stand-in for dspy.experimental.TypeSafe: scores text by keyword."""

    supports_decision_requests = True

    def __init__(self):
        self.calls = 0

    def __call__(self, state, questions):
        self.calls += 1
        text = str(state["inputs"]).lower()
        hit = "onion" in text
        answers = {}
        for name, question in questions.items():
            if question["type"] == "noul":
                answers[name] = {"noul": 0.9 if hit else 0.2}
            else:
                labels = list(question["criteria"])
                probs = {label: 0.0 for label in labels}
                probs[labels[0] if hit else labels[-1]] = 0.7
                probs[labels[-1] if hit else labels[0]] += 0.3
                answers[name] = {
                    "choice": max(probs, key=probs.get),
                    "confidence": 0.4,
                    "probabilities": probs,
                }
        return answers

    async def acall(self, state, questions):
        return self(state, questions)


class Satire(dspy.Signature):
    """Decide whether a headline is satire."""

    headline: str = dspy.InputField()
    is_satire: bool = dspy.OutputField(desc="Is this headline satire?")


class Topic(dspy.Signature):
    """Pick the section."""

    headline: str = dspy.InputField()
    section: Literal["satire", "politics", "sports"] = dspy.OutputField(
        desc="Which section?"
    )


def _data(n=12):
    headlines = [
        f"onion reports man {i} wins" if i % 2 else f"senate votes {i}"
        for i in range(n)
    ]
    X = pl.DataFrame({"headline": headlines})
    y = np.array([i % 2 == 1 for i in range(n)])
    return X, y


@pytest.mark.parametrize("use_async", [True, False])
def test_decision_lm_predict_and_proba(use_async):
    X, y = _data()
    lm = FakeDecisionLM()
    est = DSPyMator(
        program=dspy.Predict(Satire),
        target_names="is_satire",
        lm=lm,
        use_async=use_async,
        verbose=False,
    ).fit(X, y)

    assert est.lm_ is lm
    assert list(est.classes_) == [False, True]

    pred = est.predict(X)
    assert isinstance(pred, pl.DataFrame)
    assert pred["is_satire"].to_list() == y.tolist()

    proba = est.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_allclose(proba[:, 1], np.where(y, 0.9, 0.2))


def test_choice_proba_follows_classes():
    X, y = _data(4)
    est = DSPyMator(
        program=dspy.Predict(Topic),
        target_names="section",
        lm=FakeDecisionLM(),
        verbose=False,
    ).fit(X, None)

    assert list(est.classes_) == ["satire", "politics", "sports"]
    proba = est.predict_proba(X)
    assert proba.shape == (4, 3)
    assert est.predict(X)["section"].to_list() == [
        "satire" if s else "sports" for s in y
    ]
    np.testing.assert_allclose(proba[1], [0.7, 0.0, 0.3])


def test_sklearn_scorer_and_pipeline_with_decision_lm():
    X, y = _data(12)
    est = DSPyMator(
        program=dspy.Predict(Satire),
        target_names="is_satire",
        lm=FakeDecisionLM(),
        verbose=False,
    )
    pipe = make_pipeline(FunctionTransformer(lambda df: df), est)
    scores = cross_val_score(
        pipe, X, y, cv=StratifiedKFold(3), scoring="roc_auc", error_score="raise"
    )
    np.testing.assert_allclose(scores, 1.0)


def test_predict_proba_requires_decision_target():
    X, _ = _data(2)
    est = DSPyMator(
        program=dspy.Predict("headline -> summary"),
        target_names="summary",
        lm=FakeDecisionLM(),
        verbose=False,
    ).fit(X, None)
    assert not hasattr(est, "classes_")
    with pytest.raises(ValueError, match="predict_proba needs"):
        est.predict_proba(X)
