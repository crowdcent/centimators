import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


def save_preds(path, test, proba, classes):
    """Per-row probabilities for calibration analysis: text, label, p_<class>..."""
    cols = {
        f"p_{c}": np.asarray(proba, dtype=float)[:, i] for i, c in enumerate(classes)
    }
    pl.DataFrame({"text": test["text"], "label": test["label"], **cols}).write_parquet(
        path
    )


def calibration(y_true, proba, classes, bins=10) -> dict:
    """Top-label ECE, Brier score, and accuracy on the most confident 50% / 20% of rows."""
    proba = np.asarray(proba, dtype=float)
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_true = np.asarray(y_true)
    conf = proba.max(axis=1)
    correct = np.asarray(classes)[proba.argmax(axis=1)] == y_true
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece = sum(
        abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
        for b in range(bins)
        if (idx == b).any()
    )
    onehot = (y_true[:, None] == np.asarray(classes)[None, :]).astype(float)
    order = np.argsort(-conf)
    return {
        "ece": float(ece),
        "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()),
        "acc_top50": float(correct[order[: len(order) // 2]].mean()),
        "acc_top20": float(correct[order[: len(order) // 5]].mean()),
    }


def score(y_true, proba, classes) -> dict:
    """Accuracy, macro-F1, log loss and (macro one-vs-rest) AUC from class probabilities."""
    # sklearn's log_loss and multiclass AUC read probability columns in sorted label order.
    order = np.argsort(classes)
    classes = [list(classes)[i] for i in order]
    proba = np.clip(np.asarray(proba, dtype=float)[:, order], 1e-6, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_true = np.asarray(y_true)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    if len(classes) == 2:
        auc = roc_auc_score(y_true == classes[1], proba[:, 1])
    else:
        auc = roc_auc_score(y_true, proba, labels=classes, multi_class="ovr")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=classes, average="macro")),
        "log_loss": float(log_loss(y_true, proba, labels=classes)),
        "auc": float(auc),
    }
