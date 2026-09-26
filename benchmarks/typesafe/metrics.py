import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


def score(y_true, proba, classes) -> dict:
    """Accuracy, macro-F1, log loss and (macro one-vs-rest) AUC from class probabilities."""
    classes = list(classes)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_true = np.asarray(y_true)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    if len(classes) == 2:
        auc = roc_auc_score(y_true == classes[1], proba[:, 1])
    else:
        order = np.argsort(classes)  # sklearn's multiclass AUC wants sorted labels
        auc = roc_auc_score(
            y_true,
            proba[:, order],
            labels=[classes[i] for i in order],
            multi_class="ovr",
        )
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=classes, average="macro")),
        "log_loss": float(log_loss(y_true, proba, labels=classes)),
        "auc": float(auc),
    }
