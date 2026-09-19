"""Metrics and saved predictions shared by every experiment.

Scores are percentages with 2 decimals, the same scale as the published reference numbers.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def pct(x):
    return round(100 * float(x), 2)


def classification_metrics(gold, pred, classes):
    """Macro-F1 (primary), micro-F1, accuracy, per-class scores and the confusion matrix.

    `classes` fixes the order of the per-class scores and of the matrix rows (gold) and columns (predicted).
    A class that is never predicted still counts in macro-F1, with F1 = 0.
    """
    precision, recall, f1, support = precision_recall_fscore_support(gold, pred, labels=classes, zero_division=0)
    return {
        "macro_f1": pct(f1_score(gold, pred, labels=classes, average="macro", zero_division=0)),
        "micro_f1": pct(f1_score(gold, pred, labels=classes, average="micro", zero_division=0)),
        "accuracy": pct(accuracy_score(gold, pred)),
        "per_class": {c: {"precision": pct(p), "recall": pct(r), "f1": pct(f), "support": int(s)}
                      for c, p, r, f, s in zip(classes, precision, recall, f1, support)},
        "confusion_matrix": {"rows": "gold", "columns": "predicted", "labels": list(classes),
                             "values": confusion_matrix(gold, pred, labels=classes).tolist()},
    }


def save_predictions(ids, gold, pred, proba, classes, path):
    """Predictions without text: id, gold, predicted label and one probability column per class."""
    frame = pd.DataFrame({"id": list(ids), "gold": list(gold), "pred": list(pred)})
    for k, c in enumerate(classes):
        frame[f"prob_{c}"] = np.round(proba[:, k], 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
