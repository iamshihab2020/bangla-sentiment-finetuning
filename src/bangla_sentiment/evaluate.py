"""Metrics and saved predictions shared by every experiment.

Scores are percentages with 2 decimals, the same scale as the published reference numbers.
"""
import statistics

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


def label_codes(labels, classes):
    """Labels as integer class indices, in the order given by `classes`."""
    index = {c: i for i, c in enumerate(classes)}
    return np.fromiter((index[lab] for lab in labels), dtype=np.int64, count=len(labels))


def macro_f1_from_codes(gold, pred, n_classes):
    """Macro-F1 (percentage) from integer codes, counted with bincount.

    Same convention as classification_metrics: a class that is never predicted and never appears
    scores 0, as with sklearn's zero_division=0. Used by the bootstrap, which calls it thousands
    of times and cannot afford sklearn's input checking.
    """
    matrix = np.bincount(gold * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(matrix).astype(float)
    denominator = 2 * tp + (matrix.sum(axis=0) - tp) + (matrix.sum(axis=1) - tp)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(n_classes), where=denominator > 0)
    return 100 * float(f1.mean())


def bootstrap_differences(gold, preds_a, preds_b, classes, n_resamples=5000, seed=0):
    """Seed-averaged paired bootstrap of the macro-F1 difference (A minus B) on the same items.

    `preds_a` and `preds_b` hold one prediction sequence per training seed, all aligned with `gold`.
    Every resample draws items with replacement and is then used for both systems and all of their
    seeds, so items and seeds are paired and only the systems differ.

    Returns (observed difference, array of one difference per resample).
    """
    n_classes = len(classes)
    gold = label_codes(gold, classes)
    a = [label_codes(p, classes) for p in preds_a]
    b = [label_codes(p, classes) for p in preds_b]

    def difference(idx):
        g = gold[idx]
        return (statistics.mean(macro_f1_from_codes(g, p[idx], n_classes) for p in a)
                - statistics.mean(macro_f1_from_codes(g, p[idx], n_classes) for p in b))

    rng = np.random.default_rng(seed)
    observed = difference(np.arange(len(gold)))
    draws = np.array([difference(rng.integers(0, len(gold), len(gold))) for _ in range(n_resamples)])
    return observed, draws


def summarize_differences(observed, draws):
    """Percentile interval and the share of resamples on the other side of zero."""
    share_le_zero = float((draws <= 0).mean())
    share_ge_zero = float((draws >= 0).mean())  # both are 1.0 when every difference is exactly zero
    return {
        "difference": round(float(observed), 2),
        "bootstrap_mean": round(float(draws.mean()), 2),
        "ci95": [round(float(np.percentile(draws, 2.5)), 2), round(float(np.percentile(draws, 97.5)), 2)],
        "share_of_resamples_le_zero": round(share_le_zero, 4),
        "p_two_sided": round(min(1.0, 2 * min(share_le_zero, share_ge_zero)), 4),
        "resamples": int(len(draws)),
    }
