"""M1: compare Shihab's blind labels on 50 SentNoB train comments with the dataset's labels.

Usage:
    uv run python scripts/m1_label_check.py

Reads data/processed/sentnob/label_check_50.csv (your_label filled in by hand) and the answer key
label_check_50_key.csv. Writes results/m1_label_check.json with IDs and labels only, no text.
"""
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from bangla_sentiment.data import DATASETS, processed_dir
from bangla_sentiment.utils import ensure_utf8, save_result

SHORT = {"pos": "positive", "neg": "negative", "neu": "neutral"}


def read_sheet(path, classes):
    """Your labels by ID, plus the rows without a valid label. Accepts full names or pos/neg/neu, any case."""
    # Only id and your_label are needed, so text damaged by a non-UTF-8 save in Excel does not matter
    sheet = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace", dtype=str, keep_default_na=False)
    labels = sheet["your_label"].str.strip().str.lower().replace(SHORT)
    bad = sheet.loc[~labels.isin(classes), ["id", "your_label"]]
    bad.index = bad.index + 2  # Excel row number: header is row 1
    return dict(zip(sheet["id"], labels)), bad


def compare(gold, yours, classes):
    """Agreement, Cohen's kappa and the confusion matrix (rows: dataset label, columns: your label)."""
    ids = list(gold)
    g = [gold[i] for i in ids]
    y = [yours[i] for i in ids]
    return {
        "n": len(ids),
        "agreement": round(sum(a == b for a, b in zip(g, y)) / len(ids), 4),
        "cohen_kappa": round(float(cohen_kappa_score(g, y, labels=classes)), 4),
        "classes": classes,
        "confusion_matrix": {"rows": "dataset label", "columns": "your label",
                             "values": confusion_matrix(g, y, labels=classes).tolist()},
        "disagreements": [{"id": i, "dataset": gold[i], "yours": yours[i]} for i in ids if gold[i] != yours[i]],
    }


def main():
    ensure_utf8()
    spec = DATASETS["sentnob"]
    classes = list(spec.label_names.values())
    folder = processed_dir(spec)

    yours, bad = read_sheet(folder / "label_check_50.csv", classes)
    if len(bad):
        print(f"{len(bad)} of {len(yours)} rows have no valid label yet. Use: {', '.join(classes)} (or pos, neg, neu).")
        print(bad.head(10).to_string(index_names=False))
        sys.exit(1)
    key = pd.read_csv(folder / "label_check_50_key.csv", encoding="utf-8-sig", dtype=str)
    gold = dict(zip(key["id"], key["gold_label"]))
    if set(gold) != set(yours):
        sys.exit("The IDs in the sheet no longer match the answer key. Was a row added or deleted?")

    result = compare(gold, yours, classes)
    print(f"agreement {result['agreement']:.0%} ({result['n'] - len(result['disagreements'])} of {result['n']}), "
          f"Cohen's kappa {result['cohen_kappa']}")
    matrix = pd.DataFrame(result["confusion_matrix"]["values"], index=classes, columns=classes)
    print("rows: dataset label, columns: your label")
    print(matrix.to_string())
    path = save_result({"dataset": "sentnob", "split": "train", **result}, "m1_label_check")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
