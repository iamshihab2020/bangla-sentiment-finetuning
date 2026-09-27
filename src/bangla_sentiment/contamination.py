"""A2: which evaluation rows a training split could have memorized, and what that is worth.

The near-duplicate key is the one from M1 (`data.dedupe_key`), so the contaminated slice holds
exactly the evaluation rows that the 902 leaked training rows were removed for. Splitting the test
split this way separates memorization from the two things it is otherwise confounded with: a
slightly smaller training set, and a test set that might simply be harder.
"""
import random
from collections import Counter

import pandas as pd

from bangla_sentiment.data import dedupe_key


def slice_frame(train, evaluation):
    """One row per evaluation row, saying whether `train` contains a near-duplicate of it.

    Columns added to (id, label): key, contaminated, exact_copy, train_copies, train_label
    (the most frequent label among the matching training rows, ties broken alphabetically),
    train_label_agrees.
    """
    train_keys = train["text"].map(dedupe_key)
    labels_by_key, exact_texts = {}, set(train["text"])
    for key, label in zip(train_keys, train["label"]):
        if key:
            labels_by_key.setdefault(key, []).append(label)

    rows = []
    for row_id, text, label in zip(evaluation["id"], evaluation["text"], evaluation["label"]):
        key = dedupe_key(text)
        matches = labels_by_key.get(key, []) if key else []
        counts = Counter(matches)
        train_label = min(sorted(counts), key=lambda c: (-counts[c], c)) if counts else None
        rows.append({
            "id": row_id,
            "label": label,
            "key": key,
            "contaminated": bool(matches),
            "exact_copy": text in exact_texts,
            "train_copies": len(matches),
            "train_label": train_label,
            "train_label_agrees": train_label == label if train_label is not None else None,
        })
    return pd.DataFrame(rows)


def slice_summary(frame):
    """Counts that describe the split: sizes, class balance and how often the copies agree."""
    contaminated = frame[frame["contaminated"]]
    return {
        "rows": len(frame),
        "contaminated": {
            "rows": int(len(contaminated)),
            "share": round(float(len(contaminated) / len(frame)), 4),
            "exact_copies": int(contaminated["exact_copy"].sum()),
            "near_only": int((~contaminated["exact_copy"]).sum()),
            "training_copies_total": int(contaminated["train_copies"].sum()),
            "copy_label_agrees": int(contaminated["train_label_agrees"].sum()),
            "class_counts": {k: int(v) for k, v in contaminated["label"].value_counts().sort_index().items()},
        },
        "clean": {
            "rows": int((~frame["contaminated"]).sum()),
            "class_counts": {k: int(v) for k, v in
                             frame.loc[~frame["contaminated"], "label"].value_counts().sort_index().items()},
        },
        "rows_without_letters": int((frame["key"] == "").sum()),
    }


def random_removal_subset(original, cleaned, seed):
    """`original` minus as many randomly chosen rows per class as the leakage removal dropped.

    The control for A2. It has the same number of rows and the same class balance as the cleaned
    train split, so the only difference is which rows went: random ones instead of the ones that
    have a near-duplicate in val or test. If training on this costs about as much as training on
    the cleaned split, the drop was never about leakage, only about the smaller training set.
    """
    removed = Counter(original.loc[~original["id"].isin(set(cleaned["id"])), "label"])
    rng = random.Random(seed)
    keep = set()
    for label in sorted(set(original["label"])):
        ids = sorted(original.loc[original["label"] == label, "id"])
        rng.shuffle(ids)
        keep |= set(ids[removed.get(label, 0):])
    subset = original[original["id"].isin(keep)].reset_index(drop=True)
    return subset, {"rows_removed_per_class": dict(sorted(removed.items())), "rows_kept": len(subset)}
