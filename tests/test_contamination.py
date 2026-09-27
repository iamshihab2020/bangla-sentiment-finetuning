"""A2: the fast macro-F1 and the paired bootstrap, and the contaminated-slice split."""
import numpy as np
import pandas as pd

from bangla_sentiment.contamination import random_removal_subset, slice_frame, slice_summary
from bangla_sentiment.evaluate import (bootstrap_differences, classification_metrics, label_codes,
                                       macro_f1_from_codes, summarize_differences)

CLASSES = ["neutral", "positive", "negative"]


def frame(ids, texts, labels):
    return pd.DataFrame({"id": ids, "text": texts, "label": labels})


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    gold = [CLASSES[i] for i in rng.integers(0, 3, 200)]
    pred = [CLASSES[i] for i in rng.integers(0, 2, 200)]  # never predicts "negative", so its F1 is 0
    expected = classification_metrics(gold, pred, CLASSES)["macro_f1"]
    actual = macro_f1_from_codes(label_codes(gold, CLASSES), label_codes(pred, CLASSES), len(CLASSES))
    assert round(actual, 2) == expected


def test_bootstrap_difference_is_zero_for_identical_systems():
    gold = ["positive", "negative", "neutral"] * 10
    pred = ["positive", "negative", "positive"] * 10
    observed, draws = bootstrap_differences(gold, [pred], [pred], CLASSES, n_resamples=50, seed=0)
    assert observed == 0.0 and not draws.any()


def test_bootstrap_averages_over_seeds_and_pairs_the_items():
    gold = ["positive"] * 8 + ["negative"] * 8
    perfect = list(gold)
    half = ["positive"] * 16  # every negative row wrong
    observed, draws = bootstrap_differences(gold, [perfect, perfect], [perfect, half], CLASSES,
                                            n_resamples=200, seed=0)
    # A is perfect in both seeds; B averages one perfect run with one that never says "negative".
    assert observed > 0
    summary = summarize_differences(observed, draws)
    assert summary["ci95"][0] <= summary["bootstrap_mean"] <= summary["ci95"][1]
    assert summary["resamples"] == 200


def test_contaminated_slice_ignores_punctuation_and_case():
    train = frame(["t-0", "t-1"], ["ভালো লাগলো!!", "Nice One"], ["positive", "positive"])
    test = frame(["test-0", "test-1", "test-2"], ["ভালো, লাগলো", "nice one.", "অন্য কিছু"],
                 ["positive", "negative", "neutral"])
    result = slice_frame(train, test).set_index("id")
    assert result.loc["test-0", "contaminated"] and not result.loc["test-0", "exact_copy"]
    assert result.loc["test-1", "contaminated"] and result.loc["test-1", "train_label"] == "positive"
    assert not result.loc["test-1", "train_label_agrees"]  # the copy carries a different label
    assert not result.loc["test-2", "contaminated"]


def test_summary_counts_copies_and_rows_without_letters():
    train = frame(["t-0", "t-1"], ["ভালো", "ভালো"], ["positive", "positive"])
    test = frame(["test-0", "test-1"], ["ভালো", "!!!"], ["positive", "neutral"])
    summary = slice_summary(slice_frame(train, test))
    assert summary["contaminated"]["rows"] == 1
    assert summary["contaminated"]["training_copies_total"] == 2
    assert summary["contaminated"]["copy_label_agrees"] == 1
    assert summary["clean"]["rows"] == 1 and summary["rows_without_letters"] == 1


def test_random_removal_matches_the_leakage_removal_per_class():
    original = frame([f"t-{i}" for i in range(10)], [f"text {i}" for i in range(10)],
                     ["positive"] * 6 + ["negative"] * 4)
    cleaned = original[~original["id"].isin(["t-0", "t-1", "t-6"])]  # 2 positive, 1 negative removed
    subset, info = random_removal_subset(original, cleaned, seed=0)
    assert info["rows_removed_per_class"] == {"negative": 1, "positive": 2}
    assert len(subset) == len(cleaned)
    assert dict(subset["label"].value_counts()) == dict(cleaned["label"].value_counts())
    assert set(subset["id"]) != set(cleaned["id"])  # different rows went, with this seed
    assert set(subset["id"]) == set(random_removal_subset(original, cleaned, seed=0)[0]["id"])  # seeded
