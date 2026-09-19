"""Data tests: duplicate keys, stratified nested subsets, label mapping, split integrity, no text in git."""
import json
import re
from collections import Counter

import pytest

from bangla_sentiment.data import (DATASETS, SPLITS, dedupe_key, load_raw, load_split, raw_dir, splits_dir,
                                   stratified_order, training_subset)

BANGLA_CHARS = re.compile(r"[ঀ-৿]")


def prepared(name):
    return (splits_dir(DATASETS[name]) / "splits.json").exists()


def test_dedupe_key_ignores_punctuation_digits_spacing_and_emoji():
    assert dedupe_key("ভালো বই!!! 😀 ১০/১০") == dedupe_key("ভালো  বই")
    assert dedupe_key("ভালো বই") != dedupe_key("খারাপ বই")
    assert dedupe_key("!!! 123 😀") == ""


def test_stratified_order_is_nested_stratified_and_seeded():
    labels = ["a"] * 600 + ["b"] * 300 + ["c"] * 100
    ids = [f"train-{i}" for i in range(len(labels))]
    label_of = dict(zip(ids, labels))
    order = stratified_order(ids, labels, seed=0)
    assert sorted(order) == sorted(ids)
    assert order == stratified_order(ids, labels, seed=0)
    assert order != stratified_order(ids, labels, seed=1)
    for n in (10, 50, 250, 999):
        got = Counter(label_of[i] for i in order[:n])
        for lab, share in (("a", 0.6), ("b", 0.3), ("c", 0.1)):
            assert abs(got[lab] - share * n) <= 1


@pytest.mark.parametrize("name", list(DATASETS))
def test_label_mapping_matches_published_counts(name):
    spec = DATASETS[name]
    if not (raw_dir(spec) / spec.files["train"]).exists():
        pytest.skip("raw data not downloaded; run scripts/m1_prepare_data.py")
    frames = load_raw(spec)
    observed = Counter(lab for s in SPLITS for lab in frames[s]["label"])
    assert dict(observed) == spec.published_counts


@pytest.mark.parametrize("name", list(DATASETS))
def test_committed_split_files_contain_no_text(name):
    if not prepared(name):
        pytest.skip("splits not built; run scripts/m1_prepare_data.py")
    for path in splits_dir(DATASETS[name]).glob("*.json"):
        assert not BANGLA_CHARS.search(path.read_text(encoding="utf-8")), f"text found in {path}"


@pytest.mark.parametrize("name", list(DATASETS))
def test_subsets_come_from_checked_train_split_only(name):
    if not prepared(name):
        pytest.skip("splits not built; run scripts/m1_prepare_data.py")
    spec = DATASETS[name]
    splits = json.loads((splits_dir(spec) / "splits.json").read_text(encoding="utf-8"))
    dropped = set(splits["dropped_from_train_for_leakage"])
    eval_ids = set(splits["splits"]["val"]) | set(splits["splits"]["test"])
    small = training_subset(name, spec.curve_sizes[0], seed=0)  # also verifies the train checksum
    assert len(small) == spec.curve_sizes[0]
    assert all(i.startswith("train-") for i in small["id"])
    assert not (set(small["id"]) & (dropped | eval_ids))
    for split in ("val", "test"):
        load_split(name, split)  # raises if the processed split no longer matches its checksum

    # The cleaned train is the original train minus exactly the leaked rows
    original = load_split(name, "train_original")
    train = load_split(name, "train")
    assert len(original) == splits["train_original_rows"]
    assert set(original["id"]) - set(train["id"]) == dropped
    assert set(train["id"]) <= set(original["id"])
