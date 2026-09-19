"""M1: download the datasets, normalize, check duplicates and leakage, build nested subsets, audit.

Usage:
    uv run python scripts/m1_prepare_data.py

Writes (gitignored, contain dataset text):
    data/raw/<name>/                       original files at the pinned revision
    data/processed/<name>/<split>.jsonl    id, text (normalized), text_raw, label; train has leaked rows removed
    data/processed/<name>/train_original.jsonl   train with leaks, only for the M2 check against published numbers
    data/processed/sentnob/label_check_50.csv (+ _key.csv)   blind sheet for the manual label check
Writes (committed, IDs and checksums only):
    data/splits/<name>/splits.json, data/splits/<name>/train_order.json
Writes results/m1_data_audit.json.
"""
import argparse

import pandas as pd

from bangla_sentiment.data import (DATASETS, SEEDS, SPLITS, clean_text, download, duplicate_report,
                                   length_stats, load_raw, processed_dir, split_checksum, splits_dir,
                                   stratified_order, write_json, write_jsonl)
from bangla_sentiment.utils import ensure_utf8, save_result

LICENSES = {
    "sentnob": "CC BY-ND 4.0 per the Hugging Face dataset card. The GitHub repo (KhondokerIslam/SentNoB) "
               "has no license file. No text, raw or processed, may be committed.",
    "bengali_sa": "MIT per the Hugging Face card. The original repo (sazzadcsedu/BN-Dataset) states no "
                  "license, so treat it as research use only.",
}
LABEL_CHECK_SIZE = 50
LABEL_CHECK_SEED = 2026


def counts(series):
    return {k: int(v) for k, v in series.value_counts().sort_index().items()}


def prepare(spec):
    file_sha256 = download(spec)
    frames = load_raw(spec)
    rows_raw = {s: len(frames[s]) for s in SPLITS}
    for frame in frames.values():
        frame["text"] = frame["text_raw"].map(clean_text)

    observed = counts(pd.concat(frames.values())["label"])
    dup, leaked = duplicate_report(frames)
    # The original train split (leaks included) is kept only for the M2 check against published numbers
    frames["train_original"] = frames["train"]
    frames["train"] = frames["train"][~frames["train"]["id"].isin(leaked)].reset_index(drop=True)

    for split in (*SPLITS, "train_original"):
        write_jsonl(frames[split][["id", "text", "text_raw", "label"]], processed_dir(spec) / f"{split}.jsonl")

    train = frames["train"]
    orders = {f"seed_{s}": stratified_order(train["id"].tolist(), train["label"].tolist(), s) for s in SEEDS}
    source = {"repo_id": spec.repo_id, "revision": spec.revision, "file_sha256": file_sha256}
    write_json({
        "source": source,
        "label_names": {str(k): v for k, v in spec.label_names.items()},
        "splits": {s: frames[s]["id"].tolist() for s in SPLITS},
        "dropped_from_train_for_leakage": leaked,
        "train_original_rows": len(frames["train_original"]),
        "checksums": {s: split_checksum(frames[s]) for s in (*SPLITS, "train_original")},
    }, splits_dir(spec) / "splits.json")
    write_json({
        "note": "Stratified, nested orderings of the train split. A subset of size n is the first n IDs.",
        **orders,
    }, splits_dir(spec) / "train_order.json")

    label_of = dict(zip(train["id"], train["label"]))
    sizes = [n or len(train) for n in spec.curve_sizes]
    audit = {
        "source": source,
        "license": LICENSES[spec.name],
        "rows_raw": rows_raw,
        "rows_final": {s: len(frames[s]) for s in SPLITS},
        "class_counts": {s: counts(frames[s]["label"]) for s in SPLITS},
        "label_mapping": {
            "raw_to_name": {str(k): v for k, v in spec.label_names.items()},
            "published_counts": spec.published_counts,
            "observed_counts": observed,
            "matches_published": observed == spec.published_counts,
        },
        "normalizer_changed_share": {s: round(float((frames[s]["text"] != frames[s]["text_raw"]).mean()), 4)
                                     for s in SPLITS},
        "words_per_text": {s: length_stats(frames[s]["text"].str.split().str.len()) for s in SPLITS},
        "chars_per_text": {s: length_stats(frames[s]["text"].str.len()) for s in SPLITS},
        "duplicates": dup,
        "dropped_from_train_for_leakage": len(leaked),
        "curve_sizes": sizes,
        "subset_class_counts_seed_0": {str(n): counts(pd.Series([label_of[i] for i in orders["seed_0"][:n]]))
                                       for n in sizes},
    }
    return audit, frames


def write_label_check(spec, train):
    """Blind sheet for the manual label check: Shihab fills your_label without seeing the gold label."""
    sample = train.sample(n=LABEL_CHECK_SIZE, random_state=LABEL_CHECK_SEED)
    out = processed_dir(spec)
    blind = sample[["id", "text_raw"]].rename(columns={"text_raw": "text"}).assign(your_label="", notes="")
    blind.to_csv(out / "label_check_50.csv", index=False, encoding="utf-8-sig")  # utf-8-sig opens in Excel
    sample[["id", "label"]].rename(columns={"label": "gold_label"}).to_csv(
        out / "label_check_50_key.csv", index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args()
    ensure_utf8()

    audits = {}
    for name in args.datasets:
        spec = DATASETS[name]
        audits[name], frames = prepare(spec)
        if name == "sentnob":
            write_label_check(spec, frames["train"])
        a = audits[name]
        print(f"{name}: rows {a['rows_final']}, label mapping ok={a['label_mapping']['matches_published']}, "
              f"dropped from train for leakage={a['dropped_from_train_for_leakage']}")

    path = save_result({"datasets": audits}, "m1_data_audit")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
