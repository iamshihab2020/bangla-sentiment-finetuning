"""Datasets: download, normalization, duplicate and leakage checks, and nested training subsets.

Dataset text stays in data/raw/ and data/processed/ (both gitignored, see data/README.md).
Only row IDs, labels and checksums are written to data/splits/, which is committed.
"""
import hashlib
import json
import random
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from normalizer import normalize

from bangla_sentiment.utils import DATA_DIR

SPLITS = ("train", "val", "test")
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str
    revision: str  # pinned Hugging Face dataset commit
    files: dict  # split -> file name in the repo
    text_col: str
    label_col: str
    label_names: dict  # raw integer label -> label name
    published_counts: dict  # label name -> count over all splits, from the dataset's source
    curve_sizes: tuple  # learning-curve training sizes; None means the full train split


DATASETS = {
    "sentnob": DatasetSpec(
        name="sentnob",
        repo_id="khondoker/SentNoB",
        revision="7f9ab8ccd02457b79cd15d300fc1f0eece47b81b",
        files={"train": "Train.csv", "val": "Val.csv", "test": "Test.csv"},
        text_col="Data",
        label_col="Label",
        label_names={0: "neutral", 1: "positive", 2: "negative"},
        published_counts={"positive": 6410, "negative": 5709, "neutral": 3609},  # SentNoB paper, Table 2
        curve_sizes=(250, 500, 1000, 2000, 4000, None),
    ),
    "bengali_sa": DatasetSpec(
        name="bengali_sa",
        repo_id="DGurgurov/bengali_sa",
        revision="11d6e7ff3404e626640840b17af5f86fcec467b7",
        files={"train": "train.csv", "val": "dev.csv", "test": "test.csv"},
        text_col="text",
        label_col="label",
        label_names={0: "negative", 1: "positive"},
        published_counts={"positive": 8500, "negative": 3307},  # sazzadcsedu/BN-Dataset README
        curve_sizes=(250, 1000, None),
    ),
}


def raw_dir(spec):
    return DATA_DIR / "raw" / spec.name


def processed_dir(spec):
    return DATA_DIR / "processed" / spec.name


def splits_dir(spec):
    return DATA_DIR / "splits" / spec.name


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(spec):
    """Download the pinned revision into data/raw/<name>/ and return {file name: sha256}."""
    checksums = {}
    for fname in spec.files.values():
        path = hf_hub_download(spec.repo_id, fname, repo_type="dataset", revision=spec.revision,
                               local_dir=raw_dir(spec))
        checksums[fname] = sha256_file(path)
    return checksums


def load_raw(spec):
    """Original rows per split, with stable IDs '<split>-<row>' and label names. Text is unmodified."""
    frames = {}
    for split, fname in spec.files.items():
        df = pd.read_csv(raw_dir(spec) / fname)
        unknown = set(df[spec.label_col]) - set(spec.label_names)
        if unknown:
            raise ValueError(f"{spec.name}/{fname}: unknown labels {sorted(unknown)}")
        frames[split] = pd.DataFrame({
            "id": [f"{split}-{i}" for i in range(len(df))],
            "text_raw": df[spec.text_col].astype(str),
            "label": df[spec.label_col].map(spec.label_names),
        })
    return frames


def clean_text(text):
    """The one preprocessing pipeline for every model: BanglaBERT normalizer, then whitespace."""
    return " ".join(normalize(text).split())


def dedupe_key(text):
    """Near-duplicate key: letters and combining marks only, case-folded.

    Ignores punctuation, digits, spacing, emoji and symbols. Empty for texts without letters.
    """
    text = unicodedata.normalize("NFC", text).casefold()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LM")


def duplicate_report(frames):
    """Exact and near-duplicate counts within and across splits.

    Returns (report, leaked_train_ids): train rows whose near-duplicate key also appears in val or
    test. Val and test are never changed, so published numbers stay comparable.
    """
    report = {}
    for kind, keyfn in (("exact", lambda t: t), ("near", dedupe_key)):
        keys = {s: frames[s]["text"].map(keyfn) for s in SPLITS}
        within = {s: int(keys[s][keys[s] != ""].duplicated().sum()) for s in SPLITS}
        cross = {
            f"{a}_rows_also_in_{b}": int(keys[a].isin(set(keys[b]) - {""}).sum())
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        }
        train = frames["train"].assign(key=keys["train"])
        train = train[train["key"] != ""]
        conflicting = int((train.groupby("key")["label"].nunique() > 1).sum())
        report[kind] = {
            "extra_copies_within_split": within,
            "cross_split": cross,
            "train_duplicate_groups_with_conflicting_labels": conflicting,
        }
    report["texts_without_letters"] = {s: int((frames[s]["text"].map(dedupe_key) == "").sum()) for s in SPLITS}

    eval_keys = (set(frames["val"]["text"].map(dedupe_key)) | set(frames["test"]["text"].map(dedupe_key))) - {""}
    leaked = frames["train"]["text"].map(dedupe_key).isin(eval_keys)
    return report, frames["train"].loc[leaked, "id"].tolist()


def stratified_order(ids, labels, seed):
    """A seeded ordering of IDs in which every prefix is class-stratified (within 1 per class).

    Items are shuffled within each class, then interleaved by relative position (k + 0.5) / n_class.
    Taking the first n IDs gives a stratified subset, and smaller subsets are nested in larger ones.
    """
    rng = random.Random(seed)
    by_class = {}
    for i, lab in zip(ids, labels):
        by_class.setdefault(lab, []).append(i)
    keyed = []
    for lab in sorted(by_class):
        members = by_class[lab][:]
        rng.shuffle(members)
        keyed += [((k + 0.5) / len(members), rng.random(), m) for k, m in enumerate(members)]
    keyed.sort()
    return [m for _, _, m in keyed]


def split_checksum(frame):
    """Checksum over (id, label, sha1 of the original text); detects any change to a split."""
    lines = (f"{i}\t{lab}\t{hashlib.sha1(t.encode('utf-8')).hexdigest()}"
             for i, lab, t in zip(frame["id"], frame["label"], frame["text_raw"]))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def length_stats(values):
    a = np.asarray(list(values))
    return {
        "mean": round(float(a.mean()), 2),
        "median": round(float(np.median(a)), 1),
        "p95": round(float(np.percentile(a, 95)), 1),
        "p99": round(float(np.percentile(a, 99)), 1),
        "max": int(a.max()),
    }


def write_jsonl(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in frame.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f])


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


# ---- used by later milestones -------------------------------------------------------------------

def load_split(name, split):
    """Processed rows (id, text, text_raw, label) of one split, verified against the saved checksum.

    split is 'train' (leaked rows removed), 'val', 'test', or 'train_original' (M2 check only).
    """
    spec = DATASETS[name]
    frame = read_jsonl(processed_dir(spec) / f"{split}.jsonl")
    expected = read_json(splits_dir(spec) / "splits.json")["checksums"][split]
    if split_checksum(frame) != expected:
        raise RuntimeError(f"{name}/{split} does not match data/splits checksum; rerun scripts/m1_prepare_data.py")
    return frame


def training_subset(name, size, seed):
    """The first `size` rows of the saved stratified order for `seed` (size=None means all)."""
    spec = DATASETS[name]
    order = read_json(splits_dir(spec) / "train_order.json")[f"seed_{seed}"]
    ids = order if size is None else order[:size]
    train = load_split(name, "train").set_index("id")
    return train.loc[ids].reset_index()
