"""M2 / E0: majority-class and TF-IDF with logistic regression baselines on SentNoB.

Usage:
    uv run python scripts/m2_e0_baselines.py

Trains every model at every learning-curve size and seed on the cleaned train split, and once per
seed on train_original for the reproduction check against the SentNoB paper.
Settings are fixed in configs/e0.yaml and nothing is tuned, so val and test are both scored.
Runs on the CPU in a few minutes.

Writes results/e0/<run>.json, test predictions (IDs and labels, no text) to results/predictions/e0/,
and results/e0_summary.json (mean and sample standard deviation over seeds, plus the gate check).
"""
import statistics
import time

import yaml

from bangla_sentiment.baselines import build_model
from bangla_sentiment.data import DATASETS, SEEDS, load_split, read_json, splits_dir, training_subset
from bangla_sentiment.evaluate import classification_metrics, save_predictions
from bangla_sentiment.utils import CONFIGS_DIR, RESULTS_DIR, ensure_utf8, save_result, set_seed

# SentNoB paper, Table 3. The majority row has precision = recall = F1 = 41.24, which is the test
# accuracy of always answering "positive", so it can be compared exactly. The best lexical row
# (F1 64.61) has precision 57.71 and recall 73.39; micro precision and recall are always equal for
# single-label classes, so 64.61 is not micro-F1 and matches no standard metric. Context only.
PAPER_MAJORITY_ACCURACY = 41.24
PAPER_BEST_LEXICAL = {"precision": 57.71, "recall": 73.39, "f1": 64.61}


def run_one(name, model_name, cfg, train, train_split, seed, evals, classes, checksums, subset_note):
    set_seed(seed)
    model = build_model(model_name, cfg, seed)
    start = time.perf_counter()
    model.fit(train["text"], train["label"])
    train_seconds = time.perf_counter() - start
    order = [list(model.classes_).index(c) for c in classes]

    result = {
        "experiment": "E0",
        "model": model_name,
        "dataset": "sentnob",
        "train_split": train_split,
        "train_rows": len(train),
        "subset": subset_note,
        "seed": seed,
        "config": cfg,
        "device": "cpu",
        "split_checksums": {s: checksums[s] for s in (train_split, "val", "test")},
        "selection_done_on": "none (fixed settings, nothing tuned)",
        "timing": {"train_seconds": round(train_seconds, 2)},
    }
    for split, frame in evals.items():
        start = time.perf_counter()
        pred = model.predict(frame["text"])
        proba = model.predict_proba(frame["text"])[:, order]
        if split == "test":
            result["timing"]["test_inference_ms_per_example"] = round(
                1000 * (time.perf_counter() - start) / len(frame), 4)
            pred_path = RESULTS_DIR / "predictions" / "e0" / f"{name}.csv"
            save_predictions(frame["id"], frame["label"], pred, proba, classes, pred_path)
            result["test_predictions"] = pred_path.relative_to(RESULTS_DIR.parent).as_posix()
        result[split] = classification_metrics(frame["label"], pred, classes)
    save_result(result, f"e0/{name}")
    return result


def summarize(runs):
    """Mean and sample standard deviation over seeds of val and test scores, per model and size."""
    groups = {}
    for r in runs:
        groups.setdefault((r["model"], r["size"]), []).append(r)
    table = []
    for (model, size), rs in groups.items():
        row = {"model": model, "size": size, "seeds": [r["seed"] for r in rs]}
        for split in ("val", "test"):
            for metric in ("macro_f1", "micro_f1"):
                values = [r[split][metric] for r in rs]
                row[f"{split}_{metric}"] = {"mean": round(statistics.mean(values), 2),
                                            "std": round(statistics.stdev(values), 2)}
        table.append(row)
    return table


def main():
    ensure_utf8()
    cfg = yaml.safe_load((CONFIGS_DIR / "e0.yaml").read_text(encoding="utf-8"))
    spec = DATASETS["sentnob"]
    classes = list(spec.label_names.values())
    checksums = read_json(splits_dir(spec) / "splits.json")["checksums"]
    evals = {s: load_split("sentnob", s) for s in ("val", "test")}
    original = load_split("sentnob", "train_original")

    runs = []
    for model_name, model_cfg in cfg["models"].items():
        for seed in SEEDS:
            for size in spec.curve_sizes:
                train = training_subset("sentnob", size, seed)
                tag = "full" if size is None else str(size)
                note = f"data/splits/sentnob/train_order.json seed_{seed}, first {len(train)} IDs"
                r = run_one(f"{model_name}_sentnob_{tag}_s{seed}", model_name, model_cfg, train, "train",
                            seed, evals, classes, checksums, note)
                runs.append({**r, "size": tag})
                print(f"{model_name:<9} {tag:>5} seed {seed}: test macro-F1 {r['test']['macro_f1']:6.2f}  "
                      f"({r['timing']['train_seconds']}s)")
            r = run_one(f"{model_name}_sentnob_original_s{seed}", model_name, model_cfg, original,
                        "train_original", seed, evals, classes, checksums, "all rows of train_original")
            runs.append({**r, "size": "original"})
            print(f"{model_name:<9} original seed {seed}: test macro-F1 {r['test']['macro_f1']:6.2f}, "
                  f"micro-F1 {r['test']['micro_f1']:6.2f}")

    table = summarize(runs)
    row = lambda model: next(r for r in table if r["model"] == model and r["size"] == "original")
    ours = row("majority")["test_micro_f1"]["mean"]  # micro-F1 equals accuracy for single-label classes
    tfidf = row("tfidf_lr")
    gate = {
        "check": "Majority-class test accuracy equals the SentNoB paper's (same test set and label mapping)",
        "paper_majority_accuracy": PAPER_MAJORITY_ACCURACY,
        "ours_majority_accuracy": ours,
        "passed": abs(ours - PAPER_MAJORITY_ACCURACY) < 0.01,
        "context_paper_best_lexical": PAPER_BEST_LEXICAL,
        "context_ours_tfidf_original": {"accuracy": tfidf["test_micro_f1"]["mean"],
                                        "macro_f1": tfidf["test_macro_f1"]["mean"]},
    }
    path = save_result({"experiment": "E0", "dataset": "sentnob", "std": "sample (n-1) over seeds",
                        "table": table, "gate": gate}, "e0_summary")

    print("\nmodel      size      test macro-F1     test micro-F1")
    for row in table:
        m, u = row["test_macro_f1"], row["test_micro_f1"]
        print(f"{row['model']:<9} {row['size']:>8}   {m['mean']:6.2f} +- {m['std']:<5}   {u['mean']:6.2f} +- {u['std']}")
    print(f"\nM2 E0 gate: majority accuracy ours {ours} vs paper {PAPER_MAJORITY_ACCURACY}: "
          f"{'PASS' if gate['passed'] else 'FAIL'}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
