"""M2 / A2: is the leakage effect memorization, or just a smaller training set?

Usage:
    uv run python scripts/m2_a2_contamination_slices.py

Splits the test split in two, using the M1 near-duplicate key:
  contaminated  test rows that have a near-duplicate in train_original (the split the published
                numbers were trained on). These are the rows the removed training rows could have
                been memorized from.
  clean         every other test row. No training row anywhere resembles these.

Every saved E0 and E1 prediction file is then re-scored on each slice. Nothing is retrained and no
model sees anything new, so this adds no GPU time and cannot change any earlier number.

The claim it tests: if removing leaked training rows costs accuracy because of memorization, the
loss must sit on the contaminated slice. If instead it is spread evenly, the smaller training set
explains it. The headline number is the difference in differences, with a paired bootstrap.

Three training variants are compared where they exist. The third one appears only after
`m2_e1_banglabert.py --stage control` has run:
  train_original           leaks kept, the published setup
  control_random_removal   leaks kept, but as many random rows per class removed as the leakage
                           removal dropped, so it has the size of the cleaned split
  cleaned_full             leaks removed, full size

Writes results/a2_contamination_slices.json.
"""
import pandas as pd

from bangla_sentiment.contamination import slice_frame, slice_summary
from bangla_sentiment.data import DATASETS, load_split, read_json
from bangla_sentiment.evaluate import bootstrap_differences, classification_metrics, summarize_differences
from bangla_sentiment.utils import RESULTS_DIR, ensure_utf8, save_result

DATASET = "sentnob"
EXPERIMENTS = ("e0", "e1")
N_RESAMPLES = 5000
SLICES = ("full", "contaminated", "clean")
PAIRS = (("train_original", "cleaned_full"),            # the effect reported in E0 and E1
         ("train_original", "control_random_removal"),  # the smaller training set on its own
         ("control_random_removal", "cleaned_full"))    # the leaked rows on their own


def load_runs(test_ids):
    """Every saved run that has test predictions, with its predictions aligned to the test split."""
    runs = []
    for experiment in EXPERIMENTS:
        for path in sorted((RESULTS_DIR / experiment).glob("*.json")):
            result = read_json(path)
            if "test_predictions" not in result:
                continue  # validation-only runs (the learning-rate grids) never scored the test split
            predictions = pd.read_csv(RESULTS_DIR.parent / result["test_predictions"])
            if list(predictions["id"]) != list(test_ids):
                raise RuntimeError(f"{path.name}: predictions are not aligned with the test split")
            runs.append({
                "run": path.stem,
                "experiment": result["experiment"],
                "model": result["model"],
                "train_split": result["train_split"],
                "control": result.get("control"),
                "train_rows": result["train_rows"],
                "seed": result["seed"],
                "pred": predictions["pred"].tolist(),
            })
    return runs


def variant(run, full_rows):
    """Which of the three compared training variants a run belongs to, or None."""
    if run["control"] == "random_removal":
        return "control_random_removal"
    if run["train_split"] == "train_original" and run["control"] is None:
        return "train_original"
    if run["train_split"] == "train" and run["train_rows"] == full_rows:
        return "cleaned_full"
    return None


def keep_mask(frame, name):
    if name == "full":
        return [True] * len(frame)
    contaminated = frame["contaminated"].tolist()
    return contaminated if name == "contaminated" else [not c for c in contaminated]


def take(values, keep):
    return [v for v, k in zip(values, keep) if k]


def scores(gold, pred, classes, keep):
    """macro-F1, micro-F1 and accuracy on one slice."""
    metrics = classification_metrics(take(gold, keep), take(pred, keep), classes)
    return {k: metrics[k] for k in ("macro_f1", "micro_f1", "accuracy")}


def compare(runs_a, runs_b, gold, classes, keep, seed):
    """Seed-averaged paired bootstrap of (variant A minus variant B) on one slice."""
    return bootstrap_differences(take(gold, keep), [take(r["pred"], keep) for r in runs_a],
                                 [take(r["pred"], keep) for r in runs_b], classes,
                                 n_resamples=N_RESAMPLES, seed=seed)


def mean_macro_f1(runs, name):
    return round(sum(r["slices"][name]["macro_f1"] for r in runs) / len(runs), 2)


def differences(runs_a, runs_b, gold, classes, keep, seed_offset):
    """Every slice, plus the contaminated-minus-clean difference in differences."""
    out, draws = {"slices": {}}, {}
    for i, name in enumerate(SLICES):
        observed, draws[name] = compare(runs_a, runs_b, gold, classes, keep[name], seed_offset + i)
        out["slices"][name] = summarize_differences(observed, draws[name])
    observed = out["slices"]["contaminated"]["difference"] - out["slices"]["clean"]["difference"]
    out["difference_in_differences"] = summarize_differences(observed, draws["contaminated"] - draws["clean"])
    return out


def main():
    ensure_utf8()
    spec = DATASETS[DATASET]
    classes = list(spec.label_names.values())
    test = load_split(DATASET, "test")
    original = load_split(DATASET, "train_original")
    cleaned_train = load_split(DATASET, "train")

    frame = slice_frame(original, test)
    summary = slice_summary(frame)
    still_leaking = slice_summary(slice_frame(cleaned_train, test))["contaminated"]["rows"]
    if still_leaking:
        raise RuntimeError(f"{still_leaking} test rows still have a near-duplicate in the cleaned train split")
    gold = test["label"].tolist()
    keep = {name: keep_mask(frame, name) for name in SLICES}
    print(f"test rows {summary['rows']}: contaminated {summary['contaminated']['rows']} "
          f"({summary['contaminated']['share']:.1%}), clean {summary['clean']['rows']}")

    # What pure memorization could score: copy the label of the matching training row.
    matched = frame[frame["contaminated"]]
    ceiling = classification_metrics(matched["label"], matched["train_label"], classes)

    runs = load_runs(test["id"])
    for run in runs:
        run["slices"] = {name: scores(gold, run["pred"], classes, keep[name]) for name in SLICES}

    comparisons = []
    for model in dict.fromkeys(run["model"] for run in runs):
        same = [r for r in runs if r["model"] == model]
        on_cleaned = [r["train_rows"] for r in same if r["train_split"] == "train" and r["control"] is None]
        groups = {}
        for run in same:
            name = variant(run, max(on_cleaned, default=0))
            if name:
                groups.setdefault(name, []).append(run)
        groups = {k: sorted(v, key=lambda r: r["seed"]) for k, v in sorted(groups.items())}
        pairs = [(a, b) for a, b in PAIRS if a in groups and b in groups]
        if not pairs:
            continue
        seeds = [r["seed"] for r in groups[pairs[0][0]]]
        if any([r["seed"] for r in group] != seeds for group in groups.values()):
            raise RuntimeError(f"{model}: the training variants were not run on the same seeds")

        comparisons.append({
            "model": model,
            "experiment": same[0]["experiment"],
            "seeds": seeds,
            "variants": {name: {"train_rows": group[0]["train_rows"],
                                "macro_f1": {s: mean_macro_f1(group, s) for s in SLICES}}
                         for name, group in groups.items()},
            "differences": [{"a": a, "b": b,
                             **differences(groups[a], groups[b], gold, classes, keep, 10 * (i + 1))}
                            for i, (a, b) in enumerate(pairs)],
        })

    payload = {
        "analysis": "A2",
        "question": "Does removing leaked training rows cost accuracy because of memorization?",
        "dataset": DATASET,
        "definition": {
            "key": "bangla_sentiment.data.dedupe_key: NFC, case-folded, Unicode letters and marks only",
            "contaminated": "test rows with a near-duplicate in train_original",
            "clean": "all other test rows, including rows whose key is empty",
            "note": "The cleaned train split contains no near-duplicate of any test row, by construction.",
        },
        "slices": summary,
        "memorization_ceiling": {
            "description": "Copy the label of the matching training row; contaminated slice only.",
            "accuracy": ceiling["accuracy"],
            "macro_f1": ceiling["macro_f1"],
            "per_class": ceiling["per_class"],
        },
        "bootstrap": {
            "resamples": N_RESAMPLES,
            "method": "seed-averaged paired bootstrap over test rows, percentile interval. The two "
                      "slices are resampled independently for the difference in differences.",
        },
        "comparisons": comparisons,
        "runs": [{k: run[k] for k in
                  ("run", "experiment", "model", "train_split", "control", "train_rows", "seed", "slices")}
                 for run in runs],
    }
    path = save_result(payload, "a2_contamination_slices")

    print(f"\nmemorization ceiling on the contaminated slice: {ceiling['accuracy']:.2f} accuracy, "
          f"{ceiling['macro_f1']:.2f} macro-F1")
    for row in comparisons:
        print(f"\n{row['model']}  (test macro-F1, mean over seeds {row['seeds']})")
        print(f"  {'variant':<24} {'rows':>6} " + " ".join(f"{s:>13}" for s in SLICES))
        for name, v in row["variants"].items():
            print(f"  {name:<24} {v['train_rows']:>6} " + " ".join(f"{v['macro_f1'][s]:13.2f}" for s in SLICES))
        for diff in row["differences"]:
            print(f"  {diff['a']} minus {diff['b']}:")
            for name in SLICES:
                d = diff["slices"][name]
                print(f"    {name:<14} {d['difference']:7.2f}  95% CI [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]"
                      f"  p {d['p_two_sided']}")
            did = diff["difference_in_differences"]
            print(f"    contaminated minus clean: {did['difference']:.2f} "
                  f"[{did['ci95'][0]:.2f}, {did['ci95'][1]:.2f}], p {did['p_two_sided']}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
