"""M2 / E1: BanglaBERT full fine-tuning on SentNoB.

Usage:
    uv run python scripts/m2_e1_banglabert.py --stage reproduction   # gate vs the published 72.89
    uv run python scripts/m2_e1_banglabert.py --stage grid           # learning rate, 1k, seed 0
    uv run python scripts/m2_e1_banglabert.py --stage curve          # 6 sizes x 3 seeds
    uv run python scripts/m2_e1_banglabert.py --stage control        # A2 control, 3 seeds
    uv run python scripts/m2_e1_banglabert.py --stage all

Stages, in order:
  reproduction  learning-rate grid on train_original (validation only), then 3 seeds at the best
                rate with test scoring. Gate: mean test macro-F1 within 3 points of 72.89.
  grid          the same 3 learning rates on 1k rows of the cleaned train split, seed 0, validation
                only. The winner is frozen for every later E1, E4, E5 and E6 run.
  curve         the frozen rate at all 6 learning-curve sizes and 3 seeds, with test scoring.
  control       the A2 control: train_original minus the same number of rows per class that the
                leakage removal dropped, chosen at random, 3 seeds, with test scoring. It separates
                "the training set got smaller" from "the memorizable rows went away".

Writes results/e1/<run>.json, predictions to results/predictions/e1/, and the summaries
results/e1_reproduction.json, results/e1_lr_grid.json, results/e1_summary.json and
results/e1_control.json.
"""
import argparse
import statistics

import yaml

from bangla_sentiment.contamination import random_removal_subset, slice_frame, slice_summary
from bangla_sentiment.data import DATASETS, SEEDS, load_split, read_json, splits_dir, training_subset
from bangla_sentiment.evaluate import save_predictions
from bangla_sentiment.train_encoder import train_encoder
from bangla_sentiment.utils import CONFIGS_DIR, RESULTS_DIR, ensure_utf8, save_result

PUBLISHED_MACRO_F1 = 72.89  # BanglaBERT paper, Table 2 (macro-F1, average of 3 seeds)
GATE_TOLERANCE = 3.0  # PRD section 10, M2 gate
GRID_SIZE = 1000  # tuning size, PRD section 8


def run(name, cfg, classes, train, train_split, lr, seed, val, test, checksums, subset, score_test, **extra):
    result, predictions = train_encoder(cfg, classes, train, val, test, lr, seed, score_test=score_test)
    result = {"experiment": "E1", "dataset": cfg["dataset"], "train_split": train_split, "subset": subset,
              "config": {k: cfg[k] for k in ("model", "max_seq_len", "train_batch_size", "max_epochs",
                                             "min_steps", "weight_decay", "warmup_ratio", "precision")},
              "split_checksums": {s: checksums[s] for s in (train_split, "val", "test")},
              **extra, **result}
    if predictions is not None:
        path = RESULTS_DIR / "predictions" / "e1" / f"{name}.csv"
        save_predictions(predictions["ids"], predictions["gold"], predictions["pred"],
                         predictions["proba"], classes, path)
        result["test_predictions"] = path.relative_to(RESULTS_DIR.parent).as_posix()
    save_result(result, f"e1/{name}")
    test_score = f", test macro-F1 {result['test']['macro_f1']:.2f}" if score_test else ""
    print(f"  {name}: best epoch {result['best_epoch']}/{result['epochs_run']}, "
          f"val macro-F1 {result['val']['macro_f1']:.2f}{test_score} "
          f"({result['efficiency']['train_seconds']}s, {result['efficiency']['peak_vram_gb']} GB)")
    return result


def learning_rate_grid(stage, cfg, classes, train, train_split, val, test, checksums, subset):
    """Train one model per learning rate and return (best rate, one row per rate). Validation only."""
    rows = []
    for lr in cfg["learning_rates"]:
        r = run(f"{stage}_grid_lr{lr:g}_s0", cfg, classes, train, train_split, lr, 0, val, test,
                checksums, subset, score_test=False)
        rows.append({"learning_rate": lr, "val_macro_f1": r["val"]["macro_f1"], "best_epoch": r["best_epoch"]})
    best = max(rows, key=lambda r: r["val_macro_f1"])
    print(f"  chosen learning rate {best['learning_rate']:g} (val macro-F1 {best['val_macro_f1']:.2f})")
    return best["learning_rate"], rows


def mean_std(values):
    return {"mean": round(statistics.mean(values), 2),
            "std": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["reproduction", "grid", "curve", "control", "all"], default="all")
    args = ap.parse_args()

    ensure_utf8()
    cfg = yaml.safe_load((CONFIGS_DIR / "e1.yaml").read_text(encoding="utf-8"))
    spec = DATASETS[cfg["dataset"]]
    classes = list(spec.label_names.values())
    checksums = read_json(splits_dir(spec) / "splits.json")["checksums"]
    val, test = load_split(spec.name, "val"), load_split(spec.name, "test")

    if args.stage in ("reproduction", "all"):
        print("Reproduction on train_original (the split the published number was trained on)")
        original = load_split(spec.name, "train_original")
        lr, grid = learning_rate_grid("reproduction", cfg, classes, original, "train_original", val, test,
                                      checksums, "all rows of train_original")
        runs = [run(f"reproduction_original_lr{lr:g}_s{seed}", cfg, classes, original, "train_original", lr,
                    seed, val, test, checksums, "all rows of train_original", score_test=True)
                for seed in SEEDS]
        scores = mean_std([r["test"]["macro_f1"] for r in runs])
        gate = {"check": "E1 on train_original vs the BanglaBERT paper",
                "published_macro_f1": PUBLISHED_MACRO_F1, "ours_macro_f1": scores,
                "difference": round(scores["mean"] - PUBLISHED_MACRO_F1, 2), "tolerance": GATE_TOLERANCE,
                "passed": abs(scores["mean"] - PUBLISHED_MACRO_F1) <= GATE_TOLERANCE}
        save_result({"experiment": "E1", "stage": "reproduction", "learning_rate_grid": grid,
                     "chosen_learning_rate": lr, "seeds": list(SEEDS), "test_macro_f1": scores,
                     "gate": gate}, "e1_reproduction")
        print(f"M2 E1 gate: ours {scores['mean']} +- {scores['std']} vs published {PUBLISHED_MACRO_F1} "
              f"(tolerance {GATE_TOLERANCE}): {'PASS' if gate['passed'] else 'FAIL'}")
        if not gate["passed"]:
            print("Gate failed. Per the PRD, explain the gap before starting M3 "
                  "(check epochs, normalization, label mapping, max length, learning rate).")

    if args.stage in ("grid", "all"):
        print(f"Learning-rate grid on the cleaned train split, {GRID_SIZE} rows, seed 0, validation only")
        subset = training_subset(spec.name, GRID_SIZE, 0)
        lr, grid = learning_rate_grid("curve", cfg, classes, subset, "train", val, test, checksums,
                                      f"train_order.json seed_0, first {GRID_SIZE} IDs")
        save_result({"experiment": "E1", "stage": "learning_rate_grid", "train_rows": GRID_SIZE,
                     "grid": grid, "chosen_learning_rate": lr}, "e1_lr_grid")

    if args.stage in ("curve", "all"):
        lr = read_json(RESULTS_DIR / "e1_lr_grid.json")["chosen_learning_rate"]
        print(f"Learning curve at learning rate {lr:g}")
        table = []
        for size in spec.curve_sizes:
            tag = "full" if size is None else str(size)
            runs = []
            for seed in SEEDS:
                train = training_subset(spec.name, size, seed)
                runs.append(run(f"curve_{tag}_lr{lr:g}_s{seed}", cfg, classes, train, "train", lr, seed, val,
                                test, checksums, f"train_order.json seed_{seed}, first {len(train)} IDs",
                                score_test=True))
            table.append({"size": tag, "train_rows": runs[0]["train_rows"],
                          "val_macro_f1": mean_std([r["val"]["macro_f1"] for r in runs]),
                          "test_macro_f1": mean_std([r["test"]["macro_f1"] for r in runs]),
                          "test_micro_f1": mean_std([r["test"]["micro_f1"] for r in runs]),
                          "train_seconds": mean_std([r["efficiency"]["train_seconds"] for r in runs])})
        path = save_result({"experiment": "E1", "dataset": spec.name, "learning_rate": lr,
                            "seeds": list(SEEDS), "std": "sample (n-1) over seeds", "table": table},
                           "e1_summary")
        print("\nsize      rows    test macro-F1     train seconds")
        for row in table:
            f1, secs = row["test_macro_f1"], row["train_seconds"]
            print(f"{row['size']:>5} {row['train_rows']:>8}   {f1['mean']:6.2f} +- {f1['std']:<5}   {secs['mean']}")
        print(f"wrote {path}")

    if args.stage in ("control", "all"):
        lr = read_json(RESULTS_DIR / "e1_lr_grid.json")["chosen_learning_rate"]
        original, cleaned = load_split(spec.name, "train_original"), load_split(spec.name, "train")
        print(f"A2 control at learning rate {lr:g}: train_original minus random rows, "
              f"the same number per class that the leakage removal dropped")
        runs, removal, matchable = [], None, []
        for seed in SEEDS:
            train, removal = random_removal_subset(original, cleaned, seed)
            # A random removal drops a few leaked rows by chance; record how many stay memorizable.
            matchable.append(slice_summary(slice_frame(train, test))["contaminated"]["rows"])
            runs.append(run(f"control_random_removal_lr{lr:g}_s{seed}", cfg, classes, train,
                            "train_original", lr, seed, val, test, checksums,
                            f"train_original minus {len(original) - len(train)} rows drawn at random "
                            f"with seed {seed}, matched per class to the leakage removal",
                            score_test=True, control="random_removal"))
        removal["contaminated_test_rows_still_matchable"] = matchable
        removal["contaminated_test_rows_in_train_original"] = slice_summary(
            slice_frame(original, test))["contaminated"]["rows"]
        control_scores = mean_std([r["test"]["macro_f1"] for r in runs])
        reference = {
            "train_original": read_json(RESULTS_DIR / "e1_reproduction.json")["test_macro_f1"],
            "train_cleaned": next(row["test_macro_f1"] for row in
                                  read_json(RESULTS_DIR / "e1_summary.json")["table"] if row["size"] == "full"),
        }
        effects = {
            "total_original_minus_cleaned": round(reference["train_original"]["mean"] - reference["train_cleaned"]["mean"], 2),
            "size_only_original_minus_control": round(reference["train_original"]["mean"] - control_scores["mean"], 2),
            "leakage_control_minus_cleaned": round(control_scores["mean"] - reference["train_cleaned"]["mean"], 2),
        }
        path = save_result({"experiment": "E1", "stage": "control", "analysis": "A2 control",
                            "question": "Is the leakage effect just a smaller training set?",
                            "learning_rate": lr, "seeds": list(SEEDS), "removal": removal,
                            "test_macro_f1": {"control_random_removal": control_scores, **reference},
                            "effects_in_macro_f1_points": effects}, "e1_control")
        print(f"\n  train_original     {reference['train_original']['mean']:.2f} "
              f"+- {reference['train_original']['std']}")
        print(f"  random removal     {control_scores['mean']:.2f} +- {control_scores['std']}  (same size as cleaned)")
        print(f"  cleaned train      {reference['train_cleaned']['mean']:.2f} "
              f"+- {reference['train_cleaned']['std']}")
        print(f"  of the {effects['total_original_minus_cleaned']:.2f} point drop, "
              f"{effects['size_only_original_minus_control']:.2f} is the smaller training set and "
              f"{effects['leakage_control_minus_cleaned']:.2f} is the leaked rows")
        print(f"wrote {path}")
        print("  now rerun scripts/m2_a2_contamination_slices.py to add the control to the slices")


if __name__ == "__main__":
    main()
