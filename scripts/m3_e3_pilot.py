"""M3: the 1k-row QLoRA pilot, one run per candidate, on validation only.

Usage:
    uv run python scripts/m3_e3_pilot.py
    uv run python scripts/m3_e3_pilot.py --models gemma-3-1b        # a subset, for debugging
    uv run python scripts/m3_e3_pilot.py --limit 64 --models gemma-3-1b   # smoke test, not a result

This is the last piece of M3. It trains each candidate on the same 1,000 rows with the same
settings, scores the full validation split with the epoch chosen on a 500-row validation sample,
and writes everything the selection rule needs. The test split is never touched.

Shihab then picks the main LLM (PRD section 6, plus the cost rule in the decision log):
  1. highest validation macro-F1 after this run
  2. if two are within 1 point, prefer fewer tokens per word
  3. it must train inside about 5 GB
  4. if the leader's projected full curve is over about 20 GPU-hours and another candidate is
     within 2 points, pick the cheaper one and record why
TigerLLM is not eligible: it is reserved for RQ2.

Writes results/m3/qlora_pilot_<model>.json and results/m3_e3_pilot.json, and LoRA adapters under
outputs/. Runs that already have a result file are skipped, so a crash costs at most one run.
"""
import argparse

import yaml

from bangla_sentiment.data import DATASETS, load_split, read_json, stratified_order, training_subset
from bangla_sentiment.train_qlora import train_with_oom_backoff
from bangla_sentiment.utils import CONFIGS_DIR, RESULTS_DIR, ensure_utf8, save_result

VRAM_LIMIT_GB = 5.0        # PRD section 6, rule 3
CLOSE_ENOUGH = 1.0         # PRD section 6, rule 2: the tie-break band
COST_BAND = 2.0            # decision log 2026-09-25: the cost rule's band
COST_LIMIT_HOURS = 20.0    # decision log 2026-09-25: when the cost rule applies


def done(name):
    return (RESULTS_DIR / "m3" / f"{name}.json").exists()


def projected_curve_hours(cfg, spec, result, val_rows, test_rows):
    """Estimated GPU-hours for the full E3 learning curve at this run's measured speed.

    6 training sizes x 3 seeds, each trained for `max_epochs` and scored once per epoch on the
    epoch-selection sample, then once on validation and once on test. An estimate, not a result.
    """
    runs = 3 * len(spec.curve_sizes)
    full = read_json(RESULTS_DIR / "e1_summary.json")["table"][-1]["train_rows"]
    train_examples = 3 * cfg["max_epochs"] * sum(size or full for size in spec.curve_sizes)
    scored = runs * (cfg["max_epochs"] * cfg["pilot_eval_rows"] + val_rows + test_rows)
    speed = result["efficiency"]
    return round((train_examples / speed["train_examples_per_s"]
                  + scored / speed["val_examples_per_s"]) / 3600, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated subset of configs/e3.yaml")
    ap.add_argument("--limit", type=int, default=None, help="tiny run for smoke testing, not a result")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the epoch cap, to check whether 3 epochs is enough. Writes its own "
                         "result files and no summary, so the pilot results stay untouched.")
    args = ap.parse_args()

    ensure_utf8()
    cfg = yaml.safe_load((CONFIGS_DIR / "e3.yaml").read_text(encoding="utf-8"))
    spec = DATASETS[cfg["dataset"]]
    classes = list(spec.label_names.values())
    variant = read_json(RESULTS_DIR / "m3_prompt_pilot.json")["chosen_variant"]
    lr = cfg["pilot_learning_rate"]

    train = training_subset(spec.name, cfg["pilot_train_rows"], 0)
    val = load_split(spec.name, "val")
    # A stratified sample of validation, the same rows for every model, only to choose the epoch.
    order = stratified_order(val["id"].tolist(), val["label"].tolist(), 0)[:cfg["pilot_eval_rows"]]
    eval_val = val.set_index("id").loc[order].reset_index()

    models = {k: v for k, v in cfg["models"].items() if v["candidate"]}
    if args.models:
        models = {k: v for k, v in cfg["models"].items() if k in args.models.split(",")}
    prefix = ""
    if args.epochs:  # epoch-cap check: the same pilot at a different cap, in its own result files
        cfg["max_epochs"] = args.epochs
        prefix = f"e{args.epochs}_"
    if args.limit:  # smoke test: results from truncated splits are not comparable to anything, so
        prefix = "smoke_" + prefix  # they are written under their own name, and no summary is produced
        train, val, eval_val = train.head(args.limit), val.head(args.limit), eval_val.head(args.limit)
        cfg["max_epochs"], cfg["min_steps"] = 1, 1  # the min-steps rule would train a tiny set for ages

    print(f"QLoRA pilot: {len(train)} train rows, prompt {variant}, learning rate {lr:g}, "
          f"{cfg['max_epochs']} epochs, effective batch {cfg['effective_batch_size']}", flush=True)
    rows = []
    for name, cfg_model in models.items():
        run_name = f"{prefix}qlora_pilot_{name}"
        if done(run_name):
            rows.append((name, read_json(RESULTS_DIR / "m3" / f"{run_name}.json")))
            print(f"  {run_name}: already done", flush=True)
            continue
        print(f"  {run_name}: {cfg_model['repo_id']}", flush=True)
        result, _ = train_with_oom_backoff(cfg, cfg_model, classes, train, val, lr, 0, variant,
                                           eval_val=eval_val, adapter_name=f"{prefix}m3_pilot_{name}")
        result = {"experiment": "E3", "stage": "pilot", "split": "val",
                  "candidate": cfg_model["candidate"], "max_text_tokens": cfg_model["max_text_tokens"],
                  "config": {k: cfg[k] for k in ("lora", "quantization", "effective_batch_size",
                                                 "max_epochs", "min_steps", "warmup_ratio",
                                                 "weight_decay", "gradient_checkpointing",
                                                 "optimizer", "scheduler")},
                  # After the merge, so the trainer's own fields cannot overwrite the identity here
                  **result, "model": cfg_model["repo_id"], "model_key": name}
        save_result(result, f"m3/{run_name}")
        rows.append((name, result))
        speed = result["efficiency"]
        print(f"    val macro-F1 {result['val']['macro_f1']:.2f} (best epoch {result['best_epoch']}), "
              f"{speed['train_seconds'] / 60:.1f} min at {speed['train_examples_per_s']}/s, "
              f"peak {speed['peak_vram_gb']} GB", flush=True)

    if args.limit or args.epochs:
        what = "smoke test" if args.limit else f"epoch-cap check at {args.epochs} epochs"
        print(f"\n{what}: no summary written. Compare epoch_history against the pilot runs.")
        return

    test_rows = len(load_split(spec.name, "test"))
    tokenizers = read_json(RESULTS_DIR / "m1_tokenizer_stats.json")["tokenizers"]
    zero_shot = {r["model"]: r["val"]["macro_f1"] for r in
                 (read_json(p) for p in (RESULTS_DIR / "m3").glob("zeroshot_*.json"))
                 if r["variant"] == variant}
    few_shot = {r["model"]: r["val_macro_f1"]["mean"]
                for r in read_json(RESULTS_DIR / "m3_e2_summary.json")["table"]}

    table = []
    for name, result in rows:
        speed = result["efficiency"]
        table.append({
            "model": name,
            "candidate": result["candidate"],
            "qlora_1k_val_macro_f1": result["val"]["macro_f1"],
            "zero_shot_val_macro_f1": zero_shot.get(name),
            "few_shot_val_macro_f1": few_shot.get(name),
            "tokens_per_word": tokenizers[name][spec.name]["tokens_per_word"],
            "max_text_tokens": result["max_text_tokens"],
            "train_examples_per_s": speed["train_examples_per_s"],
            "train_minutes_1k": round(speed["train_seconds"] / 60, 1),
            "peak_vram_gb": speed["peak_vram_gb"],
            "fits_in_5gb": speed["peak_vram_gb"] <= VRAM_LIMIT_GB,
            "batch_size_reductions": speed.get("batch_size_reductions", []),
            "projected_full_curve_hours_estimate": projected_curve_hours(
                cfg, spec, result, len(val), test_rows),
        })
    table.sort(key=lambda r: -r["qlora_1k_val_macro_f1"])

    eligible = [r for r in table if r["candidate"] and r["fits_in_5gb"]]
    leader = eligible[0] if eligible else None
    rule = None
    if leader:
        within_1 = [r for r in eligible if leader["qlora_1k_val_macro_f1"] - r["qlora_1k_val_macro_f1"] <= CLOSE_ENOUGH]
        cheaper = [r for r in eligible
                   if leader["qlora_1k_val_macro_f1"] - r["qlora_1k_val_macro_f1"] <= COST_BAND
                   and r["projected_full_curve_hours_estimate"] < leader["projected_full_curve_hours_estimate"]]
        rule = {
            "highest_val_macro_f1": leader["model"],
            "within_1_point": [r["model"] for r in within_1],
            "fewest_tokens_per_word_among_those": min(within_1, key=lambda r: r["tokens_per_word"])["model"],
            "cost_rule_applies": leader["projected_full_curve_hours_estimate"] > COST_LIMIT_HOURS,
            "cheaper_candidates_within_2_points": [r["model"] for r in cheaper],
            "decision": "Shihab decides and records it in the PRD decision log",
        }

    path = save_result({"experiment": "E3", "stage": "pilot", "split": "val",
                        "prompt_variant": variant, "learning_rate": lr,
                        "train_rows": len(train), "epoch_selection_rows": len(eval_val),
                        "selection_rule": "PRD section 6, plus the cost rule in the decision log",
                        "note": "projected_full_curve_hours_estimate is an estimate from measured "
                                "speeds, not a measurement",
                        "table": table, "rule_inputs": rule}, "m3_e3_pilot")

    print("\nmodel          QLoRA 1k   few-shot   zero-shot   tok/word   min/1k   peak GB   curve h (est)")
    for row in table:
        print(f"{row['model']:<14} {row['qlora_1k_val_macro_f1']:8.2f} "
              f"{row['few_shot_val_macro_f1'] or 0:10.2f} {row['zero_shot_val_macro_f1'] or 0:11.2f} "
              f"{row['tokens_per_word']:10.2f} {row['train_minutes_1k']:8.1f} "
              f"{row['peak_vram_gb']:9.2f} {row['projected_full_curve_hours_estimate']:14.1f}")
    if rule:
        print(f"\nhighest validation macro-F1: {rule['highest_val_macro_f1']}")
        print(f"within 1 point: {rule['within_1_point']}, "
              f"fewest tokens per word among them: {rule['fewest_tokens_per_word_among_those']}")
        print(f"cost rule applies (over {COST_LIMIT_HOURS:g} projected hours): {rule['cost_rule_applies']}, "
              f"cheaper candidates within {COST_BAND:g} points: {rule['cheaper_candidates_within_2_points']}")
        print("Shihab picks the main LLM and records it in the decision log.")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
