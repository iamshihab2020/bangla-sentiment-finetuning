"""M3 / E2: prompt pilot, zero-shot and few-shot scoring for every LLM, on validation only.

Usage:
    uv run python scripts/m3_e2_pilot.py --stage zeroshot   # both prompt languages, all models
    uv run python scripts/m3_e2_pilot.py --stage fewshot    # frozen prompt, 3 draws of 6 examples
    uv run python scripts/m3_e2_pilot.py --stage all

The test split is never touched here. The prompt variant is chosen on validation macro-F1 averaged
over the candidate models, saved to results/m3_prompt_pilot.json, and frozen for every later run.

Scoring rule per model and variant: the label's first token when the three labels differ there,
otherwise the summed log-probability of the whole label (PRD section 7). Both are computed from the
same forward pass, and the rule used and the agreement between them are recorded.

Writes results/m3/<run>.json, results/m3_prompt_pilot.json and results/m3_e2_summary.json.
Runs that already have a result file are skipped, so a crash costs at most one run.
"""
import argparse
import statistics
import time

import torch
import yaml

from bangla_sentiment.data import DATASETS, load_split, read_json, training_subset
from bangla_sentiment.evaluate import classification_metrics
from bangla_sentiment.prompts import draw_shots, label_words
from bangla_sentiment.scoring import (echoes_input, first_tokens_differ, generate_answers, load_4bit,
                                      parse_answer, predictions_from_scores, score_labels)
from bangla_sentiment.utils import CONFIGS_DIR, RESULTS_DIR, ensure_utf8, save_result, set_seed

SHOT_POOL_SIZE = 4000  # few-shot examples are drawn from the cleaned train split, seed 0


def done(name):
    return (RESULTS_DIR / "m3" / f"{name}.json").exists()


def score_split(model, tok, frame, variant, classes, cfg_model, shots=()):
    """Predictions from both scoring rules, with timing. Returns a result dict."""
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    summed, first = score_labels(model, tok, frame["text"].tolist(), variant, classes,
                                 batch_size=cfg_model["batch_size"],
                                 max_text_tokens=cfg_model["max_text_tokens"], shots=shots)
    elapsed = time.perf_counter() - start
    distinct = first_tokens_differ(tok, label_words(variant, classes))
    rule = "first_token" if distinct else "full_label"
    pred_first, _ = predictions_from_scores(first, classes)
    pred_full, probs = predictions_from_scores(summed, classes)
    pred = pred_first if distinct else pred_full
    return {
        "scoring_rule": rule,
        "label_first_tokens_distinct": distinct,
        "rules_agree": round(sum(a == b for a, b in zip(pred_first, pred_full)) / len(pred_full), 4),
        "invalid_predictions": 0,  # scoring always returns one of the classes
        "val": classification_metrics(frame["label"], pred, classes),
        "predicted_counts": {c: pred.count(c) for c in classes},
        "timing": {"seconds": round(elapsed, 1), "examples_per_s": round(len(frame) / elapsed, 2),
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)},
    }, probs


def generation_check(model, tok, frame, variant, classes, cfg, cfg_model):
    """How often free generation fails to answer with a label word. Reported, never used as the metric."""
    sample = frame.head(cfg["generation_check"]["n_examples"])
    answers = generate_answers(model, tok, sample["text"].tolist(), variant, classes,
                               batch_size=cfg_model["batch_size"],
                               max_new_tokens=cfg["generation_check"]["max_new_tokens"],
                               max_text_tokens=cfg_model["max_text_tokens"])
    parsed = [parse_answer(a, variant, classes) for a in answers]
    valid = [(p, g) for p, g in zip(parsed, sample["label"]) if p is not None]
    return {
        "n_examples": len(sample),
        "invalid_rate": round(sum(p is None for p in parsed) / len(parsed), 4),
        "macro_f1_on_valid_answers": (classification_metrics([g for _, g in valid], [p for p, _ in valid],
                                                             classes)["macro_f1"] if valid else None),
        # Redacted when the model quotes the comment back: no dataset text in the repository
        "example_answers": ["[redacted: the model echoed the comment]" if echoes_input(a, t) else a[:60]
                            for a, t in zip(answers[:3], sample["text"][:3])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["zeroshot", "fewshot", "all"], default="all")
    ap.add_argument("--models", default=None, help="comma-separated subset of configs/e2.yaml, for debugging")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N validation rows, for debugging")
    args = ap.parse_args()

    ensure_utf8()
    set_seed(0)
    cfg = yaml.safe_load((CONFIGS_DIR / "e2.yaml").read_text(encoding="utf-8"))
    spec = DATASETS[cfg["dataset"]]
    classes = list(spec.label_names.values())
    val = load_split(spec.name, "val")
    shot_pool = training_subset(spec.name, SHOT_POOL_SIZE, 0)
    if args.models:
        cfg["models"] = {k: v for k, v in cfg["models"].items() if k in args.models.split(",")}
    if args.limit:  # debugging only: results from a truncated split are not comparable
        val = val.head(args.limit)
        cfg["generation_check"]["n_examples"] = min(cfg["generation_check"]["n_examples"], args.limit)

    if args.stage in ("zeroshot", "all"):
        rows = []
        for name, cfg_model in cfg["models"].items():
            tok = model = None
            for variant in cfg["variants"]:
                run_name = f"zeroshot_{name}_{variant}"
                if done(run_name):
                    rows.append(read_json(RESULTS_DIR / "m3" / f"{run_name}.json"))
                    print(f"  {run_name}: already done", flush=True)
                    continue
                if model is None:
                    tok, model = load_4bit(cfg_model["repo_id"])
                result, _ = score_split(model, tok, val, variant, classes, cfg_model)
                result = {"experiment": "E2", "stage": "zeroshot", "model": name,
                          "repo_id": cfg_model["repo_id"], "candidate": cfg_model["candidate"],
                          "variant": variant, "split": "val", "shots": 0,
                          "max_text_tokens": cfg_model["max_text_tokens"], **result,
                          "generation": generation_check(model, tok, val, variant, classes, cfg, cfg_model)}
                save_result(result, f"m3/{run_name}")
                rows.append(result)
                print(f"  {run_name}: val macro-F1 {result['val']['macro_f1']:.2f} "
                      f"({result['scoring_rule']}, {result['timing']['examples_per_s']}/s), "
                      f"generation invalid {result['generation']['invalid_rate']:.0%}", flush=True)
            if model is not None:
                del model
                torch.cuda.empty_cache()

        by_variant = {v: [r["val"]["macro_f1"] for r in rows if r["variant"] == v and r["candidate"]]
                      for v in cfg["variants"]}
        chosen = max(by_variant, key=lambda v: statistics.mean(by_variant[v]))
        save_result({"experiment": "E2", "stage": "prompt_pilot", "split": "val",
                     "mean_macro_f1_over_candidates": {v: round(statistics.mean(s), 2)
                                                       for v, s in by_variant.items()},
                     "per_model": {f"{r['model']}/{r['variant']}": r["val"]["macro_f1"] for r in rows},
                     "chosen_variant": chosen,
                     "max_invalid_generation_rate": max(r["generation"]["invalid_rate"] for r in rows),
                     "gate_zero_invalid_scored_predictions": all(r["invalid_predictions"] == 0 for r in rows)},
                    "m3_prompt_pilot")
        print(f"\nchosen prompt variant: {chosen} "
              f"({ {v: round(statistics.mean(s), 2) for v, s in by_variant.items()} })", flush=True)

    if args.stage in ("fewshot", "all"):
        variant = read_json(RESULTS_DIR / "m3_prompt_pilot.json")["chosen_variant"]
        print(f"\nFew-shot with the frozen prompt {variant}, "
              f"{cfg['few_shot']['per_class']} examples per class", flush=True)
        table = []
        for name, cfg_model in cfg["models"].items():
            tok = model = None
            scores = []
            for draw in cfg["few_shot"]["draws"]:
                run_name = f"fewshot_{name}_{variant}_draw{draw}"
                if done(run_name):
                    scores.append(read_json(RESULTS_DIR / "m3" / f"{run_name}.json")["val"]["macro_f1"])
                    print(f"  {run_name}: already done", flush=True)
                    continue
                if model is None:
                    tok, model = load_4bit(cfg_model["repo_id"])
                shots = draw_shots(shot_pool, classes, seed=draw, per_class=cfg["few_shot"]["per_class"])
                result, _ = score_split(model, tok, val, variant, classes, cfg_model, shots=shots)
                result = {"experiment": "E2", "stage": "fewshot", "model": name,
                          "repo_id": cfg_model["repo_id"], "candidate": cfg_model["candidate"],
                          "variant": variant, "split": "val", "shots": len(shots), "draw": draw,
                          "shot_ids": [i for i, t in zip(shot_pool["id"], shot_pool["text"])
                                       if t in {s for s, _ in shots}],
                          "max_text_tokens": cfg_model["max_text_tokens"], **result}
                save_result(result, f"m3/{run_name}")
                scores.append(result["val"]["macro_f1"])
                print(f"  {run_name}: val macro-F1 {result['val']['macro_f1']:.2f} "
                      f"({result['timing']['examples_per_s']}/s)", flush=True)
            if model is not None:
                del model
                torch.cuda.empty_cache()
            table.append({"model": name, "candidate": cfg_model["candidate"], "variant": variant,
                          "val_macro_f1": {"mean": round(statistics.mean(scores), 2),
                                           "std": round(statistics.stdev(scores), 2)}})
        path = save_result({"experiment": "E2", "stage": "fewshot", "variant": variant,
                            "shots_per_class": cfg["few_shot"]["per_class"], "table": table},
                           "m3_e2_summary")
        print("\nmodel          few-shot val macro-F1")
        for row in table:
            f1 = row["val_macro_f1"]
            print(f"{row['model']:<14} {f1['mean']:6.2f} +- {f1['std']}")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
