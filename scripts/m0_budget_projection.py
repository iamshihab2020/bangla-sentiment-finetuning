"""M0 budget projection: worst-case GPU-hours for the full experiment grid (PRD section 7).

Usage:
    uv run python scripts/m0_budget_projection.py

Reads measured speeds from results/m0_env_check.json and results/m0_encoder_check.json,
applies the run plan below, and writes results/m0_budget_projection.json.

Worst case on purpose: every example is padded to 128 tokens, and every run trains for its
maximum number of epochs. Refine at M1 with real token lengths.
"""
import json

from bangla_sentiment.utils import RESULTS_DIR, save_result

D1_FULL = 12_582  # 80% of SentNoB's 15,728 rows; exact official train count is checked at M1
D2_FULL = 8_260  # bengali_sa official train split
EVAL_EXAMPLES_PER_RUN = 4 * 1_600  # up to 3 validation passes plus 1 test pass

LLM = {"batch": 16, "min_steps": 150, "max_epochs": 3}
ENCODER = {"batch": 32, "min_steps": 150, "max_epochs": 5}
CURVE = [250, 500, 1_000, 2_000, 4_000, D1_FULL]

# (name, kind, train sizes, runs per size, priority)
PLAN = [
    ("M3 pilot: 3 candidates at 1k", "llm", [1_000], 3, "core"),
    ("E3 learning-rate grid", "llm", [1_000], 3, "core"),
    ("E3 learning curve", "llm", CURVE, 3, "core"),
    ("E4a LoRA on bf16 base", "llm", [1_000, D1_FULL], 3, "core"),
    ("E4b classification head", "llm", [250, 1_000, D1_FULL], 3, "core"),
    ("E5 TigerLLM", "llm", [500, 2_000, D1_FULL], 3, "secondary"),
    ("E6 QLoRA on bengali_sa", "llm", [250, 1_000, D2_FULL], 3, "secondary"),
    ("E4c ranks 8 and 32", "llm", [D1_FULL], 6, "low"),
    ("E4d attention-only modules", "llm", [D1_FULL], 3, "low"),
    ("E5 Gemma control (only if Gemma is not the main LLM)", "llm", [500, 2_000, D1_FULL], 3, "conditional"),
    ("E1 learning-rate grid", "encoder", [1_000], 3, "core"),
    ("E1 learning curve", "encoder", CURVE, 3, "core"),
    ("E6 BanglaBERT on bengali_sa", "encoder", [250, 1_000, D2_FULL], 3, "secondary"),
]


def trained_examples(n, cfg):
    """Examples processed by one run: max epochs, but at least min_steps optimizer steps."""
    return max(n * cfg["max_epochs"], cfg["min_steps"] * cfg["batch"])


def load(name):
    return json.loads((RESULTS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def main():
    llm, enc = load("m0_env_check"), load("m0_encoder_check")
    speed = {
        "llm": {
            "train": max(r["examples_per_s"] for r in llm["train_benchmark"] if "examples_per_s" in r),
            "eval": max(r["examples_per_s"] for r in llm["infer_benchmark"] if "examples_per_s" in r),
        },
        "encoder": {
            "train": enc["train_benchmark"]["examples_per_s"],
            "eval": max(r["examples_per_s"] for r in enc["infer_benchmark"]),
        },
    }
    cfgs = {"llm": LLM, "encoder": ENCODER}

    rows = []
    for name, kind, sizes, runs, priority in PLAN:
        n_runs = runs * len(sizes)
        train_ex = runs * sum(trained_examples(n, cfgs[kind]) for n in sizes)
        eval_ex = n_runs * EVAL_EXAMPLES_PER_RUN
        hours = (train_ex / speed[kind]["train"] + eval_ex / speed[kind]["eval"]) / 3600
        rows.append({"experiment": name, "kind": kind, "priority": priority, "runs": n_runs,
                     "train_examples": train_ex, "gpu_hours": round(hours, 1)})

    totals = {}
    for p in ["core", "secondary", "low", "conditional"]:
        sel = [r for r in rows if r["priority"] == p]
        totals[p] = {"runs": sum(r["runs"] for r in sel), "gpu_hours": round(sum(r["gpu_hours"] for r in sel), 1)}
    totals["all"] = {"runs": sum(r["runs"] for r in rows), "gpu_hours": round(sum(r["gpu_hours"] for r in rows), 1)}

    out = {
        "assumptions": {
            "padding": "every example padded to the benchmark sequence length (worst case)",
            "llm": LLM, "encoder": ENCODER, "eval_examples_per_run": EVAL_EXAMPLES_PER_RUN,
            "d1_full": D1_FULL, "d2_full": D2_FULL,
            "llm_speed_source": f"results/m0_env_check.json ({llm['model']})",
            "encoder_speed_source": f"results/m0_encoder_check.json ({enc['model']})",
        },
        "measured_examples_per_s": speed,
        "rows": rows,
        "totals": totals,
    }
    path = save_result(out, "m0_budget_projection")
    for r in rows:
        print(f"{r['gpu_hours']:>7} h  {r['runs']:>3} runs  {r['priority']:<11} {r['experiment']}")
    print("totals:", json.dumps(totals))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
