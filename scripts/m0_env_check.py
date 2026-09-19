"""M0 environment check: 4-bit load of a ~1B model, QLoRA training speed and inference speed.

Usage:
    uv run python scripts/m0_env_check.py
    uv run python scripts/m0_env_check.py --model md-nishat-008/TigerLLM-1B-it --batch-sizes 4,8,16

Writes results/m0_env_check.json. Worst case on purpose: every example is padded to --seq-len
and training uses full-sequence loss (real runs only need loss on the label tokens).
"""
import argparse
import time

import bitsandbytes as bnb
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from bangla_sentiment.utils import ensure_utf8, save_result, set_seed

BANGLA_TEXT = "ভাই আপনার ক্যামেরা মেনকে দিলেয়া একাই সব সাবার করলেন, হা হা হা। "
PROMPT = "এই মন্তব্যটি ইতিবাচক, নেতিবাচক, নাকি নিরপেক্ষ? মন্তব্য: বইটা পড়ে খুব ভালো লাগলো।"


def gb(n_bytes):
    return round(n_bytes / 1e9, 3)


def timed(step, n_steps, warmup=2):
    """Average seconds per call of step(), after warmup calls."""
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n_steps):
        step()
    torch.cuda.synchronize()
    return (time.time() - t) / n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="md-nishat-008/TigerLLM-1B-it")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-sizes", default="4,8,16")
    ap.add_argument("--infer-batch-sizes", default="16,32")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    ensure_utf8()
    set_seed(0)
    assert torch.cuda.is_available(), "CUDA is not available"
    out = {"model": args.model, "seq_len": args.seq_len}

    t = time.time()
    snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt"])
    out["download_s"] = round(time.time() - t, 1)

    # 4-bit load straight to the GPU
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    t = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_cfg, device_map={"": 0}, dtype=torch.bfloat16
    )
    out["load_s"] = round(time.time() - t, 1)
    out["vram_after_4bit_load_gb"] = gb(torch.cuda.memory_allocated())

    # Tokenizer cost on one sample (a first look; A1 does this properly on the dataset)
    n_words = len(BANGLA_TEXT.split())
    n_tokens = len(tok(BANGLA_TEXT, add_special_tokens=False)["input_ids"])
    out["sample_tokens_per_word"] = round(n_tokens / n_words, 2)

    # Generation sanity check
    enc = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(0)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=16, do_sample=False)
    out["sample_generation"] = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # QLoRA training speed
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainable, total = model.get_nb_trainable_parameters()
    out["lora_trainable_params"] = trainable
    out["lora_trainable_pct"] = round(100 * trainable / total, 3)
    model.train()
    opt = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    ids = tok(BANGLA_TEXT * 64, return_tensors="pt")["input_ids"][:, : args.seq_len]
    assert ids.shape[1] == args.seq_len, "sample text too short for seq_len"

    out["train_benchmark"] = []
    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        batch = ids.repeat(bs, 1).to(0)

        def train_step():
            loss = model(input_ids=batch, labels=batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            s = timed(train_step, args.steps)
            out["train_benchmark"].append({
                "batch_size": bs,
                "s_per_step": round(s, 3),
                "tokens_per_s": round(bs * args.seq_len / s),
                "examples_per_s": round(bs / s, 2),
                "peak_vram_gb": gb(torch.cuda.max_memory_allocated()),
            })
        except torch.OutOfMemoryError:
            out["train_benchmark"].append({"batch_size": bs, "result": "out of memory"})
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break

    # Inference speed for label-probability scoring: logits for the last position only
    del opt
    torch.cuda.empty_cache()
    model.eval()
    out["infer_benchmark"] = []
    for bs in [int(b) for b in args.infer_batch_sizes.split(",")]:
        batch = ids.repeat(bs, 1).to(0)

        def infer_step():
            with torch.no_grad():
                model(input_ids=batch, logits_to_keep=1)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            s = timed(infer_step, args.steps)
            out["infer_benchmark"].append({
                "batch_size": bs,
                "examples_per_s": round(bs / s, 2),
                "ms_per_example": round(1000 * s / bs, 2),
                "peak_vram_gb": gb(torch.cuda.max_memory_allocated()),
            })
        except torch.OutOfMemoryError:
            out["infer_benchmark"].append({"batch_size": bs, "result": "out of memory"})
            torch.cuda.empty_cache()
            break

    path = save_result(out, "m0_env_check")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
