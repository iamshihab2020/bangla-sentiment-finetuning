"""M0 environment check: 4-bit load of a ~1B model and a short QLoRA throughput benchmark.

Usage:
    python scripts/m0_env_check.py
    python scripts/m0_env_check.py --model md-nishat-008/TigerLLM-1B-it --batch-sizes 4,8,16

Writes results/m0_env_check.json. The training benchmark uses full-sequence loss on repeated
Bangla text, so it is a worst case for memory (real runs only need loss on the label tokens).
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import bitsandbytes as bnb
import peft
import torch
import transformers
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BANGLA_TEXT = "ভাই আপনার ক্যামেরা মেনকে দিলেয়া একাই সব সাবার করলেন, হা হা হা। "
PROMPT = "এই মন্তব্যটি ইতিবাচক, নেতিবাচক, নাকি নিরপেক্ষ? মন্তব্য: বইটা পড়ে খুব ভালো লাগলো।"


def gb(n_bytes):
    return round(n_bytes / 1e9, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="md-nishat-008/TigerLLM-1B-it")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-sizes", default="4,8,16")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default="results/m0_env_check.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"
    out = {
        "model": args.model,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "utf8_mode": sys.flags.utf8_mode,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bnb.__version__,
    }

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
    torch.cuda.reset_peak_memory_stats()
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

    # QLoRA training throughput
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
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for step in range(args.steps + 2):  # first 2 steps are warmup
                if step == 2:
                    torch.cuda.synchronize()
                    t = time.time()
                loss = model(input_ids=batch, labels=batch).loss
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.time() - t
            out["train_benchmark"].append({
                "batch_size": bs,
                "seq_len": args.seq_len,
                "s_per_step": round(dt / args.steps, 3),
                "tokens_per_s": round(bs * args.seq_len * args.steps / dt),
                "peak_vram_gb": gb(torch.cuda.max_memory_allocated()),
            })
        except torch.OutOfMemoryError:
            out["train_benchmark"].append({"batch_size": bs, "seq_len": args.seq_len, "result": "out of memory"})
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
