"""M0 encoder check: BanglaBERT training and inference speed on this GPU.

Usage:
    uv run python scripts/m0_encoder_check.py

Writes results/m0_encoder_check.json. Worst case on purpose: every example is padded to --seq-len.
"""
import argparse
import time

import torch
from normalizer import normalize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bangla_sentiment.utils import ensure_utf8, save_result, set_seed

BANGLA_TEXT = "ভাই আপনার ক্যামেরা মেনকে দিলেয়া একাই সব সাবার করলেন, হা হা হা। "


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
    ap.add_argument("--model", default="csebuetnlp/banglabert")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--train-batch-size", type=int, default=32)
    ap.add_argument("--infer-batch-sizes", default="1,32,128")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    ensure_utf8()
    set_seed(0)
    assert torch.cuda.is_available(), "CUDA is not available"
    out = {"model": args.model, "seq_len": args.seq_len}

    t = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3).to("cuda")
    out["load_s"] = round(time.time() - t, 1)
    out["params"] = sum(p.numel() for p in model.parameters())

    text = normalize(BANGLA_TEXT * 20)

    def make_batch(bs):
        return tok([text] * bs, truncation=True, max_length=args.seq_len, padding="max_length",
                   return_tensors="pt").to("cuda")

    # Training speed: full fine-tuning, bf16 autocast, AdamW
    bs = args.train_batch_size
    batch = make_batch(bs)
    labels = torch.zeros(bs, dtype=torch.long, device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    model.train()

    def train_step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    s = timed(train_step, args.steps)
    out["train_benchmark"] = {
        "batch_size": bs,
        "s_per_step": round(s, 3),
        "examples_per_s": round(bs / s, 1),
        "peak_vram_gb": gb(torch.cuda.max_memory_allocated()),
    }

    # Inference speed
    del opt
    torch.cuda.empty_cache()
    model.eval()
    out["infer_benchmark"] = []
    for bs in [int(b) for b in args.infer_batch_sizes.split(",")]:
        batch = make_batch(bs)

        def infer_step():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(**batch)

        torch.cuda.reset_peak_memory_stats()
        s = timed(infer_step, args.steps)
        out["infer_benchmark"].append({
            "batch_size": bs,
            "examples_per_s": round(bs / s, 1),
            "ms_per_example": round(1000 * s / bs, 3),
            "peak_vram_gb": gb(torch.cuda.max_memory_allocated()),
        })

    path = save_result(out, "m0_encoder_check")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
