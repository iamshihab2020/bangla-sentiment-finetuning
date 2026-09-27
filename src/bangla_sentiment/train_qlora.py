"""E3, E4, E5 and E6: QLoRA fine-tuning of a generative LLM for sentiment classification.

One run trains LoRA adapters on a 4-bit base, picks the best epoch on validation macro-F1, and only
then scores the test split. Nothing is ever selected on test.

Training and evaluation share one prompt path. A training example is the same prompt
`scoring.score_labels` builds, followed by the gold label word, and the loss is computed on the
label tokens only: every prompt token gets the ignore index. No end-of-turn token is appended,
because the metric reads the probability of the label word and nothing after it.

The loop is written out here rather than delegated to TRL's SFTTrainer, so that the prompt, the
masking and the validation metric are exactly the ones used everywhere else in this study.
"""
import copy
import gc
import math
import time
import traceback

import torch
from bitsandbytes.optim import PagedAdamW8bit
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training,
                  set_peft_model_state_dict)
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from bangla_sentiment.evaluate import classification_metrics
from bangla_sentiment.prompts import LABEL_WORDS, build_messages, label_words
from bangla_sentiment.scoring import first_tokens_differ, load_4bit, predictions_from_scores, score_labels, truncate
from bangla_sentiment.utils import OUTPUTS_DIR, set_seed

DEVICE = "cuda"
IGNORE = -100  # positions the loss must skip


def build_example(tok, text, label, variant, classes, max_text_tokens):
    """(token ids, per-token labels) for one training row, with the loss on the label word only."""
    messages = build_messages(truncate(tok, text, max_text_tokens), variant, classes)
    rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt = tok(rendered, add_special_tokens=False)["input_ids"]
    answer = tok(LABEL_WORDS[variant][label], add_special_tokens=False)["input_ids"]
    return prompt + answer, [IGNORE] * len(prompt) + answer


def make_loader(frame, tok, variant, classes, cfg_model, batch_size, shuffle, generator=None):
    """Batches padded on the right; padding is masked out of both attention and the loss."""
    rows = [build_example(tok, text, label, variant, classes, cfg_model["max_text_tokens"])
            for text, label in zip(frame["text"], frame["label"])]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch):
        width = max(len(ids) for ids, _ in batch)
        return {
            "input_ids": torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in batch]),
            "attention_mask": torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch]),
            "labels": torch.tensor([lab + [IGNORE] * (width - len(lab)) for _, lab in batch]),
        }

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=generator, num_workers=0)  # num_workers=0: Windows has no fork


def load_for_training(model_id, cfg):
    """4-bit base as in M0, with LoRA adapters attached and gradient checkpointing as configured."""
    tok, model = load_4bit(model_id)
    model.config.use_cache = False  # incompatible with gradient checkpointing, and unused here
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM"))
    return tok, model


def score_frame(model, tok, frame, variant, classes, cfg_model, rule):
    """Label-probability predictions for one split, using the rule frozen in M3."""
    model.eval()
    summed, first = score_labels(model, tok, frame["text"].tolist(), variant, classes,
                                 batch_size=cfg_model["eval_batch_size"],
                                 max_text_tokens=cfg_model["max_text_tokens"])
    pred, probs = predictions_from_scores(first if rule == "first_token" else summed, classes)
    return classification_metrics(frame["label"], pred, classes), pred, probs


def train_qlora(cfg, cfg_model, classes, train, val, lr, seed, variant, batch_size,
                eval_val=None, test=None, adapter_name=None):
    """Fine-tune one model. Returns (result, test predictions), predictions None unless `test` is given.

    `eval_val` is the split scored after every epoch to choose the best one. It defaults to `val`,
    and a smaller stratified sample of it keeps the epoch choice cheap on a laptop. The full `val`
    is always scored once at the end, with the chosen adapter.
    """
    set_seed(seed)
    tok, model = load_for_training(cfg_model["repo_id"], cfg)
    rule = "first_token" if first_tokens_differ(tok, label_words(variant, classes)) else "full_label"
    eval_val = val if eval_val is None else eval_val

    accumulation = max(1, cfg["effective_batch_size"] // batch_size)
    generator = torch.Generator().manual_seed(seed)
    loader = make_loader(train, tok, variant, classes, cfg_model, batch_size, shuffle=True, generator=generator)
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    epochs = max(cfg["max_epochs"], math.ceil(cfg["min_steps"] / steps_per_epoch))  # small subsets train longer
    total_steps = steps_per_epoch * epochs

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = PagedAdamW8bit(trainable, lr=lr, weight_decay=cfg["weight_decay"])
    schedule = get_cosine_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total_steps), total_steps)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history, best, train_seconds = [], {"val_macro_f1": -1.0}, 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = model(**batch).loss / accumulation
            loss.backward()
            if step % accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"])
                opt.step()
                schedule.step()
                opt.zero_grad(set_to_none=True)
        train_seconds += time.perf_counter() - epoch_start
        metrics, _, _ = score_frame(model, tok, eval_val, variant, classes, cfg_model, rule)
        history.append({"epoch": epoch, "eval_val_macro_f1": metrics["macro_f1"]})
        print(f"    epoch {epoch}/{epochs}: eval-val macro-F1 {metrics['macro_f1']:.2f}", flush=True)
        if metrics["macro_f1"] > best["val_macro_f1"]:  # no early stopping, just keep the best epoch
            best = {"epoch": epoch, "val_macro_f1": metrics["macro_f1"],
                    "state": copy.deepcopy({k: v.cpu() for k, v in get_peft_model_state_dict(model).items()})}
    peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    set_peft_model_state_dict(model, best.pop("state"))  # the epoch chosen on validation
    if adapter_name:
        model.save_pretrained(OUTPUTS_DIR / adapter_name)  # adapters only, never a full model copy
    eval_start = time.perf_counter()
    val_metrics, _, _ = score_frame(model, tok, val, variant, classes, cfg_model, rule)
    val_seconds = time.perf_counter() - eval_start

    result = {
        "model": cfg_model["repo_id"],
        "prompt_variant": variant,
        "scoring_rule": rule,
        "learning_rate": lr,
        "seed": seed,
        "train_rows": len(train),
        "epochs_run": epochs,
        "optimizer_steps": total_steps,
        "best_epoch": best["epoch"],
        "epoch_history": history,
        "epoch_selection_rows": len(eval_val),
        "selection_done_on": "val",
        "efficiency": {
            "trainable_params": sum(p.numel() for p in trainable),
            "trainable_pct": round(100 * sum(p.numel() for p in trainable)
                                   / sum(p.numel() for p in model.parameters()), 3),
            "train_batch_size": batch_size,
            "gradient_accumulation": accumulation,
            "train_seconds": round(train_seconds, 1),
            "train_examples_per_s": round(len(train) * epochs / train_seconds, 2),
            "val_seconds": round(val_seconds, 1),
            "val_examples_per_s": round(len(val) / val_seconds, 2),
            "peak_vram_gb": peak_vram_gb,
            "total_seconds": round(time.perf_counter() - start, 1),
        },
        "val": val_metrics,
    }

    predictions = None
    if test is not None:  # only after the epoch has been chosen on validation
        test_metrics, pred, probs = score_frame(model, tok, test, variant, classes, cfg_model, rule)
        result["test"] = test_metrics
        predictions = {"ids": test["id"], "gold": test["label"], "pred": pred, "proba": probs}

    del model, opt, trainable
    gc.collect()
    torch.cuda.empty_cache()
    return result, predictions


def train_with_oom_backoff(cfg, cfg_model, *args, **kwargs):
    """The PRD out-of-memory rule: halve the batch and keep the effective batch at 16, then retry.

    The sequence length is never shortened to force a fit. A model that cannot train at batch 1 is
    reported as impractical on this GPU, which is itself an RQ3 result.
    """
    batch_size = cfg_model["train_batch_size"]
    reductions = []
    while True:
        try:
            result, predictions = train_qlora(cfg, cfg_model, *args, batch_size=batch_size, **kwargs)
            result["efficiency"]["batch_size_reductions"] = reductions
            return result, predictions
        except (torch.OutOfMemoryError, RuntimeError) as error:
            if not isinstance(error, torch.OutOfMemoryError) and "out of memory" not in str(error).lower():
                raise  # some other failure: do not hide it behind a smaller batch
            traceback.clear_frames(error.__traceback__)  # frees the half-built model on the GPU
            gc.collect()
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise RuntimeError(
                    f"{cfg_model['repo_id']} runs out of memory at batch 1 and "
                    f"{cfg_model['max_text_tokens']} text tokens: impractical on this GPU") from None
            reductions.append({"from": batch_size, "to": batch_size // 2, "reason": "CUDA out of memory"})
            batch_size //= 2
            print(f"    out of memory, retrying at batch size {batch_size}", flush=True)
