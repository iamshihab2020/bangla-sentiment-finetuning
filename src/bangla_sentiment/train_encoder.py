"""E1 and E6: full fine-tuning of an encoder (BanglaBERT) for sentiment classification.

One run trains on a training subset, picks the best epoch on validation macro-F1, and only then
scores the test split with those weights. Nothing is ever selected on test.
"""
import copy
import math
import statistics
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from bangla_sentiment.evaluate import classification_metrics
from bangla_sentiment.utils import set_seed

DEVICE = "cuda"


def make_loader(frame, classes, tok, cfg, batch_size, shuffle, generator=None):
    """Batches of (text, label index). Texts are padded to the longest text in the batch."""
    index = {c: i for i, c in enumerate(classes)}
    rows = list(zip(frame["text"], [index[lab] for lab in frame["label"]]))

    def collate(batch):
        texts, labels = zip(*batch)
        out = tok(list(texts), truncation=True, max_length=cfg["max_seq_len"], padding=True, return_tensors="pt")
        out["labels"] = torch.tensor(labels)
        return out

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=generator, num_workers=0)  # num_workers=0: Windows has no fork


@torch.no_grad()
def predict(model, loader, classes):
    """Class probabilities for every row of the loader, in order."""
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        batch.pop("labels")
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), dim=-1).cpu())
    probs = torch.cat(probs).numpy()
    return probs, [classes[i] for i in probs.argmax(axis=1)]


def inference_speed(model, frame, classes, tok, cfg, batch_sizes=(1, 32), n_examples=256, passes=3):
    """Median ms per example over `passes` passes, after one warmup pass. Same text for every size."""
    sample = frame.head(n_examples)
    speed = {}
    for bs in batch_sizes:
        loader = make_loader(sample, classes, tok, cfg, bs, shuffle=False)
        predict(model, loader, classes)  # warmup
        times = []
        for _ in range(passes):
            torch.cuda.synchronize()
            start = time.perf_counter()
            predict(model, loader, classes)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        speed[f"ms_per_example_batch_{bs}"] = round(1000 * statistics.median(times) / len(sample), 3)
    return speed


def optimizer_groups(model, weight_decay):
    """No weight decay on bias and LayerNorm parameters, the standard BERT recipe."""
    no_decay = [p for n, p in model.named_parameters() if n.endswith("bias") or "LayerNorm" in n]
    decay = [p for n, p in model.named_parameters() if not (n.endswith("bias") or "LayerNorm" in n)]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def train_encoder(cfg, classes, train, val, test, lr, seed, score_test=True):
    """Fine-tune one model. Returns (result, test predictions) where predictions is None if score_test is False."""
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model"], num_labels=len(classes)).to(DEVICE)

    generator = torch.Generator().manual_seed(seed)
    train_loader = make_loader(train, classes, tok, cfg, cfg["train_batch_size"], shuffle=True, generator=generator)
    val_loader = make_loader(val, classes, tok, cfg, cfg["eval_batch_size"], shuffle=False)

    steps_per_epoch = math.ceil(len(train) / cfg["train_batch_size"])
    epochs = max(cfg["max_epochs"], math.ceil(cfg["min_steps"] / steps_per_epoch))  # small subsets train longer
    total_steps = steps_per_epoch * epochs
    opt = torch.optim.AdamW(optimizer_groups(model, cfg["weight_decay"]), lr=lr)
    schedule = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total_steps), total_steps)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history, best = [], {"val_macro_f1": -1.0}
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            opt.step()
            schedule.step()
            opt.zero_grad(set_to_none=True)
        _, val_pred = predict(model, val_loader, classes)
        val_metrics = classification_metrics(val["label"], val_pred, classes)
        history.append({"epoch": epoch, "val_macro_f1": val_metrics["macro_f1"]})
        if val_metrics["macro_f1"] > best["val_macro_f1"]:  # no early stopping, just keep the best epoch
            best = {"epoch": epoch, "val_macro_f1": val_metrics["macro_f1"], "val": val_metrics,
                    "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})}
    train_seconds = time.perf_counter() - start
    peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    model.load_state_dict(best.pop("state"))  # the epoch chosen on validation
    result = {
        "model": cfg["model"],
        "learning_rate": lr,
        "seed": seed,
        "train_rows": len(train),
        "epochs_run": epochs,
        "optimizer_steps": total_steps,
        "best_epoch": best["epoch"],
        "epoch_history": history,
        "selection_done_on": "val",
        "efficiency": {
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "train_seconds": round(train_seconds, 1),
            "peak_vram_gb": peak_vram_gb,
            **inference_speed(model, test, classes, tok, cfg),
        },
        "val": best["val"],
    }

    predictions = None
    if score_test:  # only after the epoch has been chosen on validation
        test_loader = make_loader(test, classes, tok, cfg, cfg["eval_batch_size"], shuffle=False)
        probs, pred = predict(model, test_loader, classes)
        result["test"] = classification_metrics(test["label"], pred, classes)
        predictions = {"ids": test["id"], "gold": test["label"], "pred": pred, "proba": probs}

    del model, opt
    torch.cuda.empty_cache()
    return result, predictions
