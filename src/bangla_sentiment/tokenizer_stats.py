"""A1: tokenizer cost on Bangla text (tokens per word and per character, length percentiles)."""
import re

import numpy as np

BYTE_TOKEN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")  # SentencePiece byte-fallback pieces


def tokenizer_stats(tokenizer, texts):
    """Token statistics for a list of texts, without special tokens."""
    ids = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    n_tokens = np.array([len(x) for x in ids])
    n_words = sum(len(t.split()) for t in texts)
    n_chars = sum(len(t) for t in texts)
    total = int(n_tokens.sum())

    unk_id = tokenizer.unk_token_id
    unk = sum(x.count(unk_id) for x in ids) if unk_id is not None else None
    used = sorted({i for x in ids for i in x})
    byte_ids = {i for i, tok in zip(used, tokenizer.convert_ids_to_tokens(used)) if tok and BYTE_TOKEN.match(tok)}
    byte = sum(i in byte_ids for x in ids for i in x)

    return {
        "vocab_size": len(tokenizer),
        "texts": len(texts),
        "tokens_per_word": round(total / n_words, 3),
        "tokens_per_char": round(total / n_chars, 3),
        "tokens_per_text": {
            "mean": round(float(n_tokens.mean()), 1),
            "median": round(float(np.median(n_tokens)), 1),
            "p95": round(float(np.percentile(n_tokens, 95)), 1),
            "p99": round(float(np.percentile(n_tokens, 99)), 1),
            "max": int(n_tokens.max()),
        },
        "share_over_64_tokens": round(float((n_tokens > 64).mean()), 4),
        "share_over_128_tokens": round(float((n_tokens > 128).mean()), 4),
        "unk_rate": round(unk / total, 5) if unk is not None else None,
        "byte_fallback_rate": round(byte / total, 5),
    }
