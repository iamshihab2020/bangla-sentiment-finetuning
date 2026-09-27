"""A1: tokenizer cost on Bangla text (tokens per word and per character, length percentiles).

Fertility (tokens per word) depends on how it is measured, and published Bangla figures differ from
ours mainly because of the corpus, not the tokenizer. Three measurements are reported so that ours
can be compared with anyone else's:
  all_words              every whitespace word, including Latin, digits and emoji
  bangla_words_only      only words whose letters, marks and digits are all Bengali
  distinct_bangla_words  each distinct Bangla word counted once, so common words stop dominating
The first is the headline number, and the last is the one closest to figures published on clean
corpora, because those have far more word types per token than 178k words of social media text.
"""
import re
import unicodedata

import numpy as np

BYTE_TOKEN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")  # SentencePiece byte-fallback pieces
BENGALI = range(0x0980, 0x0A00)
TYPES_PER_LINE = 50  # distinct words are joined into lines, so each one keeps a leading space


def is_bangla_word(word):
    """True when the word's letters, marks and digits are all Bengali. Punctuation is ignored."""
    core = [ch for ch in word if unicodedata.category(ch)[0] in "LMN"]
    return bool(core) and all(ord(ch) in BENGALI for ch in core)


def bangla_only(texts):
    """(texts keeping only their Bangla words, one line per 50 distinct Bangla words)."""
    kept = [" ".join(w for w in text.split() if is_bangla_word(w)) for text in texts]
    kept = [text for text in kept if text]
    types = sorted({word for text in kept for word in text.split()})
    return kept, [" ".join(types[i:i + TYPES_PER_LINE]) for i in range(0, len(types), TYPES_PER_LINE)]


def count_tokens(tokenizer, texts):
    return sum(len(x) for x in tokenizer(list(texts), add_special_tokens=False)["input_ids"])


def words_in(texts):
    return sum(len(text.split()) for text in texts)


def tokenizer_stats(tokenizer, texts):
    """Token statistics for a list of texts, without special tokens."""
    ids = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    n_tokens = np.array([len(x) for x in ids])
    n_words = words_in(texts)
    n_chars = sum(len(t) for t in texts)
    total = int(n_tokens.sum())

    unk_id = tokenizer.unk_token_id
    unk = sum(x.count(unk_id) for x in ids) if unk_id is not None else None
    used = sorted({i for x in ids for i in x})
    byte_ids = {i for i, tok in zip(used, tokenizer.convert_ids_to_tokens(used)) if tok and BYTE_TOKEN.match(tok)}
    byte = sum(i in byte_ids for x in ids for i in x)

    bangla_texts, type_lines = bangla_only(texts)
    bangla_words, type_words = words_in(bangla_texts), words_in(type_lines)

    return {
        "vocab_size": len(tokenizer),
        "texts": len(texts),
        "tokens_per_word": round(total / n_words, 3),
        "tokens_per_char": round(total / n_chars, 3),
        "fertility": {
            "definition": "tokens divided by whitespace words, on this split",
            "all_words": round(total / n_words, 3),
            "bangla_words_only": round(count_tokens(tokenizer, bangla_texts) / bangla_words, 3),
            "distinct_bangla_words": round(count_tokens(tokenizer, type_lines) / type_words, 3),
            "bangla_word_share": round(bangla_words / n_words, 4),
            "distinct_bangla_words_count": type_words,
        },
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
