"""M1 / A1: tokenizer cost of every candidate model on the normalized training text.

Usage:
    uv run python scripts/m1_tokenizer_stats.py

Downloads tokenizer files only (no model weights). Writes results/m1_tokenizer_stats.json.
"""
from transformers import AutoTokenizer

from bangla_sentiment.data import DATASETS, load_split
from bangla_sentiment.tokenizer_stats import tokenizer_stats
from bangla_sentiment.utils import ensure_utf8, save_result

TOKENIZERS = {
    "banglabert": "csebuetnlp/banglabert",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma-3-1b": "google/gemma-3-1b-it",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "tigerllm-1b": "md-nishat-008/TigerLLM-1B-it",
}


def main():
    ensure_utf8()
    texts = {name: load_split(name, "train")["text"].tolist() for name in DATASETS}
    out = {}
    for key, repo_id in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(repo_id)
        out[key] = {"repo_id": repo_id, **{name: tokenizer_stats(tok, t) for name, t in texts.items()}}
        s = out[key]["sentnob"]
        print(f"{key:<13} sentnob: {s['tokens_per_word']} tokens/word, p99 {s['tokens_per_text']['p99']} tokens")
    path = save_result({"split": "train", "tokenizers": out}, "m1_tokenizer_stats")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
