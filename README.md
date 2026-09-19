# Bangla Sentiment: Data Efficiency of Small LLMs vs BanglaBERT

How much labeled data does a small general-purpose LLM fine-tuned with QLoRA need to match a Bangla-native encoder (BanglaBERT) on noisy Bangla sentiment classification, and what does it cost on a 6 GB laptop GPU?

**Status:** M0 (environment) done, M1 (data) nearly done; see [reports/data_card.md](reports/data_card.md). There are no experimental results yet. Every number that appears here later will come from a results file produced by an actual run. See [progress.md](progress.md) for the running log.

## Research questions

- **RQ1, data efficiency (headline).** How does test macro-F1 grow with the number of labeled training examples for BanglaBERT (full fine-tuning) and for a 1B to 2B multilingual LLM (QLoRA)? Where do the learning curves cross, and what does each point cost in training time, memory and inference speed?
- **RQ2, Bangla adaptation.** Does a Bangla-adapted LLM (TigerLLM-1B-it) keep its advantage over the general model with the same architecture (Gemma-3-1B-it) after identical QLoRA fine-tuning?
- **RQ3, tokenizer cost.** How many tokens per Bangla word does each model use, and does that track cost and accuracy?

## Hardware

Everything is designed to run on one consumer laptop, using only free tools.

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 3050 laptop, 6 GB VRAM |
| System RAM | 8 GB |
| OS | Windows 11 (native, no WSL) |
| Fallback | Free Kaggle or Colab GPU, only for whole experiment groups |

## Setup

Requirements: an NVIDIA GPU with a recent driver, [uv](https://docs.astral.sh/uv/), and git. You do not need to install the CUDA Toolkit, because the PyTorch wheels include the CUDA runtime.

```powershell
git clone <this repo>
cd bangla-sentiment-finetuning
uv sync                      # creates .venv with Python 3.12 and the exact locked versions

# Windows only, once: make Python read and write files as UTF-8 (needed for Bangla), then open a new terminal
setx PYTHONUTF8 1

uv run pytest                # environment tests
uv run python scripts/m0_env_check.py   # GPU, 4-bit loading and QLoRA speed check
```

Gated models (Gemma, Llama) need a free Hugging Face account, a Read token (`uv run hf auth login`) and an accepted license on each model page.

## Compute budget (M0 projection)

This is a worst-case estimate, produced by `scripts/m0_budget_projection.py` from speeds measured on the laptop (`results/m0_env_check.json`, `results/m0_encoder_check.json`). It assumes every example is padded to 128 tokens and every run trains for its maximum number of epochs. It will be refined at M1 with real text lengths.

| Priority | Runs | GPU-hours |
|---|---|---|
| Core | 60 | 33.5 |
| Secondary | 27 | 16.5 |
| Low (cut first) | 9 | 22.5 |
| Conditional (Gemma control, only if Gemma is not the main LLM) | 9 | 9.6 |
| **All** | **105** | **82.1** |

Measured speeds behind these numbers:

| Model | Training | Inference |
|---|---|---|
| QLoRA, TigerLLM-1B in 4-bit (batch 4, 128 tokens) | 4.32 examples/s, 3.3 GB peak VRAM | 24.6 examples/s |
| BanglaBERT (batch 32, 128 tokens) | 108.3 examples/s | 447.8 examples/s |

Notes:
- The LLM hours assume a 1B model. If the main LLM chosen at M3 is the larger Qwen3.5-2B, LLM hours will be higher.
- Timings vary by up to about 25% between repeated runs on this laptop.

## Repository layout

```
configs/                 one YAML file per experiment (prompt and scoring rule frozen after the pilot)
data/                    datasets live here but are NOT committed (see data/README.md)
notebooks/               optional exploration only, never the source of reported numbers
reports/                 technical report
results/                 one JSON file per run, saved predictions (IDs and labels only), tables, plots
scripts/                 one-command runners per experiment
src/bangla_sentiment/    shared code: data, prompts, scoring, training, evaluation, statistics
tests/                   unit tests (UTF-8 handling, splits, label mapping, metrics, loss masking)
pyproject.toml, uv.lock  pinned environment
```

## Data

Datasets are not included in this repository because of their licenses. Scripts download the original files and rebuild everything locally.

| Dataset | Role | License |
|---|---|---|
| [SentNoB](https://github.com/KhondokerIslam/SentNoB) (Islam et al., 2021) | Primary (3 classes, noisy social media comments) | CC BY-ND 4.0 (per the Hugging Face card) |
| [bengali_sa](https://huggingface.co/datasets/DGurgurov/bengali_sa) (from Sazzed, 2020) | Replication (binary, YouTube drama reviews) | MIT on Hugging Face; the original repository states no license |

Only row IDs, split checksums and predictions without text are committed. To rebuild the data locally:

```powershell
uv run python scripts/m1_prepare_data.py     # download, normalize, remove leaked rows, build subsets
uv run python scripts/m1_tokenizer_stats.py  # tokenizer cost per model
uv run python scripts/m1_data_card.py        # reports/data_card.md
```

Train rows that near-duplicate a validation or test row are removed (SentNoB 902, bengali_sa 919). Validation and test are unchanged. Details are in the data card.

## Reproducibility

- Every run writes a JSON file to `results/` recording its metrics, config, seed, git commit, exact command, library versions and hardware.
- Tables and plots are generated by scripts from `results/` only.
- `uv.lock` pins every dependency.

## License

Code: MIT (see `LICENSE`). Datasets and models keep their own licenses: BanglaBERT is CC BY-NC-SA 4.0, Gemma models use the Gemma Terms of Use, Llama 3.2 uses the Llama 3.2 Community License, Qwen3.5 is Apache 2.0, and TigerLLM is listed as CC BY 4.0.
