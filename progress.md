# Progress

Running log of the project. Newest entry first. Numbers here come from files in `results/`; anything that is only an estimate is marked as one.

Timeline: 2026-09-21 to 2026-12-18 (13 weeks, about 30 hours per week).

## Milestones

| # | Milestone | Planned weeks | Status |
| --- | --- | --- | --- |
| M0 | Environment and budget | W1 (Sep 21 to Sep 27) | Done (Sep 19) |
| M1 | Data | W2 (Sep 28 to Oct 4) | Nearly done (Sep 19). Waiting on Shihab's 50-label manual check |
| M2 | Baselines E0 and E1 | W3 (Oct 5 to Oct 11) | Not started |
| M3 | Pilot and E2 | W4 (Oct 12 to Oct 18) | Not started |
| M4 | RQ1 learning curves | W5 to W6 (Oct 19 to Nov 1) | Not started |
| M5 | Ablations | W7 (Nov 2 to Nov 8) | Not started |
| M6 | Secondary experiments | W8 to W9 (Nov 9 to Nov 22) | Not started |
| M7 | Analysis and report draft | W10 to W11 (Nov 23 to Dec 6) | Not started |
| M8 | Review and release | W12 to W13 (Dec 7 to Dec 18) | Not started |

## Log

<!-- New entry template:
### YYYY-MM-DD

#### Done
#### Findings
#### Next
#### Waiting on
-->

### 2026-09-19 (M1, data)

#### Done

- **Downloads:** SentNoB and bengali_sa downloaded at pinned revisions, with a sha256 recorded for every file.
- **Label mappings verified:** the class totals match the published counts exactly. SentNoB: 0 = neutral, 1 = positive, 2 = negative. bengali_sa: 0 = negative, 1 = positive.
- **One text pipeline for every model:** the BanglaBERT normalizer plus whitespace collapsing.
- **Duplicate and leakage checks** done (exact and near), and leaked train rows removed. Final train sizes: SentNoB 11,673, bengali_sa 7,345. Val and test are unchanged.
- **Nested, stratified training subsets** for seeds 0 to 2. The worst stratification error over all subset sizes is 0.7 examples per class.
- **Tokenizer statistics (A1)** for all 5 tokenizers (`results/m1_tokenizer_stats.json`).
- **Data card** generated: `reports/data_card.md`.
- **Blind sheet for the manual label check:** `data/processed/sentnob/label_check_50.csv` (gitignored).
- **Tests:** 12 passing, including label mapping, no dataset text in committed files, subset nesting, and checksums.

#### Findings

- **Leakage in SentNoB is real.** 902 train rows (7.2%) near-duplicate a val or test row. 81% are exact copies with a median of 13 words, and 9% of those pairs carry different labels. The published 72.89 was trained with these leaks, so M2 checks reproduction on the original train split and reports the gap to the cleaned one.
- **bengali_sa leakage is mostly short common phrases.** 919 rows leak, and 91% are 3 words or fewer; one phrase appears 90 times in train.
- **SentNoB train has 1,342 extra exact copies internally, and 118 duplicate groups carry conflicting labels.** This is kept and reported.
- **Tokenizer cost differs up to 5x on the same text** (tokens per word on SentNoB): BanglaBERT 1.16, Gemma and TigerLLM 1.43 (identical tokenizer), Qwen3.5 3.38, Llama 3.2 6.24. For Llama, 23% of comments exceed 128 tokens, so `max_seq_len` must be set per model.
- **The normalizer changes about 72% of the texts** (Unicode and punctuation normalization).

#### Next

- Shihab: fill in `your_label` in `data/processed/sentnob/label_check_50.csv` (positive, negative or neutral). A script then compares it with the hidden gold labels.
- M2: E0 baselines, then E1 BanglaBERT, starting with the reproduction check on `train_original`.

#### Waiting on

- The 50-label manual check.
- The go-ahead to commit. After committing, rerun the M1 scripts so the results point to a clean commit.

### 2026-09-19 (M0)

#### Done

- Reviewed the first plan and rewrote it. It now has research questions with predictions recorded in advance, SentNoB as the primary dataset, matched learning curves for encoder and LLM, 3 seeds plus bootstrap intervals, and a dated 13-week timeline.
- Added a zero-cost rule: free tools only, nothing that needs a card.
- Accounts:
  - Hugging Face iamshihab2020 is logged in, with access granted to both gated models (Gemma 3 1B and Llama 3.2 1B, both checked from code).
  - GitHub iamshihab2020.
  - Kaggle with a verified phone (confirmed by Shihab).
- Environment: Python 3.12 via uv, PyTorch 2.14 with CUDA 13.0, all versions pinned in `uv.lock`. `PYTHONUTF8=1` is set for Bangla text.
- M0 checks, all passing:
  - bitsandbytes 4-bit works on native Windows, so WSL is not needed.
  - TigerLLM-1B-it loads in 4-bit using 0.97 GB of VRAM (`results/m0_env_check.json`).
  - QLoRA at 128 tokens, batch 4: 4.32 examples/s (553 tokens/s), 3.3 GB peak VRAM. Batch 8: 2.04 examples/s, 5.0 GB peak VRAM. Batch 16: out of memory.
  - LLM inference, scoring only the last position: 24.6 examples/s at batch 32.
  - BanglaBERT (`results/m0_encoder_check.json`): training 108.3 examples/s at batch 32, 3.2 GB peak VRAM. Inference 447.8 examples/s at batch 128.
- Worst-case compute projection (`results/m0_budget_projection.json`): 33.5 GPU-hours for core experiments and 82.1 for everything, including the low-priority ablations. Written to the README.
- Checked for newer small models. Qwen3.5-2B replaced Qwen3-1.7B, and Gemma 4 E2B was left out (5.1B total parameters). Recorded in the PRD decision log.
- Repo organized, with 4 tests passing. Private GitHub repo iamshihab2020/bangla-sentiment-finetuning.
- Fixed run tracking: result files no longer mark a run as "dirty". All three M0 result files point to clean commits.

#### Findings

- **BanglaBook rejected.** Its labels come from star ratings, it is 89.6% positive, and 43% of its reviews were machine-translated.
- **TigerLLM-1B-it is a Gemma 3 1B model.** Its `config.json` says so, even though the paper says Llama 3.2. The RQ2 control is therefore Gemma-3-1B-it.
- **SentNoB is CC BY-ND 4.0.** No dataset text, raw or processed, goes into git.
- **Zero-shot generation starts reasoning step by step instead of giving a label.** This supports scoring by label probability rather than parsing generated text.
- **Batch 4 is the practical QLoRA setting.** Batch 8 runs close to the 6 GB limit and is about 2x slower per example, probably because it spills into shared system memory.
- **Speed varies between repeated runs.** Batch 4 did 449 tokens/s in the first run (commit 053d98e) and 553 tokens/s later, a difference of up to about 25%. Always use warmup and medians, and compare timings only within one machine.
- **Early signal only, not a result:** on padded dummy text, BanglaBERT inference was about 18x faster than the 4-bit 1B LLM (447.8 vs 24.6 examples/s). Prediction H1c is tested properly in the experiments.

#### Next

- M1: download SentNoB and bengali_sa, build the data card, dedupe and leakage report, nested subsets, and tokenizer statistics for all candidate tokenizers.
- Refine the compute projection with real token lengths once the data is in.

#### Waiting on

- Shihab's go-ahead to commit and push (see the M1 entry).
