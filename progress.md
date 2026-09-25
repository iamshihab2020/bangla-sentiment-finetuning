# Progress

Running log of the project. Newest entry first. Numbers here come from files in `results/`; anything that is only an estimate is marked as one.

Timeline: 2026-09-21 to 2026-12-18 (13 weeks, about 30 hours per week).

## Milestones

| # | Milestone | Planned weeks | Status |
| --- | --- | --- | --- |
| M0 | Environment and budget | W1 (Sep 21 to Sep 27) | Done (Sep 19) |
| M1 | Data | W2 (Sep 28 to Oct 4) | Done (Sep 19) |
| M2 | Baselines E0 and E1 | W3 (Oct 5 to Oct 11) | Done (Sep 25). Both gates passed |
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

### 2026-09-25 (M2 done, E1 BanglaBERT)

#### Done

- **E1 code:** `src/bangla_sentiment/train_encoder.py`, `configs/e1.yaml`, `scripts/m2_e1_banglabert.py`, committed before the runs (`2a09114`), so every result points at a clean commit. Best epoch is chosen on validation and its weights are kept in memory; test is scored once at the end. 3 new tests, 19 passing.
- **Reproduction gate passed:** learning-rate grid on `train_original` (validation only) chose 5e-5, then 3 seeds scored **72.39 ± 1.17** test macro-F1 against the published **72.89** (`results/e1_reproduction.json`). Difference 0.5 points, tolerance 3.
- **Learning-rate grid on the cleaned split** (1k rows, seed 0, validation only): 2e-5 55.07, 3e-5 61.91, **5e-5 64.02**. 5e-5 is frozen for all later E1, E4, E5 and E6 runs (`results/e1_lr_grid.json`).
- **E1 learning curve** (test macro-F1, mean ± sd over 3 seeds, `results/e1_summary.json`): 250: 60.42 ± 1.03, 500: 61.87 ± 1.03, 1k: 63.13 ± 1.34, 2k: 63.91 ± 1.26, 4k: 65.48 ± 0.73, full (11,673): 68.13 ± 0.35.
- **Cost:** 2.3 to 2.9 GB peak VRAM per run, 38 s at 1k rows and 349 s at full data. The whole of E1 was 24 runs in about 1 hour 25 minutes.

#### Findings

- **Leakage inflates BanglaBERT by 4.26 points:** 72.39 on the original train split against 68.13 on the cleaned one. The word-counting baseline moved 5.84 points (67.87 against 62.03). So the published 72.89 is partly a leakage effect, and the study's honest number for BanglaBERT on SentNoB is about 68.
- **Pretraining is worth roughly 11,000 labeled examples here.** BanglaBERT with 250 examples scores 60.42, close to word counting trained on all 11,673 rows (62.03). Against the same 250 rows, word counting gets 43.49.
- **The curve is still rising at full data** (65.48 at 4k to 68.13 at 11.7k), so more data would still help. This is the point where adding BLP-2023 could be considered after M4.
- **At full data the best epoch was epoch 1 for two of three seeds.** The learning rate is frozen at the value tuned on 1k rows, by design and equally for the LLM, but it means the large-data runs converge in one pass and then overfit. Worth a sentence in the limitations.
- **Validation and test track each other closely** (full data: 68.01 validation against 68.13 test), which is a good sign that the selection procedure is not overfitting the validation set.

#### Next

- M3: E2 zero-shot and few-shot for all LLM candidates, prompt language pilot (P-en vs P-bn) on validation, a 1k QLoRA pilot per candidate, then Shihab picks the main LLM. Gate: label-probability scoring produces zero invalid predictions.

#### Waiting on

- Go-ahead to commit and push the E1 results.

### 2026-09-19 (M2 start, E0 baselines)

#### Done

- **Predictions frozen:** H1a to H3 kept exactly as written, recorded in the PRD decision log before the first M2 result.
- **Label check script:** `scripts/m1_label_check.py` compares the 50 blind labels with the dataset (agreement, Cohen's kappa, confusion matrix). It was tested on a fake filled-in copy and is waiting for the real labels.
- **Pushed:** all commits up to `bd005e6` are on GitHub (private repo).
- **Shared metrics:** `src/bangla_sentiment/evaluate.py` computes macro-F1, micro-F1, accuracy, per-class scores and the confusion matrix, and saves predictions without text.
- **E0 baselines:** `src/bangla_sentiment/baselines.py`, `configs/e0.yaml` and `scripts/m2_e0_baselines.py` cover 42 runs on the CPU (2 models, 6 sizes plus train_original, 3 seeds), in about a minute. Summary in `results/e0_summary.json`.
- **Tests:** 16 passing (4 new: hand-checked metrics, Bangla word splitting, majority class).

#### Findings

- **sklearn's default word splitter breaks Bangla.** It treats vowel signs as separators, so "ভালো লাগলো খুব" becomes `['গল']`. E0 uses a Bangla-aware pattern, guarded by a test. A TF-IDF baseline built with the defaults would have been quietly wrong.
- **E0 test macro-F1 (mean ± sd over 3 seeds, cleaned train):** majority 19.46 at every size; TF-IDF 43.49 ± 1.39 (250), 46.71 ± 0.82 (500), 52.10 ± 0.48 (1k), 56.20 ± 1.13 (2k), 59.31 ± 0.37 (4k), 62.03 ± 0.00 (full).
- **Leakage effect on TF-IDF:** trained on train_original (leaks included), test macro-F1 is 67.87, against 62.03 on the cleaned train, a gap of 5.84 points (accuracy 72.32 vs 66.71). Part of the gap comes from the original split having 902 more rows, but the learning curve rises only about 2.7 points from 4k to 11.7k, so most of it is very likely the leaked duplicates. E1 will show whether BanglaBERT is affected the same way.
- **The E0 gate in the PRD was wrong and has been corrected.** The SentNoB paper's 64.61 is not micro-F1: its precision (57.71) and recall (73.39) differ, which cannot happen with micro-averaging. The exact comparison is the paper's majority row: 41.24, which our majority baseline reproduces exactly (`results/e0_summary.json`), confirming the same test set and label mapping. Our TF-IDF per-class F1 on train_original is also close to the paper's Table 5.
- **50-label manual check (closes M1):** Shihab's blind labels agree with the dataset on 27 of 50 (54%), Cohen's kappa 0.30 (`results/m1_label_check.json`). Most disagreements sit on the neutral boundary: 10 comments the dataset calls positive were labeled neutral (mostly polite questions, requests and praise followed by "but"). A read-through of the 23 disagreements suggests roughly a third look like dataset label errors, a third like misses by Shihab (mostly sarcasm and complaints stated as facts), and a third are genuinely ambiguous. This is a rough reading of 50 items, not a measurement. The labels were not changed after seeing the key.
- **The 72.89 BanglaBERT reference is confirmed as macro-F1** (averaged over 3 seeds, learning rate from 2e-5 to 5e-5, 3 to 20 epochs). The PRD allows at most 5 epochs, which may matter for the E1 reproduction check.

#### Next

- E1: BanglaBERT reproduction on train_original (gate: within about 3 points of 72.89), then the learning-rate grid and all sizes and seeds on the cleaned train.

#### Waiting on

- Nothing. The corrected E0 gate was confirmed by Shihab, and the label check and E0 were committed, rerun from the clean commit and pushed.

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
- Nothing else. M1 is committed and pushed, and the M1 scripts were rerun after the commit, so both result files point to the clean commit `db101bc`. The rerun reproduced the same splits and counts.

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

- Nothing. Committed; pushing is tracked in the M1 entry.
