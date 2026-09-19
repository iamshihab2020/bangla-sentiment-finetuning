# Progress

Running log of the project. Newest entry first. Numbers here come from files in `results/`; anything that is only an estimate is marked as one.

Timeline: 2026-09-21 to 2026-12-18 (13 weeks, about 30 hours per week).

## Milestones

| # | Milestone | Planned weeks | Status |
|---|---|---|---|
| M0 | Environment and budget | W1 (Sep 21 to Sep 27) | Mostly done (Sep 19). Remaining: full-grid time estimate in README |
| M1 | Data | W2 (Sep 28 to Oct 4) | Not started |
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
**Done**
-
**Findings**
-
**Next**
-
**Waiting on**
-
-->

### 2026-09-19

**Done**

- Reviewed the first plan and rewrote it. It now has research questions with predictions recorded in advance, SentNoB as the primary dataset, matched learning curves for encoder and LLM, 3 seeds plus bootstrap intervals, and a dated 13-week timeline.
- Added a zero-cost rule: free tools only, nothing that needs a card.
- Hugging Face account iamshihab2020 is logged in. Access granted to both gated models, Gemma 3 1B and Llama 3.2 1B (both checked from code).
- Environment: Python 3.12 via uv, PyTorch 2.14 with CUDA 13.0, all versions pinned in `uv.lock`. `PYTHONUTF8=1` is set for Bangla text.
- M0 checks (`results/m0_env_check.json`):
  - bitsandbytes 4-bit works on native Windows, so WSL is not needed.
  - TigerLLM-1B-it loads in 4-bit using 0.97 GB of VRAM.
  - QLoRA at sequence length 128, batch 4: 449 tokens/s, 3.3 GB peak VRAM.
  - Batch 8: 184 tokens/s, 5.0 GB peak VRAM. Batch 16: out of memory.
- Repo organized, with 3 environment tests passing. First commit 053d98e, pushed to the private GitHub repo iamshihab2020/bangla-sentiment-finetuning.

**Findings**

- **BanglaBook rejected.** Its labels come from star ratings, it is 89.6% positive, and 43% of its reviews were machine-translated.
- **TigerLLM-1B-it is a Gemma 3 1B model.** Its `config.json` says so, even though the paper says Llama 3.2. The RQ2 control is therefore Gemma-3-1B-it.
- **SentNoB is CC BY-ND 4.0.** No dataset text, raw or processed, goes into git.
- **Zero-shot generation starts reasoning step by step instead of giving a label.** This supports scoring by label probability rather than parsing generated text.
- **Batch 8 is slower per token than batch 4.** It runs close to the 6 GB limit, probably spilling into shared system memory. Use batch 4 for now, and test the NVIDIA "Prefer No Sysmem Fallback" setting.
- **Rough estimate (worst case, not a measurement):** a full SentNoB training epoch at 449 tokens/s with every example padded to 128 tokens takes about 1 hour. Real examples are shorter, so the real number should be lower. Measure it with real data at M1 and M3.

**Next**

- Finish M0: time estimate for the full experiment grid in the README.
- M1: download SentNoB and bengali_sa, build the data card, dedupe and leakage report, nested subsets, tokenizer statistics.

**Waiting on**

- Nothing.
