# SLM vs Frontier — Entity Resolution

**Covent LLM Challenge**: Fine-tune a small language model to beat frontier models at a niche task.

**Task**: Normalize messy company names (typos, suffix variations, legal wrappers, former names) to exact SEC-registered canonical names.

**Result**: **Fine-tuned Qwen2.5-3B + RAG = 99.0% on unseen companies. Claude Opus 4.8 raw = 78.5%.** A 3B model with retrieval-grounded inference nearly matches Claude Opus 4.8 *with the same retrieval* (100%).

---

## Why this task

Entity resolution is a core real-world problem in KYC/compliance, vendor deduplication, supply-chain mapping, and SEC filing analysis. Companies appear under dozens of name variants — `Apple Inc.` vs `APPLE COMPUTER INC` (former name) vs `dba Apple, Inc., a California corporation`. Human operators spend hours normalizing these manually. A system that maps any variant to a canonical SEC-registered name, with zero hallucination and sub-second latency, has immediate production value.

The task is niche and narrow by design — exactly the kind of task where a small, fine-tuned model should beat a general-purpose frontier model if the architecture is right.

---

## The honest debugging journey

We didn't start at 99%. Here's what actually happened:

**Phase 1 — The false win**: We built a pure rule-based lookup table (`evaluate_hybrid.py`) that scored **100%** on our test set. It was fast, elegant, and completely invalid — a database lookup, not an LLM, and the test set overlapped with the database we built the rules from. This taught us that entity resolution is fundamentally a retrieval problem, but we needed an actual fine-tuned model for the challenge.

**Phase 2 — The memorization trap**: We fine-tuned Qwen2.5-3B directly on (noisy name → canonical name) pairs. On our original test set (company-level overlap with training), it scored **87.7%** — beating Claude Opus 4.8's 86.0%. We celebrated too early. When we built a truly clean holdout of 200 companies *never seen in training*, accuracy collapsed to **37.0%**. The model hadn't learned entity resolution — it had memorized 1,500 company names as a lookup table encoded in LoRA weights.

**Phase 3 — RAG-aware fine-tuning**: We pivoted to a retrieval-augmented architecture. Instead of training the model to memorize entities, we trained it to *select and format* the correct entity from retrieved candidates. The base model (Qwen2.5-3B) with RAG scored 96.6% zero-shot on unseen companies. Fine-tuning the model specifically for candidate selection (on companies disjoint from the holdout) raised this to **99.0%**.

The negative results (37.0% pure-FT on clean holdout, 100% lookup-table false win) are not hidden — they are the story. The architecture matters more than the model size.

---

## Final results (clean company-level holdout, 200 unseen companies)

All results on `data/test_clean_holdout.jsonl` — 644 examples from 200 SEC companies with **zero CIK overlap** with training/validation data. Test file checksum: `3c3003b7dc6c3f7cc5fc37e69d0fd5d396044fd770e5c5da83d18f58988e6bdd` (SHA-256; file never modified since commit `53e0b8c`). Verify with: `shasum -a 256 data/test_clean_holdout.jsonl`.

| System | Clean Identity (200) | Full 644 (hard + easy) |
|--------|---------------------|------------------------|
| **Our RAG-aware FT Qwen2.5-3B** | **99.0%** | **96.7%** |
| Claude Opus 4.8 (raw, no retrieval) | 78.5% | — |
| Claude Opus 4.8 + same RAG pipeline | 100.0% | — |
| Pure fine-tune (no RAG) — *memorization failure* | 37.0% | 33.7% |

**CIK accuracy**: Our model + RAG scores 100% CIK accuracy — the retrieval step returns the CIK from the database. Pure fine-tune (no RAG) scores 0% CIK (hallucinates random numbers).

---

## Architecture

```
User query ("dba Apple, Inc., a Delaware corporation")
       │
       ▼
┌─────────────────────────────────┐
│  Fuzzy search (rapidfuzz)       │
│  Index: 1,700 SEC companies     │
│  Scorer: token_sort_ratio       │
│  Returns top-3 candidates       │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  RAG-aware FT Qwen2.5-3B        │
│  LoRA rank 16, 800 iterations   │
│  Selects + formats from list    │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Post-processing                │
│  Fuzzy-match output to best     │
│  candidate → exact canonical    │
└─────────────────────────────────┘
       │
       ▼
  {"canonical_name": "Apple Inc.",
   "cik": 320193, ...}
```

---

## How to reproduce

### 1. Fetch data (SEC EDGAR, free, no API key)

```bash
python3 fetch_edgar.py                          # → data/edgar_raw.jsonl (1,500 companies)
python3 build_dataset_v3.py                     # → data/train_v3.jsonl, data/valid_v3.jsonl
```

### 2. Build RAG-aware training data

```bash
# Generates train_rag_v2.jsonl + valid_rag_v2.jsonl + data_rag_v2/ symlinks
# Uses the full 1,700-company fuzzy search index to create realistic
# retrieval distractors (NOT random — the key insight from Phase 3).
# Takes ~2 minutes on M1 Pro.
python3 build_rag_dataset.py
```

### 3. Train the RAG-aware model

```bash
# Requires mlx_lm, ~4.5GB GPU memory (M1 Pro / M-series Mac)
mlx_lm.lora -c train_config_rag_v2.yaml         # → adapters_rag_v2/ (800 iterations, ~30 min)
```

### 4. Evaluate

```bash
# Full evaluation on the clean holdout (200 unseen companies, 644 examples)
python3 evaluate_rag_model.py

# Company-identity subset only (200 clean examples)
python3 evaluate_rag_model.py --ci

# Quick sanity check (first 50 examples)
python3 evaluate_rag_model.py --n 50
```

### 5. Run the demo

```bash
python3 demo.py
# Prints 10 messy company names → retrieved candidates → model output
# Runs standalone, no API keys, under 60 seconds
```

---

## Files

| File | Purpose |
|------|---------|
| `demo.py` | Standalone demo — 10 examples through the full RAG pipeline |
| `evaluate_rag_model.py` | Eval harness for the RAG-aware model on the clean holdout |
| `evaluate.py` | Eval harness for pure-FT model, Claude/GPT API wrappers |
| `build_rag_dataset.py` | Build RAG-aware training data with real retrieval distractors |
| `build_dataset_v3.py` | Improved dataset builder with comma-precision training |
| `build_dataset.py` | Original dataset builder (v1/v2, superseded) |
| `fetch_edgar.py` | SEC EDGAR data fetcher (company_tickers.json + submissions API) |
| `train_config_rag_v2.yaml` | Training config for the RAG-aware fine-tuning (the winning run) |
| `train_config_v3_continue.yaml` | Training config for the pure fine-tuning (memorization experiment) |
| `data/test_clean_holdout.jsonl` | Clean holdout — 200 companies, zero train/valid overlap, SHA-256: `3c3003b7...` |
| `data/edgar_holdout.jsonl` | 200 unseen companies from SEC tickers |
| `data/rag_index_1700.jsonl` | RAG retrieval index (1,500 training + 200 holdout companies) |
| `data/edgar_raw.jsonl` | 1,500 SEC company records with former names |
| `data/train_rag_v2.jsonl` | RAG training data (6,373 examples with real retrieval distractors) |
| `data/valid_rag_v2.jsonl` | RAG validation data (1,124 examples) |
| `data_rag_v2/` | Symlink directory (train.jsonl → train_rag_v2.jsonl, for mlx_lm.lora) |
| `adapters_rag_v2/` | Trained LoRA weights (rank 16, ~53MB, 800 iterations) |
| `final_comparison.json` | Final Claude comparison results on clean holdout |
| `error_analysis.json` | Error categorization from pure fine-tune (464 failures) |

### Shelved experiments (not part of final submission)

- **`evaluate_hybrid.py`** — Rule-based lookup table. Achieved 100% on overlapping test set but is NOT a fine-tuned LLM and the test set shared companies with the lookup database. Kept as a documented "what we learned" — entity resolution is a retrieval problem, but a database without a model doesn't satisfy the challenge rules.
- **`build_dataset_rag.py`** — Early RAG experiment with random distractors (not real retrieval results). Achieved 73% — inferior because distractors didn't match the real retrieval distribution.
- **`evaluate_rag.py`** — RAG evaluation harness for the shelved experiment above.

---

## Honest limitations

- **Test size**: Claude was tested on 200 examples; our model on 644. The 200-subset comparison is apples-to-apples.
- **CIK prediction**: Our model doesn't predict CIKs from memory (0% pure-FT) — CIKs come from the retrieval database. Claude raw predicts CIKs with ~84% accuracy from pre-training knowledge.
- **Database dependency**: RAG requires a reference database at inference time. This is realistic for production entity resolution (you always have a database) but means the model can't resolve entities outside the indexed set.
- **Scalability**: Our index has 1,700 companies. A production system would index all ~10,000+ SEC-registered companies. The architecture scales linearly with index size.
- **Noise patterns**: The test noise is synthetic (suffix swaps, legal wrappers, casing). Real-world entity resolution has additional challenges (OCR errors, multilingual names, partial matches).

---

## License

MIT
