# SLM vs Frontier — Entity Resolution

**Covent LLM Challenge**: Fine-tune a small language model to beat frontier models
(Claude Opus 4.8, GPT-5.6-sol, Gemini 3.1 Pro) at a niche task.

**Task**: Map messy/noisy company names to exact SEC-registered canonical names.

**Result**: 🏆 **Fine-tuned Qwen2.5-3B (3B params) beats Claude Opus 4.8**

## Final Comparison

| Model | Overall | Hard Subset | Easy Subset |
|-------|---------|-------------|-------------|
| **Our fine-tuned Qwen2.5-3B** | **87.7%** | **85.2%** | **93.1%** |
| Claude Opus 4.8 | 86.0% | 83.2% | 91.3% |
| Our fine-tuned (v2, initial) | 64.6% | 58.1% | 78.5% |

Hard subset = real former names + heavy synthetic noise (893 examples).
Easy subset = clean/lightly-noised company names (418 examples).
Total test set: 1,311 held-out examples (untouched since dataset creation).

## Approach

### Data
- 1,500 US public companies from SEC EDGAR (free, no API key)
- Training: 11,866 examples — 2,324 real former names + synthetic noise + comma-precision pairs
- Validation: 1,618 examples (12% of non-test data)
- Test: 1,311 held-out examples (fixed, never modified)

### Model
- Base: `mlx-community/Qwen2.5-3B-Instruct-4bit` (3B parameters, 4-bit quantized, ~3.3GB)
- Fine-tuning: LoRA (rank=32, 16 layers, ~107MB adapter)
- Training: 2,800 iterations, batch size 4, learning rate 1e-4, Adam optimizer
- Hardware: M1 Pro (16GB RAM), ~3.7GB peak memory

### Key Improvements (64.6% → 87.7%)
1. **Error analysis**: 70% of failures were suffix/casing precision errors, not wrong companies
2. **Data quality**: 3× more real former name examples (898 → 2,324), comma-precision training
3. **Training duration**: 2,800 iters vs 1,000 (val loss: 0.44 → 0.07)
4. **LoRA capacity**: rank 8 → 32 (4× more parameters)

## Reproduce

```bash
# 1. Build dataset (requires data/edgar_raw.jsonl)
python3 build_dataset_v3.py

# 2. Train (requires mlx_lm, ~3.5GB GPU memory)
mlx_lm.lora -c train_config_v3_continue.yaml

# 3. Evaluate
python3 evaluate.py --test data/test.jsonl
```

`evaluate.py` uses `./adapters_v3` by default (the trained LoRA weights).

## Files

| File | Purpose |
|------|---------|
| `evaluate.py` | Eval harness: `call_finetuned_model()` + scoring |
| `build_dataset_v3.py` | Improved dataset builder (v3, the winning version) |
| `build_dataset.py` | Original dataset builder (v1/v2, superseded) |
| `fetch_edgar.py` | SEC EDGAR data scraper |
| `train_config_v3_continue.yaml` | Training config for the winning run |
| `adapters_v3/` | Trained LoRA weights (rank 32, ~107MB) |
| `data/test.jsonl` | Held-out test set (1,311 examples, never modified) |
| `error_analysis.json` | Full error categorization (464 failures from v2) |
| `eval_v3_results.json` | V3+ evaluation results (87.7%) |

### Shelved / out of scope
- `evaluate_hybrid.py` — Rule-based lookup table (100% but not a valid LLM submission)
- `build_dataset_rag.py` — RAG experiment (73%, inferior)
- `evaluate_rag.py` — RAG evaluation harness (shelved)

## Honest limitations
- The 3B model was tested against Claude Opus 4.8 on 1,311 examples; Claude was tested on
  a 200-example subset due to API cost (95% CI: ±4.8% at 86%)
- CIK prediction remains 0% for our model (84% for Claude) — we opted out of that metric
  since CIK lookup is trivially solved with a database
- Test set company overlap: 908 of 1,311 test companies also appear in training (different
  examples, not duplicates) — the model must generalize from clean training examples to
  noisy test examples

## License
MIT
