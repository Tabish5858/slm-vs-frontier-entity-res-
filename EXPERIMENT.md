# Experiment: CoT-Enhanced Pure Fine-Tune (No RAG)

**Branch**: `experiment/cot-pure-ft`  
**Date**: July 20, 2026  
**Question**: Can we improve pure-FT generalization just by scaling data + chain-of-thought training, without RAG help?

## What was tested

1. **2.3x more training data**: 31,189 examples (vs 13,485 in v3) — 28 noise wrappers, all suffix combinations, double-noise combos
2. **Chain-of-thought training**: model outputs `"Reasoning: Clean 'X' → 'Y' | Normalize 'Y' → 'Z'. Output: {JSON}"` — teaches reasoning explicitly
3. **Same model/hardware**: Qwen2.5-3B-Instruct-4bit, LoRA rank 32, M1 Pro 16GB
4. **Same holdout**: `data/test_clean_holdout.jsonl` (644 examples, 200 unseen companies, SHA-256: `3c3003b7...`, identical to main branch)

## Results

Evaluated on `data/test_clean_holdout.jsonl` — same holdout as the final submission, identical checksum.

| System | Exact match | Lenient match* | Hard (444) | CIK |
|--------|-------------|----------------|------------|-----|
| Pure-FT v3 (13K ex, no CoT) | 37.0% | — | ~8% | 0% |
| **CoT Pure-FT v4 (31K ex, +CoT)** | **51.7%** | **86.8%** | **47.7%** | 0% |
| RAG-FT v2 (w/ retrieval) | 96.7% | — | — | 100% |
| Claude Opus 4.8 (raw, no retrieval) | 78.5% | — | — | ~84% |

\* *Lenient = case-insensitive, ignore commas/dots. The model correctly identifies the right entity but can't reproduce exact SEC punctuation for unseen companies.*

## Key findings

1. **The model learned entity resolution as a skill**: 94.3% of outputs use chain-of-thought. On 86.8% of examples, the model identifies the correct company entity — it just can't match SEC-specific comma placement and capitalization that it never saw during training.

2. **Failure breakdown** (311 exact-match failures):
   - 226 (73%) are comma/punctuation precision errors — same entity, wrong SEC formatting
   - 53 are wrapper/ending differences — very close but not exact
   - 21 are genuine wrong-company picks
   - 9 are suffix mismatches
   - 2 are null outputs

3. **CIK = 0% is structural**: CIK numbers are arbitrary SEC-assigned identifiers. A 3B model cannot memorize 10,000+ CIKs, especially for companies it never saw in training. This requires a database lookup.

4. **Why exact match caps out**: SEC-registered names have specific formatting (e.g., `"Apple Inc."` vs `"APPLE INC"` vs `"Apple, Inc."`) that varies per company. The model can resolve the entity but cannot know the exact SEC filing format for an unseen company — that's factual knowledge, not learnable via reasoning.

## Reproducing

```bash
git checkout experiment/cot-pure-ft
python3 build_dataset_v4_cot.py          # → data/train_v4_cot.jsonl (31K examples)
mlx_lm.lora -c train_config_v4_cot.yaml  # → adapters_v4_cot/ (~2h on M1 Pro)
python3 evaluate_cot_model.py            # → 51.7% exact, 86.8% lenient
```

## Conclusion

Scaling data + chain-of-thought improves pure-FT by 14.7 points (37% → 51.7%). The model genuinely learns the noise-stripping skill, reaching 86.8% entity-level accuracy. But the remaining gap to 99% is structural: entity resolution requires a database. Without retrieval, exact SEC formatting and CIK prediction are impossible for unseen companies.
