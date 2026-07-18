"""
[SHELVED — NOT PART OF FINAL SUBMISSION] Rule-based entity resolution via fuzzy-search
lookup table. Achieves 100% on overlapping test set but is NOT a fine-tuned LLM and
the test set shared companies with the lookup database — violates the challenge rule
that "test set must be different from training." Kept as a documented learning:
entity resolution IS a retrieval problem, but a database without a model isn't a
valid LLM submission.

The winning submission uses RAG-aware fine-tuned Qwen2.5-3B (adapters_rag_v2/).
See README.md and demo.py for the final approach.
"""

import argparse
import json
import re
import sys
from rapidfuzz import process, fuzz


# ── name cleaner ──────────────────────────────────────────────

def strip_noise(name: str) -> str:
    """Remove wrapper text that our training taught the model to ignore."""
    name = re.sub(r'\s*\(fka\)\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\(the\s*"?Company"?\)\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*,\s*the\s*"?Company"?\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*-\s*formerly\s+known\s+as\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\(successor\s+to\s+former\s+entity\)\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*/DE/\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*,\s*a\s+Delaware\s+corporation\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*/formerly/\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^dba\s+', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^f/k/a\s+', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^formerly\s+', '', name, flags=re.IGNORECASE)
    # Trailing period removal (after noise strip)
    name = re.sub(r'\.$', '', name)
    name = re.sub(r'\.\s*$', '', name)
    return name.strip()


def normalize_suffix(name: str) -> str:
    """Normalize corporate suffixes to a canonical form."""
    name_lower = name.lower().strip()
    # Normalize AND/& variations
    name_lower = re.sub(r'\s+and\s+', ' & ', name_lower, flags=re.IGNORECASE)
    name_lower = re.sub(r'\s*&\s*', ' & ', name_lower)
    # Normalize to base form for comparison
    replacements = [
        # Multi-word before single-word (order matters!)
        (r'\bpublic\s+limited\s+company\b', 'plc'),
        (r'\binc(orporated)?\.?\b', 'inc'),
        (r'\bcorp(oration)?\.?\b', 'corp'),
        (r'\bco(mpany)?\.?\b', 'co'),
        (r'\bllc\.?\b', 'llc'),
        (r'\bl\.?l\.?c\.?\b', 'llc'),
        (r'\bltd\.?\b', 'ltd'),
        (r'\blimited\b', 'ltd'),
        (r'\bplc\.?\b', 'plc'),
        (r'\bl\.?p\.?\b', 'lp'),
        (r'\bholdings\.?\b', 'holdings'),
        (r'\bgroup\.?\b', 'group'),
        (r'\bn\.?v\.?\b', 'nv'),
        (r'\bban,?\s*corp\b', 'bancorp'),
        (r'\bs\.?a\.?\b', 'sa'),
    ]
    for pattern, replacement in replacements:
        name_lower = re.sub(pattern, replacement, name_lower, flags=re.IGNORECASE)
    # Remove punctuation for comparison
    name_lower = re.sub(r'[,.&]', '', name_lower)
    # Collapse spaces
    name_lower = re.sub(r'\s+', ' ', name_lower).strip()
    return name_lower


# ── fuzzy search (enhanced) ───────────────────────────────────

class HybridMatcher:
    def __init__(self, raw_path: str):
        self.companies = []
        with open(raw_path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("name"):
                    self.companies.append({
                        "name": r["name"],
                        "cik": r["cik"],
                        "entity_type": r.get("entityType") or "unknown",
                        "former_names": [fn.get("name","") for fn in r.get("formerNames",[])],
                        "name_norm": normalize_suffix(r["name"]),
                    })
        # Build lookup by normalized name for exact matching
        # Canonical names first (higher priority), former names second (don't overwrite)
        self.norm_lookup = {}
        for c in self.companies:
            self.norm_lookup[c["name_norm"]] = c
        for c in self.companies:
            for fn in c["former_names"]:
                if fn:
                    fn_clean = strip_noise(fn)
                    fn_norm = normalize_suffix(fn_clean)
                    if fn_norm not in self.norm_lookup:
                        self.norm_lookup[fn_norm] = c

        print(f"Indexed {len(self.companies)} companies", file=sys.stderr)

    def match(self, query: str, use_model: bool = False) -> dict | None:
        """Return {'name': ..., 'cik': ..., ...} or None.

        Strategy (no model):
        1. Clean input noise
        2. Normalize suffixes
        3. Try exact normalized match
        4. Try fuzzy top-1
        5. If top-1 score >= 85, return it
        6. Otherwise, return None (model fallback needed)
        """
        cleaned = strip_noise(query)
        norm = normalize_suffix(cleaned)

        # Step 1: exact normalized match
        if norm in self.norm_lookup:
            c = self.norm_lookup[norm]
            return {"name": c["name"], "cik": c["cik"], "entity_type": c["entity_type"]}

        # Step 2: fuzzy match against all canonical names
        choices = {}
        for i, c in enumerate(self.companies):
            choices[str(i)] = c["name"]
        # Also search against former names
        for i, c in enumerate(self.companies):
            for fn in c["former_names"]:
                if fn:
                    choices[f"{i}|fn"] = fn

        # Try multiple scorers and take the best
        results_tok = process.extract(query, {k: v for k, v in choices.items()},
                                       scorer=fuzz.token_sort_ratio, limit=1)
        results_par = process.extract(cleaned, {k: v for k, v in choices.items()},
                                       scorer=fuzz.partial_ratio, limit=1)

        best = max(results_tok + results_par, key=lambda x: x[1])
        match_str, score, idx_str = best

        if score >= 80:
            ci = int(idx_str.split("|")[0])
            c = self.companies[ci]
            return {"name": c["name"], "cik": c["cik"], "entity_type": c["entity_type"]}

        return None  # model fallback needed


# ── eval ─────────────────────────────────────────────────────

def evaluate_hybrid(test_path: str, n: int, matcher: HybridMatcher,
                    model_fn=None, model_label: str = "Hybrid"):
    examples = []
    with open(test_path) as f:
        for line in f:
            examples.append(json.loads(line))
    examples = examples[:n]

    fuzzy_hits = 0
    model_hits = 0
    model_calls = 0
    total = len(examples)

    for ex in examples:
        query = ex["messages"][1]["content"]
        expected = json.loads(ex["messages"][2]["content"])

        result = matcher.match(query, use_model=False)
        if result and result["name"] == expected["canonical_name"]:
            fuzzy_hits += 1
            continue

        # Fuzzy failed — try model fallback
        if model_fn and result is None:
            model_calls += 1
            # Use model with candidate list
            from evaluate_rag import CompanyIndex, build_rag_user_prompt, SYSTEM_PROMPT_RAG
            from evaluate import extract_json, call_finetuned_model as ft

            idx = CompanyIndex('./data/edgar_raw.jsonl')
            candidates = idx.search(query, k=5)
            rag = build_rag_user_prompt(query, candidates, 5)
            raw = ft(rag, SYSTEM_PROMPT_RAG)
            pred = extract_json(raw)
            if pred and pred.get("canonical_name", "").strip().lower() == expected["canonical_name"].strip().lower():
                model_hits += 1

    print(f"\n=== {model_label} (n={total}) ===")
    print(f"  Fuzzy matched:     {fuzzy_hits}/{total} = {fuzzy_hits/total:.1%}")
    if model_fn:
        print(f"  Model fallback:    {model_hits}/{model_calls} = {model_hits/model_calls:.1%}" if model_calls else "  Model fallback:    N/A")
    total_correct = fuzzy_hits + model_hits
    print(f"  TOTAL accuracy:    {total_correct}/{total} = {total_correct/total:.1%}")
    print(f"  Model calls:       {model_calls}/{total} ({model_calls/total:.1%})")

    # Break by difficulty
    hard = [e for e in examples if e.get("difficulty") == "hard"]
    easy = [e for e in examples if e.get("difficulty") == "easy"]

    for label, subset in [("Hard", hard), ("Easy", easy)]:
        c = 0
        for ex in subset:
            query = ex["messages"][1]["content"]
            expected = json.loads(ex["messages"][2]["content"])
            result = matcher.match(query, use_model=False)
            if result and result["name"] == expected["canonical_name"]:
                c += 1
        print(f"  Fuzzy-only {label}:  {c}/{len(subset)} = {c/len(subset):.1%}" if subset else "")

    return {
        "model": model_label,
        "fuzzy_accuracy": fuzzy_hits / total,
        "total_accuracy": total_correct / total,
        "model_call_rate": model_calls / total if model_fn else 0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="./data/test.jsonl")
    ap.add_argument("--raw", default="./data/edgar_raw.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--full", action="store_true", help="All 1311 examples")
    ap.add_argument("--with-model", action="store_true", help="Use model fallback")
    args = ap.parse_args()

    test_count = sum(1 for _ in open(args.test)) if args.full else args.n

    print(f"Evaluating {test_count} examples...")
    matcher = HybridMatcher(args.raw)

    evaluate_hybrid(args.test, test_count, matcher,
                    model_fn=None, model_label="Hybrid (fuzzy-only)")
