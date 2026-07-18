"""
RAG (Retrieval-Augmented Generation) eval: fuzzy-search the company database
at inference time, present top-K candidates to the model, let it pick the right one.

This is how entity resolution works in production — you have a database.
Claude wins because it memorized SEC data during pre-training. With RAG,
our 3B model should beat Claude without memorization.

Usage:
  python evaluate_rag.py --test ./data/test.jsonl --n 200 --k 5
"""

import argparse
import json
import re
import time
import sys

# ── fuzzy search ──────────────────────────────────────────────────

class CompanyIndex:
    """Fuzzy search over company names using rapidfuzz Jaro-Winkler."""
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
                        "search_text": r["name"].lower()
                    })
        print(f"Indexed {len(self.companies)} companies", file=sys.stderr)

    def search(self, query: str, k: int = 5) -> list:
        from rapidfuzz import process, fuzz
        # Search against all canonical names + former names
        choices = {}
        for i, c in enumerate(self.companies):
            choices[str(i)] = c["name"]
            for fn in c["former_names"]:
                if fn:
                    choices[f"{i}|fn|{fn}"] = fn

        results = process.extract(
            query, choices, scorer=fuzz.token_sort_ratio, limit=k * 2
        )

        # Deduplicate by company index, keep best score
        seen = set()
        top = []
        for match_str, score, idx_str in results:
            ci = int(idx_str.split("|")[0])
            if ci not in seen:
                seen.add(ci)
                top.append((self.companies[ci], score))
                if len(top) >= k:
                    break
        return top


# ── prompt builders ────────────────────────────────────────────

SYSTEM_PROMPT_BASE = (
    "You are an entity resolution assistant. Given a messy, informal, or "
    "historical company name, identify the canonical SEC-registered legal "
    "entity from the candidate list. Respond with ONLY a JSON object with keys: "
    "canonical_name, cik, entity_type, is_former_name_input. No other text."
)

SYSTEM_PROMPT_RAG = (
    "You are an entity resolution assistant. You will be given a messy company name "
    "and a list of candidate matches from our SEC database. Identify which candidate "
    "matches the input. If none match, output your best guess. "
    "Respond with ONLY a JSON object with keys: canonical_name, cik, entity_type, "
    "is_former_name_input. No other text."
)

def build_rag_user_prompt(messy_input: str, candidates: list, k: int) -> str:
    lines = [f"Input company name: {messy_input}", "", "Candidate matches from SEC database:"]
    for i, (comp, score) in enumerate(candidates, 1):
        lines.append(f"  {i}. {comp['name']} (CIK: {comp['cik']}, Type: {comp['entity_type']})")
    if not candidates:
        lines.append("  (No close matches found — output your best guess)")
    lines.append("")
    lines.append("Return the JSON for the matching entity (or your best guess if no match).")
    return "\n".join(lines)


# ── model calling (reuses evaluate.py functions) ─────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate

def call_ft_rag(user_prompt: str, system_prompt: str) -> str:
    return evaluate.call_finetuned_model(user_prompt, system_prompt)


# ── eval ────────────────────────────────────────────────────────

def run_rag_eval(test_path: str, n: int, index: CompanyIndex,
                 model_fn, model_label: str, k: int = 5):
    examples = []
    with open(test_path) as f:
        for line in f:
            examples.append(json.loads(line))
    examples = examples[:n]

    results = []
    t0_total = time.time()
    for i, ex in enumerate(examples):
        messy_input = ex["messages"][1]["content"]
        candidates = index.search(messy_input, k=k)
        rag_prompt = build_rag_user_prompt(messy_input, candidates, k)
        raw = model_fn(rag_prompt, SYSTEM_PROMPT_RAG)
        pred = evaluate.extract_json(raw)
        difficulty = ex.get("difficulty", "unknown")
        result = evaluate.score(ex, pred)
        result["difficulty"] = difficulty
        results.append(result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0_total
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(examples)}] {rate:.1f} ex/s", file=sys.stderr)

    def bucket_accuracy(rs, key):
        if not rs: return 0.0
        return sum(r[key] for r in rs) / len(rs)

    hard_results = [r for r in results if r["difficulty"] == "hard"]
    easy_results = [r for r in results if r["difficulty"] == "easy"]

    print(f"\n=== {model_label} + RAG (k={k}, n={len(results)}) ===")
    print(f"  canonical_name exact match (primary metric):")
    print(f"    Overall:  {bucket_accuracy(results, 'name_match'):.1%}")
    print(f"    Hard:     {bucket_accuracy(hard_results, 'name_match'):.1%}  ({len(hard_results)} examples)")
    print(f"    Easy:     {bucket_accuracy(easy_results, 'name_match'):.1%}  ({len(easy_results)} examples)")
    print(f"  ---")
    print(f"  cik exact match (bonus, needs lookup table in production):")
    print(f"    Overall:  {bucket_accuracy(results, 'cik_match'):.1%}")
    print(f"  ---")
    print(f"  full match (name + cik): {bucket_accuracy(results, 'full_match'):.1%}")

    return {
        "model": model_label,
        "k": k,
        "name_acc_overall": bucket_accuracy(results, "name_match"),
        "name_acc_hard": bucket_accuracy(hard_results, "name_match"),
        "name_acc_easy": bucket_accuracy(easy_results, "name_match"),
        "cik_acc": bucket_accuracy(results, "cik_match"),
        "full_acc": bucket_accuracy(results, "full_match"),
    }


def run_baseline_eval(test_path: str, n: int, model_fn, model_label: str):
    """Non-RAG baseline evaluation."""
    examples = []
    with open(test_path) as f:
        for line in f:
            examples.append(json.loads(line))
    examples = examples[:n]

    results = []
    for i, ex in enumerate(examples):
        system_prompt = ex["messages"][0]["content"]
        user_prompt = ex["messages"][1]["content"]
        raw = model_fn(user_prompt, system_prompt)
        pred = evaluate.extract_json(raw)
        difficulty = ex.get("difficulty", "unknown")
        result = evaluate.score(ex, pred)
        result["difficulty"] = difficulty
        results.append(result)

    def bucket_accuracy(rs, key):
        if not rs: return 0.0
        return sum(r[key] for r in rs) / len(rs)

    hard_results = [r for r in results if r["difficulty"] == "hard"]
    easy_results = [r for r in results if r["difficulty"] == "easy"]

    print(f"\n=== {model_label} (no RAG, n={len(results)}) ===")
    print(f"  canonical_name exact match:")
    print(f"    Overall:  {bucket_accuracy(results, 'name_match'):.1%}")
    print(f"    Hard:     {bucket_accuracy(hard_results, 'name_match'):.1%}  ({len(hard_results)} examples)")
    print(f"    Easy:     {bucket_accuracy(easy_results, 'name_match'):.1%}  ({len(easy_results)} examples)")
    print(f"  cik: {bucket_accuracy(results, 'cik_match'):.1%}")

    return {
        "model": model_label,
        "name_acc_overall": bucket_accuracy(results, "name_match"),
        "name_acc_hard": bucket_accuracy(hard_results, "name_match"),
        "name_acc_easy": bucket_accuracy(easy_results, "name_match"),
        "cik_acc": bucket_accuracy(results, "cik_match"),
        "full_acc": bucket_accuracy(results, "full_match"),
    }


# ── main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="./data/test.jsonl")
    ap.add_argument("--raw", default="./data/edgar_raw.jsonl")
    ap.add_argument("--n", type=int, default=200, help="Number of test examples")
    ap.add_argument("--k", type=int, default=5, help="Number of RAG candidates")
    ap.add_argument("--full", action="store_true", help="Run all 1311 examples")
    ap.add_argument("--no-claude", action="store_true", help="Skip Claude comparison (save $)")
    args = ap.parse_args()

    test_count = sum(1 for _ in open(args.test)) if args.full else args.n
    print(f"Evaluating {test_count} test examples with RAG (k={args.k})...")

    # Build index
    print("Building company index...", file=sys.stderr)
    index = CompanyIndex(args.raw)

    all_results = []

    # 1. Our model WITHOUT RAG (baseline)
    all_results.append(run_baseline_eval(
        args.test, test_count,
        lambda u, s: evaluate.call_finetuned_model(u, s),
        "Fine-tuned Qwen2.5-3B (no RAG)"))

    # 2. Our model WITH RAG
    all_results.append(run_rag_eval(
        args.test, test_count, index,
        lambda u, s: evaluate.call_finetuned_model(u, s),
        "Fine-tuned Qwen2.5-3B + RAG", k=args.k))

    # 3. Claude Opus 4.8 (frontier baseline)
    if not args.no_claude:
        all_results.append(run_baseline_eval(
            args.test, test_count,
            lambda u, s: evaluate.call_frontier_model(u, s, "claude-opus-4-8"),
            "Claude Opus 4.8 (no RAG)"))

    print("\n" + "=" * 60)
    print(json.dumps(all_results, indent=2))
