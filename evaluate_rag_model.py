"""
Evaluate the RAG-aware fine-tuned model on the clean holdout.

Loads CompanyIndex + fine-tuned model → runs the full RAG pipeline
(fuzzy search → model candidate selection → post-process) on every
example in data/test_clean_holdout.jsonl.

Usage:
  python3 evaluate_rag_model.py [--n 200] [--k 3]
"""

import argparse
import json
import re
import sys
import time
import os

# ── Imports ──
try:
    from rapidfuzz import process, fuzz
except ImportError:
    print("ERROR: pip install rapidfuzz", file=sys.stderr)
    sys.exit(1)

try:
    from mlx_lm import load as mlx_load
    from mlx_lm import generate
except ImportError:
    print("ERROR: pip install mlx-lm", file=sys.stderr)
    sys.exit(1)


# ── Fuzzy search index (identical to demo.py) ──

class CompanyIndex:
    def __init__(self, path):
        self.companies = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("name"):
                    self.companies.append({
                        "name": r["name"],
                        "cik": r["cik"],
                        "entity_type": r.get("entity_type") or "unknown",
                        "former_names": r.get("former_names", []),
                        "name_norm": self._normalize(r["name"]),
                    })

    @staticmethod
    def _normalize(s):
        s = s.lower().strip().replace(",", "").replace(".", "")
        s = s.replace(" & ", " and ")
        return re.sub(r"\s+", " ", s)

    def search(self, query, k=3):
        q_norm = self._normalize(query)
        choices = {}
        for i, c in enumerate(self.companies):
            choices[f"{i}|n"] = c["name_norm"]
            for fn in c["former_names"]:
                if fn:
                    choices[f"{i}|fn"] = fn.lower()

        results = process.extract(
            q_norm, choices, scorer=fuzz.token_sort_ratio, limit=k * 4
        )

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


# ── Prompt ──

SYSTEM_PROMPT = (
    "You are an entity resolution assistant. Given a messy company name "
    "and candidates from our SEC database, pick the ONE best matching "
    "candidate. Respond with ONLY this exact JSON format, copying "
    "canonical_name VERBATIM from the candidate list: "
    '{"canonical_name":"...","cik":...,"entity_type":"...",'
    '"is_former_name_input":false}'
)


def build_user_prompt(query, candidates):
    lines = []
    for i, (c, score) in enumerate(candidates, 1):
        lines.append(f"  {i}. canonical_name: \"{c['name']}\"  cik: {c['cik']}")
    return (
        f"Company: {query}\n\n"
        f"Candidates (pick one, copy EXACTLY):\n" +
        "\n".join(lines) +
        "\n\nReturn the JSON for the matching candidate."
    )


# ── JSON extraction ──

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ── Post-processing ──

def resolve_output(raw_json, candidates):
    """Ensure the output matches an exact candidate from the list."""
    if raw_json is None:
        return None

    # Try exact CIK match first
    raw_cik = raw_json.get("cik")
    for c, score in candidates:
        if c["cik"] == raw_cik:
            return {
                "canonical_name": c["name"],
                "cik": c["cik"],
                "entity_type": c["entity_type"],
                "is_former_name_input": raw_json.get("is_former_name_input", False),
            }

    # Fall back on name match
    raw_name = raw_json.get("canonical_name", "")
    best_score = 0
    best = None
    for c, score in candidates:
        s = fuzz.token_sort_ratio(raw_name.lower(), c["name"].lower())
        if s > best_score and s >= 70:
            best_score = s
            best = {
                "canonical_name": c["name"],
                "cik": c["cik"],
                "entity_type": c["entity_type"],
                "is_former_name_input": raw_json.get("is_former_name_input", False),
            }

    return best


# ── Main evaluation ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test_clean_holdout.jsonl")
    ap.add_argument("--index", default="data/rag_index_1700.jsonl")
    ap.add_argument("--adapter", default="./adapters_rag_v2")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    ap.add_argument("--n", type=int, default=0, help="Limit eval to N examples (0 = all)")
    ap.add_argument("--k", type=int, default=3, help="Number of retrieval candidates")
    ap.add_argument("--ci", "--company-identity", action="store_true", dest="company_identity",
                    help="Score only clean-identity examples (easy=True)")
    args = ap.parse_args()

    # Load index
    print("Loading company index...", file=sys.stderr)
    index = CompanyIndex(args.index)
    print(f"  {len(index.companies)} companies indexed", file=sys.stderr)

    # Load model
    print("Loading fine-tuned model...", file=sys.stderr)
    if not os.path.exists(args.adapter):
        print(f"ERROR: adapter not found at {args.adapter}", file=sys.stderr)
        print("Train first: mlx_lm.lora -c train_config_rag_v2.yaml", file=sys.stderr)
        sys.exit(1)

    model, tokenizer = mlx_load(args.model, adapter_path=args.adapter)
    print("  ready", file=sys.stderr)

    # Load test data
    with open(args.test) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    if args.n > 0:
        examples = examples[:args.n]

    if args.company_identity:
        examples = [e for e in examples if e.get("easy") or e.get("difficulty") == "easy"]

    # Evaluate
    total = len(examples)
    correct_name = 0
    correct_cik = 0
    hard_correct = 0
    hard_total = 0
    easy_correct = 0
    easy_total = 0
    retrieval_hits = 0

    start_time = time.time()

    for i, ex in enumerate(examples):
        query = ex["messages"][1]["content"]
        expected = json.loads(ex["messages"][2]["content"])
        expected_name = expected.get("canonical_name", "")
        expected_cik = expected.get("cik", 0)
        difficulty = ex.get("difficulty", "easy")

        # Retrieval
        candidates = index.search(query, k=args.k)
        top1 = candidates[0] if candidates else None
        if top1 and top1[0]["name"] == expected_name:
            retrieval_hits += 1

        # Model inference
        user_prompt = build_user_prompt(query, candidates)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=256,
            verbose=False,
        )

        # Post-process
        raw_json = extract_json(output)
        resolved = resolve_output(raw_json, candidates)

        # Score
        is_correct = resolved and resolved["canonical_name"] == expected_name
        is_cik_correct = resolved and resolved["cik"] == expected_cik

        if is_correct:
            correct_name += 1
        if is_cik_correct:
            correct_cik += 1

        if difficulty == "hard":
            hard_total += 1
            if is_correct:
                hard_correct += 1
        else:
            easy_total += 1
            if is_correct:
                easy_correct += 1

        # Progress
        if (i + 1) % 50 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{i+1}/{total}] name={correct_name}/{i+1} ({correct_name/(i+1):.1%}) "
                f"cik={correct_cik}/{i+1} ({correct_cik/(i+1):.1%}) "
                f"retrieval={retrieval_hits}/{i+1} ({retrieval_hits/(i+1):.1%}) "
                f"| {rate:.1f} ex/s",
                file=sys.stderr,
            )

    elapsed = time.time() - start_time

    # Report
    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Test file:      {args.test}")
    print(f"  Examples:       {total}")
    print(f"  Time:           {elapsed:.1f}s ({elapsed/total:.2f}s per example)")
    print()
    print(f"  Canonical name:  {correct_name}/{total} = {correct_name/total:.1%}")
    print(f"  CIK:             {correct_cik}/{total} = {correct_cik/total:.1%}")
    print(f"  Retrieval top-1: {retrieval_hits}/{total} = {retrieval_hits/total:.1%}")
    print()

    if easy_total:
        print(f"  Easy examples:   {easy_correct}/{easy_total} = {easy_correct/easy_total:.1%}")
    if hard_total:
        print(f"  Hard examples:   {hard_correct}/{hard_total} = {hard_correct/hard_total:.1%}")

    print()
    print(f"  Model: {args.model} + {args.adapter}")
    print(f"  Candidates: k={args.k}")
    print("=" * 60)


if __name__ == "__main__":
    main()
