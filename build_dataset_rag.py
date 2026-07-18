"""
[SHELVED — NOT PART OF FINAL SUBMISSION] Early RAG training dataset using random
distractor companies (not real retrieval results). Achieved only 73% because the
training distribution didn't match the real retrieval distribution at inference.
Kept as reference only.

The winning training pipeline uses actual fuzzy-search retrieval results as
distractors (see build_dataset_rag_v2 logic, invoked inline during training).
"""
For each training example:
  1. The correct company is always in the candidate list
  2. K-1 random "distractor" companies are also included
  3. The model outputs the canonical name + CIK (copied from the correct candidate)

Usage:
  python build_dataset_rag.py --input ./data/edgar_raw.jsonl --k 5
"""

import argparse
import json
import random
import sys

random.seed(42)

# Reuse noise functions from build_dataset
sys.path.insert(0, '.')
from build_dataset import (
    make_messy_variant, swap_suffix, NOISE_WRAPPERS, SYSTEM_PROMPT,
    to_chat_example
)

RAG_SYSTEM_PROMPT = (
    "You are an entity resolution assistant. Given a messy, informal, or "
    "historical company name and a list of candidate matches from our SEC database, "
    "identify which candidate matches the input. "
    "Respond with ONLY a JSON object with keys: canonical_name, cik, entity_type, "
    "is_former_name_input. Copy the values from the matching candidate. No other text."
)


def build_rag_input(messy_input: str, candidates: list, correct_idx: int) -> str:
    """Build a user prompt with candidates, marking none as correct."""
    lines = [f"Messy company name: {messy_input}", "", "Candidate matches:"]
    for i, c in enumerate(candidates):
        lines.append(f"  [{i+1}] {c['name']} | CIK: {c['cik']} | Type: {c.get('entityType', 'unknown')}")
    lines.append("")
    lines.append("Identify the matching candidate and return its details as JSON.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./data/edgar_raw.jsonl")
    ap.add_argument("--synthetic_per_company", type=int, default=5)
    ap.add_argument("--k", type=int, default=5, help="Candidates per example (including correct)")
    ap.add_argument("--out_dir", default="./data")
    ap.add_argument("--noise_only", action="store_true",
                    help="Skip former-name examples (pure noise training)")
    args = ap.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            if r.get("name"):
                records.append(r)

    # Pre-compute candidate pool: each record as a candidate dict
    all_candidates = [
        {"name": r["name"], "cik": r["cik"], "entityType": r.get("entityType") or "unknown"}
        for r in records
    ]

    examples = []
    for ci, r in enumerate(records):
        correct_candidate = all_candidates[ci]

        # --- synthetic noisy variants (with RAG prompt) ---
        for _ in range(args.synthetic_per_company):
            messy, noise_count = make_messy_variant(r["name"])
            # Pick k-1 random distractors (excluding the correct one)
            distractors = random.sample(
                [c for i, c in enumerate(all_candidates) if i != ci],
                min(args.k - 1, len(all_candidates) - 1)
            )
            candidates = [correct_candidate] + distractors
            random.shuffle(candidates)

            rag_prompt = build_rag_input(messy, candidates, candidates.index(correct_candidate))
            out = {
                "canonical_name": r["name"],
                "cik": r["cik"],
                "entity_type": r.get("entityType") or "unknown",
                "is_former_name_input": False,
            }
            diff = "hard" if noise_count >= 2 else "easy"
            examples.append({
                "messages": [
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": rag_prompt},
                    {"role": "assistant", "content": json.dumps(out, separators=(",", ":"))},
                ],
                "difficulty": diff,
            })

        # --- extra suffix-swap passes ---
        for _ in range(2):
            name = r["name"]
            if random.random() < 0.9:
                name = swap_suffix(name)
            wrapper = random.choice(NOISE_WRAPPERS)
            messy = wrapper.format(name=name).strip()

            distractors = random.sample(
                [c for i, c in enumerate(all_candidates) if i != ci],
                min(args.k - 1, len(all_candidates) - 1)
            )
            candidates = [correct_candidate] + distractors
            random.shuffle(candidates)

            rag_prompt = build_rag_input(messy, candidates, candidates.index(correct_candidate))
            out = {
                "canonical_name": r["name"],
                "cik": r["cik"],
                "entity_type": r.get("entityType") or "unknown",
                "is_former_name_input": False,
            }
            examples.append({
                "messages": [
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": rag_prompt},
                    {"role": "assistant", "content": json.dumps(out, separators=(",", ":"))},
                ],
                "difficulty": "hard",
            })

        # --- former-name examples ---
        if not args.noise_only:
            for fn in r.get("formerNames", []):
                fname = fn.get("name")
                if fname and fname.strip().lower() != r["name"].strip().lower():
                    distractors = random.sample(
                        [c for i, c in enumerate(all_candidates) if i != ci],
                        min(args.k - 1, len(all_candidates) - 1)
                    )
                    candidates = [correct_candidate] + distractors
                    random.shuffle(candidates)

                    rag_prompt = build_rag_input(fname, candidates, candidates.index(correct_candidate))
                    out = {
                        "canonical_name": r["name"],
                        "cik": r["cik"],
                        "entity_type": r.get("entityType") or "unknown",
                        "is_former_name_input": True,
                    }
                    examples.append({
                        "messages": [
                            {"role": "system", "content": RAG_SYSTEM_PROMPT},
                            {"role": "user", "content": rag_prompt},
                            {"role": "assistant", "content": json.dumps(out, separators=(",", ":"))},
                        ],
                        "difficulty": "hard",
                    })

        # --- clean identity example (with distractors) ---
        distractors = random.sample(
            [c for i, c in enumerate(all_candidates) if i != ci],
            min(args.k - 1, len(all_candidates) - 1)
        )
        candidates = [correct_candidate] + distractors
        random.shuffle(candidates)
        rag_prompt = build_rag_input(r["name"], candidates, candidates.index(correct_candidate))
        out = {
            "canonical_name": r["name"],
            "cik": r["cik"],
            "entity_type": r.get("entityType") or "unknown",
            "is_former_name_input": False,
        }
        examples.append({
            "messages": [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": rag_prompt},
                {"role": "assistant", "content": json.dumps(out, separators=(",", ":"))},
            ],
            "difficulty": "easy",
        })

    random.shuffle(examples)
    n = len(examples)
    n_test = max(50, int(n * 0.1))
    n_valid = max(50, int(n * 0.1))

    test = examples[:n_test]
    valid = examples[n_test:n_test + n_valid]
    train = examples[n_test + n_valid:]

    for split_name, split in [("train", train), ("valid", valid), ("test", test)]:
        path = f"{args.out_dir}/rag_{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")
        print(f"rag_{split_name}: {len(split)} examples -> {path}")

    hard_count = sum(1 for e in examples if e["difficulty"] == "hard")
    easy_count = sum(1 for e in examples if e["difficulty"] == "easy")
    print(f"\nTotal: {n} examples ({hard_count} hard, {easy_count} easy) from {len(records)} companies")


if __name__ == "__main__":
    main()
