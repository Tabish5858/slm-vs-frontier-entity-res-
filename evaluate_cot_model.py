"""
Evaluate the CoT-enhanced pure-FT model on the clean holdout (no RAG).

Unlike the RAG pipeline, this model receives only the messy company name
and must produce the canonical name from memory/pattern recognition.

Usage:
  python3 evaluate_cot_model.py [--n 200] [--all]
"""

import argparse
import json
import re
import sys
import time
import os

try:
    from mlx_lm import load as mlx_load
    from mlx_lm import generate
except ImportError:
    print("ERROR: pip install mlx-lm", file=sys.stderr)
    sys.exit(1)

SYSTEM_PROMPT = (
    "You are an entity resolution assistant. Given a messy, informal, or "
    "historical company name, identify the canonical SEC-registered legal "
    "entity. First clean the name (remove wrapper text, normalize suffixes), "
    "then output the result as a JSON object."
)


def extract_json(text):
    """Extract JSON from chain-of-thought output."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    # Try to find JSON after "Output:" or at the end
    match = re.search(r'\{[^{}]*"canonical_name"[^{}]*\}', text, re.DOTALL)
    if not match:
        # Fallback: last JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test_clean_holdout.jsonl")
    ap.add_argument("--adapter", default="./adapters_v4_cot")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    ap.add_argument("--n", type=int, default=0, help="Limit to N examples (0=all)")
    ap.add_argument("--all", action="store_true", help="Evaluate all 644 examples")
    ap.add_argument("--sample", type=int, default=0, help="Random sample of N (for quick test)")
    args = ap.parse_args()

    # Load model
    print("Loading CoT fine-tuned model...", file=sys.stderr)
    if not os.path.exists(args.adapter):
        print(f"ERROR: adapter not found at {args.adapter}", file=sys.stderr)
        sys.exit(1)
    model, tokenizer = mlx_load(args.model, adapter_path=args.adapter)
    print("  ready", file=sys.stderr)

    # Load test data
    with open(args.test) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    if args.sample > 0:
        import random
        random.seed(42)
        examples = random.sample(examples, min(args.sample, len(examples)))
    elif args.n > 0:
        examples = examples[:args.n]

    total = len(examples)
    correct_name = 0
    correct_cik = 0
    hard_correct = 0
    hard_total = 0
    easy_correct = 0
    easy_total = 0
    cot_emitted = 0  # Count how many times model used chain-of-thought

    start_time = time.time()
    failures = []

    for i, ex in enumerate(examples):
        query = ex["messages"][1]["content"]
        expected = json.loads(ex["messages"][2]["content"])
        expected_name = expected.get("canonical_name", "")
        expected_cik = expected.get("cik", 0)
        difficulty = ex.get("difficulty", "easy")

        # Model inference (no RAG, pure direct)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        raw_output = generate(
            model, tokenizer, prompt=prompt, max_tokens=256, verbose=False
        )

        # Count chain-of-thought usage
        if "Reasoning:" in raw_output:
            cot_emitted += 1

        pred = extract_json(raw_output)
        pred_name = pred.get("canonical_name", "") if pred else ""
        pred_cik = pred.get("cik", 0) if pred else 0

        # Score
        is_correct = pred_name == expected_name
        is_cik_correct = pred_cik == expected_cik

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

        if not is_correct:
            failures.append({
                "difficulty": difficulty,
                "query": query[:100],
                "expected": expected_name,
                "got": pred_name,
                "cot_used": "Reasoning:" in raw_output,
                "raw": raw_output[:200],
            })

        # Progress
        if (i + 1) % 50 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{i+1}/{total}] name={correct_name}/{i+1} ({correct_name/(i+1):.1%}) "
                f"cik={correct_cik}/{i+1} ({correct_cik/(i+1):.1%}) "
                f"CoT={cot_emitted}/{i+1} | {rate:.1f} ex/s",
                file=sys.stderr,
            )

    elapsed = time.time() - start_time

    # Report
    print()
    print("=" * 60)
    print("CoT PURE-FT EVALUATION (NO RAG)")
    print("=" * 60)
    print(f"  Test file:      {args.test}")
    print(f"  Examples:       {total}")
    print(f"  Time:           {elapsed:.1f}s ({elapsed/total:.2f}s per example)")
    print(f"  CoT emitted:    {cot_emitted}/{total} ({cot_emitted/total:.1%})")
    print()
    print(f"  Canonical name:  {correct_name}/{total} = {correct_name/total:.1%}")
    print(f"  CIK:             {correct_cik}/{total} = {correct_cik/total:.1%}")
    print()

    if easy_total:
        print(f"  Easy examples:   {easy_correct}/{easy_total} = {easy_correct/easy_total:.1%}")
    if hard_total:
        print(f"  Hard examples:   {hard_correct}/{hard_total} = {hard_correct/hard_total:.1%}")

    # Comparison
    print()
    print("  BASELINE COMPARISON:")
    print(f"    Pure-FT v3 (no RAG, no CoT):      37.0% estimated")
    print(f"    CoT Pure-FT v4 (no RAG, +CoT):    {correct_name/total:.1%}")
    print(f"    RAG-FT v2 (w/ retrieval):         96.7%")
    print("  " + "=" * 56)

    # Save failures for analysis
    if failures:
        result_file = "cot_holdout_results.json"
        with open(result_file, "w") as f:
            json.dump({
                "accuracy": correct_name / total,
                "cik_accuracy": correct_cik / total,
                "hard_accuracy": hard_correct / hard_total if hard_total else 0,
                "easy_accuracy": easy_correct / easy_total if easy_total else 0,
                "cot_rate": cot_emitted / total,
                "total": total,
                "correct": correct_name,
                "failures": failures,
            }, f, indent=2)
        print(f"\n  Failures saved: {result_file}")


if __name__ == "__main__":
    main()
