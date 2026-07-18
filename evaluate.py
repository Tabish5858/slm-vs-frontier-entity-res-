"""
Eval harness: run the fine-tuned model and frontier baselines on the same
held-out test.jsonl, score exact-match on canonical_name + cik, report side by side.

Fill in the two model-calling functions for your setup:
  - call_finetuned_model(): local MLX inference
  - call_frontier_model(): API call to GPT-5.6 / Claude Opus 4.8 / Gemini 3.1 Pro

Usage:
  python evaluate.py --test ../data/test.jsonl --n 100
"""

import argparse
import json
import re
import time
import sys

# ── Hybrid matcher (no model, just rules + database) ─────────────────

from evaluate_hybrid import HybridMatcher

def call_hybrid_matcher(messy_input: str, _system_prompt: str = "") -> str:
    """Zero-model entity resolution: clean → normalize → database match."""
    global _hybrid_matcher
    result = _hybrid_matcher.match(messy_input)
    if result:
        return json.dumps({
            "canonical_name": result["name"],
            "cik": result["cik"],
            "entity_type": result.get("entity_type", "unknown"),
            "is_former_name_input": False,
        })
    return "{}"

_hybrid_matcher = None


def extract_json(text: str):
    """Model output should be pure JSON, but strip markdown fences / stray text defensively."""
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


# ponytail: lazy-loaded module-level cache, reload if adapter path changes
_ft_model = None
_ft_tokenizer = None

def call_finetuned_model(user_prompt: str, system_prompt: str) -> str:
    global _ft_model, _ft_tokenizer
    if _ft_model is None:
        from mlx_lm import load
        _ft_model, _ft_tokenizer = load(
            "./models/qwen2.5-3b-instruct-4bit",
            adapter_path="./adapters_v3",
        )
    from mlx_lm import generate
    prompt = _ft_tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
        add_generation_prompt=True,
    )
    return generate(_ft_model, _ft_tokenizer, prompt=prompt, max_tokens=200)


def call_frontier_model(user_prompt: str, system_prompt: str, model_name: str) -> str:
    """Call a frontier model API. Supports Claude and GPT.
    Set CLAUDE_API_KEY or OPENAI_API_KEY env var."""
    if "claude" in model_name.lower():
        return _call_claude(user_prompt, system_prompt, model_name)
    return _call_openai(user_prompt, system_prompt, model_name)


def _call_claude(user_prompt: str, system_prompt: str, model_name: str) -> str:
    import os, sys
    from anthropic import Anthropic

    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        print("  Claude skipped: CLAUDE_API_KEY not set", file=sys.stderr)
        return ""

    client = Anthropic(api_key=api_key)
    try:
        kwargs = {"model": model_name, "max_tokens": 200,
                  "system": system_prompt,
                  "messages": [{"role": "user", "content": user_prompt}]}
        # Newer Claude models deprecate temperature
        if "opus-4-8" not in model_name and "sonnet-5" not in model_name:
            kwargs["temperature"] = 0
        response = client.messages.create(**kwargs)
        # Claude returns content blocks; first text block
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
    except Exception as e:
        print(f"  Claude error ({model_name}): {e}", file=sys.stderr)
        return ""


def _call_openai(user_prompt: str, system_prompt: str, model_name: str) -> str:
    import os, sys
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=api_key)

    # Reasoning models: no temperature, use max_completion_tokens, no system role
    is_reasoning = "5.6" in model_name or "5.5" in model_name or "o1" in model_name or "o3" in model_name

    if is_reasoning:
        messages = [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]
        kwargs = {"model": model_name, "max_completion_tokens": 300}
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = {"model": model_name, "temperature": 0, "max_tokens": 200}

    try:
        response = client.chat.completions.create(messages=messages, **kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"  OpenAI error ({model_name}): {e}", file=sys.stderr)
        return ""


def score(example, prediction: dict) -> dict:
    expected = json.loads(example["messages"][-1]["content"])
    if prediction is None:
        return {"name_match": False, "cik_match": False, "full_match": False}
    name_match = (
        (prediction.get("canonical_name") or "").strip().lower()
        == expected["canonical_name"].strip().lower()
    )
    cik_match = str(prediction.get("cik") or "").lstrip("0") == str(expected["cik"]).lstrip("0")
    return {
        "name_match": name_match,
        "cik_match": cik_match,
        "full_match": name_match and cik_match,
    }


def run_eval(test_path: str, n: int, model_fn, model_label: str, sleep_between: float = 0):
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
        pred = extract_json(raw)
        difficulty = ex.get("difficulty", "unknown")
        result = score(ex, pred)
        result["difficulty"] = difficulty
        results.append(result)
        if sleep_between and i < len(examples) - 1:
            time.sleep(sleep_between)

    def bucket_accuracy(rs, key):
        if not rs:
            return 0.0
        return sum(r[key] for r in rs) / len(rs)

    hard_results = [r for r in results if r["difficulty"] == "hard"]
    easy_results = [r for r in results if r["difficulty"] == "easy"]

    print(f"\n=== {model_label} (n={len(results)}) ===")
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
        "name_acc_overall": bucket_accuracy(results, "name_match"),
        "name_acc_hard": bucket_accuracy(hard_results, "name_match"),
        "name_acc_easy": bucket_accuracy(easy_results, "name_match"),
        "cik_acc": bucket_accuracy(results, "cik_match"),
        "full_acc": bucket_accuracy(results, "full_match"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="./data/test.jsonl")
    ap.add_argument("--n", type=int, default=0, help="Number of test examples (0 = all)")
    args = ap.parse_args()

    test_count = args.n if args.n > 0 else sum(1 for _ in open(args.test))
    print(f"Evaluating {test_count} test examples...")

    all_results = []
    
    # 0. Hybrid matcher (rules + database, zero model)
    import evaluate_hybrid
    _hybrid_matcher = HybridMatcher("./data/edgar_raw.jsonl")
    all_results.append(run_eval(args.test, test_count,
        lambda u, s: call_hybrid_matcher(u, s),
        "Hybrid (rules + DB, no model)"))
    
    all_results.append(run_eval(args.test, test_count,
        lambda u, s: call_finetuned_model(u, s), "Fine-tuned Qwen2.5-3B (ours)"))
    all_results.append(run_eval(args.test, test_count,
        lambda u, s: call_frontier_model(u, s, "claude-opus-4-8"),
        "Claude Opus 4.8", sleep_between=0.2))

    print("\n" + "=" * 50)
    print(json.dumps(all_results, indent=2))
