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
            adapter_path="./adapters",
        )
    from mlx_lm import generate
    prompt = _ft_tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
        add_generation_prompt=True,
    )
    return generate(_ft_model, _ft_tokenizer, prompt=prompt, max_tokens=200)


def call_frontier_model(user_prompt: str, system_prompt: str, model_name: str) -> str:
    """Call a frontier model API. Currently supports Gemini via REST.
    Set GEMINI_API_KEY env var before running."""
    if "gemini" in model_name:
        return _call_gemini(user_prompt, system_prompt, model_name)
    raise NotImplementedError(f"Frontier model '{model_name}' not configured. Add API key + SDK.")


def _call_gemini(user_prompt: str, system_prompt: str, model_name: str) -> str:
    import os, json, subprocess, sys

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY environment variable")

    model_map = {
        "gemini-3.1-pro": "gemini-3.1-pro-preview",
        "gemini-3-pro": "gemini-3-pro-preview",
        "gemini-3-flash": "gemini-3-flash-preview",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.0-flash": "gemini-2.0-flash",
    }
    model_id = model_map.get(model_name, model_name)

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={api_key}"
    )
    body = json.dumps({
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 512},
    })

    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "45", url,
             "-H", "Content-Type: application/json",
             "-d", body],
            capture_output=True, text=True, timeout=50,
        )
        if result.returncode != 0:
            print(f"  curl failed (rc={result.returncode}): {result.stderr[:200]}", file=sys.stderr)
            return ""
        data = json.loads(result.stdout)
        if "error" in data:
            print(f"  Gemini API error: {data['error'].get('message','')[:200]}", file=sys.stderr)
            return ""
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except subprocess.TimeoutExpired:
        print(f"  Gemini call timed out", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"  Gemini call failed: {e}", file=sys.stderr)
        return ""


def score(example, prediction: dict) -> dict:
    expected = json.loads(example["messages"][-1]["content"])
    if prediction is None:
        return {"name_match": False, "cik_match": False, "full_match": False}
    name_match = (
        prediction.get("canonical_name", "").strip().lower()
        == expected["canonical_name"].strip().lower()
    )
    cik_match = str(prediction.get("cik", "")).lstrip("0") == str(expected["cik"]).lstrip("0")
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
    all_results.append(run_eval(args.test, test_count,
        lambda u, s: call_finetuned_model(u, s), "Fine-tuned Qwen2.5-3B (ours)"))
    all_results.append(run_eval(args.test, test_count,
        lambda u, s: call_frontier_model(u, s, "gemini-3.1-pro"),
        "Gemini 3.1 Pro", sleep_between=1.0))

    print("\n" + "=" * 50)
    print(json.dumps(all_results, indent=2))
