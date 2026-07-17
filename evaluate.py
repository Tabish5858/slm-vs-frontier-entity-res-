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
    """Call a frontier model API. Currently supports Gemini via google-genai SDK.
    Set GEMINI_API_KEY env var before running."""
    if model_name.startswith("gemini"):
        return _call_gemini(user_prompt, system_prompt, model_name)
    raise NotImplementedError(f"Frontier model '{model_name}' not configured. Add API key + SDK.")


def _call_gemini(user_prompt: str, system_prompt: str, model_name: str) -> str:
    import os, json, urllib.request, urllib.error, sys

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY environment variable")

    model_map = {
        "gemini-3.1-pro": "gemini-3.1-pro-preview",
        "gemini-3-pro": "gemini-3-pro-preview",
        "gemini-3-flash": "gemini-3-flash-preview",
        "gemini-flash": "gemini-flash-latest",
    }
    model_id = model_map.get(model_name, model_name)

    # ponytail: systemInstruction hangs on v1beta; inline system prompt instead
    combined = system_prompt + "\n\nUser input: " + user_prompt

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    body = {
        "contents": [{"parts": [{"text": combined}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 200},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else str(e)
        print(f"  Gemini API error ({e.code}): {error_body[:300]}", file=sys.stderr)
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
    cik_match = str(prediction.get("cik", "")) == str(expected["cik"])
    return {
        "name_match": name_match,
        "cik_match": cik_match,
        "full_match": name_match and cik_match,
    }


def run_eval(test_path: str, n: int, model_fn, model_label: str):
    examples = []
    with open(test_path) as f:
        for line in f:
            examples.append(json.loads(line))
    examples = examples[:n]

    results = []
    for ex in examples:
        system_prompt = ex["messages"][0]["content"]
        user_prompt = ex["messages"][1]["content"]
        raw = model_fn(user_prompt, system_prompt)
        pred = extract_json(raw)
        difficulty = ex.get("difficulty", "unknown")
        result = score(ex, pred)
        result["difficulty"] = difficulty
        results.append(result)

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
    # TODO: uncomment when frontier API keys are configured
    # all_results.append(run_eval(args.test, test_count, lambda u,s: call_frontier_model(u,s,"gpt-5.6"), "GPT-5.6"))
    # all_results.append(run_eval(args.test, test_count, lambda u,s: call_frontier_model(u,s,"claude-opus-4-8"), "Claude Opus 4.8"))
    # all_results.append(run_eval(args.test, test_count, lambda u,s: call_frontier_model(u,s,"gemini-3.1-pro"), "Gemini 3.1 Pro"))

    print("\n" + "=" * 50)
    print(json.dumps(all_results, indent=2))
