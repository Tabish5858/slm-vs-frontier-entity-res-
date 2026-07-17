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
    """
    TODO: wire this up to whichever frontier API you have keys for.
    Keep the SAME system_prompt and user_prompt as the fine-tuned model gets --
    apples-to-apples is the whole point of this eval.
    """
    raise NotImplementedError


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
        results.append(score(ex, pred))

    name_acc = sum(r["name_match"] for r in results) / len(results)
    cik_acc = sum(r["cik_match"] for r in results) / len(results)
    full_acc = sum(r["full_match"] for r in results) / len(results)

    print(f"\n=== {model_label} (n={len(results)}) ===")
    print(f"  canonical_name exact match: {name_acc:.1%}")
    print(f"  cik exact match:            {cik_acc:.1%}")
    print(f"  full match (both):         {full_acc:.1%}")
    return {"model": model_label, "name_acc": name_acc, "cik_acc": cik_acc, "full_acc": full_acc}


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
