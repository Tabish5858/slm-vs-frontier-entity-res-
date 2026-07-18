"""
demo.py — Standalone RAG entity resolution demo.

Runs 10 messy company names through the full RAG pipeline:
  fuzzy search → candidate selection → final canonical name + CIK.

Requirements: mlx_lm, rapidfuzz, data/rag_index_1700.jsonl,
              adapters_rag_v2/ (trained LoRA weights).
No API keys. No network calls. Runs in under 60 seconds on M-series Mac.
"""

import json
import re
import sys
import os
import time

# ── Hardcoded test queries ──
# Each is a realistic messy company name that a human might type.
DEMO_QUERIES = [
    "dba Apple, Inc., a Delaware corporation",
    "NVIDIA CORPORATION",
    "Tesla, Inc. (formerly known as Tesla Motors, Inc.)",
    "MICROSOFT CORP /DE/",
    "JPMorgan Chase & Co, the Company",
    "Walmart Inc.",
    "Berkshire Hathaway Inc /formerly/",
    "meta platforms inc",
    "Exxon Mobil Corporation (successor to former entity)",
    "COCA-COLA CO",
]

# ── Imports (lazy, so error messages are clear) ──
try:
    from rapidfuzz import process, fuzz
except ImportError:
    print("ERROR: rapidfuzz not installed. Run: pip install rapidfuzz")
    sys.exit(1)

try:
    import mlx_lm
except ImportError:
    print("ERROR: mlx_lm not installed. Run: pip install mlx-lm")
    sys.exit(1)


# ── Fuzzy search index ──
class CompanyIndex:
    """Fuzzy search over 1,700 SEC companies using rapidfuzz token_sort_ratio."""

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
def resolve_to_candidate(model_output, candidates):
    """Fuzzy-match model output to the best candidate in the list."""
    if not candidates:
        return None
    out = model_output.lower().replace(",", "").replace(".", "").strip()
    # Try exact match first
    for comp, _score in candidates:
        if comp["name"].lower().replace(",", "").replace(".", "").strip() == out:
            return comp
    # Fuzzy fallback
    best_score, best_comp = 0, None
    for comp, _score in candidates:
        s = fuzz.token_sort_ratio(out, comp["name"].lower())
        if s > best_score:
            best_score, best_comp = s, comp
    return best_comp if best_score > 75 else None


# ── System prompt (same as training/inference) ──
SYSTEM_PROMPT = (
    "You are an entity resolution assistant. Given a messy company name and "
    "candidates from our SEC database, pick the ONE best matching candidate. "
    "Respond with ONLY this exact JSON format, copying canonical_name VERBATIM "
    "from the candidate list: "
    '{"canonical_name":"...","cik":...,"entity_type":"...","is_former_name_input":false}'
)


def main():
    t0 = time.time()

    # ── Load index ──
    index_path = "data/rag_index_1700.jsonl"
    if not os.path.exists(index_path):
        print(f"ERROR: Index not found at {index_path}")
        print("Run the data pipeline first (see README).")
        sys.exit(1)
    print(f"Loading company index...", end=" ", flush=True)
    index = CompanyIndex(index_path)
    print(f"{len(index.companies)} companies indexed.")

    # ── Load model ──
    adapter_path = "./adapters_rag_v2"
    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter not found at {adapter_path}")
        print("Train the model first: mlx_lm.lora -c train_config_rag_v2.yaml")
        sys.exit(1)
    print(f"Loading RAG-aware fine-tuned model...", end=" ", flush=True)
    model, tokenizer = mlx_lm.load(
        "mlx-community/Qwen2.5-3B-Instruct-4bit",
        adapter_path=adapter_path,
    )
    print("ready.")
    t_load = time.time()
    print(f"Load time: {t_load - t0:.1f}s\n")

    # ── Run demo queries ──
    print("=" * 72)
    print("  RAG ENTITY RESOLUTION DEMO")
    print("=" * 72)

    total_time = 0.0

    for i, query in enumerate(DEMO_QUERIES, 1):
        t_query = time.time()

        # Step 1: Retrieve candidates
        candidates = index.search(query, k=3)

        # Step 2: Build RAG prompt
        lines = [
            f"Company: {query}",
            "",
            "Candidates (pick one, copy EXACTLY):",
        ]
        for j, (comp, score) in enumerate(candidates, 1):
            lines.append(
                f"  {j}. canonical_name: \"{comp['name']}\"  "
                f"cik: {comp['cik']}  (score: {score:.0f})"
            )
        lines.extend(["", "Return the JSON for the matching candidate."])
        rag_prompt = "\n".join(lines)

        # Step 3: Model inference
        prompt_tokens = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": rag_prompt},
            ],
            add_generation_prompt=True,
        )
        raw_output = mlx_lm.generate(
            model, tokenizer, prompt=prompt_tokens, max_tokens=100
        )
        pred = extract_json(raw_output)

        # Step 4: Post-process
        if pred:
            best = resolve_to_candidate(
                pred.get("canonical_name", ""), candidates
            )
            if best:
                pred["canonical_name"] = best["name"]
                pred["cik"] = best["cik"]

        elapsed = time.time() - t_query
        total_time += elapsed

        # ── Print result ──
        print(f"\n─── Query {i} ───")
        print(f"  Input:       {query}")

        print(f"  Top-3 retrieval candidates:")
        for j, (comp, score) in enumerate(candidates, 1):
            marker = " ← BEST" if j == 1 else ""
            print(f"    [{j}] {comp['name']}  (CIK: {comp['cik']}, score: {score:.0f}){marker}")

        print(f"  Model raw output:")
        for line in raw_output.strip().split("\n")[:3]:
            print(f"    {line.strip()}")

        if pred:
            name = pred.get("canonical_name", "?")
            cik = pred.get("cik", "?")
            print(f"  → RESOLVED: {name}  (CIK: {cik})")
        else:
            print(f"  → FAILED: could not parse model output")

        print(f"  Time: {elapsed:.1f}s")

    # ── Summary ──
    t_total = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  {len(DEMO_QUERIES)} queries in {t_total:.1f}s "
          f"(avg {total_time/len(DEMO_QUERIES):.1f}s/query)")
    print(f"  Model: Qwen2.5-3B-Instruct-4bit + LoRA (adapters_rag_v2)")
    print(f"  No API keys. No network calls. All local inference.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
