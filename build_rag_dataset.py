"""
Build RAG-aware training data: (messy name + fuzzy-search candidates) -> correct answer.

Generates data/train_rag_v2.jsonl and data/valid_rag_v2.jsonl from
data/edgar_raw.jsonl (1,500 companies) and data/edgar_holdout.jsonl (200 unseen).
Holdout companies are excluded from training/validation splits.

Each training example:
  - system: entity resolution instructions
  - user: messy company name + top-5 fuzzy search candidates (1 correct + 4 distractors)
  - assistant: JSON of the correct candidate

Usage:
  python3 build_rag_dataset.py
"""

import argparse
import json
import random
import re
import sys

# Seed for reproducibility
random.seed(42)

try:
    from rapidfuzz import process, fuzz
except ImportError:
    print("ERROR: pip install rapidfuzz", file=sys.stderr)
    sys.exit(1)


# ── noise generators (mirrors build_dataset_v3.py patterns) ──

SUFFIX_VARIANTS = {
    "incorporated": ["Inc.", "Inc", "INC", "Incorporated"],
    "inc.": ["Inc", "INC", "Incorporated", "Inc."],
    "inc": ["Inc.", "INC", "Incorporated"],
    "corporation": ["Corp.", "Corp", "CORP", "Corporation"],
    "corp.": ["Corp", "CORP", "Corporation", "Corp."],
    "corp": ["Corp.", "CORP", "Corporation"],
    "company": ["Co.", "Co", "CO", "Company"],
    "co.": ["Co", "CO", "Company", "Co."],
    "limited liability company": ["LLC", "L.L.C.", "Llc"],
    "llc": ["L.L.C.", "Llc", "LLC"],
    "limited": ["Ltd.", "Ltd", "LTD", "Limited"],
    "ltd.": ["Ltd", "LTD", "Limited", "Ltd."],
    "holdings": ["Holdings", "Holdings", "HOLDINGS"],
    "group": ["Group", "Grp", "GROUP"],
    "l.p.": ["LP", "L.P.", "Limited Partnership"],
    "plc": ["PLC", "Public Limited Company", "plc"],
    "n.v.": ["N.V.", "NV", "Naamloze Vennootschap"],
}

NOISE_WRAPPERS = [
    "{name}",
    "{name}.",
    "{name},",
    "  {name}  ",
    "{name} (fka)",
    "{name} - formerly known as",
    "dba {name}",
    "{name}, the Company",
    '{name} (the "Company")',
    "{name} (successor to former entity)",
    "{name} /DE/",
    "{name}, a Delaware corporation",
    "{name} /formerly/",
]

SYSTEM_PROMPT = (
    "You are an entity resolution assistant. Given a messy company name "
    "and candidates from our SEC database, pick the ONE best matching "
    "candidate. Respond with ONLY this exact JSON format, copying "
    "canonical_name VERBATIM from the candidate list: "
    '{"canonical_name":"...","cik":...,"entity_type":"...",'
    '"is_former_name_input":false}'
)


def swap_suffix(name: str) -> str:
    lower = name.lower()
    for key, variants in SUFFIX_VARIANTS.items():
        if lower.endswith(key):
            base = name[: -len(key)].rstrip(", ")
            new_suffix = random.choice(variants)
            sep = random.choice([" ", ", "])
            return f"{base}{sep}{new_suffix}"
    return name


def noisy_casing(name: str) -> str:
    choice = random.random()
    if choice < 0.25:
        return name.upper()
    if choice < 0.45:
        return name.lower()
    if choice < 0.55:
        return name.title()
    return name


def remove_comma_before_suffix(name: str) -> str:
    return re.sub(
        r",\s*(Inc\.?|Corp\.?|Ltd\.?|LLC|L\.L\.C\.|PLC|Co\.?|LP|L\.P\.|N\.V\.)\b",
        r" \1",
        name,
        flags=re.IGNORECASE,
    )


# ── fuzzy index (same as demo.py) ──

class CompanyIndex:
    def __init__(self, paths):
        self.companies = []
        for path in paths:
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("name"):
                        self.companies.append({
                            "name": r["name"],
                            "cik": r["cik"],
                            "entity_type": r.get("entity_type") or "unknown",
                            "former_names": [fn.get("name", "") for fn in r.get("formerNames", [])],
                            "name_norm": self._normalize(r["name"]),
                        })

    @staticmethod
    def _normalize(s):
        s = s.lower().strip().replace(",", "").replace(".", "")
        s = s.replace(" & ", " and ")
        return re.sub(r"\s+", " ", s)

    def search(self, query, k=5):
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


def format_candidates(candidates):
    """Format a list of (company_dict, score) tuples for the prompt."""
    lines = []
    for i, (c, score) in enumerate(candidates, 1):
        lines.append(f"  {i}. canonical_name: \"{c['name']}\"  cik: {c['cik']}")
    return "\n".join(lines)


def build_user_prompt(messy_input, candidates):
    cand_text = format_candidates(candidates)
    return (
        f"Company: {messy_input}\n\n"
        f"Candidates (pick one, copy EXACTLY):\n{cand_text}\n\n"
        f"Return the JSON for the matching candidate."
    )


def generate_examples(companies, index, holdout_ciks, examples_per_company=5):
    """Generate RAG training examples from company records.

    For each company (not in holdout), creates examples_per_company noisy queries,
    retrieves top-5 candidates from the full index, and ensures the correct
    answer is among them.
    """
    examples = []
    holdout = set(holdout_ciks)

    for r in companies:
        cik = r["cik"]
        if cik in holdout:
            continue

        canonical = r["name"]
        if not canonical:
            continue

        entity_type = r.get("entityType") or "unknown"

        generated = 0
        attempts = 0

        while generated < examples_per_company and attempts < examples_per_company * 3:
            attempts += 1

            # Generate a noisy query
            name = canonical
            if random.random() < 0.5:
                name = swap_suffix(name)
            if random.random() < 0.3:
                name = noisy_casing(name)
            if random.random() < 0.15 and "," in canonical:
                name = remove_comma_before_suffix(canonical)
                name = swap_suffix(name) if random.random() < 0.5 else name

            wrapper = random.choice(NOISE_WRAPPERS)
            query = wrapper.format(name=name).strip()

            if not query or query == canonical:
                continue

            # Add former name queries (33% chance)
            former_names = [fn.get("name", "") for fn in r.get("formerNames", [])]
            if former_names and random.random() < 0.33:
                fn_name = random.choice(former_names)
                if fn_name and fn_name.strip().lower() != canonical.strip().lower():
                    query = fn_name
                    if random.random() < 0.4:
                        wrapper = random.choice(NOISE_WRAPPERS)
                        query = wrapper.format(name=fn_name).strip()

            # Retrieve candidates from full index
            candidates = index.search(query, k=5)
            if len(candidates) < 2:
                continue

            # Check if correct company is in candidates
            correct_idx = None
            for i, (c, score) in enumerate(candidates):
                if c["cik"] == cik and c["name"] == canonical:
                    correct_idx = i
                    break

            if correct_idx is None:
                continue  # correct company not in top-5, skip

            # Build output
            output = {
                "canonical_name": canonical,
                "cik": cik,
                "entity_type": entity_type,
                "is_former_name_input": any(
                    fn in query for fn in former_names if fn
                ),
            }

            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(query, candidates)},
                    {"role": "assistant", "content": json.dumps(output, separators=(",", ":"))},
                ],
                "difficulty": "hard" if "former" in query.lower() or "/DE/" in query else "easy",
            }
            examples.append(example)
            generated += 1

    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edgar", default="data/edgar_raw.jsonl")
    ap.add_argument("--holdout", default="data/edgar_holdout.jsonl")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--per_company", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print("Loading companies...")
    companies = []
    with open(args.edgar) as f:
        for line in f:
            companies.append(json.loads(line))

    holdout_ciks = set()
    with open(args.holdout) as f:
        for line in f:
            r = json.loads(line)
            holdout_ciks.add(r["cik"])

    print(f"  {len(companies)} training companies, {len(holdout_ciks)} holdout CIKs")

    print("Building fuzzy search index...")
    index = CompanyIndex([args.edgar, args.holdout])
    print(f"  {len(index.companies)} companies indexed")

    print(f"Generating RAG training examples (~{args.per_company}/company)...")
    examples = generate_examples(companies, index, holdout_ciks, args.per_company)
    print(f"  {len(examples)} examples generated")

    # Split: 85% train, 15% valid
    random.shuffle(examples)
    n_valid = max(100, int(len(examples) * 0.15))
    valid = examples[:n_valid]
    train = examples[n_valid:]

    for split_name, split in [("train_rag_v2", train), ("valid_rag_v2", valid)]:
        path = f"{args.out_dir}/{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")
        print(f"  {path}: {len(split)} examples")

    # Create symlink directory for mlx_lm.lora
    import os
    data_dir = "data_rag_v2"
    os.makedirs(data_dir, exist_ok=True)

    # Remove old links, create new relative ones
    for name in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
        link = os.path.join(data_dir, name)
        if os.path.islink(link):
            os.unlink(link)
        elif os.path.exists(link):
            os.remove(link)

        if name == "train.jsonl":
            target = "../data/train_rag_v2.jsonl"
        elif name == "valid.jsonl":
            target = "../data/valid_rag_v2.jsonl"
        else:
            target = "../data/test.jsonl"
        os.symlink(target, link)

    print(f"\nSymlinks created in {data_dir}/ → data/train_rag_v2.jsonl, valid_rag_v2.jsonl, test.jsonl")
    print("Ready for: mlx_lm.lora -c train_config_rag_v2.yaml")


if __name__ == "__main__":
    main()
