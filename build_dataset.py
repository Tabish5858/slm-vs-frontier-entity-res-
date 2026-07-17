"""
Turn raw EDGAR company records into (messy_name -> canonical structured entity) pairs.

Two sources of "messy" input per company:
  1. REAL former names EDGAR recorded (actual historical renames -- the strongest signal)
  2. SYNTHETIC noise applied to the canonical name (suffix swaps, casing, punctuation,
     spacing, "dba"-style wrapper text) -- gives volume beyond what former names alone provide

Output schema (what the model must produce), kept compact for exact-match eval:
  {
    "canonical_name": "...",
    "cik": 320193,              <- unique ID, best field for scoring exact match
    "entity_type": "corporation" | "LLC" | ... (SEC's own entityType field, may be blank),
    "is_former_name_input": true/false
  }

Writes:
  data/train.jsonl
  data/valid.jsonl
  data/test.jsonl          <- held out, untouched by training, used for the eval vs frontier models
in MLX chat-format (messages list) ready for `mlx_lm.lora`.
"""

import argparse
import json
import random
import re

random.seed(42)

SYSTEM_PROMPT = (
    "You are an entity resolution assistant. Given a messy, informal, or "
    "historical company name, identify the canonical SEC-registered legal "
    "entity. Respond with ONLY a JSON object with keys: canonical_name, cik, "
    "entity_type, is_former_name_input. No other text."
)

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
    "holdings": ["Holdings", "Hldgs", "HOLDINGS"],
    "group": ["Group", "Grp", "GROUP"],
    # New: failure-analysis patterns
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
    "{name} (the \"Company\")",
    # New: patterns from failure analysis
    "{name} (successor to former entity)",
    "{name} /DE/",
    "{name}, a Delaware corporation",
    "{name} /formerly/",
]


def swap_suffix(name: str) -> str:
    """Randomly swap a trailing corporate suffix with an equivalent variant."""
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


def punctuation_noise(name: str) -> str:
    name = re.sub(r"\s*&\s*", random.choice([" & ", " and ", " AND "]), name)
    if random.random() < 0.3:
        name = name.replace(",", "")
    if random.random() < 0.2:
        name = re.sub(r"\s+", "  ", name)  # double spaces
    return name


def make_messy_variant(canonical: str) -> tuple[str, int]:
    """Return (messy_name, noise_count). Aggressive noise for suffix-heavy failures."""
    name = canonical
    noise_count = 0
    # Boost suffix swap — 58% of failures are suffix mismatches
    if random.random() < 0.85:
        name = swap_suffix(name)
        noise_count += 1
    if random.random() < 0.50:
        name = punctuation_noise(name)
        noise_count += 1
    if random.random() < 0.40:
        name = noisy_casing(name)
        noise_count += 1
    wrapper = random.choice(NOISE_WRAPPERS)
    name = wrapper.format(name=name)
    return name.strip(), noise_count


def build_output(record, is_former):
    return {
        "canonical_name": record["name"],
        "cik": record["cik"],
        "entity_type": record.get("entityType") or "unknown",
        "is_former_name_input": is_former,
    }


def to_chat_example(messy_input: str, output_obj: dict, difficulty: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": messy_input},
            {"role": "assistant", "content": json.dumps(output_obj, separators=(",", ":"))},
        ],
        "difficulty": difficulty,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/edgar_raw.jsonl")
    ap.add_argument("--synthetic_per_company", type=int, default=5,
                     help="How many synthetic noisy variants to generate per company")
    ap.add_argument("--out_dir", default="../data")
    args = ap.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))

    examples = []
    for r in records:
        if not r.get("name"):
            continue

        # 1. Real former-name pairs (strongest, real-world signal)
        for fn in r.get("formerNames", []):
            fname = fn.get("name")
            if fname and fname.strip().lower() != r["name"].strip().lower():
                out = build_output(r, is_former=True)
                examples.append(to_chat_example(fname, out, "hard"))

        # 2. Synthetic noisy variants of the canonical name
        for _ in range(args.synthetic_per_company):
            messy, noise_count = make_messy_variant(r["name"])
            out = build_output(r, is_former=False)
            diff = "hard" if noise_count >= 2 else "easy"
            examples.append(to_chat_example(messy, out, diff))

        # 2b. Extra suffix-swap-only passes — 58% of failures are suffix mismatches
        for _ in range(2):
            name = r["name"]
            if random.random() < 0.9:
                name = swap_suffix(name)
            wrapper = random.choice(NOISE_WRAPPERS)
            out = build_output(r, is_former=False)
            examples.append(to_chat_example(wrapper.format(name=name).strip(), out, "hard"))

        # 3. One clean "identity" example so the model learns not to over-correct
        #    already-clean canonical names
        out = build_output(r, is_former=False)
        examples.append(to_chat_example(r["name"], out, "easy"))

    random.shuffle(examples)
    n = len(examples)
    n_test = max(50, int(n * 0.1))
    n_valid = max(50, int(n * 0.1))

    test = examples[:n_test]
    valid = examples[n_test:n_test + n_valid]
    train = examples[n_test + n_valid:]

    for split_name, split in [("train", train), ("valid", valid), ("test", test)]:
        path = f"{args.out_dir}/{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")
        print(f"{split_name}: {len(split)} examples -> {path}")

    print(f"\nTotal: {n} examples from {len(records)} companies")


if __name__ == "__main__":
    main()
