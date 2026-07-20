"""
build_dataset_v4_cot.py — Enhanced pure-FT dataset with chain-of-thought.

Key changes from v3:
  1. Chain-of-thought output: model learns to reason before JSON output
  2. 5x more examples per company (target ~50K total)
  3. Extended noise wrappers (25+ patterns)
  4. All suffix combinations for every company
  5. Progressive difficulty curriculum

Output: data/train_v4_cot.jsonl, data/valid_v4_cot.jsonl
"""

import argparse
import hashlib
import json
import random
import re

random.seed(42)

SYSTEM_PROMPT = (
    "You are an entity resolution assistant. Given a messy, informal, or "
    "historical company name, identify the canonical SEC-registered legal "
    "entity. First clean the name (remove wrapper text, normalize suffixes), "
    "then output the result as a JSON object."
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
    "l.p.": ["LP", "L.P.", "Limited Partnership"],
    "plc": ["PLC", "Public Limited Company", "plc"],
    "n.v.": ["N.V.", "NV", "Naamloze Vennootschap"],
}

# Extended noise wrappers (was 13, now 28)
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
    # New wrappers for v4
    "f/k/a {name}",
    "{name}, a California corporation",
    "{name}, d/b/a",
    "formerly {name}",
    "Company: {name}",
    "Legal name: {name}",
    "{name} (dba)",
    "doing business as {name}",
    "{name} (SEC registrant)",
    "{name}, a Nevada corporation",
    "{name}, a New York corporation",
    "{name} (and subsidiaries)",
    "{name} and its subsidiaries",
    "{name}, a publicly traded company",
    "{name}, the registrant",
]

LIGHT_WRAPPERS = [
    "{name}",
    "{name}.",
    "{name},",
    "  {name}  ",
]


def swap_suffix(name: str) -> str:
    """Swap suffix to a random variant."""
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


def strip_noise(name: str) -> str:
    """Clean wrapper text to show intermediate reasoning step."""
    cleaned = name
    cleaned = re.sub(r'\s*\(fka\)\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*-\s*formerly\s+known\s+as\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*\(the\s*"?Company"?\)\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*,\s*the\s*"?Company"?\s*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*\(successor\s+to\s+former\s+entity\)\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*/DE/\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*,\s*a\s+(Delaware|California|Nevada|New\s+York)\s+corporation\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*/formerly/\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^dba\s+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^f/k/a\s+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^formerly\s+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*\(dba\)\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^doing\s+business\s+as\s+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^Company:\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^Legal\s+name:\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*\(SEC\s+registrant\)\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*\(and\s+subsidiaries\)\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+and\s+its\s+subsidiaries\s*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*,\s+a\s+publicly\s+traded\s+company\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*,\s+the\s+registrant\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*,\s*d/b/a\s*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip().rstrip(".,")
    return cleaned


def normalize_suffix(name: str) -> str:
    """Normalize suffix to standard form for intermediate step."""
    lower = name.lower()
    suffix_map = {
        'incorporated': 'Inc.',
        'inc': 'Inc.',
        'inc.': 'Inc.',
        'corporation': 'Corp.',
        'corp': 'Corp.',
        'corp.': 'Corp.',
        'company': 'Co.',
        'co': 'Co.',
        'co.': 'Co.',
        'limited': 'Ltd.',
        'ltd': 'Ltd.',
        'ltd.': 'Ltd.',
        'limited liability company': 'LLC',
        'llc': 'LLC',
        'l.l.c.': 'LLC',
        'holdings': 'Holdings',
        'group': 'Group',
        'plc': 'PLC',
        'l.p.': 'LP',
        'lp': 'LP',
        'n.v.': 'NV',
        'nv': 'NV',
    }
    for suffix, canonical in suffix_map.items():
        if lower.endswith(suffix):
            base = name[: -len(suffix)].rstrip(", ")
            return f"{base} {canonical}"
    return name


def build_output(record, is_former):
    return {
        "canonical_name": record["name"],
        "cik": record["cik"],
        "entity_type": record.get("entityType") or "unknown",
        "is_former_name_input": is_former,
    }


def build_cot_response(messy_input: str, canonical_name: str, output_obj: dict) -> str:
    """Build chain-of-thought response."""
    cleaned = strip_noise(messy_input)
    normalized = normalize_suffix(cleaned)

    # Build reasoning steps
    steps = []
    if cleaned != messy_input.strip():
        steps.append(f"Clean '{messy_input}' → '{cleaned}'")
    if normalized != cleaned:
        steps.append(f"Normalize '{cleaned}' → '{normalized}'")

    if steps:
        reasoning = " | ".join(steps)
        return f"Reasoning: {reasoning}. Output: {json.dumps(output_obj, separators=(',', ':'))}"
    else:
        return json.dumps(output_obj, separators=(",", ":"))


def to_chat(messy_input: str, output_obj: dict, difficulty: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": messy_input},
            {"role": "assistant",
             "content": build_cot_response(messy_input, output_obj["canonical_name"], output_obj)},
        ],
        "difficulty": difficulty,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/edgar_raw.jsonl")
    ap.add_argument("--holdout", default="data/edgar_holdout.jsonl")
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    # Load holdout CIKs for exclusion
    holdout_ciks = set()
    with open(args.holdout) as f:
        for line in f:
            r = json.loads(line)
            holdout_ciks.add(r["cik"])

    print(f"Holdout CIKs: {len(holdout_ciks)}")

    # Load EDGAR records
    records = []
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            if r["cik"] not in holdout_ciks:
                records.append(r)

    print(f"Training companies: {len(records)} (excluding {len(holdout_ciks)} holdout)")

    examples = []

    for r in records:
        canonical = r["name"]
        if not canonical:
            continue

        out = build_output(r, is_former=False)

        # ============================================================
        # 1. Former names (3x each: clean, suffix-swap, wrapper)
        # ============================================================
        for fn in r.get("formerNames", []):
            fname = fn.get("name")
            if not fname or fname.strip().lower() == canonical.strip().lower():
                continue

            fout = build_output(r, is_former=True)

            # 1a. Clean former name
            examples.append(to_chat(fname, fout, "hard"))

            # 1b. Suffix-swapped former name
            noisy_fn = swap_suffix(fname)
            if noisy_fn != fname:
                examples.append(to_chat(noisy_fn, fout, "hard"))

            # 1c. Wrapped former name
            wrapper = random.choice(NOISE_WRAPPERS)
            wrapped = wrapper.format(name=fname).strip()
            if wrapped != fname:
                examples.append(to_chat(wrapped, fout, "hard"))

        # ============================================================
        # 2. Suffix-precision passes (6 per company, was 3)
        # ============================================================
        for _ in range(6):
            name = swap_suffix(canonical)
            wrapper = random.choice(LIGHT_WRAPPERS)
            messy = wrapper.format(name=name).strip()
            examples.append(to_chat(messy, out, "hard"))

        # ============================================================
        # 3. Full noise passes (6 per company, was 3)
        # ============================================================
        for _ in range(6):
            name = canonical
            if random.random() < 0.7:
                name = swap_suffix(name)
            if random.random() < 0.5:
                name = noisy_casing(name)
            wrapper = random.choice(NOISE_WRAPPERS)
            messy = wrapper.format(name=name).strip()
            examples.append(to_chat(messy, out, "hard"))

        # ============================================================
        # 4. Comma-precision passes (3 per company, was 2)
        # ============================================================
        if "," in canonical:
            for _ in range(3):
                no_comma = remove_comma_before_suffix(canonical)
                if no_comma != canonical:
                    wrapper = random.choice(LIGHT_WRAPPERS)
                    messy = wrapper.format(name=no_comma).strip()
                    examples.append(to_chat(messy, out, "hard"))

        # ============================================================
        # 5. All-suffix variants (NEW — one per suffix type)
        #    "Apple Inc" → "APPLE INCORPORATED" → canonical
        # ============================================================
        for suffix_forms in [
            ["INCORPORATED", "Incorporated", "Incorporated."],
            ["CORPORATION", "Corporation", "Corporation."],
            ["COMPANY", "Company", "Company."],
        ]:
            for form in suffix_forms:
                # Try replacing the last word with this suffix form
                parts = canonical.rsplit(" ", 1)
                if len(parts) == 2 and parts[1].lower() in ["inc.", "inc", "corp.", "corp", "co.", "co"]:
                    variant = f"{parts[0]} {form}"
                    examples.append(to_chat(variant, out, "hard"))
                    break  # one per suffix category

        # ============================================================
        # 6. Double-noise combos (NEW — 3 per company)
        #    wrapper + suffix swap + casing all at once
        # ============================================================
        for _ in range(3):
            name = canonical
            name = swap_suffix(name)
            name = noisy_casing(name)
            wrapper = random.choice(NOISE_WRAPPERS)
            messy = wrapper.format(name=name).strip()
            examples.append(to_chat(messy, out, "hard"))

        # ============================================================
        # 7. Clean identity (1 per company, unchanged)
        # ============================================================
        examples.append(to_chat(canonical, out, "easy"))

    print(f"Generated {len(examples)} examples")

    # Split: 85% train, 15% valid
    random.shuffle(examples)
    n_valid = max(200, int(len(examples) * 0.12))
    valid = examples[:n_valid]
    train = examples[n_valid:]

    for split_name, split in [("train_v4_cot", train), ("valid_v4_cot", valid)]:
        path = f"{args.out_dir}/{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")

    # Stats — extract JSON from CoT response
    def extract_json_from_response(text):
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    th = sum(1 for e in train if e["difficulty"] == "hard")
    te = sum(1 for e in train if e["difficulty"] == "easy")
    vh = sum(1 for e in valid if e["difficulty"] == "hard")
    ve = sum(1 for e in valid if e["difficulty"] == "easy")
    tf = sum(1 for e in train if extract_json_from_response(e["messages"][2]["content"]).get("is_former_name_input"))

    print(f"\ntrain_v4_cot: {len(train)} ({th} hard, {te} easy, {tf} former)")
    print(f"valid_v4_cot: {len(valid)} ({vh} hard, {ve} easy)")

    # Create symlink directory
    import os
    data_dir = "data_v4"
    os.makedirs(data_dir, exist_ok=True)
    for name, target in [
        ("train.jsonl", "../data/train_v4_cot.jsonl"),
        ("valid.jsonl", "../data/valid_v4_cot.jsonl"),
        ("test.jsonl", "../data/test.jsonl"),
    ]:
        link = os.path.join(data_dir, name)
        if os.path.islink(link):
            os.unlink(link)
        elif os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)

    print(f"\nSymlinks: {data_dir}/ → data/train_v4_cot.jsonl, valid_v4_cot.jsonl, test.jsonl")


if __name__ == "__main__":
    main()
