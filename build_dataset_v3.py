"""
build_dataset_v3.py — improved data pipeline for suffix-precision + former-name coverage.

Key changes from v2:
  1. 3x per former name (clean + suffix-swap + wrapper) instead of 1x
  2. Reduced synthetic noise aggressiveness; added explicit suffix-precision passes
  3. Comma-insertion training: for names like "DOCUSIGN, INC.", train pairs
     where input lacks comma but output has it
  4. Test.jsonl is NEVER touched — reads it only to build an exclusion set
     so train/valid never contain exact duplicates of test inputs

Output: data/train_v3.jsonl, data/valid_v3.jsonl (test.jsonl stays unchanged)
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
    "{name} (successor to former entity)",
    "{name} /DE/",
    "{name}, a Delaware corporation",
    "{name} /formerly/",
]

# Lighter wrappers for suffix-precision passes (don't distract with irrelevant text)
LIGHT_WRAPPERS = [
    "{name}",
    "{name}.",
    "{name},",
    "  {name}  ",
]


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
    """Remove comma before common suffixes: 'Foo, Inc.' -> 'Foo Inc.'"""
    return re.sub(r',\s*(Inc\.?|Corp\.?|Ltd\.?|LLC|L\.L\.C\.|PLC|Co\.?|LP|L\.P\.|N\.V\.)\b',
                  r' \1', name, flags=re.IGNORECASE)


def build_output(record, is_former):
    return {
        "canonical_name": record["name"],
        "cik": record["cik"],
        "entity_type": record.get("entityType") or "unknown",
        "is_former_name_input": is_former,
    }


def to_chat(messy_input: str, output_obj: dict, difficulty: str) -> dict:
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
    ap.add_argument("--input", default="data/edgar_raw.jsonl")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--test_file", default="data/test.jsonl")
    args = ap.parse_args()

    # --- Load exclusion set from test.jsonl (never modify test, just read it) ---
    test_inputs = set()
    test_hasher = hashlib.sha256()
    with open(args.test_file) as f:
        for line in f:
            test_hasher.update(line.encode())
            ex = json.loads(line)
            test_inputs.add(ex["messages"][1]["content"])
    test_hash = test_hasher.hexdigest()
    print(f"Test set: {len(test_inputs)} unique inputs, sha256={test_hash}")

    # --- Load EDGAR records ---
    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))

    examples = []
    skipped = 0

    for r in records:
        canonical = r["name"]
        if not canonical:
            continue

        # ============================================================
        # 1. REAL FORMER NAMES × 3 (was × 1)
        #    For each former name: clean, suffix-swapped, wrapper
        # ============================================================
        for fn in r.get("formerNames", []):
            fname = fn.get("name")
            if not fname or fname.strip().lower() == canonical.strip().lower():
                continue

            out = build_output(r, is_former=True)

            # 1a. Clean former name → canonical
            if fname not in test_inputs:
                examples.append(to_chat(fname, out, "hard"))
            else:
                skipped += 1

            # 1b. Former name with suffix swap → canonical
            noisy_fn = swap_suffix(fname)
            if noisy_fn != fname and noisy_fn not in test_inputs:
                examples.append(to_chat(noisy_fn, out, "hard"))
            elif noisy_fn in test_inputs:
                skipped += 1

            # 1c. Former name with wrapper → canonical
            wrapper = random.choice(NOISE_WRAPPERS)
            wrapped_fn = wrapper.format(name=fname).strip()
            if wrapped_fn != fname and wrapped_fn not in test_inputs:
                examples.append(to_chat(wrapped_fn, out, "hard"))
            elif wrapped_fn in test_inputs:
                skipped += 1

        # ============================================================
        # 2. SYNTHETIC — SUFFIX-PRECISION PASSES (3 per company)
        #    Light noise: just suffix swap + light wrapper → exact canonical
        #    These teach the model to output the EXACT canonical name
        #    regardless of suffix/wrapper variations.
        # ============================================================
        for _ in range(3):
            name = swap_suffix(canonical)
            wrapper = random.choice(LIGHT_WRAPPERS)
            messy = wrapper.format(name=name).strip()
            out = build_output(r, is_former=False)
            if messy not in test_inputs:
                examples.append(to_chat(messy, out, "hard"))
            else:
                skipped += 1

        # ============================================================
        # 3. SYNTHETIC — FULL NOISE (3 per company)
        #    Aggressive noise for robustness training
        # ============================================================
        for _ in range(3):
            name = canonical
            if random.random() < 0.7:
                name = swap_suffix(name)
            if random.random() < 0.4:
                name = noisy_casing(name)
            wrapper = random.choice(NOISE_WRAPPERS)
            messy = wrapper.format(name=name).strip()
            out = build_output(r, is_former=False)
            if messy not in test_inputs:
                examples.append(to_chat(messy, out, "hard"))
            else:
                skipped += 1

        # ============================================================
        # 4. COMMA-PRECISION PASSES (for companies whose canonical
        #    name contains a comma before the suffix)
        #    "DOCUSIGN INC" → "DOCUSIGN, INC."
        #    This directly addresses the #1 failure mode (312 cases).
        # ============================================================
        if "," in canonical:
            for _ in range(2):
                # Remove comma: "DOCUSIGN, INC." → "DOCUSIGN INC."
                no_comma = remove_comma_before_suffix(canonical)
                if no_comma != canonical:
                    wrapper = random.choice(LIGHT_WRAPPERS)
                    messy = wrapper.format(name=no_comma).strip()
                    out = build_output(r, is_former=False)
                    if messy not in test_inputs:
                        examples.append(to_chat(messy, out, "hard"))
                    else:
                        skipped += 1

        # ============================================================
        # 5. CLEAN IDENTITY (1 per company)
        #    Canonical → canonical, teaches model not to over-correct
        # ============================================================
        out = build_output(r, is_former=False)
        if canonical not in test_inputs:
            examples.append(to_chat(canonical, out, "easy"))
        else:
            skipped += 1

    print(f"Generated {len(examples)} examples (skipped {skipped} matching test)")

    # --- Split: 85% train, 15% valid (no test — test.jsonl stays untouched) ---
    random.shuffle(examples)
    n = len(examples)
    n_valid = max(100, int(n * 0.12))

    valid = examples[:n_valid]
    train = examples[n_valid:]

    for split_name, split in [("train_v3", train), ("valid_v3", valid)]:
        path = f"{args.out_dir}/{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")

    # Stats
    th = sum(1 for e in train if e["difficulty"] == "hard")
    te = sum(1 for e in train if e["difficulty"] == "easy")
    vh = sum(1 for e in valid if e["difficulty"] == "hard")
    ve = sum(1 for e in valid if e["difficulty"] == "easy")
    tf = sum(1 for e in train if json.loads(e["messages"][2]["content"]).get("is_former_name_input"))
    vf = sum(1 for e in valid if json.loads(e["messages"][2]["content"]).get("is_former_name_input"))

    print(f"\ntrain_v3: {len(train)} ({th} hard, {te} easy, {tf} real former)")
    print(f"valid_v3: {len(valid)} ({vh} hard, {ve} easy, {vf} real former)")
    print(f"\nTest.jsonl UNCHANGED — sha256: {test_hash}")
    print("Verify with: shasum -a 256 data/test.jsonl")


if __name__ == "__main__":
    main()
