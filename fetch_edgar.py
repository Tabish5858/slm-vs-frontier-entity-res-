"""
Pull real US entity data from SEC EDGAR — free, no API key, no captcha.

Data source:
  https://www.sec.gov/files/company_tickers.json        (list of ~10k CIKs + names)
  https://data.sec.gov/submissions/CIK##########.json    (per-company name history)

SEC asks for a descriptive User-Agent with contact email on every request,
and to keep request rate reasonable (we stay well under their 10 req/sec limit).

Usage:
  python fetch_edgar.py --email you@example.com --n 1200 --out ../data/edgar_raw.jsonl
"""

import argparse
import json
import time
import sys
import urllib.request
import urllib.error

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

REQUEST_DELAY_SEC = 0.15  # ~6-7 req/sec, safely under SEC's 10/sec limit


def fetch_json(url: str, user_agent: str, retries: int = 3):
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code == 403:
                wait = 2 ** attempt
                print(f"  rate limited ({e.code}), waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            print(f"  error fetching {url}: {e}, retrying...", file=sys.stderr)
            time.sleep(1 + attempt)
    raise RuntimeError(f"Failed to fetch {url} after {retries} retries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="Your email, required in User-Agent by SEC")
    ap.add_argument("--n", type=int, default=1200, help="Number of companies to pull")
    ap.add_argument("--out", default="../data/edgar_raw.jsonl")
    args = ap.parse_args()

    user_agent = f"Covent-LLM-Challenge-Research {args.email}"

    print("Fetching company ticker list...")
    tickers_data = fetch_json(TICKERS_URL, user_agent)
    companies = list(tickers_data.values())  # each: {cik_str, ticker, title}
    print(f"  got {len(companies)} companies total")

    n = min(args.n, len(companies))
    subset = companies[:n]

    out_path = args.out
    written = 0
    with open(out_path, "w") as f:
        for i, c in enumerate(subset):
            cik = int(c["cik_str"])
            url = SUBMISSIONS_URL.format(cik=cik)
            try:
                sub = fetch_json(url, user_agent)
            except Exception as e:
                print(f"  skip CIK {cik}: {e}", file=sys.stderr)
                continue

            record = {
                "cik": cik,
                "name": sub.get("name"),
                "entityType": sub.get("entityType"),
                "sic": sub.get("sic"),
                "sicDescription": sub.get("sicDescription"),
                "tickers": sub.get("tickers", []),
                "exchanges": sub.get("exchanges", []),
                "formerNames": sub.get("formerNames", []),  # [{"name":..., "from":..., "to":...}]
            }
            f.write(json.dumps(record) + "\n")
            written += 1

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n} processed, {written} written")

            time.sleep(REQUEST_DELAY_SEC)

    print(f"Done. Wrote {written} records to {out_path}")


if __name__ == "__main__":
    main()
