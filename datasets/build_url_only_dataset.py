#!/usr/bin/env python3
"""Build URL-only phishing/benign train-eval datasets for Unsloth.

The task is intentionally limited to the URL string. No page fetch, text
extraction, image extraction, or network access is performed.

Default output uses the largest exact 3:1 benign:phishing split possible from
the cleaned CSV rows:

  benign_count = all cleaned Label=good rows
  phishing_count = floor(benign_count / 3)

Rows are written as chat `messages` JSONL. The assistant target is the same
TextVerdict JSON schema used by the text branch, with URL-only reasons.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from urllib.parse import urlparse


SYSTEM_PROMPT = """\
You are a defensive cybersecurity analyst specializing in URL-only phishing detection.

You must judge whether the given URL is likely to be a phishing site or a normal benign website using only the URL string. Do not assume page content, images, or live network behavior.

Respond with a single JSON object matching this schema:
{
  "risk_score": float in [0, 1],
  "confidence": float in [0, 1],
  "reasons": [string, ...],
  "suspicious_phrases": [string, ...],
  "detected_language": string | null
}

Scoring:
- phishing or suspicious URL -> risk_score high
- normal benign URL -> risk_score low
- use confidence to reflect uncertainty from URL-only evidence
"""

CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
HOST_RE = re.compile(r"^[a-z0-9.-]+$", re.I)
IPV4_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

FREE_HOSTS = (
    "vercel.app",
    "netlify.app",
    "pages.dev",
    "github.io",
    "web.app",
    "firebaseapp.com",
    "duckdns.org",
    "wasmer.app",
    "glitch.me",
    "repl.co",
    "onrender.com",
    "workers.dev",
    "ipfs.dweb.link",
    "ipfs.io",
    "surge.sh",
    "weeblysite.com",
    "blogspot.com",
)

SUSPICIOUS_TLDS = {
    "top",
    "xyz",
    "store",
    "guru",
    "fun",
    "live",
    "click",
    "link",
    "online",
    "site",
    "shop",
    "icu",
    "cyou",
    "rest",
    "autos",
    "sbs",
    "ml",
    "ga",
    "cf",
    "gq",
    "tk",
    "bi",
    "buzz",
    "monster",
    "quest",
    "lol",
    "bond",
}

BRANDS = {
    "netflix": "netflix.com",
    "roblox": "roblox.com",
    "uniswap": "uniswap.org",
    "kucoin": "kucoin.com",
    "coinbase": "coinbase.com",
    "binance": "binance.com",
    "metamask": "metamask.io",
    "paypal": "paypal.com",
    "ebay": "ebay.com",
    "apple": "apple.com",
    "icloud": "icloud.com",
    "microsoft": "microsoft.com",
    "office": "microsoft.com",
    "amazon": "amazon.com",
    "instagram": "instagram.com",
    "whatsapp": "whatsapp.com",
    "facebook": "facebook.com",
    "tmobile": "t-mobile.com",
    "trezor": "trezor.io",
    "ledger": "ledger.com",
    "telegram": "telegram.org",
    "americanexpress": "americanexpress.com",
    "skype": "skype.com",
}

CRED_PATH_RE = re.compile(
    r"(login|signin|sign-in|auth|verify|secure|account|wallet|connect|pay|"
    r"confirm|update|unlock|recover|claim|gift|prize|airdrop|password|oauth)",
    re.I,
)


def registrable_domain(host: str | None) -> str:
    if not host:
        return ""
    parts = host.strip(".").lower().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host.lower()


def normalize_url(raw: str) -> str | None:
    value = raw.strip().strip('"').strip("'")
    if not value or CONTROL_RE.search(value) or any(ch.isspace() for ch in value):
        return None
    if "://" not in value:
        value = "https://" + value
    parsed = urlparse(value)
    host = (parsed.hostname or "").strip(".").lower()
    if not host or "." not in host or not HOST_RE.match(host):
        return None
    if len(host) > 253 or any(len(part) > 63 for part in host.split(".")):
        return None
    return value


def load_clean_urls(csv_path: Path) -> tuple[list[str], list[str], dict[str, int]]:
    good: list[str] = []
    bad: list[str] = []
    seen_good: set[str] = set()
    seen_bad: set[str] = set()
    raw_counts = {"good": 0, "bad": 0, "skipped": 0}

    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("Label") or "").strip().lower()
            url = normalize_url(row.get("URL") or "")
            if label not in {"good", "bad"} or url is None:
                raw_counts["skipped"] += 1
                continue
            raw_counts[label] += 1
            if label == "good" and url not in seen_good:
                seen_good.add(url)
                good.append(url)
            elif label == "bad" and url not in seen_bad:
                seen_bad.add(url)
                bad.append(url)
    return good, bad, raw_counts


def url_features(url: str) -> tuple[list[str], list[str]]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    domain = registrable_domain(host)
    path_q = f"{parsed.path}?{parsed.query}".lower()
    reasons: list[str] = []
    phrases: list[str] = []

    free_host = next((h for h in FREE_HOSTS if host == h or host.endswith("." + h)), None)
    if free_host:
        reasons.append(f"Hosted on free or instant deployment platform ({free_host}).")

    tld = host.rsplit(".", 1)[-1] if "." in host else host
    if tld in SUSPICIOUS_TLDS:
        reasons.append(f"Uses a low-reputation or commonly abused TLD (.{tld}).")

    if IPV4_RE.match(host):
        reasons.append("Uses a bare IPv4 host instead of a normal domain.")

    if host.startswith("xn--") or ".xn--" in host:
        reasons.append("Contains punycode, which may indicate homograph impersonation.")

    for token, apex in BRANDS.items():
        if token in host and domain != apex:
            reasons.append(
                f"Contains brand token '{token}' but registrable domain is '{domain}', not '{apex}'."
            )
            phrases.append(token)
            break

    digits = sum(c.isdigit() for c in host)
    if digits >= 5:
        reasons.append("Host contains many digits, consistent with disposable domains.")
    if host.count("-") >= 3:
        reasons.append("Host contains many hyphens, a common cybersquatting pattern.")
    if CRED_PATH_RE.search(path_q):
        reasons.append("Path suggests credential, wallet, payment, verification, or claim flow.")
    if len(host) >= 40:
        reasons.append("Host is unusually long.")
    return reasons, phrases[:20]


def verdict(url: str, label: str) -> dict[str, object]:
    reasons, phrases = url_features(url)
    if label == "phishing":
        score = 0.78 + min(0.04 * len(reasons), 0.17)
        confidence = 0.72 + min(0.03 * len(reasons), 0.18)
        if not reasons:
            reasons = ["URL is labeled phishing in the source dataset; URL-only features are otherwise limited."]
            confidence = 0.62
    else:
        score = 0.05
        confidence = 0.78
        if reasons:
            score = 0.18
            confidence = 0.58
            reasons = [
                "URL is labeled good in the source dataset despite containing some weak URL-only risk features.",
                *reasons[:4],
            ]
        else:
            reasons = ["URL is labeled good in the source dataset and has no strong URL-only phishing feature."]

    return {
        "risk_score": round(min(max(score, 0.0), 1.0), 3),
        "confidence": round(min(max(confidence, 0.0), 1.0), 3),
        "reasons": reasons[:10],
        "suspicious_phrases": phrases,
        "detected_language": None,
    }


def make_row(url: str, label: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Classify this URL using only the URL string. Return JSON only.\n"
                    f"URL: {url}"
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(verdict(url, label), ensure_ascii=False, separators=(",", ":")),
            },
        ]
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def write_urls(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("url,label\n")
        for url, label in rows:
            f.write(json.dumps(url, ensure_ascii=False) + f",{label}\n")


def split_by_label(
    rows: list[tuple[str, str]],
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    rng = random.Random(seed)
    grouped = {"benign": [], "phishing": []}
    for row in rows:
        grouped[row[1]].append(row)

    train: list[tuple[str, str]] = []
    eval_rows: list[tuple[str, str]] = []
    for label, items in grouped.items():
        rng.shuffle(items)
        eval_count = max(1, round(len(items) * eval_ratio))
        eval_rows.extend(items[:eval_count])
        train.extend(items[eval_count:])
    rng.shuffle(train)
    rng.shuffle(eval_rows)
    return train, eval_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("phishing_site_urls.csv"))
    parser.add_argument("--out", type=Path, default=Path("training_data/url_only_3to1"))
    parser.add_argument("--ratio", type=float, default=3.0, help="benign:phishing ratio")
    parser.add_argument(
        "--total-rows",
        type=int,
        default=None,
        help="Optional total selected rows before train/eval split.",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    if args.ratio <= 0:
        raise SystemExit("--ratio must be positive")
    args.out.mkdir(parents=True, exist_ok=True)

    good, bad, raw_counts = load_clean_urls(args.csv)
    if args.total_rows is not None and args.total_rows < 2:
        raise SystemExit("--total-rows must be at least 2")

    if args.total_rows is None:
        selected_good_count = len(good)
        selected_bad_count = int(len(good) / args.ratio)
    else:
        selected_bad_count = int(args.total_rows / (args.ratio + 1))
        selected_good_count = args.total_rows - selected_bad_count

    selected_good_count = min(selected_good_count, len(good))
    selected_bad_count = min(selected_bad_count, len(bad))
    if selected_bad_count < 1 or selected_good_count < 1:
        raise SystemExit("not enough good URLs for requested ratio")

    rng = random.Random(args.seed)
    rng.shuffle(good)
    rng.shuffle(bad)
    selected_good = good[:selected_good_count]
    selected_bad = bad[:selected_bad_count]

    rows = [(url, "benign") for url in selected_good] + [
        (url, "phishing") for url in selected_bad
    ]
    train_pairs, eval_pairs = split_by_label(
        rows,
        eval_ratio=args.eval_ratio,
        seed=args.seed + 1,
    )

    train_rows = [make_row(url, label) for url, label in train_pairs]
    eval_rows = [make_row(url, label) for url, label in eval_pairs]
    write_jsonl(args.out / "url_train.jsonl", train_rows)
    write_jsonl(args.out / "url_eval.jsonl", eval_rows)
    write_urls(args.out / "url_train_sources.csv", train_pairs)
    write_urls(args.out / "url_eval_sources.csv", eval_pairs)

    summary = {
        "source_csv": str(args.csv),
        "task": "URL-only phishing classification",
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "requested_benign_to_phishing_ratio": args.ratio,
        "requested_total_rows": args.total_rows,
        "raw_cleanable_counts": raw_counts,
        "deduped_clean_counts": {
            "good": len(good),
            "bad": len(bad),
        },
        "selected_counts": {
            "benign": len(selected_good),
            "phishing": len(selected_bad),
            "benign_to_phishing_ratio": round(len(selected_good) / len(selected_bad), 6),
        },
        "train_counts": {
            "rows": len(train_pairs),
            "benign": sum(label == "benign" for _url, label in train_pairs),
            "phishing": sum(label == "phishing" for _url, label in train_pairs),
        },
        "eval_counts": {
            "rows": len(eval_pairs),
            "benign": sum(label == "benign" for _url, label in eval_pairs),
            "phishing": sum(label == "phishing" for _url, label in eval_pairs),
        },
        "files": {
            "train": "url_train.jsonl",
            "eval": "url_eval.jsonl",
            "train_sources": "url_train_sources.csv",
            "eval_sources": "url_eval_sources.csv",
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out / "README.md").write_text(
        "\n".join(
            [
                "# URL-only 3:1 Dataset",
                "",
                "This dataset trains an LLM to classify phishing vs benign using only the URL string.",
                "No page fetch, page text, image, or network-derived feature is used.",
                "",
                "## Files",
                "",
                "- `url_train.jsonl`: training split.",
                "- `url_eval.jsonl`: evaluation split.",
                "- `url_train_sources.csv`: URL + label provenance for train.",
                "- `url_eval_sources.csv`: URL + label provenance for eval.",
                "- `summary.json`: counts and split details.",
                "",
                "## Labels",
                "",
                "- `Label=good` from the CSV is converted to benign.",
                "- `Label=bad` from the CSV is converted to phishing.",
                "",
                "The selected data uses an exact 3:1 benign:phishing ratio, bounded by the number of clean benign URLs available.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
