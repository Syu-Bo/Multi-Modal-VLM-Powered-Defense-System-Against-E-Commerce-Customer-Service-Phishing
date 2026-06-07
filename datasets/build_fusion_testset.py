"""Build an aligned fusion test set: pages that have BOTH page text and screenshots.

Fusion only works on pages present in both branches. The per-branch valid splits
overlap on only ~16 pages, which is too small to draw conclusions. This script
pools text (train+valid jsonl) and visual (train+valid parquet), finds the shared
Page URLs, and writes aligned subsets keyed by Page URL:

  datasets/fusion_test/text_fusion.jsonl       one text row per shared page
  datasets/fusion_test/visual_fusion.parquet   all screenshots for shared pages
  datasets/fusion_test/pages.csv               url, label per shared page

Then on the 8b/GPU box:
  EVAL_DATA=datasets/fusion_test/text_fusion.jsonl   python eval/run_eval_v2.py   (text)
  EVAL_DATA=datasets/fusion_test/visual_fusion.parquet python eval/run_eval_visual.py
  python eval/run_eval_fusion.py \
    --text-results eval/results/text_fusion_results.json \
    --text-data datasets/fusion_test/text_fusion.jsonl \
    --visual-results eval/results/visual_fusion_results.json
"""

import csv
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "fusion_test"
URL_RE = re.compile(r"Page URL:\s*(\S+)")
ALERT = 0.7


def page_url(text):
    m = URL_RE.search(text or "")
    return m.group(1).strip() if m else None


def gold_label(answer):
    a = json.loads(answer) if isinstance(answer, str) else (answer or {})
    try:
        return 1 if float(a.get("risk_score", 0.0)) >= ALERT else 0
    except Exception:
        return 0


def load_text():
    """url -> (raw_jsonl_row, label). First occurrence wins."""
    out = {}
    # 優先用 clean 版（已去重 / 修錯標 / 無網域洩漏）；不存在才回退原始檔
    pairs = [("text_train_clean.jsonl", "text_valid_clean.jsonl"),
             ("text_train.jsonl", "text_valid.jsonl")]
    names = next((p for p in pairs if (HERE / p[0]).exists()), pairs[-1])
    for name in names:
        p = HERE / name
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            url = page_url(row["messages"][1]["content"])
            if url and url not in out:
                out[url] = (row, gold_label(row["messages"][2]["content"]))
    return out


def load_visual():
    """url -> list[parquet_row]. schema captured from first table."""
    groups = {}
    schema = None
    for name in ("visual_train_studio.parquet", "visual_valid_studio.parquet"):
        p = HERE / name
        if not p.exists():
            continue
        table = pq.read_table(p)
        schema = schema or table.schema
        for row in table.to_pylist():
            url = page_url(row.get("prompt", "") or "")
            if url:
                groups.setdefault(url, []).append(row)
    return groups, schema


def main():
    text = load_text()
    visual, schema = load_visual()
    shared = sorted(set(text) & set(visual))
    if not shared:
        raise SystemExit("No shared pages between text and visual data.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # text_fusion.jsonl — one text row per shared page
    with (OUT_DIR / "text_fusion.jsonl").open("w", encoding="utf-8") as f:
        for url in shared:
            f.write(json.dumps(text[url][0], ensure_ascii=False) + "\n")

    # visual_fusion.parquet — all screenshots for shared pages
    visual_rows = [r for url in shared for r in visual[url]]
    pq.write_table(pa.Table.from_pylist(visual_rows, schema=schema),
                   OUT_DIR / "visual_fusion.parquet")

    # pages.csv — label per page (text gold is canonical page-level truth)
    n_phish = 0
    with (OUT_DIR / "pages.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "label", "n_images"])
        for url in shared:
            lbl = text[url][1]
            n_phish += lbl
            w.writerow([url, "phishing" if lbl else "benign", len(visual[url])])

    print(f"shared pages         : {len(shared)}")
    print(f"  phishing / benign  : {n_phish} / {len(shared) - n_phish}")
    print(f"text rows written    : {len(shared)}  -> fusion_test/text_fusion.jsonl")
    print(f"visual rows written  : {len(visual_rows)}  -> fusion_test/visual_fusion.parquet")
    print(f"pages.csv            : fusion_test/pages.csv")


if __name__ == "__main__":
    main()
