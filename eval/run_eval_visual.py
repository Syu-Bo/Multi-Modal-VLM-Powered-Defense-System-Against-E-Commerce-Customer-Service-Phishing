"""Visual branch evaluation.

Reads a Unsloth-style parquet (columns: image{bytes,path}, prompt, answer, ...),
sends each image + text prompt to an OpenAI-compatible vision endpoint, parses
risk_score, thresholds at 0.7, and reports P/R/F1/FPR.

The per-sample Page URL is stored in the output so run_eval_fusion.py can join
visual and text results on the same page.

NOTE on labels: the parquet gold labels are *weak page-level* labels (an image is
marked phishing iff it was scraped from a phishing page, regardless of whether the
image itself shows a visual phishing cue). Treat the resulting metrics accordingly.

Env overrides:
  VISION_MODEL     model name              (default: qwen2-vl)
  VISION_BASE_URL  OpenAI-compatible /v1   (default: http://localhost:11434/v1)
  VISION_API_KEY   api key                 (default: ollama)
  EVAL_DATA        parquet path            (default: datasets/visual_valid_studio.parquet)

Examples:
  python eval/run_eval_visual.py --dry-run --limit 3      # no model call; checks decode/payload
  VISION_MODEL=qwen2-vl python eval/run_eval_visual.py
"""

import argparse
import base64
import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ALERT_THRESHOLD = 0.7

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.environ.get("EVAL_DATA", PROJECT_ROOT / "datasets" / "visual_valid_studio.parquet"))
OUTPUT_DIR = PROJECT_ROOT / "eval" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = Path(os.environ.get("EVAL_OUTPUT", OUTPUT_DIR / "visual_results.json"))

VISION_MODEL = os.environ.get("VISION_MODEL", "qwen2-vl")
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "http://localhost:11434/v1")
VISION_API_KEY = os.environ.get("VISION_API_KEY", "ollama")

VISUAL_SYSTEM_PROMPT = """\
You are a defensive cybersecurity analyst specializing in visual phishing detection
for e-commerce and brand-impersonation pages. Inspect the image together with the
page URL/domain context provided. Decide whether the image is evidence of brand
impersonation or a phishing UI (e.g. an official logo while the registrable domain
does not match the brand, a fake login/payment screen).

Respond with a single JSON object only (no markdown, no prose):
{
  "risk_score": float in [0, 1],
  "confidence": float in [0, 1],
  "reasons": [string, ...]
}
When uncertain, prefer lower scores.
"""

URL_RE = re.compile(r"Page URL:\s*(\S+)")


def page_url_from(text):
    m = URL_RE.search(text or "")
    return m.group(1).strip() if m else None


def to_data_url(image_bytes):
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")


def extract_json(text):
    try:
        return json.loads(text), True
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group()), True
        except Exception:
            pass
    return None, False


def risk_to_label(risk_score):
    try:
        return 1 if float(risk_score) >= ALERT_THRESHOLD else 0
    except Exception:
        return 0


def call_vision(prompt_text, data_url):
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "system", "content": VISUAL_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text + "\n\nReturn JSON only."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
        "temperature": 0,
        "seed": 42,
    }
    r = requests.post(
        VISION_BASE_URL.rstrip("/") + "/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {VISION_API_KEY}"},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    ap.add_argument("--dry-run", action="store_true", help="decode + build payload, do NOT call the model")
    args = ap.parse_args()

    rows = pq.read_table(DATA_PATH).to_pylist()
    if args.limit:
        rows = rows[: args.limit]

    y_true, y_pred, latencies = [], [], []
    json_valid = 0
    details = []

    for idx, row in enumerate(tqdm(rows, desc=f"Visual[{'dry-run' if args.dry_run else VISION_MODEL}]")):
        prompt_text = row.get("prompt", "") or ""
        gold = row.get("answer")
        gold = json.loads(gold) if isinstance(gold, str) else (gold or {})
        true_label = risk_to_label(gold.get("risk_score", 0.0))
        url = page_url_from(prompt_text)

        try:
            data_url = to_data_url(row["image"]["bytes"])
        except Exception as e:
            details.append({"id": idx, "page_url": url, "error": f"decode: {e}"})
            y_true.append(true_label); y_pred.append(0); latencies.append(0.0)
            continue

        if args.dry_run:
            # Validate decode + payload sizing only.
            details.append({"id": idx, "page_url": url, "data_url_len": len(data_url),
                            "true_label": true_label})
            y_true.append(true_label); y_pred.append(true_label); latencies.append(0.0)
            json_valid += 1
            continue

        start = time.time()
        try:
            raw = call_vision(prompt_text, data_url)
            latency_ms = (time.time() - start) * 1000
            pred_json, ok = extract_json(raw)
            if ok:
                json_valid += 1
                pred_risk = pred_json.get("risk_score", 0.0)
            else:
                pred_json, pred_risk = None, 0.0
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            raw, pred_json, pred_risk, ok = str(e), None, 0.0, False

        pred_label = risk_to_label(pred_risk)
        y_true.append(true_label); y_pred.append(pred_label); latencies.append(latency_ms)
        details.append({
            "id": idx, "page_url": url,
            "true_risk_score": gold.get("risk_score", 0.0), "pred_risk_score": pred_risk,
            "true_label": true_label, "pred_label": pred_label,
            "json_valid": ok, "latency_ms": latency_ms, "pred_json": pred_json,
        })

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "model": "dry-run" if args.dry_run else VISION_MODEL,
        "dataset": str(DATA_PATH),
        "threshold": ALERT_THRESHOLD,
        "total_samples": len(rows),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "fpr": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "json_schema_compliance": round(json_valid / len(rows), 4) if rows else 0.0,
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": details}, f, ensure_ascii=False, indent=2)

    print("\n========== Visual Branch Evaluation Metrics ==========")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n結果已儲存到：{OUTPUT_PATH}")
    if args.dry_run:
        print("（dry-run：未呼叫模型，僅驗證解碼與 payload 組裝）")


if __name__ == "__main__":
    main()
