"""URL-only branch evaluation.

Reads a chat-format JSONL where each row's messages are [system, user, assistant]
and the assistant content is the gold TextVerdict JSON. Sends system + user to an
Ollama model, parses risk_score, thresholds at 0.7, and reports P/R/F1/FPR.

Env overrides:
  EVAL_MODEL      Ollama model name      (default: llama3.2:latest)
  EVAL_DATA       path to eval jsonl     (default: datasets/training_data/url_only_1to1/url_eval.jsonl)
  OLLAMA_URL      generate endpoint      (default: http://localhost:11434/api/generate)
"""

import json
import os
import time
import re
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ALERT_THRESHOLD = 0.7

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(
    os.environ.get(
        "EVAL_DATA",
        PROJECT_ROOT / "datasets" / "training_data" / "url_only_1to1" / "url_eval.jsonl",
    )
)
OUTPUT_DIR = PROJECT_ROOT / "eval" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "url_only_results.json"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.environ.get("EVAL_MODEL", "llama3.2:latest")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_gold_json(content):
    return content if isinstance(content, dict) else json.loads(content)


def extract_json_from_text(text):
    try:
        return json.loads(text), True
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()), True
        except Exception:
            pass
    return None, False


def risk_to_label(risk_score):
    try:
        score = float(risk_score)
    except Exception:
        score = 0.0
    return 1 if score >= ALERT_THRESHOLD else 0


def call_model(system_prompt, user_prompt):
    prompt = f"{system_prompt}\n\n{user_prompt}\n\nReturn JSON only. No markdown, no explanation."
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "seed": 42},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["response"]


def main():
    samples = load_jsonl(DATA_PATH)
    y_true, y_pred, latencies = [], [], []
    json_valid_count = 0
    detail_results = []

    for idx, item in enumerate(tqdm(samples, desc=f"Eval[{MODEL_NAME}]")):
        messages = item["messages"]
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        gold_json = parse_gold_json(messages[2]["content"])
        true_label = risk_to_label(gold_json.get("risk_score", 0.0))

        start = time.time()
        try:
            raw_output = call_model(system_prompt, user_prompt)
            latency_ms = (time.time() - start) * 1000
            pred_json, is_valid = extract_json_from_text(raw_output)
            if is_valid:
                json_valid_count += 1
                pred_risk = pred_json.get("risk_score", 0.0)
            else:
                pred_json, pred_risk = None, 0.0
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            raw_output, pred_json, pred_risk, is_valid = str(e), None, 0.0, False

        pred_label = risk_to_label(pred_risk)
        y_true.append(true_label)
        y_pred.append(pred_label)
        latencies.append(latency_ms)
        detail_results.append({
            "id": idx,
            "true_risk_score": gold_json.get("risk_score", 0.0),
            "pred_risk_score": pred_risk,
            "true_label": true_label,
            "pred_label": pred_label,
            "json_valid": is_valid,
            "latency_ms": latency_ms,
            "user_prompt": user_prompt,
            "pred_json": pred_json,
        })

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "model": MODEL_NAME,
        "dataset": str(DATA_PATH),
        "threshold": ALERT_THRESHOLD,
        "total_samples": len(samples),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "fpr": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "json_schema_compliance": round(json_valid_count / len(samples), 4),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": detail_results}, f, ensure_ascii=False, indent=2)

    print("\n========== URL-only Evaluation Metrics ==========")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n結果已儲存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
