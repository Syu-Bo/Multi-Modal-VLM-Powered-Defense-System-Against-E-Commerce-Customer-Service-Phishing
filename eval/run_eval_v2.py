import os
import json
import time
import re
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# =========================
# 基本設定
# =========================
ALERT_THRESHOLD = 0.7

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.environ.get("EVAL_DATA", PROJECT_ROOT / "datasets" / "text_valid.jsonl"))
OUTPUT_DIR = PROJECT_ROOT / "eval" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = Path(os.environ.get("EVAL_OUTPUT", OUTPUT_DIR / "prompt_v2_results.json"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.environ.get("EVAL_MODEL", "llama3.1:8b")


# =========================
# 讀取 JSONL
# =========================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# =========================
# 從 assistant 標準答案取 JSON
# =========================
def parse_gold_json(content):
    if isinstance(content, dict):
        return content
    return json.loads(content)


# =========================
# 從模型輸出中擷取 JSON
# =========================
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


# =========================
# risk_score 轉類別
# 1 = phishing
# 0 = legitimate
# =========================
def risk_to_label(risk_score):
    try:
        score = float(risk_score)
    except Exception:
        score = 0.0

    return 1 if score >= ALERT_THRESHOLD else 0


# =========================
# 呼叫 Ollama llama3
# =========================
def call_llama3(system_prompt, user_prompt):
    prompt = f"""
{system_prompt}

Additional detection rule:
If the visible text mentions a known brand, but the domain or URL does not match the official domain, treat it as a strong phishing signal.

Domain-brand mismatch examples:
1. Text contains "Shopee" or "蝦皮", but the domain is not shopee.tw or shopee.com.
2. Text contains "momo", but the domain is not momoshop.com.tw.
3. Text contains "LINE Pay", but the domain is not line.me.
4. Text contains bank, delivery, payment, or shopping platform names, but the URL is unrelated or suspicious.

When domain-brand mismatch is found:
- Set risk_score to at least 0.80.
- Set impersonated_brand to the mentioned brand.
- Add "[DOMAIN_MISMATCH]" in reasons.

User input:
{user_prompt}

請只輸出 JSON，不要輸出 markdown，不要輸出說明文字。
JSON schema:
{{
  "risk_score": 0.0,
  "confidence": 0.0,
  "reasons": [],
  "impersonated_brand": null
}}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    return response.json()["response"]


# =========================
# 主評估流程
# =========================
def main():
    samples = load_jsonl(DATA_PATH)

    y_true = []
    y_pred = []
    latencies = []
    json_valid_count = 0

    detail_results = []

    for idx, item in enumerate(tqdm(samples, desc="Evaluating")):
        messages = item["messages"]

        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        gold_content = messages[2]["content"]

        gold_json = parse_gold_json(gold_content)
        true_risk_score = gold_json.get("risk_score", 0.0)
        true_label = risk_to_label(true_risk_score)

        start_time = time.time()

        try:
            raw_output = call_llama3(system_prompt, user_prompt)
            latency_ms = (time.time() - start_time) * 1000

            pred_json, is_valid_json = extract_json_from_text(raw_output)

            if is_valid_json:
                json_valid_count += 1
                pred_risk_score = pred_json.get("risk_score", 0.0)
            else:
                pred_json = None
                pred_risk_score = 0.0

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            raw_output = str(e)
            pred_json = None
            pred_risk_score = 0.0
            is_valid_json = False

        pred_label = risk_to_label(pred_risk_score)

        y_true.append(true_label)
        y_pred.append(pred_label)
        latencies.append(latency_ms)

        detail_results.append({
            "id": idx,
            "true_risk_score": true_risk_score,
            "pred_risk_score": pred_risk_score,
            "true_label": true_label,
            "pred_label": pred_label,
            "json_valid": is_valid_json,
            "latency_ms": latency_ms,
            "gold_json": gold_json,
            "pred_json": pred_json,
            "raw_output": raw_output
        })

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    json_schema_compliance = json_valid_count / len(samples)
    p95_latency = float(np.percentile(latencies, 95))

    metrics = {
        "model": MODEL_NAME,
        "dataset": str(DATA_PATH),
        "threshold": ALERT_THRESHOLD,
        "total_samples": len(samples),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "json_schema_compliance": round(float(json_schema_compliance), 4),
        "p95_latency_ms": round(float(p95_latency), 2)
    }

    output = {
        "metrics": metrics,
        "results": detail_results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n========== Evaluation Metrics ==========")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n結果已儲存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()