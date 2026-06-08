"""In-process evaluation with a HuggingFace causal LM (no Ollama / no server).

Loads meta-llama/Llama-3.1-8B-Instruct via transformers and evaluates a chat JSONL whose
messages are [system, user, assistant] (assistant content = gold TextVerdict JSON). Works
for BOTH the text branch and the URL branch - point EVAL_DATA at the right file. Stores
per-sample Page URL so run_eval_fusion.py can join.

Use the Instruct model (default) to compare with the teammate's Ollama `llama3.1:8b`
(also Instruct). A non-instruct base model has no chat template; the script falls back to
a plain prompt and prints a warning, but its results are sanity-check only.

Env:
  HF_MODEL    model id/path     (default: meta-llama/Llama-3.1-8B-Instruct)
  EVAL_DATA   chat jsonl        (default: datasets/text_valid_clean.jsonl)
  EVAL_OUTPUT output json       (default: eval/results/hf_<model>_results.json)
  LOAD_4BIT   "1" -> 4-bit via bitsandbytes (else bf16/fp16; fp16 ~16GB fits a 3090)
  PROMPT_V2   "1" -> append the Domain-Brand Mismatch rule to the system prompt (= v2)

Examples:
  EVAL_DATA=datasets/text_valid_clean.jsonl python eval/run_eval_hf.py --limit 5
  EVAL_DATA=datasets/training_data/url_only_1to1/url_eval.jsonl \
    EVAL_OUTPUT=eval/results/url_hf_results.json python eval/run_eval_hf.py
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ALERT_THRESHOLD = 0.7
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.environ.get("EVAL_DATA", PROJECT_ROOT / "datasets" / "text_valid_clean.jsonl"))
OUTPUT_DIR = PROJECT_ROOT / "eval" / "results"
HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LOAD_4BIT = os.environ.get("LOAD_4BIT") == "1"
PROMPT_V2 = os.environ.get("PROMPT_V2") == "1"

URL_RE = re.compile(r"Page URL:\s*(\S+)")

# v2 規則：與 Ollama run_eval_v2.py 一致的 Domain-Brand Mismatch 附加規則。
V2_RULE = """

Additional detection rule:
If the visible text mentions a known brand, but the domain or URL does not match the official domain, treat it as a strong phishing signal.

Domain-brand mismatch examples:
1. Text contains "Shopee" or "蝦皮", but the domain is not shopee.tw or shopee.com.
2. Text contains "momo", but the domain is not momoshop.com.tw.
3. Text contains "LINE Pay", but the domain is not line.me.
4. Text contains bank, delivery, payment, or shopping platform names, but the URL is unrelated or suspicious.

When domain-brand mismatch is found:
- Set risk_score to at least 0.80.
- Add "[DOMAIN_MISMATCH]" in reasons."""


def page_url_from(text):
    m = URL_RE.search(text or "")
    return m.group(1).strip() if m else None


def parse_gold(content):
    return content if isinstance(content, dict) else json.loads(content)


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


def risk_to_label(risk):
    try:
        return 1 if float(risk) >= ALERT_THRESHOLD else 0
    except Exception:
        return 0


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    kwargs = {"device_map": "auto"}
    if LOAD_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, **kwargs)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok, model


def build_inputs(tok, system_msg, user_msg, model_device):
    """Chat template if available; else (base model) a plain concatenated prompt."""
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [system_msg, user_msg], add_generation_prompt=True,
            return_tensors="pt").to(model_device)
    prompt = f"{system_msg['content']}\n\n{user_msg['content']}\n\nJSON:"
    return tok(prompt, return_tensors="pt").input_ids.to(model_device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    import torch
    from tqdm import tqdm

    if "instruct" not in HF_MODEL.lower():
        print(f"⚠️  '{HF_MODEL}' 看起來是 BASE 模型（非 Instruct），結果僅供 sanity check。")

    samples = load_jsonl(DATA_PATH)
    if args.limit:
        samples = samples[: args.limit]

    out_path = Path(os.environ.get(
        "EVAL_OUTPUT",
        OUTPUT_DIR / f"hf_{HF_MODEL.split('/')[-1].lower()}_results.json"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {HF_MODEL} (4bit={LOAD_4BIT}) ...")
    tok, model = build_model()

    y_true, y_pred, latencies = [], [], []
    json_valid = 0
    details = []

    for idx, item in enumerate(tqdm(samples, desc=f"HF[{HF_MODEL.split('/')[-1]}]")):
        msgs = item["messages"]
        gold = parse_gold(msgs[2]["content"])
        true_label = risk_to_label(gold.get("risk_score", 0.0))
        url = page_url_from(msgs[1]["content"])

        system_msg = msgs[0]
        if PROMPT_V2:
            system_msg = {"role": "system", "content": msgs[0]["content"] + V2_RULE}
        inputs = build_inputs(tok, system_msg, msgs[1], model.device)
        start = time.time()
        with torch.no_grad():
            gen = model.generate(inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        latency_ms = (time.time() - start) * 1000
        raw = tok.decode(gen[0][inputs.shape[1]:], skip_special_tokens=True)

        pred_json, ok = extract_json(raw)
        if ok:
            json_valid += 1
            pred_risk = pred_json.get("risk_score", 0.0)
        else:
            pred_json, pred_risk = None, 0.0
        pred_label = risk_to_label(pred_risk)

        y_true.append(true_label)
        y_pred.append(pred_label)
        latencies.append(latency_ms)
        details.append({
            "id": idx, "page_url": url,
            "true_risk_score": gold.get("risk_score", 0.0), "pred_risk_score": pred_risk,
            "true_label": true_label, "pred_label": pred_label,
            "json_valid": ok, "latency_ms": latency_ms, "pred_json": pred_json,
        })

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "model": HF_MODEL, "load_4bit": LOAD_4BIT, "prompt_v2": PROMPT_V2,
        "dataset": str(DATA_PATH), "threshold": ALERT_THRESHOLD,
        "total_samples": len(samples),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "fpr": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "json_schema_compliance": round(json_valid / len(samples), 4) if samples else 0.0,
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": details}, f, ensure_ascii=False, indent=2)

    print("\n========== HF Evaluation Metrics ==========")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n結果已儲存到：{out_path}")


if __name__ == "__main__":
    main()
