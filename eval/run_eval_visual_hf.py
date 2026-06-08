"""In-process visual-branch evaluation with a HF vision-language model (no server).

Loads Qwen/Qwen2-VL-7B-Instruct via transformers and evaluates the visual parquet
(columns: image{bytes,path}, prompt, answer, ...) image-by-image: image + text prompt
-> JSON risk_score, thresholded at 0.7. Stores per-sample Page URL for fusion join.

Same output format as run_eval_visual.py so run_eval_fusion.py can consume it.

NOTE on labels: the parquet labels are WEAK page-level labels (an image is "phishing"
iff scraped from a phishing page, regardless of visual cue) - metrics are limited by
this, by design. The main value is enabling the fusion comparison.

Env:
  VISION_HF_MODEL  model id/path (default: Qwen/Qwen2-VL-7B-Instruct; open, ~16GB fp16)
  EVAL_DATA        parquet path  (default: datasets/visual_valid_studio.parquet)
  EVAL_OUTPUT      output json   (default: eval/results/visual_hf_results.json)
  LOAD_4BIT        "1" -> 4-bit via bitsandbytes (大幅省 VRAM，OOM 時建議開)
  VISION_MAX_PIXELS  每張圖最大像素 (default 512*28*28; OOM 再調小)

Examples:
  EVAL_DATA=datasets/visual_valid_studio.parquet python eval/run_eval_visual_hf.py --limit 3
  EVAL_DATA=datasets/fusion_test/visual_fusion.parquet \
    EVAL_OUTPUT=eval/results/visual_fusion_results.json python eval/run_eval_visual_hf.py
"""

import argparse
import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ALERT_THRESHOLD = 0.7
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.environ.get("EVAL_DATA", PROJECT_ROOT / "datasets" / "visual_valid_studio.parquet"))
OUTPUT_DIR = PROJECT_ROOT / "eval" / "results"
VISION_HF_MODEL = os.environ.get("VISION_HF_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
LOAD_4BIT = os.environ.get("LOAD_4BIT") == "1"
# 限制每張圖的最大像素 -> 限制 vision token 數 -> 控制激活記憶體 (避免 OOM)。
# 預設 512*28*28；OOM 再調小，清晰大圖可調大。
VISION_MAX_PIXELS = int(os.environ.get("VISION_MAX_PIXELS", 512 * 28 * 28))
VISION_MIN_PIXELS = int(os.environ.get("VISION_MIN_PIXELS", 64 * 28 * 28))

URL_RE = re.compile(r"Page URL:\s*(\S+)")

VISUAL_SYSTEM_PROMPT = """\
You are a defensive cybersecurity analyst specializing in visual phishing detection
for e-commerce and brand-impersonation pages. Inspect the image together with the
page URL/domain context. Decide whether the image is evidence of brand impersonation
or a phishing UI (e.g. an official logo while the registrable domain does not match
the brand, a fake login/payment screen).

Respond with a single JSON object only (no markdown, no prose):
{"risk_score": float in [0,1], "confidence": float in [0,1], "reasons": [string, ...]}
When uncertain, prefer lower scores."""


def page_url_from(text):
    m = URL_RE.search(text or "")
    return m.group(1).strip() if m else None


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


def build_model():
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(
        VISION_HF_MODEL, min_pixels=VISION_MIN_PIXELS, max_pixels=VISION_MAX_PIXELS)
    kwargs = {"device_map": "auto"}
    if LOAD_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2VLForConditionalGeneration.from_pretrained(VISION_HF_MODEL, **kwargs)
    model.eval()
    return proc, model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from tqdm import tqdm

    rows = pq.read_table(DATA_PATH).to_pylist()
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(os.environ.get(
        "EVAL_OUTPUT", OUTPUT_DIR / "visual_hf_results.json"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {VISION_HF_MODEL} (4bit={LOAD_4BIT}) ...")
    proc, model = build_model()

    y_true, y_pred, latencies = [], [], []
    json_valid = 0
    details = []

    for idx, row in enumerate(tqdm(rows, desc=f"VisualHF[{VISION_HF_MODEL.split('/')[-1]}]")):
        prompt_text = row.get("prompt", "") or ""
        gold = row.get("answer")
        gold = json.loads(gold) if isinstance(gold, str) else (gold or {})
        true_label = risk_to_label(gold.get("risk_score", 0.0))
        url = page_url_from(prompt_text)

        try:
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            # Qwen2-VL 要求長寬 >= 28 (patch factor)；追蹤像素等過小圖先放大。
            w, h = img.size
            if w < 28 or h < 28:
                img = img.resize((max(w, 28), max(h, 28)))
        except Exception as e:
            details.append({"id": idx, "page_url": url, "error": f"decode: {e}"})
            y_true.append(true_label); y_pred.append(0); latencies.append(0.0)
            continue

        messages = [
            {"role": "system", "content": VISUAL_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text + "\n\nReturn JSON only."},
            ]},
        ]
        chat = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=[chat], images=[img], return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        latency_ms = (time.time() - start) * 1000
        trimmed = gen[0][inputs["input_ids"].shape[1]:]
        raw = proc.decode(trimmed, skip_special_tokens=True)

        pred_json, ok = extract_json(raw)
        if ok:
            json_valid += 1
            pred_risk = pred_json.get("risk_score", 0.0)
        else:
            pred_json, pred_risk = None, 0.0
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
        "model": VISION_HF_MODEL, "load_4bit": LOAD_4BIT,
        "dataset": str(DATA_PATH), "threshold": ALERT_THRESHOLD,
        "total_samples": len(rows),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "fpr": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "json_schema_compliance": round(json_valid / len(rows), 4) if rows else 0.0,
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": details}, f, ensure_ascii=False, indent=2)

    print("\n========== Visual Branch (HF) Metrics ==========")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n結果已儲存到：{out_path}")


if __name__ == "__main__":
    main()
