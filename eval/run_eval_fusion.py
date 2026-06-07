"""Fusion evaluation — does multimodal beat single-modality?

Joins an already-computed TEXT branch result and VISUAL branch result on the same
Page URL, then reports text-only / visual-only / fused metrics on the matched
subset, plus a grid search over the visual weight Wv (Wt = 1 - Wv).

This script makes NO model calls. Run run_eval (text) and run_eval_visual first.

Ground-truth label per page is taken from the text branch (fallback: visual).
Score thresholded at 0.7 -> phishing.

Inputs (all configurable):
  --text-results    default eval/results/prompt_v2_results.json
  --text-data       default datasets/text_valid.jsonl   (to recover Page URL by id)
  --visual-results  default eval/results/visual_results.json
"""

import argparse
import json
import re
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ALERT_THRESHOLD = 0.7
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
URL_RE = re.compile(r"Page URL:\s*(\S+)")


def page_url_from(text):
    m = URL_RE.search(text or "")
    return m.group(1).strip() if m else None


def label(score):
    try:
        return 1 if float(score) >= ALERT_THRESHOLD else 0
    except Exception:
        return 0


def metrics_for(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "fpr": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
    }


def load_text(results_path, data_path):
    """Return {page_url: {"score": float, "true": int}} for the text branch."""
    res = json.load(open(results_path, encoding="utf-8"))["results"]
    data = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    out = {}
    for r in res:
        idx = r.get("id")
        url = r.get("page_url")
        if url is None and idx is not None and idx < len(data):
            url = page_url_from(data[idx]["messages"][1]["content"])
        if url is None:
            continue
        out[url] = {"score": float(r.get("pred_risk_score", 0.0)),
                    "true": int(r.get("true_label", label(r.get("true_risk_score", 0.0))))}
    return out


def load_visual(results_path):
    res = json.load(open(results_path, encoding="utf-8"))["results"]
    out = {}
    for r in res:
        url = r.get("page_url")
        if url is None or "pred_risk_score" not in r:
            continue
        out[url] = {"score": float(r["pred_risk_score"]),
                    "true": int(r.get("true_label", 0))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text-results", type=Path, default=RESULTS_DIR / "prompt_v2_results.json")
    ap.add_argument("--text-data", type=Path, default=PROJECT_ROOT / "datasets" / "text_valid.jsonl")
    ap.add_argument("--visual-results", type=Path, default=RESULTS_DIR / "visual_results.json")
    ap.add_argument("--weights", type=str, default="0.3,0.4,0.5,0.55,0.6,0.7",
                    help="comma-separated Wv values to grid-search")
    args = ap.parse_args()

    text = load_text(args.text_results, args.text_data)
    visual = load_visual(args.visual_results)
    shared = sorted(set(text) & set(visual))

    if not shared:
        raise SystemExit(
            f"No shared Page URLs between text ({len(text)}) and visual ({len(visual)}).\n"
            "Make sure both eval result files cover overlapping pages."
        )

    y_true, t_score, v_score, disagree = [], [], [], 0
    for url in shared:
        t, v = text[url], visual[url]
        if t["true"] != v["true"]:
            disagree += 1
        y_true.append(t["true"])  # text branch is canonical page-level truth
        t_score.append(t["score"])
        v_score.append(v["score"])

    text_only = metrics_for(y_true, [label(s) for s in t_score])
    visual_only = metrics_for(y_true, [label(s) for s in v_score])

    grid = []
    for wv in [float(x) for x in args.weights.split(",")]:
        wt = 1.0 - wv
        fused_pred = [label(wv * v + wt * t) for v, t in zip(v_score, t_score)]
        m = metrics_for(y_true, fused_pred)
        m["Wv"], m["Wt"] = wv, round(wt, 3)
        grid.append(m)
    best = max(grid, key=lambda m: m["f1"])

    out = {
        "matched_pages": len(shared),
        "text_pages": len(text),
        "visual_pages": len(visual),
        "label_disagreements": disagree,
        "text_only": text_only,
        "visual_only": visual_only,
        "fusion_grid": grid,
        "best_fusion": best,
        "sources": {
            "text_results": str(args.text_results),
            "visual_results": str(args.visual_results),
        },
    }
    out_path = RESULTS_DIR / "fusion_results.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def row(name, m):
        return f"{name:<14} {m.get('n','-'):>4}  {m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}  {m['fpr']:.3f}"

    print("\n========== Fusion Comparison (matched pages = %d) ==========" % len(shared))
    print(f"{'branch':<14} {'n':>4}  {'prec':>5}  {'rec':>5}  {'f1':>5}  {'fpr':>5}")
    print(row("text-only", text_only))
    print(row("visual-only", visual_only))
    for m in grid:
        print(row(f"fuse Wv={m['Wv']}", m))
    print(f"\nBest fusion: Wv={best['Wv']} Wt={best['Wt']}  F1={best['f1']}")
    print(f"label disagreements text vs visual: {disagree}/{len(shared)}")
    print(f"\n結果已儲存到：{out_path}")


if __name__ == "__main__":
    main()
