"""Clean + leakage-free re-split of the text branch dataset.

Problems addressed:
  1. Data leakage: 10 registrable domains appear in BOTH text_train and text_valid,
     violating the project's own "no domain across splits" rule.
  2. A few clear mislabels in the 0.4-0.7 grey zone (e.g. barclays-banking.net, a
     blatant Barclays-impersonation domain, labelled below the 0.7 phishing threshold).

What it does (writes NEW files; never overwrites the originals):
  - pools text_train.jsonl + text_valid.jsonl
  - applies a small, documented set of label corrections
  - groups samples by registrable domain and re-splits so NO domain crosses splits,
    stratified by label, with a fixed seed
  - writes text_train_clean.jsonl / text_valid_clean.jsonl + audit_text.md

Subjective calls are kept minimal: only unambiguous brand-impersonation domains are
auto-corrected; everything else in the grey zone is FLAGGED for human review, not changed.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL_RE = re.compile(r"Page URL:\s*(\S+)")
SEED = 20260607
ALERT = 0.7
TARGET_VALID_POS = 16   # 上限：valid 想要的釣魚正樣本數（受總量 51 限制）
TARGET_VALID_NEG = 24   # valid 想要的正常樣本數

# 明確錯標自動修正：registrable domain -> (new_risk, 說明)。僅限無爭議的品牌仿冒。
AUTO_FIX = {
    "barclays-banking.net": (0.88, "Brand-domain mismatch: impersonates Barclays bank on a non-official domain."),
}


def reg_domain(url):
    m = re.search(r"https?://([^/]+)", url or "")
    host = m.group(1).lower() if m else ""
    return ".".join(host.split(".")[-2:]) if host else ""


def page_url(text):
    m = URL_RE.search(text or "")
    return m.group(1).strip() if m else None


def load(path):
    rows = []
    for line in Path(path).open(encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main():
    pool = load(HERE / "text_train.jsonl") + load(HERE / "text_valid.jsonl")

    cleaned, flagged = [], []
    samples = []
    seen_urls = set()
    n_dup = 0
    for r in pool:
        url = page_url(r["messages"][1]["content"])
        # 去重：同一 URL 只保留第一筆
        if url in seen_urls:
            n_dup += 1
            continue
        seen_urls.add(url)
        dom = reg_domain(url)
        gold = json.loads(r["messages"][2]["content"])
        risk = float(gold.get("risk_score", 0.0))

        # 1) 自動修正明確錯標
        if dom in AUTO_FIX:
            new_risk, why = AUTO_FIX[dom]
            if abs(new_risk - risk) > 1e-6:
                gold["risk_score"] = new_risk
                gold.setdefault("reasons", []).insert(0, f"[CORRECTED] {why}")
                r = json.loads(json.dumps(r))  # copy
                r["messages"][2]["content"] = json.dumps(gold, ensure_ascii=False)
                cleaned.append((url, risk, new_risk, why))
                risk = new_risk

        # 2) 標記灰區供人工確認（不改）
        if 0.4 <= risk < ALERT:
            flagged.append((url, risk, gold.get("reasons", [])[:1]))

        samples.append({"url": url, "dom": dom, "risk": risk, "row": r,
                        "label": 1 if risk >= ALERT else 0})

    # 依網域分組
    by_dom = {}
    for s in samples:
        by_dom.setdefault(s["dom"], []).append(s)

    # 每個網域標記是否含釣魚
    dom_items = list(by_dom.items())
    # 確定性洗牌（不可用 random 模組的時鐘；用 seed 排序鍵）
    def shuffle_key(name):
        h = 0
        for ch in f"{SEED}:{name}":
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        return h
    dom_items.sort(key=lambda kv: shuffle_key(kv[0]))

    valid_doms, train_doms = set(), set()
    vpos = vneg = 0
    for dom, items in dom_items:
        pos = sum(s["label"] for s in items)
        neg = len(items) - pos
        # 優先把含釣魚的網域放進 valid，直到達標
        if pos > 0 and vpos < TARGET_VALID_POS:
            valid_doms.add(dom); vpos += pos; vneg += neg
        elif pos == 0 and vneg < TARGET_VALID_NEG:
            valid_doms.add(dom); vneg += neg
        else:
            train_doms.add(dom)

    train_rows = [s["row"] for s in samples if s["dom"] in train_doms]
    valid_rows = [s["row"] for s in samples if s["dom"] in valid_doms]

    def stats(rows):
        pos = 0
        for r in rows:
            if float(json.loads(r["messages"][2]["content"]).get("risk_score", 0)) >= ALERT:
                pos += 1
        return len(rows), pos

    def write(path, rows):
        with Path(path).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write(HERE / "text_train_clean.jsonl", train_rows)
    write(HERE / "text_valid_clean.jsonl", valid_rows)

    # 洩漏檢查
    leak = train_doms & valid_doms
    tn, tp = stats(train_rows)
    vn, vp = stats(valid_rows)

    lines = []
    lines.append("# 文字資料清洗 + 重切 稽核報告\n")
    lines.append(f"- 來源：text_train.jsonl + text_valid.jsonl（共 {len(pool)} 筆，去除重複 {n_dup} 筆）")
    lines.append(f"- 重切後 train：{tn} 筆（釣魚 {tp} / 正常 {tn-tp}）")
    lines.append(f"- 重切後 valid：{vn} 筆（釣魚 {vp} / 正常 {vn-vp}）")
    lines.append(f"- 跨 split 網域洩漏：{len(leak)}（目標 0）\n")
    lines.append("## 自動修正的錯標（已套用）")
    if cleaned:
        for url, old, new, why in cleaned:
            lines.append(f"- `{url}`：risk {old} → {new}　{why}")
    else:
        lines.append("- （無）")
    lines.append("\n## 灰區待人工確認（0.4 ≤ risk < 0.7，未改動）")
    for url, risk, reasons in sorted(flagged, key=lambda x: -x[1]):
        lines.append(f"- risk={risk:.2f} `{url}`　{reasons}")
    (HERE / "audit_text.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n寫出：text_train_clean.jsonl / text_valid_clean.jsonl / audit_text.md")


if __name__ == "__main__":
    main()
