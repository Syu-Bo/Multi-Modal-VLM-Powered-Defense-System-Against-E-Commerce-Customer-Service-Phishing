"""Playbook generator - turn a high-risk phishing verdict into an actionable response.

Given a detected phishing page (URL + optional brand/risk/reasons), produce a Markdown
playbook with three parts:
  1. 網域調查指令 (Whois / DNS)  - deterministic templates, no model needed
  2. 165 檢舉信草稿            - LLM few-shot (Traditional Chinese)
  3. 使用者防詐宣導訊息         - LLM few-shot (Traditional Chinese)

The LLM parts use an Ollama endpoint and degrade gracefully: with --no-llm (or if the
endpoint is unreachable) they fall back to static templates.

Usage:
  python playbook/playbook_generator.py --url https://shoppe88.com/login --brand Shopee
  python playbook/playbook_generator.py --input eval/results/fusion_results.json
  python playbook/playbook_generator.py --url ... --no-llm     # 純模板，不呼叫模型

Env:
  EVAL_MODEL  Ollama model (default: llama3.2:latest)
  OLLAMA_URL  generate endpoint (default: http://localhost:11434/api/generate)
"""

import argparse
import json
import os
import re
from urllib.parse import urlparse

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("EVAL_MODEL", "llama3.2:latest")


def reg_domain(url):
    host = (urlparse(url).hostname or "").lower()
    return ".".join(host.split(".")[-2:]) if host else host


def call_llm(prompt):
    import requests
    payload = {"model": MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.3, "seed": 42}}
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["response"].strip()


# ---------- 1. 網域調查（模板） ----------
def whois_section(url, domain):
    return f"""## 1. 網域調查指令 (Whois / DNS)

```bash
whois {domain}                      # 註冊人、註冊商、建立日期（新註冊網域風險高）
dig {domain} +short                 # 解析 IP
dig {domain} NS +short              # 名稱伺服器
nslookup {domain}
curl -sI "{url}"                    # 觀察 HTTP header / 重導
```

威脅情報查詢：
- VirusTotal: https://www.virustotal.com/gui/domain/{domain}
- urlscan.io: https://urlscan.io/search/#domain%3A{domain}
- abuse 聯絡：以 whois 結果中的 registrar abuse email 進行下架檢舉
"""


# ---------- 2. 165 檢舉信（LLM + 模板 fallback） ----------
def report_letter(url, domain, brand, use_llm):
    if use_llm:
        prompt = f"""你是協助民眾向台灣「165 反詐騙諮詢專線」檢舉釣魚網站的助理。
請依下列範例格式，為新案件撰寫一封簡潔、條列清楚的繁體中文檢舉信，只輸出信件內容。

範例（仿冒蝦皮）：
主旨：檢舉仿冒蝦皮（Shopee）釣魚網站 shoppe88.com
內容：
1. 檢舉網址：https://shoppe88.com/login
2. 仿冒對象：蝦皮購物 Shopee（官方網域 shopee.tw）
3. 詐騙手法：以假客服訊息誘導賣家掃描 QR Code，導向仿冒登入頁竊取帳密。
4. 請求：懇請協助通報網域下架並列入警示名單，避免民眾受害。

新案件：
- 檢舉網址：{url}
- 仿冒對象：{brand or '（未指明品牌）'}（網域 {domain}）
"""
        try:
            return "## 2. 165 檢舉信草稿\n\n" + call_llm(prompt) + "\n"
        except Exception as e:
            note = f"\n> （LLM 不可用，改用模板：{e}）\n"
            return "## 2. 165 檢舉信草稿" + note + "\n" + _letter_template(url, domain, brand)
    return "## 2. 165 檢舉信草稿\n\n" + _letter_template(url, domain, brand)


def _letter_template(url, domain, brand):
    b = brand or "（未指明品牌）"
    return f"""主旨：檢舉仿冒 {b} 釣魚網站 {domain}

內容：
1. 檢舉網址：{url}
2. 仿冒對象：{b}
3. 詐騙手法：仿冒官方頁面，誘導民眾輸入帳號密碼或付款資訊。
4. 請求：懇請協助通報網域下架並列入警示名單，避免民眾受害。
"""


# ---------- 3. 使用者防詐宣導（LLM + 模板 fallback） ----------
def user_advisory(url, domain, brand, use_llm):
    if use_llm:
        prompt = f"""你是防詐騙宣導員。請用繁體中文，為以下釣魚網站寫一段 3 至 4 句、語氣親切但明確的
警示訊息，提醒民眾不要輸入個資，並給出查證管道。只輸出訊息本身。

仿冒對象：{brand or '某知名品牌'}
可疑網址：{url}（網域 {domain}）
"""
        try:
            return "## 3. 使用者防詐宣導訊息\n\n" + call_llm(prompt) + "\n"
        except Exception as e:
            note = f"\n> （LLM 不可用，改用模板：{e}）\n"
            return "## 3. 使用者防詐宣導訊息" + note + "\n" + _advisory_template(domain, brand)
    return "## 3. 使用者防詐宣導訊息\n\n" + _advisory_template(domain, brand)


def _advisory_template(domain, brand):
    b = brand or "知名品牌"
    return f"""⚠️ 詐騙警示：網站 {domain} 疑似仿冒「{b}」官方頁面。
請勿在此輸入帳號、密碼或信用卡資訊。{b} 不會以此類連結要求驗證。
如有疑慮，請撥打 165 反詐騙專線，或直接從官方 App／官網查證。
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", help="phishing URL")
    ap.add_argument("--brand", default=None, help="impersonated brand")
    ap.add_argument("--risk", type=float, default=None, help="fused risk score")
    ap.add_argument("--input", help="JSON file with url/brand/risk (e.g. a threat report)")
    ap.add_argument("--no-llm", action="store_true", help="只用模板，不呼叫 LLM")
    ap.add_argument("--out", help="輸出 Markdown 路徑（預設印到 stdout）")
    args = ap.parse_args()

    url, brand, risk = args.url, args.brand, args.risk
    if args.input:
        data = json.load(open(args.input, encoding="utf-8"))
        url = url or data.get("url") or data.get("page_url")
        brand = brand or data.get("impersonated_brand") or data.get("brand")
        risk = risk if risk is not None else data.get("risk_score")
    if not url:
        ap.error("需要 --url 或含 url 的 --input")

    domain = reg_domain(url)
    use_llm = not args.no_llm
    header = f"""# 釣魚事件處置 Playbook

- 目標網址：`{url}`
- 註冊網域：`{domain}`
- 仿冒品牌：{brand or '未指明'}
- 風險分數：{risk if risk is not None else '未提供'}

> 自動產生之初步處置建議，請網管/資安人員審閱後執行。
"""
    parts = [header,
             whois_section(url, domain),
             report_letter(url, domain, brand, use_llm),
             user_advisory(url, domain, brand, use_llm)]
    md = "\n".join(parts)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"已輸出：{args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
