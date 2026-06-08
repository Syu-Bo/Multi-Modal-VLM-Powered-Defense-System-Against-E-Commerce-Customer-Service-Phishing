"""QR-code phishing analysis - close the proposal's #1 blind spot.

A pure-text LLM cannot read a URL hidden inside an image's QR code (proposal case 1:
the Shopee fake-customer-service + QR-code scam). This module decodes QR codes from a
page's images and flags the high-confidence signal where the decoded URL points to a
DIFFERENT registrable domain than the page itself.

Decoder: prefers pyzbar (more robust, needs the zbar system lib); falls back to
OpenCV's built-in QRCodeDetector (no extra system dependency).

Output (QRVerdict), designed to feed the fusion module as a strong / veto signal:
  {
    "qr_present": bool,
    "qr_payloads": [str, ...],
    "decoded_domains": [str, ...],
    "domain_mismatch": bool,
    "risk_score": float,        # [0,1]
    "confidence": float,
    "reasons": [str, ...]
  }

CLI:
  python qr/qr_decode.py --image path.png --page-url https://shopee.tw/...
  python qr/qr_decode.py --image path.png --page-domain shopee.tw
"""

import argparse
import json
import re
from urllib.parse import urlparse


def reg_domain(host_or_url):
    s = host_or_url or ""
    if "://" in s or "/" in s or s.startswith("www."):
        host = (urlparse(s if "://" in s else "http://" + s).hostname or "").lower()
    else:
        host = s.lower()
    host = host.strip(".")
    return ".".join(host.split(".")[-2:]) if host.count(".") >= 1 else host


def _decode_pyzbar(img_path):
    try:
        from pyzbar.pyzbar import decode
        from PIL import Image
    except Exception:
        return None  # not available
    results = decode(Image.open(img_path))
    return [r.data.decode("utf-8", "replace") for r in results
            if r.type == "QRCODE"]


def _decode_opencv(img_path):
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        return []
    det = cv2.QRCodeDetector()
    ok, decoded, _pts, _ = det.detectAndDecodeMulti(img)
    if not ok:
        return []
    return [d for d in decoded if d]


def decode_qr(img_path):
    """Return list of decoded QR payload strings (deduped, order-preserving)."""
    payloads = _decode_pyzbar(img_path)
    if payloads is None:  # pyzbar unavailable -> OpenCV
        payloads = _decode_opencv(img_path)
    seen, out = set(), []
    for p in payloads:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


URL_LIKE = re.compile(r"\b((?:https?://|www\.)[^\s]+|[a-z0-9-]+\.[a-z]{2,}(?:/[^\s]*)?)", re.I)


def analyze_qr(img_path, page_domain):
    payloads = decode_qr(img_path)
    page_domain = reg_domain(page_domain) if page_domain else ""

    decoded_domains, reasons = [], []
    mismatch = False
    has_url_payload = False
    insecure = False

    for p in payloads:
        m = URL_LIKE.search(p)
        if not m:
            continue
        has_url_payload = True
        token = m.group(1)
        dom = reg_domain(token)
        if dom:
            decoded_domains.append(dom)
        if token.lower().startswith("http://"):
            insecure = True
        if page_domain and dom and dom != page_domain:
            mismatch = True
            reasons.append(
                f"QR code points to '{dom}', which differs from the page domain "
                f"'{page_domain}' - classic in-image redirect."
            )

    if not payloads:
        risk, conf = 0.0, 0.6
        reasons.append("No QR code decoded in this image.")
    elif mismatch:
        risk, conf = 0.9, 0.85
    elif has_url_payload and not page_domain:
        risk, conf = 0.55, 0.5
        reasons.append("QR code contains a URL; page domain unknown, cannot verify match.")
    elif has_url_payload:
        risk, conf = 0.3, 0.6
        reasons.append("QR code URL matches the page domain.")
    else:
        risk, conf = 0.2, 0.5
        reasons.append("QR code present but payload is not a URL.")

    if insecure and risk < 0.9:
        risk = min(0.9, risk + 0.1)
        reasons.append("QR URL uses insecure http://.")

    return {
        "qr_present": bool(payloads),
        "qr_payloads": payloads,
        "decoded_domains": sorted(set(decoded_domains)),
        "domain_mismatch": mismatch,
        "risk_score": round(risk, 2),
        "confidence": round(conf, 2),
        "reasons": reasons,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="image file containing a possible QR code")
    ap.add_argument("--page-url", help="the page the image was found on")
    ap.add_argument("--page-domain", help="registrable domain of the page (overrides --page-url)")
    args = ap.parse_args()

    page_domain = args.page_domain or args.page_url or ""
    verdict = analyze_qr(args.image, page_domain)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
