import json
import os
import re
import ipaddress
import unicodedata
from urllib.parse import urlparse, parse_qs, unquote

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get("MODEL_ID", "amazon.nova-pro-v1:0")

brt = boto3.client("bedrock-runtime", region_name=REGION)

SYSTEM_INSTRUCTIONS = """You are TrustCheck, a fraud and scam risk classifier.
You MUST respond in English.
Return ONLY valid JSON (no markdown, no extra text) in this schema:
{
  "risk_score": 0-100,
  "risk_level": "low" | "medium" | "high",
  "summary": "one sentence",
  "red_flags": [{"type": "string", "severity": "low|medium|high", "evidence": "string"}],
  "recommended_actions": ["string", "string"],
  "safe_reply": "string",
  "url": "string",
  "normalized_url": "string",
  "host": "string",
  "scheme": "string",
  "tld": "string|null",
  "url_flags": [{"flag": "string", "severity": "low|medium|high", "evidence": "string"}],
  "url_risk_score": 0-100
}
If you are uncertain, set risk_level to "medium" and explain uncertainty in summary.
"""

SUSPICIOUS_TLDS = {
    "zip", "mov", "top", "xyz", "icu", "click", "country", "kim", "loan", "work",
    "tk", "gq", "cf", "ml", "ga", "monster", "quest", "rest", "cfd"
}

URL_SHORTENERS = {
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "is.gd", "cutt.ly", "rebrand.ly",
    "buff.ly", "ow.ly", "rb.gy", "shorturl.at"
}

SUSPICIOUS_PATH_WORDS = {
    "login", "signin", "verify", "verification", "update", "secure", "account",
    "wallet", "bank", "payment", "confirm", "password", "reset", "otp", "code",
    "2fa", "security", "support", "unlock"
}

SUSPICIOUS_QUERY_KEYS = {
    "redirect", "redir", "return", "continue", "next", "url", "target", "dest",
    "destination", "goto", "callback"
}

def _parse_body(event):
    body = event.get("body", event)
    if isinstance(body, str):
        try:
            return json.loads(body)
        except Exception:
            return {}
    return body if isinstance(body, dict) else {}

def _normalize_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    # If no scheme, assume http (and we'll penalize non-https anyway)
    if "://" not in raw:
        raw = "http://" + raw
    return raw

def _extract_host_parts(host: str):
    host = (host or "").strip().lower().rstrip(".")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    parts = host.split(".") if host else []
    tld = parts[-1] if len(parts) >= 2 else None
    return host, parts, tld

def _is_ip_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except Exception:
        return False

def _has_non_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return False
    except Exception:
        return True

def _unicode_suspicion(host: str) -> bool:
    # Flag if host has mixed scripts or suspicious unicode (homoglyph risk)
    # Simple heuristic: any non-ascii in host
    return _has_non_ascii(host)

def _count_percent_encoding(url: str) -> int:
    return len(re.findall(r"%[0-9A-Fa-f]{2}", url))

def _safe_unquote(s: str) -> str:
    try:
        return unquote(s)
    except Exception:
        return s

def _build_flags(parsed, normalized_url: str):
    flags = []
    score = 0

    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "")
    query = (parsed.query or "")

    host_norm, parts, tld = _extract_host_parts(host)

    # 1) Scheme checks
    if scheme != "https":
        flags.append({"flag": "non_https", "severity": "high", "evidence": f"URL scheme is '{scheme or 'missing'}', not HTTPS."})
        score += 25

    # 2) IP literal host
    if host_norm and _is_ip_host(host_norm):
        flags.append({"flag": "ip_in_host", "severity": "high", "evidence": f"Host is an IP address: {host_norm}."})
        score += 25

    # 3) Shorteners
    if host_norm in URL_SHORTENERS:
        flags.append({"flag": "url_shortener", "severity": "medium", "evidence": f"Uses a known URL shortener: {host_norm}."})
        score += 12

    # 4) Suspicious TLD
    if tld and tld in SUSPICIOUS_TLDS:
        flags.append({"flag": "suspicious_tld", "severity": "medium", "evidence": f"TLD '.{tld}' is commonly abused in phishing."})
        score += 10

    # 5) Too many subdomains
    if len(parts) >= 5:  # e.g. a.b.c.d.example.com
        flags.append({"flag": "many_subdomains", "severity": "medium", "evidence": f"Host has many subdomains ({len(parts)-2} subdomain levels)."})
        score += 10

    # 6) Punycode / IDN
    if "xn--" in host_norm:
        flags.append({"flag": "punycode_host", "severity": "high", "evidence": "Host contains punycode 'xn--', potential homograph attack."})
        score += 20

    if host_norm and _unicode_suspicion(host_norm):
        flags.append({"flag": "non_ascii_host", "severity": "medium", "evidence": "Host contains non-ASCII characters (possible homoglyph risk)."})
        score += 12

    # 7) @ in URL (userinfo trick)
    if "@" in normalized_url.split("://", 1)[-1].split("/", 1)[0]:
        flags.append({"flag": "userinfo_in_url", "severity": "high", "evidence": "URL contains '@' in the authority component (can hide real destination)."})
        score += 20

    # 8) Suspicious keywords in path/query
    path_l = _safe_unquote(path).lower()
    query_l = _safe_unquote(query).lower()

    for w in SUSPICIOUS_PATH_WORDS:
        if w in path_l:
            flags.append({"flag": "suspicious_path_keyword", "severity": "medium", "evidence": f"Path contains '{w}'."})
            score += 5
            break

    qs = parse_qs(query_l, keep_blank_values=True)
    for k in qs.keys():
        k_clean = k.lower()
        if k_clean in SUSPICIOUS_QUERY_KEYS:
            flags.append({"flag": "redirect_parameter", "severity": "medium", "evidence": f"Query contains redirect-like parameter '{k_clean}'."})
            score += 8
            break

    # 9) Heavy percent-encoding
    pct = _count_percent_encoding(normalized_url)
    if pct >= 10:
        flags.append({"flag": "heavy_encoding", "severity": "medium", "evidence": f"URL contains many percent-encoded sequences ({pct})."})
        score += 8

    # 10) Very long URL
    if len(normalized_url) >= 140:
        flags.append({"flag": "very_long_url", "severity": "low", "evidence": f"URL is very long ({len(normalized_url)} chars)."})
        score += 4

    # Bound score
    score = max(0, min(100, score))
    return flags, score, host_norm, scheme, tld

def _risk_level(score: int) -> str:
    if score >= 70:
        return "high"
    if score >= 35:
        return "medium"
    return "low"

def _rules_to_redflags(url_flags, url_risk_score):
    red = []
    # Map strongest url_flags to red_flags
    for f in url_flags[:5]:
        sev = f["severity"]
        red.append({"type": f"url:{f['flag']}", "severity": sev, "evidence": f["evidence"]})
    # Add a general note if score is high
    if url_risk_score >= 70:
        red.append({"type": "suspicious_link", "severity": "high", "evidence": "Multiple URL signals indicate phishing/redirect risk."})
    return red

def _bedrock_finalize(raw_url, normalized_url, host, scheme, tld, url_flags, url_risk_score):
    # Keep prompt small and deterministic
    flags_compact = [{"flag": f["flag"], "severity": f["severity"], "evidence": f["evidence"]} for f in url_flags]

    user_prompt = (
        SYSTEM_INSTRUCTIONS
        + "\n\nURL_INPUT:\n" + raw_url
        + "\n\nNORMALIZED_URL:\n" + normalized_url
        + "\n\nHOST:\n" + (host or "")
        + "\n\nSCHEME:\n" + (scheme or "")
        + "\n\nTLD:\n" + (tld or "null")
        + "\n\nURL_RISK_SCORE:\n" + str(url_risk_score)
        + "\n\nURL_FLAGS_JSON:\n" + json.dumps(flags_compact)
    )

    resp = brt.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={"maxTokens": 900, "temperature": 0.2},
    )
    out_text = resp["output"]["message"]["content"][0]["text"]
    return out_text

def lambda_handler(event, context):
    data = _parse_body(event)
    raw_url = (data.get("url") or "").strip()

    if not raw_url:
        return {
            "statusCode": 400,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": "Missing 'url' field"})
        }

    try:
        normalized = _normalize_url(raw_url)
        parsed = urlparse(normalized)

        url_flags, url_risk_score, host, scheme, tld = _build_flags(parsed, normalized)

        # Start with a rules-only payload (fallback)
        rules_payload = {
            "risk_score": url_risk_score,
            "risk_level": _risk_level(url_risk_score),
            "summary": "URL was analyzed using heuristic checks.",
            "red_flags": _rules_to_redflags(url_flags, url_risk_score),
            "recommended_actions": [
                "Do not open the link if you did not request it or it came unexpectedly.",
                "Verify the website by typing the official domain manually or contacting support via official channels."
            ],
            "safe_reply": "I won't open links from unverified sources. Please provide official contact information instead.",
            "url": raw_url,
            "normalized_url": normalized,
            "host": host,
            "scheme": scheme,
            "tld": tld,
            "url_flags": url_flags,
            "url_risk_score": url_risk_score
        }

        # If we have meaningful signals, ask Nova Pro to produce a clean final JSON explanation.
        out_text = _bedrock_finalize(raw_url, normalized, host, scheme, tld, url_flags, url_risk_score)

        try:
            payload = json.loads(out_text)
        except Exception:
            payload = None

        if isinstance(payload, dict):
            # Ensure required fields exist even if model forgets something
            payload.setdefault("url", raw_url)
            payload.setdefault("normalized_url", normalized)
            payload.setdefault("host", host)
            payload.setdefault("scheme", scheme)
            payload.setdefault("tld", tld)
            payload.setdefault("url_flags", url_flags)
            payload.setdefault("url_risk_score", url_risk_score)

            # If model didn't set risk_score/risk_level, fallback to rules
            if "risk_score" not in payload:
                payload["risk_score"] = rules_payload["risk_score"]
            if "risk_level" not in payload:
                payload["risk_level"] = rules_payload["risk_level"]
            if "red_flags" not in payload:
                payload["red_flags"] = rules_payload["red_flags"]
            if "recommended_actions" not in payload:
                payload["recommended_actions"] = rules_payload["recommended_actions"]
            if "safe_reply" not in payload:
                payload["safe_reply"] = rules_payload["safe_reply"]
            if "summary" not in payload:
                payload["summary"] = rules_payload["summary"]

            return {
                "statusCode": 200,
                "headers": {"content-type": "application/json"},
                "body": json.dumps(payload)
            }

        # Fallback (rules-only)
        return {
            "statusCode": 200,
            "headers": {"content-type": "application/json"},
            "body": json.dumps(rules_payload)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": "unhandled_exception", "details": str(e), "url": raw_url})
        }
