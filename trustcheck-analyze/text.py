import json
import os
import boto3

# Явно берем регион, чтобы не ловить сюрпризы
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
  "safe_reply": "string"
}
Mask any card numbers or OTP codes if they appear in the input.
"""

def _parse_body(event):
    body = event.get("body", event)
    if isinstance(body, str):
        try:
            return json.loads(body)
        except Exception:
            return {"text": body}
    if isinstance(body, dict):
        return body
    return {}

def lambda_handler(event, context):
    data = _parse_body(event)
    text = (data.get("text") or "").strip()

    if not text:
        return {
            "statusCode": 400,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": "Missing 'text' field"})
        }

    user_prompt = f"{SYSTEM_INSTRUCTIONS}\n\nTEXT:\n{text}"

    print("Calling Bedrock...", MODEL_ID)
    # Nova models are designed to be used with the Converse API. :contentReference[oaicite:1]{index=1}
    resp = brt.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={"maxTokens": 800, "temperature": 0.2},
    )
    
    print("Bedrock responded")
    out_text = resp["output"]["message"]["content"][0]["text"]

    try:
        payload = json.loads(out_text)
    except Exception:
        payload = {"raw": out_text}

    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload)
    }
