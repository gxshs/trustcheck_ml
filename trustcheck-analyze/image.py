import json
import os
import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get("MODEL_ID", "amazon.nova-pro-v1:0")
BUCKET = os.environ["UPLOAD_BUCKET"]

s3 = boto3.client("s3", region_name=REGION)
rek = boto3.client("rekognition", region_name=REGION)
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
  "extracted_text": "string"
}
If extracted_text is empty or too short (<20 chars), set risk_level to "medium" and explain uncertainty in summary.
Mask any card numbers or OTP codes if they appear in extracted_text.
"""

def _parse_body(event):
    body = event.get("body", event)
    if isinstance(body, str):
        try:
            return json.loads(body)
        except Exception:
            return {}
    return body if isinstance(body, dict) else {}

def _join_text(detections):
    # Keep LINEs for readability; fallback to WORD if no lines.
    lines = []
    words = []
    for d in detections:
        t = d.get("Type")
        txt = d.get("DetectedText", "")
        conf = d.get("Confidence", 0)
        if conf < 70:  # drop super noisy detections
            continue
        if t == "LINE":
            lines.append(txt)
        elif t == "WORD":
            words.append(txt)

    if lines:
        return "\n".join(lines).strip()
    return " ".join(words).strip()

def lambda_handler(event, context):
    data = _parse_body(event)
    s3_key = (data.get("s3Key") or "").strip()

    if not s3_key:
        return {
            "statusCode": 400,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": "Missing 's3Key' field"})
        }

    # Rekognition can read S3 objects directly
    ocr = rek.detect_text(
        Image={"S3Object": {"Bucket": BUCKET, "Name": s3_key}}
    )
    extracted = _join_text(ocr.get("TextDetections", []))

    user_prompt = f"{SYSTEM_INSTRUCTIONS}\n\nEXTRACTED_TEXT:\n{extracted}"

    resp = brt.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={"maxTokens": 900, "temperature": 0.2},
    )

    out_text = resp["output"]["message"]["content"][0]["text"]

    try:
        payload = json.loads(out_text)
    except Exception:
        payload = {"raw": out_text, "extracted_text": extracted}

    # Ensure extracted_text is always returned for debugging
    if isinstance(payload, dict) and "extracted_text" not in payload:
        payload["extracted_text"] = extracted

    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload)
    }
