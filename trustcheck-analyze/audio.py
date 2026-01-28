import json
import os
import time
import uuid
import urllib.request

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ["UPLOAD_BUCKET"]
MODEL_ID = os.environ.get("MODEL_ID", "amazon.nova-pro-v1:0")

transcribe = boto3.client("transcribe", region_name=REGION)
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
  "transcript_text": "string",
  "transcribe_job": "string",
  "language_code": "string|null"
}
If transcript_text is empty or too short, set risk_level to "medium" and explain uncertainty.
Mask any card numbers or OTP codes if they appear in transcript_text.
"""

def _parse_body(event):
    body = event.get("body", event)
    if isinstance(body, str):
        try:
            return json.loads(body)
        except Exception:
            return {}
    return body if isinstance(body, dict) else {}

def _guess_media_format(s3_key: str) -> str:
    key = s3_key.lower()
    for ext in ("mp3", "mp4", "wav", "flac", "ogg", "amr", "webm", "m4a"):
        if key.endswith("." + ext):
            return ext
    return "mp3"

def _download_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read()
    return json.loads(raw.decode("utf-8"))

def _bedrock_analyze(transcript_text: str, job_name: str, detected_lang: str | None):
    # safety: do not send megabytes into the model
    if len(transcript_text) > 12000:
        transcript_text = transcript_text[:12000] + "\n[TRUNCATED]"

    user_prompt = f"{SYSTEM_INSTRUCTIONS}\n\nTRANSCRIBE_JOB: {job_name}\n\nTRANSCRIPT_TEXT:\n{transcript_text}"

    resp = brt.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={"maxTokens": 900, "temperature": 0.2},
    )
    out_text = resp["output"]["message"]["content"][0]["text"]

    try:
        payload = json.loads(out_text)
    except Exception:
        payload = {"raw": out_text}

    if isinstance(payload, dict):
        payload.setdefault("transcript_text", transcript_text)
        payload.setdefault("transcribe_job", job_name)
        payload.setdefault("language_code", detected_lang)

    return payload

def lambda_handler(event, context):
    data = _parse_body(event)

    # Either start a new job with s3Key OR poll an existing job with transcribeJob
    s3_key = (data.get("s3Key") or "").strip()
    existing_job = (data.get("transcribeJob") or "").strip()
    language_code = (data.get("languageCode") or "").strip()  # optional

    if not s3_key and not existing_job:
        return {
            "statusCode": 400,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": "Provide either 's3Key' or 'transcribeJob'."})
        }

    try:
        if existing_job:
            job_name = existing_job
        else:
            media_format = _guess_media_format(s3_key)
            s3_uri = f"s3://{BUCKET}/{s3_key}"  # Transcribe expects S3 location. :contentReference[oaicite:2]{index=2}
            job_name = f"trustcheck-{int(time.time())}-{uuid.uuid4().hex[:12]}"

            kwargs = dict(
                TranscriptionJobName=job_name,
                Media={"MediaFileUri": s3_uri},
                MediaFormat=media_format,
            )

            if language_code:
                kwargs["LanguageCode"] = language_code
            else:
                kwargs["IdentifyLanguage"] = True
                kwargs["LanguageOptions"] = ["en-US", "ru-RU", "kk-KZ"]

            transcribe.start_transcription_job(**kwargs)

        # Poll for completion (MVP)
        deadline = time.time() + 90  # tune as needed
        status = None
        result_uri = None
        detected_lang = None

        while time.time() < deadline:
            resp = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            job = resp["TranscriptionJob"]
            status = job["TranscriptionJobStatus"]
            detected_lang = job.get("LanguageCode")

            if status == "COMPLETED":
                # If you don't set OutputBucketName, Transcribe returns a presigned TranscriptFileUri. :contentReference[oaicite:3]{index=3}
                result_uri = job["Transcript"]["TranscriptFileUri"]
                break

            if status == "FAILED":
                return {
                    "statusCode": 500,
                    "headers": {"content-type": "application/json"},
                    "body": json.dumps({
                        "error": "Transcribe job failed",
                        "transcribe_job": job_name,
                        "failureReason": job.get("FailureReason")
                    })
                }

            time.sleep(2)

        if not result_uri:
            return {
                "statusCode": 202,
                "headers": {"content-type": "application/json"},
                "body": json.dumps({
                    "status": status or "IN_PROGRESS",
                    "transcribe_job": job_name
                })
            }

        transcript_json = _download_json(result_uri)
        transcript_text = ""
        try:
            transcript_text = transcript_json["results"]["transcripts"][0]["transcript"]
        except Exception:
            transcript_text = ""

        payload = _bedrock_analyze(transcript_text, job_name, detected_lang)

        return {
            "statusCode": 200,
            "headers": {"content-type": "application/json"},
            "body": json.dumps(payload)
        }

    except Exception as e:
        # Make errors visible instead of "Internal Server Error"
        return {
            "statusCode": 500,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({
                "error": "unhandled_exception",
                "details": str(e),
                "s3Key": s3_key,
                "transcribe_job": existing_job
            })
        }
