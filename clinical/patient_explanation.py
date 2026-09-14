"""Generate constrained patient language from an existing model result."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "gemini-2.5-flash"


def _fallback(result):
    return {
        "patient_summary": result.get("summary", "A screening result was created."),
        "next_steps": result.get(
            "recommendation", "Follow the next-step guidance shown in the report."
        ),
        "urgent_warning": (
            "Seek urgent care for severe pain, breathing difficulty, sudden vision loss, "
            "rapid worsening, or other emergency symptoms."
        ),
        "generated": False,
    }


def explain_model_result(result, modality, api_key=None, model_name=None):
    fallback = _fallback(result)
    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback

    model_name = model_name or os.getenv("PROBSTRIP_EXPLANATION_MODEL", DEFAULT_MODEL)
    context = {
        "imaging_category": modality,
        "detected_finding": result.get("primary_condition"),
        "score_label": result.get("score_label", "Model confidence"),
        "model_score_percent": result.get("model_score", result.get("risk_score")),
        "model_summary": result.get("summary"),
        "approved_next_step": result.get("recommendation"),
        "population_scope": result.get("population_scope"),
    }
    prompt = (
        "Rewrite only the supplied screening result in clear patient-friendly language. "
        "Do not inspect an image, add a new disease, claim certainty, prescribe medication, "
        "or change the approved next step. Keep each field under 45 words.\n\n"
        + json.dumps(context, ensure_ascii=True)
    )
    schema = {
        "type": "OBJECT",
        "properties": {
            "patient_summary": {"type": "STRING"},
            "next_steps": {"type": "STRING"},
            "urgent_warning": {"type": "STRING"},
        },
        "required": ["patient_summary", "next_steps", "urgent_warning"],
    }
    body = json.dumps(
        {
            "system_instruction": {
                "parts": [
                    {
                        "text": (
                            "You explain structured medical screening outputs. Never create "
                            "or upgrade a diagnosis beyond the supplied model result."
                        )
                    }
                ]
            },
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 250,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
    ).encode("utf-8")
    request = Request(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent",
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        explanation = json.loads(text)
        if not all(
            isinstance(explanation.get(key), str) and explanation[key].strip()
            for key in ("patient_summary", "next_steps", "urgent_warning")
        ):
            return fallback
        return {**explanation, "generated": True}
    except (HTTPError, URLError, KeyError, IndexError, json.JSONDecodeError, TimeoutError):
        return fallback
