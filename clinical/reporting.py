import html
import json
from datetime import datetime, timezone

def make_case_id(timestamp=None):
    timestamp = timestamp or datetime.now(timezone.utc)
    return timestamp.strftime("PS-%Y%m%d-%H%M%S-%f")[:-3]


def make_report_payload(
    case_id, quality, measures, outcome, settings, language="English"
):
    safety_notice = (
        "यह स्क्रीनिंग परिणाम अंतिम निदान नहीं है। अचानक दृष्टि हानि, तेज आंख दर्द "
        "या तेजी से बिगड़ते लक्षणों पर तुरंत चिकित्सा सहायता लें।"
        if language == "Hindi"
        else (
            "This screening result is not a final diagnosis. Seek urgent care for sudden "
            "vision loss, severe eye pain, new flashes, or rapidly worsening symptoms."
        )
    )
    return {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "intended_use": "AI-assisted retinal vessel screening",
        "diagnosis": None,
        "diagnosis_message": "Retinal vessel screening completed.",
        "quality": quality.to_dict(),
        "vessel_measures": measures,
        "review_outcome": outcome,
        "model_settings": settings,
        "patient_language": language,
        "safety_notice": safety_notice,
    }


def report_as_json(payload):
    exported = dict(payload)
    exported.pop("quality", None)
    return json.dumps(exported, indent=2)


def report_as_html(payload):
    measures = payload["vessel_measures"]
    diagnosis = payload.get("diagnosis") or {}
    diag_block = ""
    if diagnosis:
        explanation = diagnosis.get("patient_explanation") or {}
        score_text = (
            f"{diagnosis.get('model_score')}%"
            if diagnosis.get("model_score") is not None
            else diagnosis.get("score_text", "Screening completed")
        )
        diag_block = (
            "<h2>Screening result</h2>"
            f"<p><strong>Finding:</strong> {html.escape(diagnosis.get('primary_condition', ''))}</p>"
            f"<p><strong>{html.escape(diagnosis.get('score_label', 'Status'))}:</strong> {html.escape(score_text)}</p>"
            f"<p>{html.escape(explanation.get('patient_summary') or diagnosis.get('summary', ''))}</p>"
            f"<p><strong>Next step:</strong> {html.escape(explanation.get('next_steps') or diagnosis.get('recommendation', ''))}</p>"
        )

    rows = "".join(
        "<tr><td>{}</td><td>{:.2f}%</td></tr>".format(
            html.escape(key.replace("_", " ")), float(value)
        )
        for key, value in measures.items()
        if isinstance(value, (int, float))
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>ProbStrip patient "
        f"report {html.escape(payload['case_id'])}</title><style>body{{font-family:Helvetica,"
        "Arial,sans-serif;margin:2rem;color:#1B2A26}}table{{border-collapse:collapse;width:100%;"
        "margin:1rem 0}}th,td{{border:1px solid #D6DFDB;padding:8px;text-align:left}}th{{"
        "background-color:#E3ECE8}}.notice{{background-color:#FDF5E6;border-left:4px solid "
        "#D97706;padding:12px;margin:1rem 0}}</style></head><body><h1>ProbStrip retinal "
        f"report</h1><p><strong>Reference:</strong> {html.escape(payload['case_id'])}</p>"
        f"<p><strong>Created:</strong> {html.escape(payload['created_at'])}</p><div class='notice'>"
        f"{html.escape(payload['safety_notice'])}</div>{diag_block}"
        f"<h2>Screening status</h2><p>{html.escape(payload.get('diagnosis_message', 'Screening completed.'))}</p>"
        f"<h2>Supporting vessel measurements</h2><table><tr><th>Measure</th><th>Value</th>"
        f"</tr>{rows}</table></body></html>"
    )


def report_as_fhir(payload):
    measures = payload.get("vessel_measures", {})
    notes = [{"text": payload.get("safety_notice", "")}]
    return json.dumps(
        {
            "resourceType": "DiagnosticReport",
            "id": payload.get("case_id", "").lower(),
            "status": "preliminary",
            "category": [{"text": "AI-assisted retinal image analysis"}],
            "code": {"text": "Retinal vessel mapping and diagnostic risk assessment"},
            "effectiveDateTime": payload.get("created_at", ""),
            "conclusion": (payload.get("diagnosis") or {}).get(
                "primary_condition", payload.get("diagnosis_message", "")
            ),
            "note": notes,
            "result": [{"reference": "#visible-vessel-coverage"}],
            "contained": [
                {
                    "resourceType": "Observation",
                    "id": "visible-vessel-coverage",
                    "status": "preliminary",
                    "code": {"text": "Visible vessel coverage"},
                    "valueQuantity": {
                        "value": measures.get("visible_vessel_coverage_percent", 0.0),
                        "unit": "%",
                    },
                }
            ],
        },
        indent=2,
    )
