import html
import json
from datetime import datetime, timezone

from clinical.diagnosis import assess_retinal_diagnosis


def make_case_id(timestamp=None):
    timestamp = timestamp or datetime.now(timezone.utc)
    return timestamp.strftime("PS-%Y%m%d-%H%M%S-%f")[:-3]


def make_report_payload(
    case_id, quality, measures, outcome, settings, language="English"
):
    safety_notice = (
        "यह शोध प्रोटोटाइप बीमारी का निदान नहीं करता और योग्य स्वास्थ्य पेशेवर "
        "द्वारा जांच का स्थान नहीं ले सकता।"
        if language == "Hindi"
        else (
            "This decision support analysis is intended for clinician review and must not replace "
            "an examination by a qualified healthcare professional."
        )
    )
    diagnosis = assess_retinal_diagnosis(measures)
    return {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "intended_use": "Decision support for retinal vessel mapping and microvascular disease risk assessment",
        "diagnosis": diagnosis,
        "diagnosis_message": diagnosis.get("summary", ""),
        "quality": quality.to_dict(),
        "research_measures": measures,
        "review_outcome": outcome,
        "model_settings": settings,
        "patient_language": language,
        "safety_notice": safety_notice,
    }


def report_as_json(payload):
    return json.dumps(payload, indent=2)


def report_as_html(payload):
    quality = payload["quality"]
    clinician_review = payload.get("clinician_review") or {}
    measures = (
        clinician_review.get("reviewed_research_measures")
        or payload["research_measures"]
    )
    outcome = payload["review_outcome"]
    diagnosis = payload.get("diagnosis") or {}
    review_block = ""
    if clinician_review.get("status") == "reviewed":
        note = html.escape(clinician_review.get("note") or "No note provided.")
        impression = html.escape(
            clinician_review.get("impression") or "No diagnosis entered."
        )
        recommendation = html.escape(
            clinician_review.get("recommendation") or "Follow local clinical guidance."
        )
        review_block = (
            "<h2>Clinician review</h2><p><strong>Impression or diagnosis:</strong> "
            f"{impression}</p><p><strong>Notes:</strong> {note}</p><p><strong>"
            f"Recommended next step:</strong> {recommendation}</p>"
        )

    diag_block = ""
    if diagnosis:
        diag_block = (
            "<h2>Diagnostic Risk Assessment</h2>"
            f"<p><strong>Primary Finding:</strong> {html.escape(diagnosis.get('primary_condition', ''))} "
            f"({html.escape(diagnosis.get('risk_level', ''))} Risk - {diagnosis.get('risk_score', 0)}%)</p>"
            f"<p>{html.escape(diagnosis.get('summary', ''))}</p>"
            f"<p><strong>Recommendation:</strong> {html.escape(diagnosis.get('recommendation', ''))}</p>"
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
        f"{html.escape(payload['safety_notice'])}</div>{diag_block}<h2>Image quality: "
        f"{quality['score']}/100</h2><p>{html.escape(outcome.get('explanation', ''))}</p>"
        f"{review_block}<h2>Vascular measurements</h2><table><tr><th>Measure</th><th>Value</th>"
        f"</tr>{rows}</table></body></html>"
    )


def report_as_fhir(payload):
    review = payload.get("clinician_review") or {}
    measures = review.get("reviewed_research_measures") or payload.get("research_measures", {})
    notes = [{"text": payload.get("safety_notice", "")}]
    if review.get("status") == "reviewed":
        notes.append(
            {
                "text": "Clinician session review: "
                + (review.get("note") or "No note provided.")
            }
        )
    diag = payload.get("diagnosis") or {}
    if diag:
        notes.append(
            {
                "text": f"Diagnostic evaluation: {diag.get('primary_condition', '')} - {diag.get('risk_level', '')} Risk ({diag.get('risk_score', 0)}%)"
            }
        )
    return json.dumps(
        {
            "resourceType": "DiagnosticReport",
            "id": payload.get("case_id", "").lower(),
            "status": "preliminary",
            "category": [{"text": "Clinical decision support retinal image analysis"}],
            "code": {"text": "Retinal vessel mapping and diagnostic risk assessment"},
            "effectiveDateTime": payload.get("created_at", ""),
            "conclusion": review.get("impression") or payload.get("diagnosis_message", ""),
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
