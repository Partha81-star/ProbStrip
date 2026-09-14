import html
import json
from datetime import datetime, timezone

def make_general_report_payload(
    case_id,
    modality,
    quality,
    edge_area,
    fingerprint,
    image=None,
    automated_diagnosis=None,
):
    observations = []
    failed = [check["name"] for check in quality["checks"] if not check["ok"]]
    if failed:
        observations.append(
            "Technical review recommended for: " + ", ".join(failed) + "."
        )
    else:
        observations.append("The image passed the available technical quality checks.")
    observations.append(
        f"A structure-edge view was created; visible edges cover {edge_area:.2f}% of the image."
    )
    return {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "imaging_type": modality,
        "image_fingerprint": fingerprint,
        "report_status": "automated screening report",
        "automated_diagnosis": automated_diagnosis,
        "diagnosis_message": (
            "Automated disease screening completed."
            if automated_diagnosis
            else "No disease-specific model is configured for this imaging category."
        ),
        "analysis_scope": "Automated screening for supported findings.",
        "technical_quality": quality,
        "visible_edge_area_percent": edge_area,
        "technical_observations": observations,
        "safety_notice": (
            "Contrast and edge views can hide or emphasize features and must be compared "
            "with the original image. Correlate with clinical history and patient symptoms."
        ),
    }



def general_report_as_json(payload):
    exported = dict(payload)
    exported.pop("technical_quality", None)
    return json.dumps(exported, indent=2)


def general_report_as_html(payload):
    automated = payload.get("automated_diagnosis") or {}
    review_html = (
        "<h2>Detection result</h2>"
        "<p>No disease-specific detection model was used for this imaging category.</p>"
    )
    if automated.get("status") == "evaluated":
        finding = html.escape(automated.get("primary_condition") or "No finding recorded.")
        summary = html.escape(automated.get("summary") or "")
        recommendation = html.escape(automated.get("recommendation") or "")
        model_score = automated.get("model_score")
        if model_score is None and automated.get("detections"):
            model_score = automated.get("risk_score")
        confidence = (
            f"{model_score}%"
            if model_score is not None
            else automated.get(
                "score_text", "No detection above the configured threshold"
            )
        )
        score_label = html.escape(automated.get("score_label") or "Model confidence")
        explanation = automated.get("patient_explanation") or {}
        patient_summary = html.escape(explanation.get("patient_summary") or summary)
        next_steps = html.escape(explanation.get("next_steps") or recommendation)
        urgent_warning = html.escape(explanation.get("urgent_warning") or "")
        review_html = (
            f"<h2>Detected finding</h2><p><b>Result:</b> {finding}</p>"
            f"<p><b>{score_label}:</b> {confidence}</p>"
            f"<p>{patient_summary}</p><p><b>Recommended next step:</b> {next_steps}</p>"
            f"<p>{urgent_warning}</p>"
        )
    return (
        f"<!DOCTYPE html><html><body><h1>ProbStrip Clinical Image Report</h1>"
        f"{review_html}</body></html>"
    )
