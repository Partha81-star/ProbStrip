import html
import json
from datetime import datetime, timezone

def make_general_report_payload(case_id, modality, quality, edge_area, fingerprint, image=None):
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
        "report_status": "clinical report awaiting clinician confirmation",
        "automated_diagnosis": None,
        "diagnosis_message": "Diagnosis status: awaiting clinician confirmation.",
        "analysis_scope": "AI-assisted image views with clinician-documented diagnosis.",
        "technical_quality": quality,
        "visible_edge_area_percent": edge_area,
        "technical_observations": observations,
        "clinician_review": None,
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
    clinician = payload.get("clinician_review") or {}
    review_html = "<h2>Diagnosis</h2><p>Awaiting confirmation by a qualified clinician.</p>"
    if clinician.get("status") == "reviewed":
        diagnosis = html.escape(clinician.get("impression") or "No diagnosis entered.")
        rec = html.escape(clinician.get("recommendation") or "None entered.")
        review_html = f"<h2>Clinician-confirmed diagnosis</h2><p><b>Diagnosis:</b> {diagnosis}</p><p><b>Next step:</b> {rec}</p>"
    return (
        f"<!DOCTYPE html><html><body><h1>ProbStrip Clinical Image Report</h1>"
        f"{review_html}</body></html>"
    )
