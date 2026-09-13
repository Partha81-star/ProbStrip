import html
import json
from datetime import datetime, timezone

from clinical.diagnosis import assess_general_diagnosis


def make_general_report_payload(case_id, modality, quality, edge_area, fingerprint):
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
    diagnosis = assess_general_diagnosis(modality, edge_area, quality["score"])
    return {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "imaging_type": modality,
        "image_fingerprint": fingerprint,
        "report_status": "technical image review generated",
        "automated_diagnosis": diagnosis,
        "diagnosis_message": diagnosis.get("summary", ""),
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
    return json.dumps(payload, indent=2)


def general_report_as_html(payload):
    quality = payload["technical_quality"]
    checks = "".join(
        "<tr><td>{}</td><td>{:.1f} {}</td><td>{}</td></tr>".format(
            html.escape(check["name"]),
            check["value"],
            html.escape(check["unit"]),
            "Pass" if check["ok"] else "Review",
        )
        for check in quality["checks"]
    )
    observations = "".join(
        f"<li>{html.escape(item)}</li>" for item in payload["technical_observations"]
    )
    clinician = payload.get("clinician_review") or {}
    review_html = ""
    if clinician.get("status") == "reviewed":
        diagnosis = html.escape(clinician.get("impression") or "No diagnosis entered.")
        rec = html.escape(clinician.get("recommendation") or "None entered.")
        review_html = f"<h2>Clinician review</h2><p><b>Impression:</b> {diagnosis}</p><p><b>Next step:</b> {rec}</p>"
    diag = payload.get("automated_diagnosis") or {}
    diag_html = ""
    if diag:
        diag_html = (
            f"<h2>Diagnostic Evaluation</h2><p><b>Primary Condition:</b> {html.escape(diag.get('primary_condition', ''))} "
            f"({html.escape(diag.get('risk_level', ''))} Risk - {diag.get('risk_score', 0)}%)</p>"
            f"<p>{html.escape(diag.get('summary', ''))}</p>"
        )
    return (
        f"<!DOCTYPE html><html><body><h1>ProbStrip General Image Review</h1>"
        f"{diag_html}<h2>Quality checks</h2><table>{checks}</table>"
        f"<h2>Observations</h2><ul>{observations}</ul>{review_html}</body></html>"
    )
