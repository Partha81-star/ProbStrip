import html
import json
from datetime import datetime, timezone


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
    return {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "imaging_type": modality,
        "image_fingerprint": fingerprint,
        "report_status": "technical image review generated",
        "automated_diagnosis": None,
        "diagnosis_message": (
            "No automated diagnosis is available for this imaging type. A qualified "
            "clinician may add an impression below."
        ),
        "technical_quality": quality,
        "visible_edge_area_percent": edge_area,
        "technical_observations": observations,
        "clinician_review": None,
        "safety_notice": (
            "Contrast and edge views can hide or emphasize features and must be compared "
            "with the original image. This prototype does not detect or exclude disease."
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
    if clinician.get("status") == "reviewed":
        diagnosis = html.escape(clinician.get("impression") or "No diagnosis entered.")
        clinical_block = (
            "<h2>Clinician assessment</h2>"
            f"<p><strong>Impression or diagnosis:</strong> {diagnosis}</p>"
            f"<p><strong>Observations:</strong> {html.escape(clinician.get('observations') or 'None entered.')}</p>"
            f"<p><strong>Recommended next step:</strong> {html.escape(clinician.get('recommendation') or 'Follow local clinical guidance.')}</p>"
            f"<p><small>Recorded: {html.escape(clinician.get('reviewed_at', ''))}</small></p>"
        )
    else:
        clinical_block = (
            "<h2>Clinical interpretation</h2><p><strong>Awaiting review by a qualified "
            "healthcare professional.</strong> No automated diagnosis was generated.</p>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ProbStrip image review {html.escape(payload['case_id'])}</title>
<style>body{{font-family:Arial,sans-serif;max-width:760px;margin:32px auto;padding:0 20px;color:#17202a;line-height:1.55}}h1{{font-size:26px}}h2{{font-size:18px;margin-top:28px}}.notice{{border-left:5px solid #d97706;background:#fff7ed;padding:14px}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #d8dee4;padding:9px;text-align:left}}small{{color:#52606d}}</style></head>
<body><h1>ProbStrip {html.escape(payload['imaging_type'])} report</h1>
<small>Reference: {html.escape(payload['case_id'])}</small>
<div class="notice"><strong>Preliminary image review</strong><br>{html.escape(payload['safety_notice'])}</div>
<h2>Technical observations</h2><ul>{observations}</ul>
{clinical_block}
<h2>Image quality</h2><p>Score: {quality['score']}/100. {html.escape(quality['status'])}</p>
<table><thead><tr><th>Check</th><th>Value</th><th>Result</th></tr></thead><tbody>{checks}</tbody></table>
<p><small>Image fingerprint: {html.escape(payload['image_fingerprint'])}</small></p></body></html>"""
