from io import BytesIO

import numpy as np
from pypdf import PdfReader

from clinical.general_reporting import (
    general_report_as_html,
    general_report_as_json,
    make_general_report_payload,
)
from clinical.pdf_reporting import general_report_as_pdf


def _quality():
    return {
        "score": 80,
        "status": "Usable with caution",
        "checks": [
            {"name": "Contrast", "value": 12.5, "unit": "%", "ok": True}
        ],
    }


def test_general_report_is_created_without_a_configured_model():
    payload = make_general_report_payload(
        "PS-TEST", "Bone or joint X-ray", _quality(), 4.2, "abc123"
    )

    report = general_report_as_html(payload)

    assert payload["report_status"] == "automated screening report"
    assert payload["automated_diagnosis"] is None
    assert "No disease-specific model" in payload["diagnosis_message"]
    assert "No disease-specific detection model" in report
    assert "technical_quality" not in general_report_as_json(payload)
    assert "Quality checks" not in report


def test_general_report_includes_confirmed_clinician_impression():
    payload = make_general_report_payload(
        "PS-TEST", "Chest X-ray", _quality(), 3.1, "abc123"
    )
    payload["clinician_review"] = {
        "status": "reviewed",
        "impression": "Clinician-entered impression",
        "observations": "Compared with original image.",
        "recommendation": "Follow up.",
        "reviewed_at": "2026-09-11T00:00:00Z",
    }

    assert "Clinician-entered impression" in general_report_as_html(payload)


def test_general_pdf_prioritizes_diagnosis_and_omits_quality_details():
    payload = make_general_report_payload(
        "PS-TEST", "Chest X-ray", _quality(), 3.1, "abc123"
    )
    payload["clinician_review"] = {
        "status": "reviewed",
        "impression": "Clinician-confirmed pneumonia",
        "recommendation": "Arrange treatment review.",
    }
    image = np.full((64, 64, 3), 100, dtype=np.uint8)

    pdf = general_report_as_pdf(payload, image, image)
    text = "\n".join(
        page.extract_text() or "" for page in PdfReader(BytesIO(pdf)).pages
    )

    assert "Clinician-confirmed pneumonia" in text
    assert "Image-quality" not in text
    assert "Quality score" not in text


def test_general_pdf_reports_automated_fracture_confidence():
    diagnosis = {
        "status": "evaluated",
        "primary_condition": "Fracture pattern detected",
        "risk_level": "High",
        "risk_score": 82.4,
        "summary": "The model localized one fracture candidate.",
        "recommendation": "Arrange prompt orthopedic assessment.",
        "detections": [{"box": [2, 3, 20, 22], "confidence": 0.824}],
    }
    payload = make_general_report_payload(
        "PS-FRACTURE",
        "Bone or joint X-ray",
        _quality(),
        2.5,
        "abc123",
        automated_diagnosis=diagnosis,
    )
    image = np.full((64, 64, 3), 100, dtype=np.uint8)

    pdf = general_report_as_pdf(payload, image, image)
    text = "\n".join(
        page.extract_text() or "" for page in PdfReader(BytesIO(pdf)).pages
    )

    assert "Fracture pattern detected" in text
    assert "Model confidence: 82.4%" in text
    assert "Awaiting confirmation" not in text
