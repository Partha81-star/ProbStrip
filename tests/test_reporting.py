import json
from io import BytesIO

import numpy as np
from pypdf import PdfReader

from clinical.pdf_reporting import retinal_report_as_pdf
from clinical.quality import assess_image_quality
from clinical.reporting import (
    make_report_payload,
    report_as_fhir,
    report_as_html,
    report_as_json,
)


def _payload():
    quality = assess_image_quality(np.zeros((64, 64, 3), dtype=np.uint8))
    measures = {
        "visible_vessel_coverage_percent": 5.5,
        "central_vessel_coverage_percent": 6.0,
        "low_confidence_area_percent": 2.0,
        "mean_model_variance": 0.001,
        "maximum_model_variance": 0.02,
    }
    outcome = {
        "level": "ready",
        "title": "Ready for review",
        "explanation": "The vessel map was completed.",
        "next_step": "Keep the report with the original image.",
    }
    payload = make_report_payload("PS-TEST", quality, measures, outcome, {})
    payload["diagnosis"] = {
        "status": "evaluated",
        "primary_condition": "Retinal vessel map completed",
        "risk_level": "Low",
        "risk_score": None,
        "model_score": None,
        "score_label": "Screening status",
        "score_text": "Vessel map completed",
        "summary": "Visible retinal vessels were mapped.",
        "recommendation": "Keep the result with the original image.",
    }
    return payload


def test_report_contains_retinal_screening_result():
    payload = _payload()

    assert payload["diagnosis"]["primary_condition"] == "Retinal vessel map completed"
    assert "Retinal vessel map completed" in report_as_html(payload)
    exported = json.loads(report_as_json(payload))
    assert "vessel_measures" in exported
    assert "research_measures" not in exported
    assert "quality" not in exported


def test_fhir_export_is_preliminary_and_ai_assisted():
    fhir = json.loads(report_as_fhir(_payload()))

    assert fhir["resourceType"] == "DiagnosticReport"
    assert fhir["status"] == "preliminary"
    assert "AI-assisted" in fhir["category"][0]["text"]


def test_patient_pdf_is_valid_and_contains_safety_content():
    payload = _payload()
    image = np.full((128, 128, 3), 110, dtype=np.uint8)

    pdf = retinal_report_as_pdf(payload, image, image)
    reader = PdfReader(BytesIO(pdf))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert pdf.startswith(b"%PDF-")
    assert len(reader.pages) >= 1
    assert "Retinal vessel map completed" in text.replace("\n", " ")
    assert "research" not in text.lower()
    assert "Image-quality" not in text
    assert "Quality score" not in text
