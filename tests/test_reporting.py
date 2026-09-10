import json

import numpy as np

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
        "explanation": "A research explanation.",
        "next_step": "See a clinician.",
    }
    return make_report_payload("PS-TEST", quality, measures, outcome, {})


def test_report_never_claims_a_diagnosis():
    payload = _payload()

    assert payload["diagnosis"] is None
    assert "No diagnosis" in report_as_html(payload)
    assert json.loads(report_as_json(payload))["diagnosis"] is None


def test_fhir_export_is_preliminary_and_research_labeled():
    fhir = json.loads(report_as_fhir(_payload()))

    assert fhir["resourceType"] == "DiagnosticReport"
    assert fhir["status"] == "preliminary"
    assert "Research" in fhir["category"][0]["text"]

