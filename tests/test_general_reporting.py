from clinical.general_reporting import (
    general_report_as_html,
    make_general_report_payload,
)


def _quality():
    return {
        "score": 80,
        "status": "Usable with caution",
        "checks": [
            {"name": "Contrast", "value": 12.5, "unit": "%", "ok": True}
        ],
    }


def test_general_report_is_created_without_claiming_diagnosis():
    payload = make_general_report_payload(
        "PS-TEST", "Bone or joint X-ray", _quality(), 4.2, "abc123"
    )

    report = general_report_as_html(payload)

    assert payload["report_status"] == "technical image review generated"
    assert payload["automated_diagnosis"] is None
    assert "Awaiting review" in report


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
