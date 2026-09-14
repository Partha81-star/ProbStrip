import json

import clinical.patient_explanation as patient_explanation


def _result():
    return {
        "primary_condition": "Pneumonia pattern detected",
        "model_score": 82.0,
        "score_label": "Pneumonia model score",
        "summary": "The score exceeded the operating threshold.",
        "recommendation": "Arrange prompt medical assessment.",
    }


def test_explanation_falls_back_without_a_secret(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    explanation = patient_explanation.explain_model_result(
        _result(), "Chest X-ray"
    )

    assert explanation["generated"] is False
    assert explanation["next_steps"] == "Arrange prompt medical assessment."


def test_explanation_accepts_only_structured_response(monkeypatch):
    response_text = json.dumps(
        {
            "patient_summary": "The screening model found a pneumonia pattern.",
            "next_steps": "Arrange prompt medical assessment.",
            "urgent_warning": "Seek urgent help for breathing difficulty.",
        }
    )
    api_response = {
        "candidates": [{"content": {"parts": [{"text": response_text}]}}]
    }

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(api_response).encode("utf-8")

    monkeypatch.setattr(patient_explanation, "urlopen", lambda request, timeout: _Response())

    explanation = patient_explanation.explain_model_result(
        _result(), "Chest X-ray", api_key="test-key"
    )

    assert explanation["generated"] is True
    assert "pneumonia pattern" in explanation["patient_summary"]
