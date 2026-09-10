import numpy as np

from clinical.analysis import calculate_research_measures, review_outcome


def test_research_measures_are_bounded_percentages():
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    vessels = np.zeros((64, 64), dtype=bool)
    vessels[:, 30:34] = True
    uncertain = np.zeros_like(vessels)
    uncertain[10:20, 10:20] = True
    variance = uncertain.astype(np.float32) * 0.03

    measures = calculate_research_measures(image, vessels, uncertain, variance)

    assert 0 <= measures["visible_vessel_coverage_percent"] <= 100
    assert 0 <= measures["low_confidence_area_percent"] <= 100


def test_review_outcome_abstains_on_bad_capture():
    outcome = review_outcome("Retake recommended", 0)

    assert outcome["level"] == "retake"
    assert "should not be interpreted" in outcome["explanation"]


def test_review_outcome_escalates_high_uncertainty():
    outcome = review_outcome("Suitable for mapping", 20)

    assert outcome["level"] == "review"

