import numpy as np

from clinical.calibration import (
    build_calibration_profile,
    calibration_error,
    optimal_dice_threshold,
    uncertainty_review_threshold,
)


def test_optimal_threshold_finds_separable_predictions():
    probabilities = np.array([0.05, 0.2, 0.65, 0.9])
    targets = np.array([0, 0, 1, 1])

    threshold, dice = optimal_dice_threshold(probabilities, targets)

    assert 0.2 < threshold <= 0.65
    assert dice == 1.0


def test_uncertainty_threshold_captures_high_variance_errors():
    variances = np.linspace(0, 1, 100)
    errors = variances > 0.9

    threshold, review_fraction, captured = uncertainty_review_threshold(
        variances, errors, review_fraction=0.1
    )

    assert threshold > 0.8
    assert 0.09 <= review_fraction <= 0.11
    assert captured == 1.0


def test_calibration_profile_is_explicitly_not_clinically_validated():
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    targets = np.array([0, 0, 1, 1])
    variances = np.array([0.001, 0.002, 0.003, 0.004])

    profile = build_calibration_profile(
        probabilities, targets, variances, dataset_name="test"
    )

    assert "not clinically validated" in profile["status"]
    assert profile["dataset"] == "test"
    assert calibration_error(probabilities, targets) >= 0

