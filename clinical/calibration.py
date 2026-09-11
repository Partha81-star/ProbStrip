from datetime import datetime, timezone

import numpy as np


def _as_flat(values):
    return np.asarray(values, dtype=np.float64).reshape(-1)


def calibration_error(probabilities, targets, bins=15):
    probabilities = _as_flat(probabilities)
    targets = _as_flat(targets)
    if probabilities.size != targets.size or probabilities.size == 0:
        raise ValueError("probabilities and targets must have equal non-zero size")
    boundaries = np.linspace(0, 1, bins + 1)
    error = 0.0
    for index in range(bins):
        include_right = index == bins - 1
        selected = (probabilities >= boundaries[index]) & (
            probabilities <= boundaries[index + 1]
            if include_right
            else probabilities < boundaries[index + 1]
        )
        if selected.any():
            error += selected.mean() * abs(
                targets[selected].mean() - probabilities[selected].mean()
            )
    return float(error)


def optimal_dice_threshold(probabilities, targets, thresholds=None):
    probabilities = _as_flat(probabilities)
    targets = _as_flat(targets) >= 0.5
    thresholds = np.asarray(
        thresholds if thresholds is not None else np.linspace(0.2, 0.8, 61)
    )
    best_threshold = 0.5
    best_dice = -1.0
    for threshold in thresholds:
        prediction = probabilities >= threshold
        intersection = np.logical_and(prediction, targets).sum()
        dice = (2 * intersection + 1e-8) / (prediction.sum() + targets.sum() + 1e-8)
        if dice > best_dice:
            best_threshold = float(threshold)
            best_dice = float(dice)
    return best_threshold, best_dice


def uncertainty_review_threshold(variances, errors, review_fraction=0.1):
    variances = _as_flat(variances)
    errors = _as_flat(errors) >= 0.5
    if variances.size != errors.size or variances.size == 0:
        raise ValueError("variances and errors must have equal non-zero size")
    if not 0 < review_fraction < 1:
        raise ValueError("review_fraction must be between 0 and 1")
    threshold = float(np.quantile(variances, 1 - review_fraction))
    flagged = variances >= threshold
    captured = float(errors[flagged].sum() / max(int(errors.sum()), 1))
    return threshold, float(flagged.mean()), captured


def build_calibration_profile(
    probabilities,
    targets,
    variances,
    review_fraction=0.1,
    dataset_name="unspecified validation dataset",
):
    probabilities = _as_flat(probabilities)
    targets = _as_flat(targets)
    variances = _as_flat(variances)
    threshold, dice = optimal_dice_threshold(probabilities, targets)
    errors = (probabilities >= threshold) != (targets >= 0.5)
    uncertainty_threshold, actual_review, captured = uncertainty_review_threshold(
        variances, errors, review_fraction
    )
    return {
        "status": "validation-derived; not clinically validated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "sampled_pixel_count": int(probabilities.size),
        "decision_threshold": round(threshold, 6),
        "uncertainty_threshold": round(uncertainty_threshold, 8),
        "validation_dice": round(dice, 6),
        "expected_calibration_error": round(
            calibration_error(probabilities, targets), 6
        ),
        "target_review_fraction": review_fraction,
        "actual_review_fraction": round(actual_review, 6),
        "pixel_error_capture_fraction": round(captured, 6),
        "warning": (
            "Thresholds are valid only for the documented validation population and "
            "must not be treated as regulatory or clinical calibration."
        ),
    }

