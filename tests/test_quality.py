import cv2
import numpy as np

from clinical.quality import (
    QualityCheck,
    assess_image_quality,
    retinal_field_mask,
    summarize_quality,
)


def test_blank_image_is_stopped_by_quality_gate():
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    result = assess_image_quality(image)

    assert result.status == "Retake recommended"
    assert result.score < 55


def test_fundus_field_mask_finds_circular_image_area():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(image, (128, 128), 105, (90, 115, 80), thickness=-1)

    mask = retinal_field_mask(image)

    assert 0.45 < float(mask.mean()) < 0.65
    assert mask[128, 128]
    assert not mask[0, 0]


def test_fundus_field_ignores_white_screenshot_border():
    image = np.full((300, 360, 3), 245, dtype=np.uint8)
    image[:, 20:340] = 3
    cv2.circle(image, (180, 150), 130, (225, 92, 20), thickness=-1)
    cv2.circle(image, (180, 150), 75, (190, 65, 18), thickness=-1)
    cv2.circle(image, (265, 135), 18, (250, 185, 75), thickness=-1)
    cv2.line(image, (70, 150), (285, 125), (85, 30, 15), thickness=4)

    result = assess_image_quality(image)
    framing = next(check for check in result.checks if check.name == "Retina framing")

    assert 45 < framing.value < 75
    assert framing.status == "Good"
    assert result.status != "Retake recommended"


def test_quality_result_is_serializable():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(image, (64, 64), 52, (80, 110, 75), thickness=-1)
    cv2.line(image, (20, 64), (108, 64), (20, 25, 20), thickness=3)

    payload = assess_image_quality(image).to_dict()

    assert payload["status"] in {
        "Suitable for mapping",
        "Usable with caution",
        "Retake recommended",
    }
    assert len(payload["checks"]) == 5


def test_one_failed_check_allows_cautious_mapping_when_rest_are_strong():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(image, (128, 128), 105, (170, 90, 35), thickness=-1)
    cv2.circle(image, (128, 128), 45, (255, 255, 255), thickness=-1)
    for offset in range(-70, 71, 20):
        cv2.line(image, (45, 128 + offset), (210, 128 - offset), (60, 22, 10), 2)

    result = assess_image_quality(image)
    retakes = sum(check.status == "Retake" for check in result.checks)

    assert retakes == 1
    assert result.score >= 55
    assert result.status == "Usable with caution"


def test_high_score_can_never_be_labeled_retake():
    checks = tuple(
        QualityCheck(f"Check {index}", 0, "score", status, "Review capture.")
        for index, status in enumerate(
            ["Retake", "Retake", "Good", "Good", "Good"], start=1
        )
    )

    status, _ = summarize_quality(checks, score=82)

    assert status == "Usable with caution"
