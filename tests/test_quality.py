import cv2
import numpy as np

from clinical.quality import assess_image_quality, retinal_field_mask


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

