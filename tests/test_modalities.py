import cv2
import numpy as np

from clinical.modalities import analyze_general_image, assess_general_image_quality


def test_general_imaging_creates_same_size_review_views():
    image = np.zeros((180, 220, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 15), (170, 165), (170, 170, 170), thickness=-1)
    cv2.circle(image, (110, 90), 32, (45, 45, 45), thickness=-1)

    result = analyze_general_image(image, "Bone or joint X-ray")

    assert result["enhanced"].shape == image.shape[:2]
    assert result["overlay"].shape == image.shape
    assert result["edge_area_percent"] > 0


def test_blank_general_image_is_marked_poor_quality():
    result = assess_general_image_quality(np.zeros((128, 128), dtype=np.uint8))

    assert result["status"] == "Poor technical quality"
    assert result["score"] < 60
