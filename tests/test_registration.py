import cv2
import numpy as np

from clinical.registration import register_followup, vessel_change_map


def _reference_image():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(image, (64, 64), 54, (95, 120, 85), thickness=-1)
    cv2.line(image, (20, 64), (108, 64), (20, 35, 20), thickness=4)
    cv2.line(image, (64, 20), (64, 108), (30, 45, 25), thickness=3)
    cv2.circle(image, (82, 52), 7, (180, 190, 160), thickness=-1)
    return image


def test_registration_aligns_translated_followup():
    reference = _reference_image()
    transform = np.float32([[1, 0, 5], [0, 1, -4]])
    moving = cv2.warpAffine(reference, transform, (128, 128))
    before = np.mean(np.abs(reference.astype(float) - moving.astype(float)))

    result = register_followup(reference, moving)
    after = np.mean(
        np.abs(reference.astype(float) - result.aligned_image.astype(float))
    )

    assert result.success
    assert result.score > 0.8
    assert after < before


def test_change_map_labels_stable_and_changed_pixels():
    reference = np.zeros((16, 16), dtype=bool)
    followup = np.zeros_like(reference)
    reference[4:8, 4] = True
    followup[4:8, 4] = True
    followup[8:12, 8] = True

    change = vessel_change_map(reference, followup)

    assert change["stable"].sum() == 4
    assert change["appeared"].sum() == 4
    assert change["disappeared"].sum() == 0
    assert change["changed_vessel_percent"] == 50.0
