import cv2
import numpy as np

from clinical.biomarkers import (
    calculate_vascular_biomarkers,
    fractal_dimension,
    reviewed_mask,
    skeletonize,
)


def test_skeletonize_reduces_thick_line_to_centerline():
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.line(mask, (8, 32), (55, 32), 1, thickness=7)

    skeleton = skeletonize(mask)

    assert 35 <= int(skeleton.sum()) <= 60
    assert skeleton.sum() < mask.sum()


def test_biomarkers_describe_branching_map():
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.circle(image, (48, 48), 44, (100, 120, 90), thickness=-1)
    mask = np.zeros((96, 96), dtype=np.uint8)
    cv2.line(mask, (15, 48), (80, 48), 1, 3)
    cv2.line(mask, (48, 48), (48, 15), 1, 3)

    measures = calculate_vascular_biomarkers(image, mask)

    assert measures["skeleton_length_pixels"] > 60
    assert measures["branch_region_count"] >= 1
    assert 0 <= measures["fractal_dimension"] <= 2


def test_reviewed_mask_changes_threshold_and_removes_small_regions():
    probability = np.zeros((32, 32), dtype=np.float32)
    probability[5:15, 5:15] = 0.8
    probability[25, 25] = 0.9

    mask = reviewed_mask(probability, threshold=0.7, minimum_area=4)

    assert mask[8, 8]
    assert not mask[25, 25]


def test_empty_fractal_dimension_is_zero():
    assert fractal_dimension(np.zeros((32, 32), dtype=bool)) == 0.0

