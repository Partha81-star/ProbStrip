import cv2
import numpy as np

from clinical.quality import retinal_field_mask


def skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """Return a morphological skeleton without optional OpenCV contrib modules."""
    image = (np.asarray(binary_mask) > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(image):
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = cv2.erode(image, element)

    return skeleton > 0


def _cluster_count(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return int(sum(stats[index, cv2.CC_STAT_AREA] >= 1 for index in range(1, count)))


def fractal_dimension(binary_mask: np.ndarray) -> float:
    """Estimate 2D box-counting dimension for a binary vessel map."""
    image = np.asarray(binary_mask, dtype=bool)
    if min(image.shape) < 4 or image.sum() < 2:
        return 0.0

    max_power = int(np.floor(np.log2(min(image.shape))))
    sizes = 2 ** np.arange(1, max_power)
    counts = []
    valid_sizes = []
    for size in sizes:
        padded_h = int(np.ceil(image.shape[0] / size) * size)
        padded_w = int(np.ceil(image.shape[1] / size) * size)
        padded = np.zeros((padded_h, padded_w), dtype=bool)
        padded[: image.shape[0], : image.shape[1]] = image
        blocks = padded.reshape(
            padded_h // size, size, padded_w // size, size
        ).any(axis=(1, 3))
        count = int(blocks.sum())
        if count > 0:
            valid_sizes.append(size)
            counts.append(count)

    if len(counts) < 2:
        return 0.0
    slope = np.polyfit(np.log(1 / np.asarray(valid_sizes)), np.log(counts), 1)[0]
    return float(max(0.0, min(2.0, slope)))


def calculate_vascular_biomarkers(
    rgb_image: np.ndarray, binary_mask: np.ndarray
) -> dict[str, float | int]:
    """Calculate descriptive research morphology, not diagnostic biomarkers."""
    field = retinal_field_mask(rgb_image)
    vessels = np.logical_and(np.asarray(binary_mask, dtype=bool), field)
    skeleton = skeletonize(vessels)

    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighborhood = cv2.filter2D(
        skeleton.astype(np.uint8), cv2.CV_16U, neighbor_kernel
    )
    endpoint_pixels = np.logical_and(skeleton, neighborhood == 2)
    branch_pixels = np.logical_and(skeleton, neighborhood >= 4)

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        vessels.astype(np.uint8), connectivity=8
    )
    meaningful_components = sum(
        stats[index, cv2.CC_STAT_AREA] >= 4
        for index in range(1, component_count)
    )

    vessel_area = int(vessels.sum())
    skeleton_length = int(skeleton.sum())
    mean_width = vessel_area / max(skeleton_length, 1)
    field_area = max(int(field.sum()), 1)

    return {
        "fractal_dimension": round(fractal_dimension(skeleton), 4),
        "skeleton_length_pixels": skeleton_length,
        "estimated_mean_width_pixels": round(float(mean_width), 3),
        "endpoint_region_count": _cluster_count(endpoint_pixels),
        "branch_region_count": _cluster_count(branch_pixels),
        "vessel_component_count": int(meaningful_components),
        "skeleton_density_percent": round(skeleton_length / field_area * 100, 3),
    }


def remove_small_components(binary_mask: np.ndarray, minimum_area: int) -> np.ndarray:
    mask = (np.asarray(binary_mask) > 0).astype(np.uint8)
    if minimum_area <= 1:
        return mask.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask, dtype=bool)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == index] = True
    return cleaned


def reviewed_mask(probability: np.ndarray, threshold: float, minimum_area: int):
    initial = np.asarray(probability) >= threshold
    return remove_small_components(initial, minimum_area)

