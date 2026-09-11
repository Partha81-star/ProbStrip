import cv2
import numpy as np

from clinical.biomarkers import calculate_vascular_biomarkers
from clinical.quality import retinal_field_mask


def model_input_channel(rgb_image: np.ndarray) -> np.ndarray:
    """Use the green fundus channel while preserving color for display."""
    if rgb_image.ndim == 2:
        return rgb_image
    return rgb_image[:, :, 1]


def resize_scan(image: np.ndarray, size=(256, 256)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def build_visuals(
    rgb_image: np.ndarray,
    mean_prediction: np.ndarray,
    variance_map: np.ndarray,
    decision_threshold: float,
    uncertainty_threshold: float,
):
    binary = mean_prediction >= decision_threshold
    uncertain = variance_map >= uncertainty_threshold

    overlay = rgb_image.copy()
    vessel_color = np.zeros_like(overlay)
    vessel_color[:, :] = (20, 184, 166)
    overlay[binary] = (
        0.58 * overlay[binary] + 0.42 * vessel_color[binary]
    ).astype(np.uint8)
    overlay[uncertain] = (
        0.45 * overlay[uncertain] + 0.55 * np.array([245, 158, 11])
    ).astype(np.uint8)

    uncertainty = np.clip(
        variance_map / max(float(np.percentile(variance_map, 99)), 1e-8), 0, 1
    )
    heat = cv2.applyColorMap((uncertainty * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return binary, uncertain, overlay, heat


def calculate_research_measures(
    rgb_image: np.ndarray,
    binary_mask: np.ndarray,
    uncertain_mask: np.ndarray,
    variance_map: np.ndarray,
):
    field = retinal_field_mask(rgb_image)
    field_pixels = max(int(field.sum()), 1)
    vessel_density = float(np.logical_and(binary_mask, field).sum() / field_pixels)
    uncertainty_ratio = float(np.logical_and(uncertain_mask, field).sum() / field_pixels)

    h, w = binary_mask.shape
    yy, xx = np.ogrid[:h, :w]
    radius = min(h, w) * 0.25
    central = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 <= radius**2
    central_field = np.logical_and(central, field)
    central_density = float(
        np.logical_and(binary_mask, central_field).sum()
        / max(int(central_field.sum()), 1)
    )

    measures = {
        "visible_vessel_coverage_percent": round(vessel_density * 100, 2),
        "central_vessel_coverage_percent": round(central_density * 100, 2),
        "low_confidence_area_percent": round(uncertainty_ratio * 100, 2),
        "mean_model_variance": round(float(np.mean(variance_map[field])), 7),
        "maximum_model_variance": round(float(np.max(variance_map[field])), 7),
    }
    measures.update(calculate_vascular_biomarkers(rgb_image, binary_mask))
    return measures


def review_outcome(quality_status: str, low_confidence_percent: float):
    if quality_status == "Retake recommended":
        return {
            "level": "retake",
            "title": "A clearer retinal image is needed",
            "explanation": (
                "The capture-quality check found issues that can make the vessel map "
                "unreliable. This result should not be interpreted clinically."
            ),
            "next_step": "Ask the imaging professional to repeat the retinal photograph.",
        }
    if low_confidence_percent >= 12:
        return {
            "level": "review",
            "title": "Clinician review is especially important",
            "explanation": (
                "The model was unsure in a noticeable part of the image. Amber areas "
                "show where its vessel map needs closer review."
            ),
            "next_step": "Share the original image and this report with an eye-care professional.",
        }
    return {
        "level": "ready",
        "title": "Vessel map ready for clinician review",
        "explanation": (
            "The software produced a vessel map with limited flagged uncertainty. "
            "Only a qualified clinician can interpret what it means for your health."
        ),
        "next_step": "Discuss the image during your normal eye-care appointment.",
    }
