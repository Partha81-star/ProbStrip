from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RegistrationResult:
    success: bool
    method: str
    score: float
    aligned_image: np.ndarray
    aligned_mask: np.ndarray | None
    matrix: np.ndarray
    message: str


def _gray_float(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.astype(np.float32) / 255.0


def register_followup(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    moving_mask: np.ndarray | None = None,
) -> RegistrationResult:
    """Affine-register a follow-up image to a reference using ECC."""
    height, width = reference_image.shape[:2]
    moving = cv2.resize(moving_image, (width, height), interpolation=cv2.INTER_AREA)
    reference_gray = _gray_float(reference_image)
    moving_gray = _gray_float(moving)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        150,
        1e-6,
    )

    try:
        score, warp = cv2.findTransformECC(
            reference_gray,
            moving_gray,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            3,
        )
        aligned = cv2.warpAffine(
            moving,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )
        aligned_mask = None
        if moving_mask is not None:
            resized_mask = cv2.resize(
                moving_mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            aligned_mask = cv2.warpAffine(
                resized_mask,
                warp,
                (width, height),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
            ).astype(bool)
        success = bool(np.isfinite(score) and score >= 0.45)
        message = (
            "Images aligned for research comparison."
            if success
            else "Alignment confidence is too low for change measurements."
        )
        return RegistrationResult(
            success,
            "ECC affine",
            round(float(score), 4),
            aligned,
            aligned_mask,
            warp,
            message,
        )
    except cv2.error:
        return RegistrationResult(
            False,
            "ECC affine",
            0.0,
            moving,
            moving_mask,
            warp,
            "The images could not be aligned. Compare them manually.",
        )


def vessel_change_map(reference_mask: np.ndarray, aligned_followup_mask: np.ndarray):
    reference = np.asarray(reference_mask, dtype=bool)
    followup = np.asarray(aligned_followup_mask, dtype=bool)
    appeared = np.logical_and(followup, np.logical_not(reference))
    disappeared = np.logical_and(reference, np.logical_not(followup))
    stable = np.logical_and(reference, followup)
    visualization = np.zeros((*reference.shape, 3), dtype=np.uint8)
    visualization[stable] = (30, 160, 145)
    visualization[appeared] = (37, 99, 235)
    visualization[disappeared] = (220, 70, 70)
    denominator = max(int(np.logical_or(reference, followup).sum()), 1)
    return {
        "stable": stable,
        "appeared": appeared,
        "disappeared": disappeared,
        "visualization": visualization,
        "changed_vessel_percent": round(
            float(np.logical_xor(reference, followup).sum()) / denominator * 100,
            2,
        ),
    }
