from dataclasses import asdict, dataclass

import cv2
import numpy as np

QUALITY_POLICY_VERSION = "2026.09.11"


@dataclass(frozen=True)
class QualityCheck:
    name: str
    value: float
    unit: str
    status: str
    guidance: str


@dataclass(frozen=True)
class ImageQualityResult:
    score: int
    status: str
    summary: str
    checks: tuple[QualityCheck, ...]
    policy_version: str = QUALITY_POLICY_VERSION

    def to_dict(self):
        result = asdict(self)
        result["checks"] = [asdict(check) for check in self.checks]
        return result


def _largest_component_mask(candidate: np.ndarray) -> np.ndarray | None:
    candidate = candidate.astype(np.uint8) * 255
    kernel_size = max(3, int(round(min(candidate.shape) * 0.01)) | 1)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        np.ones((kernel_size, kernel_size), np.uint8),
    )
    contours, _ = cv2.findContours(
        candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(candidate, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask > 0


def _largest_foreground_mask(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image)

    # Fundus photographs are chromatic while screenshot borders and labels are
    # usually white, gray, or black. Starting with saturation prevents those
    # borders from being joined to the retinal field during morphology.
    if image.ndim == 3 and image.shape[2] == 3:
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
        chromatic = (hsv[:, :, 1] >= 20) & (hsv[:, :, 2] >= 12)
        color_mask = _largest_component_mask(chromatic)
        if color_mask is not None and float(color_mask.mean()) >= 0.08:
            return color_mask

    threshold = max(5, int(np.percentile(gray, 2)))
    intensity_mask = _largest_component_mask(gray > threshold)
    if intensity_mask is None:
        return np.ones_like(gray, dtype=bool)
    return intensity_mask


def retinal_field_mask(image: np.ndarray) -> np.ndarray:
    return _largest_foreground_mask(image)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _check(name, value, unit, good, acceptable, guidance):
    if good(value):
        status = "Good"
    elif acceptable(value):
        status = "Review"
    else:
        status = "Retake"
    return QualityCheck(name, float(value), unit, status, guidance)


def summarize_quality(checks, score):
    retake_checks = [check.name for check in checks if check.status == "Retake"]
    review_checks = [check.name for check in checks if check.status == "Review"]
    retake_count = len(retake_checks)

    if score < 55 or (retake_count >= 2 and score < 75):
        reasons = ", ".join(retake_checks or review_checks)
        return (
            "Retake recommended",
            "Mapping was stopped because these capture checks need improvement: "
            f"{reasons}.",
        )
    if retake_count >= 1 or review_checks or score < 82:
        reasons = ", ".join(retake_checks + review_checks)
        return (
            "Usable with caution",
            "The image can be mapped, but a clinician should review: "
            f"{reasons or 'overall capture quality'}.",
        )
    return (
        "Suitable for mapping",
        "The image passed the automated capture-quality checks.",
    )


def assess_image_quality(image: np.ndarray) -> ImageQualityResult:
    """Estimate acquisition quality without making a clinical judgment."""
    gray = _to_gray(image)
    field = _largest_foreground_mask(image)
    pixels = gray[field] if np.any(field) else gray.reshape(-1)

    brightness = float(np.mean(pixels) / 255.0)
    contrast = float(np.std(pixels) / 255.0)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F)[field].var())
    glare = float(np.mean(pixels >= 248))
    coverage = float(np.mean(field))

    checks = (
        _check(
            "Brightness",
            brightness * 100,
            "%",
            lambda value: 20 <= value <= 80,
            lambda value: 10 <= value <= 90,
            "Use even lighting and avoid an image that is very dark or washed out.",
        ),
        _check(
            "Contrast",
            contrast * 100,
            "%",
            lambda value: value >= 12,
            lambda value: value >= 7,
            "Make sure vessels are visibly distinct from the retinal background.",
        ),
        _check(
            "Sharpness",
            sharpness,
            "score",
            lambda value: value >= 45,
            lambda value: value >= 18,
            "Hold the imaging device steady and focus on the retina.",
        ),
        _check(
            "Glare",
            glare * 100,
            "%",
            lambda value: value <= 3,
            lambda value: value <= 10,
            "Reduce reflections and bright white patches before repeating the image.",
        ),
        _check(
            "Retina framing",
            coverage * 100,
            "%",
            lambda value: 35 <= value <= 92,
            lambda value: 20 <= value <= 98,
            "Center the circular retinal field and keep its edges visible.",
        ),
    )

    points = {"Good": 20, "Review": 11, "Retake": 2}
    score = int(round(sum(points[check.status] for check in checks)))
    # A high aggregate score must never produce a contradictory hard stop. Two
    # genuinely poor checks can stop a low-scoring image, while a single weak
    # check always allows mapping with an explicit caution.
    status, summary = summarize_quality(checks, score)

    return ImageQualityResult(score, status, summary, checks)
