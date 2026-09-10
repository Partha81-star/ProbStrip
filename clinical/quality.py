from dataclasses import asdict, dataclass

import cv2
import numpy as np


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

    def to_dict(self):
        result = asdict(self)
        result["checks"] = [asdict(check) for check in self.checks]
        return result


def _largest_foreground_mask(gray: np.ndarray) -> np.ndarray:
    threshold = max(5, int(np.percentile(gray, 2)))
    candidate = (gray > threshold).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
    )
    contours, _ = cv2.findContours(
        candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.ones_like(gray, dtype=bool)
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask > 0


def retinal_field_mask(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image)
    return _largest_foreground_mask(gray)


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


def assess_image_quality(image: np.ndarray) -> ImageQualityResult:
    """Estimate acquisition quality without making a clinical judgment."""
    gray = _to_gray(image)
    field = _largest_foreground_mask(gray)
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
    statuses = {check.status for check in checks}
    if "Retake" in statuses or score < 55:
        status = "Retake recommended"
        summary = "The image may not be reliable enough for vessel mapping."
    elif "Review" in statuses or score < 82:
        status = "Usable with caution"
        summary = "The image can be mapped, but a clinician should review its quality."
    else:
        status = "Suitable for mapping"
        summary = "The image passed the automated capture-quality checks."

    return ImageQualityResult(score, status, summary, checks)

