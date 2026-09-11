import cv2
import numpy as np


MODALITIES = {
    "Bone or joint X-ray": {
        "short": "bone X-ray",
        "view": "Bone and joint structure view",
        "guidance": "Use the original radiograph export when possible. Include the full bone or joint and its side marker.",
    },
    "Chest X-ray": {
        "short": "chest X-ray",
        "view": "Chest structure view",
        "guidance": "Include the full chest, side marker, and exposure field. Do not crop the lung edges.",
    },
    "CT image": {
        "short": "CT image",
        "view": "CT structure view",
        "guidance": "Upload a de-identified PNG or JPEG export using the clinically relevant window.",
    },
    "MRI image": {
        "short": "MRI image",
        "view": "MRI structure view",
        "guidance": "Upload a de-identified export and preserve the sequence and orientation outside this prototype.",
    },
    "Ultrasound image": {
        "short": "ultrasound image",
        "view": "Ultrasound structure view",
        "guidance": "Use an original still frame with depth and orientation labels visible.",
    },
    "Skin or external photo": {
        "short": "external photograph",
        "view": "Surface detail view",
        "guidance": "Use even lighting, include a scale when appropriate, and avoid identifying features.",
    },
}


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)


def assess_general_image_quality(image: np.ndarray) -> dict:
    """Describe technical image quality without assessing anatomy or disease."""
    gray = _as_gray(image)
    brightness = float(np.mean(gray) / 255.0)
    contrast = float(np.std(gray) / 255.0)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped_dark = float(np.mean(gray <= 3))
    clipped_light = float(np.mean(gray >= 252))

    checks = [
        {
            "name": "Brightness",
            "value": round(brightness * 100, 1),
            "unit": "%",
            "ok": 0.08 <= brightness <= 0.92,
        },
        {
            "name": "Contrast",
            "value": round(contrast * 100, 1),
            "unit": "%",
            "ok": contrast >= 0.06,
        },
        {
            "name": "Sharpness",
            "value": round(sharpness, 1),
            "unit": "score",
            "ok": sharpness >= 12,
        },
        {
            "name": "Clipped dark area",
            "value": round(clipped_dark * 100, 1),
            "unit": "%",
            "ok": clipped_dark <= 0.45,
        },
        {
            "name": "Clipped light area",
            "value": round(clipped_light * 100, 1),
            "unit": "%",
            "ok": clipped_light <= 0.45,
        },
    ]
    passed = sum(check["ok"] for check in checks)
    score = int(round(100 * passed / len(checks)))
    status = "Good technical quality" if passed == len(checks) else (
        "Usable with caution" if passed >= 3 else "Poor technical quality"
    )
    return {"score": score, "status": status, "checks": checks}


def create_structure_views(
    image: np.ndarray,
    clip_limit: float = 2.5,
    edge_sensitivity: float = 1.0,
    invert: bool = False,
) -> dict:
    """Create modality-neutral enhancement and edge views for human review."""
    gray = _as_gray(image)
    working = cv2.bitwise_not(gray) if invert else gray
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(working)
    smooth = cv2.GaussianBlur(enhanced, (5, 5), 0)
    median = max(float(np.median(smooth)), 1.0)
    scale = max(edge_sensitivity, 0.25)
    lower = int(np.clip((0.55 / scale) * median, 5, 220))
    upper = int(np.clip((1.35 / scale) * median, lower + 1, 255))
    edges = cv2.Canny(smooth, lower, upper)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    base_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    overlay = base_rgb.copy()
    edge_pixels = edges > 0
    overlay[edge_pixels] = (
        0.35 * overlay[edge_pixels] + 0.65 * np.array([14, 165, 164])
    ).astype(np.uint8)

    return {
        "enhanced": enhanced,
        "edges": edges,
        "overlay": overlay,
        "edge_area_percent": round(float(np.mean(edge_pixels) * 100), 2),
    }


def analyze_general_image(
    image: np.ndarray,
    modality: str,
    clip_limit: float = 2.5,
    edge_sensitivity: float = 1.0,
    invert: bool = False,
) -> dict:
    if modality not in MODALITIES:
        raise ValueError(f"Unsupported imaging type: {modality}")
    return {
        "modality": modality,
        "quality": assess_general_image_quality(image),
        **create_structure_views(image, clip_limit, edge_sensitivity, invert),
    }
