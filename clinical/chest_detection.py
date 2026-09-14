"""PneumoniaMNIST inference for the supported chest X-ray workflow."""

import cv2
import numpy as np
import torch


PNEUMONIA_THRESHOLD = 0.29


def detect_pneumonia(image, model, threshold=PNEUMONIA_THRESHOLD):
    if image is None or image.ndim not in (2, 3):
        raise ValueError("A readable chest X-ray image is required.")
    if not 0 < threshold < 1:
        raise ValueError("The pneumonia threshold must be between zero and one.")

    gray = (
        cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if image.ndim == 3
        else image.astype(np.uint8)
    )
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().div_(255.0).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        score = float(torch.sigmoid(model(tensor).squeeze()).item())

    positive = score >= threshold
    score_percent = round(score * 100, 1)
    if positive:
        finding = {
            "status": "evaluated",
            "primary_condition": "Pneumonia pattern detected",
            "risk_level": "High" if score >= 0.75 else "Moderate",
            "risk_score": score_percent,
            "model_score": score_percent,
            "score_label": "Pneumonia model score",
            "summary": (
                "The chest X-ray classifier produced a score above its validated "
                f"operating threshold of {threshold * 100:.0f}%."
            ),
            "recommendation": (
                "Arrange prompt medical assessment, especially with fever, cough, "
                "breathing difficulty, low oxygen level, or worsening symptoms."
            ),
        }
    else:
        finding = {
            "status": "evaluated",
            "primary_condition": "No pneumonia pattern detected",
            "risk_level": "Low",
            "risk_score": score_percent,
            "model_score": score_percent,
            "score_label": "Pneumonia model score",
            "summary": (
                "The chest X-ray classifier produced a score below its validated "
                f"operating threshold of {threshold * 100:.0f}%."
            ),
            "recommendation": (
                "A low score does not exclude pneumonia or another chest condition. "
                "Seek medical assessment when symptoms are persistent or severe."
            ),
        }

    finding.update(
        {
            "differential_diagnoses": [],
            "detections": [],
            "model": "PneumoniaMNIST compact classifier",
            "population_scope": "Pediatric pneumonia-versus-normal chest X-ray screening",
        }
    )
    return {"finding": finding}
