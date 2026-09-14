from pathlib import Path

import cv2
import numpy as np


def detect_fractures(image, model, confidence_threshold=0.25):
    """Run FracAtlas localization and return report-ready detections."""
    if image is None or image.ndim != 3:
        raise ValueError("A color X-ray image is required for fracture detection.")

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    prediction = model.predict(
        source=bgr,
        conf=confidence_threshold,
        imgsz=640,
        device="cpu",
        verbose=False,
    )[0]
    boxes = prediction.boxes
    detections = []
    if boxes is not None:
        coordinates = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        for xyxy, confidence in zip(coordinates, confidences):
            detections.append(
                {
                    "box": [round(float(value), 1) for value in xyxy],
                    "confidence": round(float(confidence), 4),
                }
            )

    overlay = image.copy()
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = (int(value) for value in detection["box"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (220, 38, 38), 3)
        label = f"Fracture {index}: {detection['confidence'] * 100:.1f}%"
        cv2.putText(
            overlay,
            label,
            (max(4, x1), max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 38, 38),
            2,
            cv2.LINE_AA,
        )

    confidence = max((item["confidence"] for item in detections), default=0.0)
    if detections:
        finding = {
            "status": "evaluated",
            "primary_condition": "Fracture pattern detected",
            "risk_level": "High",
            "risk_score": round(confidence * 100, 1),
            "summary": f"The model localized {len(detections)} fracture candidate(s) in the X-ray.",
            "recommendation": "Arrange prompt orthopedic or emergency assessment and correlate with the original radiograph and examination.",
            "differential_diagnoses": [],
            "detections": detections,
            "model": "FracAtlas YOLOv8 localization",
        }
    else:
        finding = {
            "status": "evaluated",
            "primary_condition": "No fracture pattern detected",
            "risk_level": "Low",
            "risk_score": 0.0,
            "summary": "The model did not localize a fracture candidate above its configured confidence threshold.",
            "recommendation": "A negative model result does not rule out fracture; use clinical assessment when symptoms or injury history remain concerning.",
            "differential_diagnoses": [],
            "detections": [],
            "model": "FracAtlas YOLOv8 localization",
        }

    return {"finding": finding, "overlay": overlay}


def fracture_model_available(model_path):
    return Path(model_path).is_file()
