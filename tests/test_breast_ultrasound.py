import cv2
import numpy as np

from clinical.breast_ultrasound import detect_breast_ultrasound
from clinical.general_reporting import make_general_report_payload
from clinical.pdf_reporting import general_report_as_pdf


def test_normal_breast_ultrasound():
    # Synthetic normal ultrasound: smooth horizontal tissue layers without focal mass
    img = np.full((300, 400, 3), 120, dtype=np.uint8)
    # Add subtle tissue texture
    img[40:80, :] = 140   # fibroglandular
    img[80:120, :] = 110  # parenchymal
    img[120:200, :] = 135 # deep glandular

    result = detect_breast_ultrasound(img)
    finding = result["finding"]
    assert finding["status"] == "evaluated"
    assert finding["risk_level"] == "Low"
    assert "Normal" in finding["primary_condition"]
    assert finding["model_score"] >= 80.0
    assert result["overlay"].shape == img.shape
    assert result["mask"].shape == (300, 400)


def test_benign_looking_breast_lesion():
    # Synthetic benign lesion: wider-than-tall (horizontal) circumscribed hypoechoic mass
    img = np.full((300, 400, 3), 130, dtype=np.uint8)
    # Draw oval hypoechoic mass (width 80, height 40 -> W/H = 2.0)
    cv2.ellipse(img, (200, 150), (40, 20), 0, 0, 360, (45, 45, 45), -1)

    result = detect_breast_ultrasound(img)
    finding = result["finding"]
    assert finding["status"] == "evaluated"
    assert finding["risk_level"] == "Moderate"
    assert "Benign" in finding["primary_condition"]
    assert len(finding["detections"]) == 1
    assert finding["detections"][0]["aspect_ratio"] > 1.0
    assert result["mask"].sum() > 0


def test_malignant_looking_breast_lesion():
    # Synthetic malignant lesion: taller-than-wide (vertical) irregular hypoechoic mass
    img = np.full((300, 400, 3), 140, dtype=np.uint8)
    # Draw vertical irregular nodule (width 26, height 60 -> W/H = 0.43)
    cv2.ellipse(img, (200, 150), (13, 30), 0, 0, 360, (30, 30, 30), -1)
    # Add spiculation / irregularity
    cv2.circle(img, (185, 140), 10, (30, 30, 30), -1)
    cv2.circle(img, (215, 160), 10, (30, 30, 30), -1)

    result = detect_breast_ultrasound(img)
    finding = result["finding"]
    assert finding["status"] == "evaluated"
    assert finding["risk_level"] == "High"
    assert "malignant" in finding["primary_condition"].lower() or "suspicious" in finding["primary_condition"].lower()
    assert len(finding["detections"]) == 1


def test_breast_ultrasound_pdf_generation():
    img = np.full((300, 400, 3), 120, dtype=np.uint8)
    cv2.ellipse(img, (200, 150), (40, 20), 0, 0, 360, (45, 45, 45), -1)

    result = detect_breast_ultrasound(img)
    payload = make_general_report_payload(
        case_id="BUSI-TEST-001",
        modality="Breast ultrasound",
        quality={"score": 90, "checks": []},
        edge_area=2.5,
        fingerprint="busi12345678",
        image=img,
        automated_diagnosis=result["finding"],
    )

    pdf_bytes = general_report_as_pdf(payload, img, result["overlay"])
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
