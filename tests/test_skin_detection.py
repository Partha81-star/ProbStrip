import cv2
import numpy as np

from clinical.skin_detection import detect_skin_lesion
from clinical.general_reporting import make_general_report_payload
from clinical.pdf_reporting import general_report_as_pdf


def test_benign_skin_lesion():
    # Synthetic circular symmetric uniform mole
    img = np.full((300, 300, 3), 220, dtype=np.uint8)
    cv2.circle(img, (150, 150), 45, (60, 45, 35), -1)

    result = detect_skin_lesion(img)
    finding = result["finding"]
    assert finding["status"] == "evaluated"
    assert finding["risk_level"] == "Low"
    assert "Benign" in finding["primary_condition"]
    assert result["mask"].sum() > 0


def test_atypical_skin_lesion():
    # Synthetic asymmetric irregular multi-tone lesion
    img = np.full((300, 300, 3), 220, dtype=np.uint8)
    cv2.ellipse(img, (150, 150), (60, 25), 35, 0, 360, (40, 30, 20), -1)
    # Irregular borders and multi-color blotches
    cv2.circle(img, (175, 130), 20, (20, 15, 10), -1)
    cv2.circle(img, (130, 165), 18, (120, 40, 30), -1)

    result = detect_skin_lesion(img)
    finding = result["finding"]
    assert finding["status"] == "evaluated"
    assert finding["risk_level"] == "High"
    assert "atypical" in finding["primary_condition"].lower() or "melanoma" in finding["primary_condition"].lower()


def test_skin_pdf_generation():
    img = np.full((300, 300, 3), 220, dtype=np.uint8)
    cv2.circle(img, (150, 150), 45, (60, 45, 35), -1)

    result = detect_skin_lesion(img)
    payload = make_general_report_payload(
        case_id="ISIC-TEST-001",
        modality="Skin or external photo",
        quality={"score": 85, "checks": []},
        edge_area=1.8,
        fingerprint="isic12345678",
        image=img,
        automated_diagnosis=result["finding"],
    )

    pdf_bytes = general_report_as_pdf(payload, img, result["overlay"])
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
