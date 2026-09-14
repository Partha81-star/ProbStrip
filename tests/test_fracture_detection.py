import numpy as np
import torch

from clinical.fracture_detection import detect_fractures


class _Boxes:
    def __init__(self, boxes, confidences):
        self.xyxy = torch.tensor(boxes, dtype=torch.float32)
        self.conf = torch.tensor(confidences, dtype=torch.float32)


class _Prediction:
    def __init__(self, boxes):
        self.boxes = boxes


class _Model:
    def __init__(self, boxes, confidences):
        self.boxes = boxes
        self.confidences = confidences

    def predict(self, **kwargs):
        assert kwargs["conf"] == 0.25
        assert kwargs["device"] == "cpu"
        return [_Prediction(_Boxes(self.boxes, self.confidences))]


def test_fracture_detector_reports_and_draws_detected_box():
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    result = detect_fractures(image, _Model([[10, 12, 70, 65]], [0.864]))

    assert result["finding"]["primary_condition"] == "Fracture pattern detected"
    assert result["finding"]["risk_score"] == 86.4
    assert result["finding"]["detections"][0]["box"] == [10.0, 12.0, 70.0, 65.0]
    assert np.any(result["overlay"] != image)


def test_fracture_detector_does_not_turn_a_negative_result_into_a_rule_out():
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    result = detect_fractures(image, _Model([], []))

    assert result["finding"]["primary_condition"] == "No fracture pattern detected"
    assert "does not rule out" in result["finding"]["recommendation"]
    assert np.array_equal(result["overlay"], image)
