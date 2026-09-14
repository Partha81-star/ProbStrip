import numpy as np
import torch

from clinical.chest_detection import detect_pneumonia


class _ConstantModel(torch.nn.Module):
    def __init__(self, probability):
        super().__init__()
        probability = torch.tensor(float(probability))
        self.logit = torch.logit(probability)

    def forward(self, inputs):
        assert inputs.shape == (1, 1, 28, 28)
        return self.logit.reshape(1, 1)


def test_pneumonia_detector_reports_positive_model_score():
    image = np.full((120, 160, 3), 100, dtype=np.uint8)

    result = detect_pneumonia(image, _ConstantModel(0.83))["finding"]

    assert result["primary_condition"] == "Pneumonia pattern detected"
    assert result["model_score"] == 83.0
    assert result["score_label"] == "Pneumonia model score"


def test_pneumonia_detector_preserves_negative_result_limit():
    image = np.full((120, 160, 3), 100, dtype=np.uint8)

    result = detect_pneumonia(image, _ConstantModel(0.1))["finding"]

    assert result["primary_condition"] == "No pneumonia pattern detected"
    assert result["model_score"] == 10.0
    assert "does not exclude" in result["recommendation"]
