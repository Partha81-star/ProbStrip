import numpy as np

from training.metrics import sensitivity, specificity


def test_sensitivity_and_specificity():
    probabilities = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.6])
    targets = np.array([1, 1, 0, 0, 0, 1])

    assert sensitivity(probabilities, targets) == 1.0
    assert np.isclose(specificity(probabilities, targets), 2 / 3)
