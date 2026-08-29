import numpy as np
import torch


def dice_similarity_coefficient(preds, targets, threshold=0.5, smooth=1e-6):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    binary_preds = (preds > threshold).astype(np.float32)
    binary_targets = (targets > threshold).astype(np.float32)

    intersection = np.sum(binary_preds * binary_targets)
    total = np.sum(binary_preds) + np.sum(binary_targets)
    return float((2.0 * intersection + smooth) / (total + smooth))


def intersection_over_union(preds, targets, threshold=0.5, smooth=1e-6):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    binary_preds = (preds > threshold).astype(np.float32)
    binary_targets = (targets > threshold).astype(np.float32)

    intersection = np.sum(binary_preds * binary_targets)
    union = np.sum(binary_preds) + np.sum(binary_targets) - intersection
    return float((intersection + smooth) / (union + smooth))


def expected_calibration_error(probs, targets, n_bins=10):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy().flatten()
    else:
        probs = np.asarray(probs).flatten()

    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy().flatten()
    else:
        targets = np.asarray(targets).flatten()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(probs)

    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        count = np.sum(in_bin)
        if count > 0:
            acc = np.mean(targets[in_bin])
            conf = np.mean(probs[in_bin])
            ece += np.abs(acc - conf) * (count / total_samples)

    return float(ece)
