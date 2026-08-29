import torch
import torch.nn as nn
import numpy as np


class StochasticInferenceEngine:
    def __init__(
        self,
        model,
        num_samples=20,
        decision_threshold=0.5,
        uncertainty_threshold=0.03,
        device="cpu",
    ):
        self.model = model.to(device)
        self.num_samples = num_samples
        self.decision_threshold = decision_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.device = device

    def _activate_mc_dropout(self):
        self.model.eval()
        if hasattr(self.model, "enable_mc_dropout"):
            self.model.enable_mc_dropout()
        else:
            for module in self.model.modules():
                if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                    module.train()

    def predict_stochastic(self, input_tensor):
        self._activate_mc_dropout()

        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device)

        stochastic_passes = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                logits = self.model(input_tensor)
                probs = torch.sigmoid(logits)
                stochastic_passes.append(probs)

        stochastic_stack = torch.stack(stochastic_passes, dim=0)

        mean_probs = torch.mean(stochastic_stack, dim=0)
        variance_map = torch.var(stochastic_stack, dim=0)

        eps = 1e-8
        entropy_map = -(
            mean_probs * torch.log2(mean_probs + eps)
            + (1.0 - mean_probs) * torch.log2(1.0 - mean_probs + eps)
        )

        binary_mask = (mean_probs >= self.decision_threshold).float()
        low_confidence_mask = (variance_map >= self.uncertainty_threshold).float()

        return {
            "mean_prediction": mean_probs,
            "binary_mask": binary_mask,
            "variance_map": variance_map,
            "entropy_map": entropy_map,
            "low_confidence_mask": low_confidence_mask,
            "stochastic_passes": stochastic_stack,
        }
