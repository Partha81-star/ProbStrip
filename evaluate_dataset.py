import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from clinical.analysis import model_input_channel, resize_scan
from clinical.biomarkers import calculate_vascular_biomarkers
from clinical.calibration import build_calibration_profile
from clinical.quality import retinal_field_mask
from data.discovery import discover_image_mask_pairs
from data.preprocessing import MedicalImagePreprocessor
from inference.mc_dropout_inference import StochasticInferenceEngine
from models.probabilistic_unet import ProbabilisticUNet
from training.metrics import (
    dice_similarity_coefficient,
    expected_calibration_error,
    intersection_over_union,
    sensitivity,
    specificity,
)


def load_model(checkpoint_path, device):
    model = ProbabilisticUNet(
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256],
        strip_kernel_size=7,
        dropout_prob=0.2,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    return model


def evaluate(args):
    pairs = discover_image_mask_pairs(args.dataset_dir)
    if not pairs:
        raise ValueError("No image/mask pairs were found")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = load_model(args.checkpoint, device)
    engine = StochasticInferenceEngine(
        model,
        num_samples=args.samples,
        decision_threshold=0.5,
        uncertainty_threshold=0.02,
        device=device,
    )
    preprocessor = MedicalImagePreprocessor(use_clahe=True, norm_mode="minmax")
    records = []
    sampled_probabilities = []
    sampled_targets = []
    sampled_variances = []
    random_generator = np.random.default_rng(args.seed)

    for image_path, mask_path in pairs:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = resize_scan(rgb)
        target = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) >= 128
        tensor = preprocessor.process(model_input_channel(rgb))
        result = engine.predict_stochastic(tensor)
        probability = result["mean_prediction"].squeeze().cpu().numpy()
        variance = result["variance_map"].squeeze().cpu().numpy()
        field = retinal_field_mask(rgb)
        predicted_mask = probability >= 0.5
        predicted_morphology = calculate_vascular_biomarkers(rgb, predicted_mask)
        target_morphology = calculate_vascular_biomarkers(rgb, target)

        records.append(
            {
                "image": image_path.name,
                "dice": dice_similarity_coefficient(probability, target),
                "iou": intersection_over_union(probability, target),
                "ece": expected_calibration_error(probability[field], target[field]),
                "sensitivity": sensitivity(probability[field], target[field]),
                "specificity": specificity(probability[field], target[field]),
                "mean_variance": float(variance[field].mean()),
                "fractal_dimension_error": abs(
                    predicted_morphology["fractal_dimension"]
                    - target_morphology["fractal_dimension"]
                ),
                "branch_region_absolute_error": abs(
                    predicted_morphology["branch_region_count"]
                    - target_morphology["branch_region_count"]
                ),
            }
        )
        field_indices = np.flatnonzero(field.reshape(-1))
        sample_count = min(args.pixels_per_image, field_indices.size)
        selected = random_generator.choice(
            field_indices, size=sample_count, replace=False
        )
        sampled_probabilities.append(probability.reshape(-1)[selected])
        sampled_targets.append(target.reshape(-1)[selected])
        sampled_variances.append(variance.reshape(-1)[selected])

    if not records:
        raise ValueError("Images were found but none could be decoded")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(records)
    dataframe.to_csv(output_dir / "per_image_metrics.csv", index=False)
    profile = build_calibration_profile(
        np.concatenate(sampled_probabilities),
        np.concatenate(sampled_targets),
        np.concatenate(sampled_variances),
        review_fraction=args.review_fraction,
        dataset_name=Path(args.dataset_dir).name,
    )
    profile["image_count"] = len(records)
    profile["summary"] = {
        column: round(float(dataframe[column].mean()), 6)
        for column in (
            "dice",
            "iou",
            "ece",
            "sensitivity",
            "specificity",
            "mean_variance",
            "fractal_dimension_error",
            "branch_region_absolute_error",
        )
    }
    with (output_dir / "calibration_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)
    print(json.dumps(profile, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ProbStrip on an independent vessel-segmentation dataset."
    )
    parser.add_argument("dataset_dir", help="Directory containing Images/ and Masks/")
    parser.add_argument(
        "--checkpoint", default="checkpoints/latest_model.pth", help="Model checkpoint"
    )
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--pixels-per-image", type=int, default=10000)
    parser.add_argument("--review-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
