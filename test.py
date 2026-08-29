import argparse
import os
import cv2
import numpy as np
import torch

from data.preprocessing import MedicalImagePreprocessor
from models.probabilistic_unet import ProbabilisticUNet
from inference.mc_dropout_inference import StochasticInferenceEngine
from training.metrics import (
    dice_similarity_coefficient,
    intersection_over_union,
    expected_calibration_error,
)


def load_model(checkpoint_path, device):
    model = ProbabilisticUNet(
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256],
        strip_kernel_size=7,
        dropout_prob=0.2,
    ).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    return model


def test_single_image(
    image_path,
    mask_path=None,
    checkpoint_path="checkpoints/latest_model.pth",
    output_dir="test_results",
    num_samples=20,
    uncertainty_threshold=0.02,
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(checkpoint_path, device)
    engine = StochasticInferenceEngine(
        model=model,
        num_samples=num_samples,
        decision_threshold=0.5,
        uncertainty_threshold=uncertainty_threshold,
        device=device,
    )

    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    raw_img = cv2.resize(raw_img, (256, 256))

    preprocessor = MedicalImagePreprocessor(use_clahe=True, norm_mode="minmax")
    img_tensor = preprocessor.process(raw_img)

    results = engine.predict_stochastic(img_tensor)

    mean_pred = results["mean_prediction"].squeeze().cpu().numpy()
    variance_map = results["variance_map"].squeeze().cpu().numpy()
    low_conf = results["low_confidence_mask"].squeeze().cpu().numpy()

    var_norm = (
        (variance_map - variance_map.min())
        / (variance_map.max() - variance_map.min() + 1e-8)
        * 255
    ).astype(np.uint8)
    var_color = cv2.applyColorMap(var_norm, cv2.COLORMAP_JET)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_input.png"),
        raw_img,
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_pred.png"),
        (mean_pred * 255).astype(np.uint8),
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_uncertainty.png"),
        var_color,
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_low_confidence_flagged.png"),
        (low_conf * 255).astype(np.uint8),
    )

    if mask_path and os.path.exists(mask_path):
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        gt_mask = (gt_mask > 127).astype(np.float32)

        dsc = dice_similarity_coefficient(mean_pred, gt_mask)
        iou = intersection_over_union(mean_pred, gt_mask)
        ece = expected_calibration_error(mean_pred, gt_mask)

        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_ground_truth.png"),
            (gt_mask * 255).astype(np.uint8),
        )

        print(f"=== Evaluation Results for {base_name} ===")
        print(f"Dice Similarity (DSC): {dsc:.4f}")
        print(f"Intersection over Union (IoU): {iou:.4f}")
        print(f"Expected Calibration Error (ECE): {ece:.4f}")
    else:
        print(f"Inference complete for {base_name}. Visualizations saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default=r"C:\Users\parth\Documents\Prob-strip-dataset\Images\Image_14L.jpg",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=r"C:\Users\parth\Documents\Prob-strip-dataset\Masks\Image_14L_1stHO.png",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/latest_model.pth",
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="test_results")
    args = parser.parse_args()

    test_single_image(
        image_path=args.image,
        mask_path=args.mask,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.samples,
    )


if __name__ == "__main__":
    main()
