import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import ElongatedStructureDataset
from models.probabilistic_unet import ProbabilisticUNet
from training.trainer import ProbStripTrainer
from inference.mc_dropout_inference import StochasticInferenceEngine


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_visualization(
    input_img, gt_mask, mean_pred, variance_map, low_conf_mask, save_dir
):
    os.makedirs(save_dir, exist_ok=True)

    img_np = (input_img.squeeze().cpu().numpy() * 255).astype(np.uint8)
    gt_np = (gt_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    mean_np = (mean_pred.squeeze().cpu().numpy() * 255).astype(np.uint8)

    var_raw = variance_map.squeeze().cpu().numpy()
    var_norm = (
        (var_raw - var_raw.min())
        / (var_raw.max() - var_raw.min() + 1e-8)
        * 255
    ).astype(np.uint8)
    var_color = cv2.applyColorMap(var_norm, cv2.COLORMAP_JET)

    flagged_np = (low_conf_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(save_dir, "01_input_image.png"), img_np)
    cv2.imwrite(os.path.join(save_dir, "02_ground_truth.png"), gt_np)
    cv2.imwrite(os.path.join(save_dir, "03_mean_prediction.png"), mean_np)
    cv2.imwrite(os.path.join(save_dir, "04_uncertainty_variance.png"), var_color)
    cv2.imwrite(os.path.join(save_dir, "05_low_confidence_flagged.png"), flagged_np)


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = ElongatedStructureDataset(length=64, image_size=(128, 128))
    val_dataset = ElongatedStructureDataset(length=16, image_size=(128, 128))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = ProbabilisticUNet(
        in_channels=1,
        out_channels=1,
        features=[16, 32, 64, 128],
        strip_kernel_size=7,
        dropout_prob=0.2,
    )

    trainer = ProbStripTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-3,
        device=device,
        checkpoint_dir="checkpoints",
    )

    trainer.train(num_epochs=5)

    inference_engine = StochasticInferenceEngine(
        model=model,
        num_samples=20,
        decision_threshold=0.5,
        uncertainty_threshold=0.02,
        device=device,
    )

    sample_img, sample_gt = val_dataset[0]
    results = inference_engine.predict_stochastic(sample_img)

    save_visualization(
        input_img=sample_img,
        gt_mask=sample_gt,
        mean_pred=results["mean_prediction"],
        variance_map=results["variance_map"],
        low_conf_mask=results["low_confidence_mask"],
        save_dir="outputs",
    )


if __name__ == "__main__":
    main()
