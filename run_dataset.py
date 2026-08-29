import glob
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import ElongatedStructureDataset
from models.probabilistic_unet import ProbabilisticUNet
from training.trainer import ProbStripTrainer
from inference.mc_dropout_inference import StochasticInferenceEngine


def load_chase_db1_dataset(dataset_dir):
    images_dir = os.path.join(dataset_dir, "Images")
    masks_dir = os.path.join(dataset_dir, "Masks")

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )

    matched_images = []
    matched_masks = []

    for img_p in image_paths:
        base_name = os.path.basename(img_p)
        name_no_ext = os.path.splitext(base_name)[0]

        possible_masks = [
            os.path.join(masks_dir, f"{name_no_ext}_1stHO.png"),
            os.path.join(masks_dir, f"{name_no_ext}_2ndHO.png"),
            os.path.join(masks_dir, f"{name_no_ext}.png"),
            os.path.join(masks_dir, base_name),
        ]

        found_mask = None
        for pm in possible_masks:
            if os.path.exists(pm):
                found_mask = pm
                break

        if found_mask is not None:
            matched_images.append(img_p)
            matched_masks.append(found_mask)

    return matched_images, matched_masks


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
    dataset_dir = r"C:\Users\parth\Documents\Prob-strip-dataset"
    img_paths, mask_paths = load_chase_db1_dataset(dataset_dir)

    split_idx = int(0.8 * len(img_paths))

    train_dataset = ElongatedStructureDataset(
        image_paths=img_paths[:split_idx],
        mask_paths=mask_paths[:split_idx],
        image_size=(256, 256),
        use_clahe=True,
        augment=True,
    )

    val_dataset = ElongatedStructureDataset(
        image_paths=img_paths[split_idx:],
        mask_paths=mask_paths[split_idx:],
        image_size=(256, 256),
        use_clahe=True,
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ProbabilisticUNet(
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256],
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

    trainer.train(num_epochs=10)

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
        save_dir="outputs_real_dataset",
    )


if __name__ == "__main__":
    main()
