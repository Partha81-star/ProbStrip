import glob
import os
import torch
from torch.utils.data import DataLoader
from data.dataset import ElongatedStructureDataset


def create_chase_db1_dataloaders(
    data_dir="dataset/chase_db1", image_size=(256, 256), batch_size=4, train_ratio=0.8
):
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    mask_paths = []

    for img_p in image_paths:
        base_name = os.path.basename(img_p).replace(".jpg", "")
        mask_p = os.path.join(data_dir, "MASKS", f"{base_name}_1stHO.png")
        if os.path.exists(mask_p):
            mask_paths.append(mask_p)

    split_idx = int(train_ratio * len(image_paths))

    train_ds = ElongatedStructureDataset(
        image_paths=image_paths[:split_idx],
        mask_paths=mask_paths[:split_idx],
        image_size=image_size,
        use_clahe=True,
        augment=True,
    )

    val_ds = ElongatedStructureDataset(
        image_paths=image_paths[split_idx:],
        mask_paths=mask_paths[split_idx:],
        image_size=image_size,
        use_clahe=True,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def create_ultrasound_nerve_dataloaders(
    data_dir="dataset/ultrasound_nerve",
    image_size=(256, 256),
    batch_size=4,
    train_ratio=0.8,
):
    all_tif = sorted(
        glob.glob(os.path.join(data_dir, "*.tif"))
        + glob.glob(os.path.join(data_dir, "**", "*.tif"), recursive=True)
    )
    mask_paths = [p for p in all_tif if p.endswith("_mask.tif")]
    image_paths = [p for p in all_tif if not p.endswith("_mask.tif")]

    matched_images = []
    matched_masks = []

    for img_p in image_paths:
        base = img_p[:-4]
        mask_p = f"{base}_mask.tif"
        if os.path.exists(mask_p):
            matched_images.append(img_p)
            matched_masks.append(mask_p)

    split_idx = int(train_ratio * len(matched_images))

    train_ds = ElongatedStructureDataset(
        image_paths=matched_images[:split_idx],
        mask_paths=matched_masks[:split_idx],
        image_size=image_size,
        use_clahe=True,
        augment=True,
    )

    val_ds = ElongatedStructureDataset(
        image_paths=matched_images[split_idx:],
        mask_paths=matched_masks[split_idx:],
        image_size=image_size,
        use_clahe=True,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def create_generic_dataloaders(
    images_dir, masks_dir, image_size=(256, 256), batch_size=4, train_ratio=0.8
):
    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png"))
        + glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.tif"))
    )
    mask_paths = sorted(
        glob.glob(os.path.join(masks_dir, "*.png"))
        + glob.glob(os.path.join(masks_dir, "*.jpg"))
        + glob.glob(os.path.join(masks_dir, "*.tif"))
    )

    split_idx = int(train_ratio * len(image_paths))

    train_ds = ElongatedStructureDataset(
        image_paths=image_paths[:split_idx],
        mask_paths=mask_paths[:split_idx],
        image_size=image_size,
        use_clahe=True,
        augment=True,
    )

    val_ds = ElongatedStructureDataset(
        image_paths=image_paths[split_idx:],
        mask_paths=mask_paths[split_idx:],
        image_size=image_size,
        use_clahe=True,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
