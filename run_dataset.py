import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import ElongatedStructureDataset
from data.discovery import discover_image_mask_pairs, group_aware_split
from models.probabilistic_unet import ProbabilisticUNet
from training.trainer import ProbStripTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ProbStrip with a subject-group-aware validation split."
    )
    parser.add_argument("dataset_dir", help="Directory containing Images/ and Masks/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main(args):
    pairs = discover_image_mask_pairs(args.dataset_dir)
    if len(pairs) < 2:
        raise ValueError("At least two image/mask pairs are required")
    train_pairs, validation_pairs = group_aware_split(
        pairs, args.validation_fraction, args.seed
    )
    print(
        f"Discovered {len(pairs)} pairs: {len(train_pairs)} train, "
        f"{len(validation_pairs)} validation"
    )

    def dataset(selected_pairs, augment):
        return ElongatedStructureDataset(
            image_paths=[str(pair[0]) for pair in selected_pairs],
            mask_paths=[str(pair[1]) for pair in selected_pairs],
            image_size=(256, 256),
            use_clahe=True,
            augment=augment,
        )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        dataset(train_pairs, True),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        dataset(validation_pairs, False),
        batch_size=args.batch_size,
        shuffle=False,
    )
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = ProbabilisticUNet(
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256],
        strip_kernel_size=7,
        dropout_prob=0.2,
    )
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    trainer = ProbStripTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        lr=args.learning_rate,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        metadata={
            "dataset_root_name": Path(args.dataset_dir).resolve().name,
            "pair_count": len(pairs),
            "train_count": len(train_pairs),
            "validation_count": len(validation_pairs),
            "validation_fraction": args.validation_fraction,
            "split_seed": args.seed,
            "image_size": [256, 256],
            "model_features": [32, 64, 128, 256],
            "strip_kernel_size": 7,
            "dropout_probability": 0.2,
        },
    )
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main(parse_args())
