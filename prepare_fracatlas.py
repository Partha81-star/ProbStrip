"""Prepare the official FracAtlas release for reproducible YOLO training."""

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

import yaml


SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create deterministic image-level stratified FracAtlas splits."
    )
    parser.add_argument("dataset_dir", type=Path, help="Extracted FracAtlas directory")
    parser.add_argument("output_dir", type=Path, help="Destination for the YOLO dataset")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _stratified_split(rows, seed):
    rng = random.Random(seed)
    splits = {name: [] for name in SPLIT_RATIOS}
    for fractured in (0, 1):
        group = [row for row in rows if int(row["fractured"]) == fractured]
        rng.shuffle(group)
        train_end = round(len(group) * SPLIT_RATIOS["train"])
        val_end = train_end + round(len(group) * SPLIT_RATIOS["val"])
        splits["train"].extend(group[:train_end])
        splits["val"].extend(group[train_end:val_end])
        splits["test"].extend(group[val_end:])
    for rows_in_split in splits.values():
        rng.shuffle(rows_in_split)
    return splits


def _source_image(dataset_dir, row):
    class_dir = "Fractured" if int(row["fractured"]) else "Non_fractured"
    path = dataset_dir / "images" / class_dir / row["image_id"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing image listed by dataset.csv: {path}")
    return path


def _link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare(dataset_dir, output_dir, seed):
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    csv_path = dataset_dir / "dataset.csv"
    labels_dir = dataset_dir / "Annotations" / "YOLO"
    if not csv_path.is_file() or not labels_dir.is_dir():
        raise FileNotFoundError(
            "Expected dataset.csv and Annotations/YOLO inside the extracted FracAtlas directory."
        )

    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("dataset.csv contains no records.")

    splits = _stratified_split(rows, seed)
    manifest = []
    for split, split_rows in splits.items():
        for row in split_rows:
            image_source = _source_image(dataset_dir, row)
            label_source = labels_dir / f"{Path(row['image_id']).stem}.txt"
            if not label_source.is_file():
                raise FileNotFoundError(f"Missing YOLO annotation: {label_source}")
            _link_or_copy(image_source, output_dir / "images" / split / image_source.name)
            _link_or_copy(label_source, output_dir / "labels" / split / label_source.name)
            manifest.append(
                {
                    "image_id": row["image_id"],
                    "split": split,
                    "fractured": int(row["fractured"]),
                }
            )

    yaml_path = output_dir / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {
                "path": output_dir.as_posix(),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "fractured"},
            },
            stream,
            sort_keys=False,
        )

    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "dataset": "FracAtlas v7",
                "seed": seed,
                "ratios": SPLIT_RATIOS,
                "records": manifest,
            },
            stream,
            indent=2,
        )

    print(f"Prepared {len(manifest)} images at {output_dir}")
    for split in SPLIT_RATIOS:
        members = [item for item in manifest if item["split"] == split]
        positives = sum(item["fractured"] for item in members)
        print(
            f"{split}: {len(members)} images "
            f"({positives} fractured, {len(members) - positives} non-fractured)"
        )
    print(f"Training data configuration: {yaml_path}")


if __name__ == "__main__":
    args = parse_args()
    prepare(args.dataset_dir, args.output_dir, args.seed)
