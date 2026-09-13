import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ClassificationRecord:
    image_path: Path
    label: float
    group: str
    sample_id: str


def load_binary_records(
    image_dir,
    labels_csv,
    id_column,
    label_column,
    group_column=None,
    image_extension=".jpg",
    path_column=None,
):
    image_dir = Path(image_dir).expanduser().resolve()
    labels_csv = Path(labels_csv).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Label file not found: {labels_csv}")

    records = []
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = row[id_column].strip()
            relative_path = row.get(path_column, "").strip() if path_column else ""
            image_path = image_dir / (relative_path or f"{sample_id}{image_extension}")
            if not image_path.is_file():
                continue
            group = (row.get(group_column, "") if group_column else "").strip()
            records.append(
                ClassificationRecord(
                    image_path=image_path,
                    label=float(row[label_column]),
                    group=group or sample_id,
                    sample_id=sample_id,
                )
            )
    if not records:
        raise ValueError("No labeled images matched the supplied CSV and image directory")
    return records


def group_stratified_split(records, validation_fraction=0.15, seed=42):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    grouped = {}
    for record in records:
        grouped.setdefault(record.group, []).append(record)
    positive = []
    negative = []
    for group, members in grouped.items():
        destination = positive if any(item.label >= 0.5 for item in members) else negative
        destination.append(group)
    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    val_groups = set()
    for groups in (positive, negative):
        count = min(len(groups), max(1, round(len(groups) * validation_fraction)))
        val_groups.update(groups[:count])
    train = [record for record in records if record.group not in val_groups]
    validation = [record for record in records if record.group in val_groups]
    if not train or not validation:
        raise ValueError("The grouped split produced an empty training or validation set")
    return train, validation


class BinaryImageDataset(Dataset):
    def __init__(self, records, image_size=224, augment=False):
        self.records = list(records)
        self.image_size = int(image_size)
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {record.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if self.augment:
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
            if random.random() < 0.3:
                gain = random.uniform(0.85, 1.15)
                bias = random.uniform(-12, 12)
                image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)
        tensor = tensor.sub_(0.5).div_(0.25)
        return tensor, torch.tensor(record.label, dtype=torch.float32)
