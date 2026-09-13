import csv

import cv2
import numpy as np
import torch

from clinical.model_registry import model_task_status
from data.classification_dataset import BinaryImageDataset, group_stratified_split, load_binary_records
from models.medical_classifier import CompactMedicalClassifier
from prepare_medical_dataset import prepare_chexpert, prepare_mura
from train_classifier import binary_metrics


def test_binary_dataset_and_grouped_split(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    labels = tmp_path / "labels.csv"
    rows = []
    for group, label in (("patient-a", 1), ("patient-b", 1), ("patient-c", 0), ("patient-d", 0)):
        for eye in ("left", "right"):
            sample_id = f"{group}-{eye}"
            cv2.imwrite(str(image_dir / f"{sample_id}.jpg"), np.full((24, 24, 3), 80 + label * 80, np.uint8))
            rows.append({"id": sample_id, "label": label, "patient": group})
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label", "patient"])
        writer.writeheader()
        writer.writerows(rows)

    records = load_binary_records(image_dir, labels, "id", "label", "patient")
    train, validation = group_stratified_split(records, validation_fraction=0.5, seed=4)

    assert {item.group for item in train}.isdisjoint({item.group for item in validation})
    image, label = BinaryImageDataset(train, image_size=32)[0]
    assert image.shape == (3, 32, 32)
    assert label.ndim == 0


def test_classifier_output_and_metrics():
    model = CompactMedicalClassifier()
    output = model(torch.zeros((2, 3, 64, 64)))
    metrics = binary_metrics([0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1])

    assert output.shape == (2, 1)
    assert metrics["auc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0


def test_registry_reports_missing_and_available_models(tmp_path):
    missing = model_task_status("Chest X-ray", tmp_path)
    assert missing["available"] is False

    checkpoint = tmp_path / "checkpoints" / "classifiers" / "skin_malignancy_isic2024" / "latest_model.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    available = model_task_status("Skin or external photo", tmp_path)
    assert available["available"] is True
    assert available["clinical_readiness"] == "research-only"


def test_mura_and_chexpert_manifest_adapters(tmp_path):
    mura = tmp_path / "mura"
    mura.mkdir()
    (mura / "train_image_paths.csv").write_text(
        "MURA-v1.1/train/XR_WRIST/patient00001/study1_positive/image1.png\n",
        encoding="utf-8",
    )
    mura_output = tmp_path / "mura.csv"
    prepare_mura(mura, mura_output)
    assert "patient00001" in mura_output.read_text(encoding="utf-8")

    chexpert = tmp_path / "chexpert"
    chexpert.mkdir()
    (chexpert / "train.csv").write_text(
        "Path,Pleural Effusion\nCheXpert-v1.0-small/train/patient00002/study1/view1.jpg,-1\n",
        encoding="utf-8",
    )
    chexpert_output = tmp_path / "chexpert.csv"
    prepare_chexpert(chexpert, "Pleural Effusion", "positive", chexpert_output)
    text = chexpert_output.read_text(encoding="utf-8")
    assert "patient00002" in text
    assert ",1,Pleural Effusion" in text
