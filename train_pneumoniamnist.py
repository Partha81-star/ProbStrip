"""Train the lightweight PneumoniaMNIST chest X-ray research benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.medical_classifier import CompactMedicalClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/classifiers"))
    args = parser.parse_args()
    data = np.load(args.dataset)
    required = {"train_images", "train_labels", "val_images", "val_labels", "test_images", "test_labels"}
    if not required.issubset(data.files):
        raise ValueError(f"Expected MedMNIST keys, found: {data.files}")

    def tensors(split):
        images = torch.from_numpy(data[f"{split}_images"].astype("float32") / 255.0).unsqueeze(1)
        labels = torch.from_numpy(data[f"{split}_labels"].reshape(-1).astype("float32"))
        return TensorDataset(images, labels)

    train, val, test = tensors("train"), tensors("val"), tensors("test")
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    model = CompactMedicalClassifier(in_channels=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    counts = np.bincount(data["train_labels"].reshape(-1), minlength=2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([counts[0] / max(counts[1], 1)]))
    best_auc, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images).squeeze(1), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for images, batch_labels in val_loader:
                probs.extend(torch.sigmoid(model(images).squeeze(1)).tolist())
                labels.extend(batch_labels.tolist())
        order = np.argsort(probs)
        y = np.asarray(labels)[order]
        ranks = np.arange(1, len(y) + 1)
        auc = float((ranks[y == 1].sum() - (y == 1).sum() * ((y == 1).sum() + 1) / 2) / max((y == 1).sum() * (y == 0).sum(), 1))
        print(f"Epoch {epoch + 1}/{args.epochs} - validation AUROC: {auc:.4f}")
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.cpu() for k, v in model.state_dict().items()}
    output = args.checkpoint_root / "pneumoniamnist_pneumonia" 
    output.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output / "best_model.pth")
    torch.save(model.state_dict(), output / "latest_model.pth")
    report = {"task_id": "pneumoniamnist_pneumonia", "dataset": "PneumoniaMNIST", "samples": {"train": len(train), "validation": len(val), "test": len(test)}, "best_validation_auc": round(best_auc, 6), "release_status": "research_only", "dataset_license": "CC BY 4.0", "dataset_source_url": "https://zenodo.org/records/10519652", "limitations": ["28x28 pediatric chest X-ray benchmark", "not clinically validated", "not a diagnosis or treatment recommendation"]}
    (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
