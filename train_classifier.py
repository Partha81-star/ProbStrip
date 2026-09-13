import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.classification_dataset import BinaryImageDataset, group_stratified_split, load_binary_records
from models.medical_classifier import CompactMedicalClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train a provenance-aware binary medical-image research classifier.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--group-column")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-license", required=True)
    parser.add_argument("--dataset-source-url", required=True)
    parser.add_argument("--intended-population", required=True)
    parser.add_argument("--image-extension", default=".jpg")
    parser.add_argument("--path-column")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-root", default="checkpoints/classifiers")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def binary_metrics(probabilities, labels, threshold=0.5):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = probabilities >= threshold
    positives = labels == 1
    negatives = ~positives
    tp = int(np.sum(predictions & positives))
    tn = int(np.sum(~predictions & negatives))
    fp = int(np.sum(predictions & negatives))
    fn = int(np.sum(~predictions & positives))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    order = np.argsort(probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    ranks = np.empty_like(probabilities, dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_probabilities[end] == sorted_probabilities[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_ranks = float(np.sum(ranks[positives]))
    auc = (positive_ranks - np.sum(positives) * (np.sum(positives) + 1) / 2) / max(np.sum(positives) * np.sum(negatives), 1)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        selected = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if np.any(selected):
            ece += np.mean(selected) * abs(np.mean(probabilities[selected]) - np.mean(labels[selected]))
    return {
        "threshold": round(float(threshold), 6),
        "auc": round(float(auc), 6),
        "sensitivity": round(float(sensitivity), 6),
        "specificity": round(float(specificity), 6),
        "balanced_accuracy": round(float((sensitivity + specificity) / 2), 6),
        "brier_score": round(float(np.mean((probabilities - labels) ** 2)), 6),
        "ece": round(float(ece), 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def select_balanced_threshold(probabilities, labels):
    candidates = np.linspace(0.01, 0.99, 99)
    scored = [binary_metrics(probabilities, labels, threshold) for threshold in candidates]
    return max(scored, key=lambda item: (item["balanced_accuracy"], item["sensitivity"], item["specificity"]))["threshold"]


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_total = 0.0
    probabilities = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images).squeeze(1)
            loss_total += criterion(logits, targets).item() * images.size(0)
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(targets.cpu().int().tolist())
    threshold = select_balanced_threshold(probabilities, labels)
    return {
        "loss": loss_total / len(loader.dataset),
        "threshold_source": "validation balanced-accuracy optimization",
        **binary_metrics(probabilities, labels, threshold),
    }


def save_checkpoint(path, model, metadata):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    records = load_binary_records(
        args.dataset_dir,
        args.labels_csv,
        args.id_column,
        args.label_column,
        args.group_column,
        args.image_extension,
        args.path_column,
    )
    if args.max_samples:
        rng = random.Random(args.seed)
        positives = [record for record in records if record.label >= 0.5]
        negatives = [record for record in records if record.label < 0.5]
        keep_negative = max(0, args.max_samples - len(positives))
        records = positives + rng.sample(negatives, min(keep_negative, len(negatives)))
    train_records, validation_records = group_stratified_split(records, args.validation_fraction, args.seed)
    train_dataset = BinaryImageDataset(train_records, args.image_size, augment=True)
    validation_dataset = BinaryImageDataset(validation_records, args.image_size, augment=False)
    positive_count = sum(record.label >= 0.5 for record in train_records)
    negative_count = len(train_records) - positive_count
    weights = [
        1.0 / max(positive_count if record.label >= 0.5 else negative_count, 1)
        for record in train_records
    ]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = CompactMedicalClassifier().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    checkpoint_dir = Path(args.checkpoint_root) / args.task_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_digest = hashlib.sha256(
        "\n".join(f"{record.sample_id}:{record.label}:{record.group}" for record in sorted(records, key=lambda item: item.sample_id)).encode()
    ).hexdigest()
    base_metadata = {
        "task_id": args.task_id,
        "dataset_name": args.dataset_name,
        "dataset_license": args.dataset_license,
        "dataset_source_url": args.dataset_source_url,
        "intended_population": args.intended_population,
        "dataset_manifest_sha256": manifest_digest,
        "sample_count": len(records),
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "positive_count": sum(record.label >= 0.5 for record in records),
        "negative_count": sum(record.label < 0.5 for record in records),
        "patient_or_lesion_grouped_split": True,
        "seed": args.seed,
        "image_size": args.image_size,
        "architecture": "CompactMedicalClassifier",
        "intended_use": "Research evaluation only; not validated for diagnosis or treatment",
    }
    history = []
    best_auc = -1.0
    best_epoch = 0
    without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()
        metrics = evaluate(model, validation_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": running_loss / len(train_dataset), **metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["auc"] > best_auc + 1e-4:
            best_auc = metrics["auc"]
            best_epoch = epoch
            without_improvement = 0
            save_checkpoint(checkpoint_dir / "best_model.pth", model, {**base_metadata, "best_epoch": best_epoch, "validation_metrics": metrics})
        else:
            without_improvement += 1
        if without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.", flush=True)
            break
    best = torch.load(checkpoint_dir / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    final_metrics = evaluate(model, validation_loader, criterion, device)
    save_checkpoint(checkpoint_dir / "latest_model.pth", model, {**base_metadata, "best_epoch": best_epoch, "validation_metrics": final_metrics})
    report = {
        "release_status": "research_only",
        "release_blockers": [
            "independent external test set not evaluated",
            "subgroup and acquisition-shift performance not established",
            "prospective clinical validation not completed",
            "regulatory status not established",
        ],
        "metadata": base_metadata,
        "best_epoch": best_epoch,
        "validation_metrics": final_metrics,
        "history": history,
    }
    (checkpoint_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved best research checkpoint and report to {checkpoint_dir}")


if __name__ == "__main__":
    main(parse_args())
