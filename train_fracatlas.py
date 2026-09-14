"""Train and evaluate a compact fracture-localization model on FracAtlas."""

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(APP_ROOT / ".ultralytics"))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO on prepared FracAtlas data.")
    parser.add_argument("dataset_yaml", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=APP_ROOT / "checkpoints" / "fracture" / "fracatlas_local_best.pt",
    )
    return parser.parse_args()


def _numeric_metrics(results):
    return {
        key: round(float(value), 6)
        for key, value in results.results_dict.items()
        if isinstance(value, (int, float))
    }


def main(args):
    dataset_yaml = args.dataset_yaml.resolve()
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Prepared dataset configuration not found: {dataset_yaml}")

    run_root = APP_ROOT / "training_runs"
    model = YOLO(args.model)
    train_results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.image_size,
        workers=args.workers,
        device="cpu",
        project=str(run_root),
        name="fracatlas_local",
        exist_ok=True,
        patience=max(3, min(10, args.epochs)),
        seed=args.seed,
        deterministic=True,
        cache=False,
        plots=False,
        verbose=True,
    )

    best_checkpoint = Path(train_results.save_dir) / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Training did not create {best_checkpoint}")
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, args.output_checkpoint)

    evaluated_model = YOLO(str(args.output_checkpoint))
    test_results = evaluated_model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=args.image_size,
        batch=args.batch,
        workers=args.workers,
        device="cpu",
        project=str(run_root),
        name="fracatlas_local_test",
        exist_ok=True,
        plots=False,
        verbose=True,
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "FracAtlas v7",
        "dataset_yaml": dataset_yaml.name,
        "initial_weights": args.model,
        "epochs_requested": args.epochs,
        "image_size": args.image_size,
        "batch": args.batch,
        "seed": args.seed,
        "device": "cpu",
        "checkpoint": args.output_checkpoint.as_posix(),
        "test_metrics": _numeric_metrics(test_results),
        "validation_scope": (
            "Internal image-level holdout only; not external or prospective clinical validation."
        ),
    }
    report_path = args.output_checkpoint.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(parse_args())
