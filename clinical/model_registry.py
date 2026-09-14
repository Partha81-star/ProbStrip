import json
import os
from pathlib import Path


MODEL_TASKS = {
    "Bone or joint X-ray": {
        "task_id": "mura_abnormality",
        "task": "Musculoskeletal study abnormality research classification",
        "dataset": "MURA",
    },
    "Chest X-ray": {
        "task_id": "pneumoniamnist_pneumonia",
        "task": "Pediatric chest radiograph pneumonia screening classification",
        "dataset": "PneumoniaMNIST",
    },
    "CT image": {
        "task_id": "lidc_lung_nodule",
        "task": "Thoracic CT lung-nodule research analysis",
        "dataset": "LIDC-IDRI",
    },
    "MRI image": {
        "task_id": "mri_task_unselected",
        "task": "A body region and clinical task must be selected before training",
        "dataset": "Not selected",
    },
    "Ultrasound image": {
        "task_id": "ultrasound_task_unselected",
        "task": "An organ and clinical task must be selected before training",
        "dataset": "Not selected",
    },
    "Skin or external photo": {
        "task_id": "skin_malignancy_isic2024",
        "task": "Skin-lesion malignancy research classification",
        "dataset": "ISIC 2024 Permissive",
    },
}


def model_task_status(modality, app_root):
    task = dict(MODEL_TASKS[modality])
    checkpoint = Path(app_root) / "checkpoints" / "classifiers" / task["task_id"] / "latest_model.pth"
    task["checkpoint"] = str(checkpoint)
    task["available"] = checkpoint.is_file()
    task["clinical_readiness"] = "research-only" if task["available"] else "not trained"
    report_path = checkpoint.parent / "training_report.json"
    task["release_status"] = "research_only"
    task["validation_metrics"] = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            task["release_status"] = report.get("release_status", "research_only")
            task["validation_metrics"] = report.get("validation_metrics", {})
        except (OSError, ValueError):
            task["release_status"] = "invalid_report"
    task["patient_inference_enabled"] = (
        task["available"]
        and task["release_status"] == "externally_validated"
        and os.getenv("PROBSTRIP_ENABLE_VALIDATED_MODELS") == "1"
    )
    return task
