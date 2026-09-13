from pathlib import Path


MODEL_TASKS = {
    "Bone or joint X-ray": {
        "task_id": "mura_abnormality",
        "task": "Musculoskeletal study abnormality research classification",
        "dataset": "MURA",
    },
    "Chest X-ray": {
        "task_id": "chexpert_findings",
        "task": "Chest radiograph finding research classification",
        "dataset": "CheXpert",
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
    return task
