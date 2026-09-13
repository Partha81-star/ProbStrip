# ProbStrip dataset and task registry

ProbStrip keeps raw medical datasets outside Git. Only manifests, training code,
model cards, evaluation reports, and LFS-tracked research checkpoints belong in
the repository. Dataset access terms and required attribution must be preserved.

## Current tasks

| App category | Defined research task | Dataset | Access status |
| --- | --- | --- | --- |
| Retinal fundus | Vessel segmentation | CHASE_DB1-style local set | Trained locally; provenance remains limited |
| Bone or joint X-ray | Study-level normal/abnormal classification | MURA | Stanford AIMI account approval required |
| Chest X-ray | Multi-label radiographic finding classification | CheXpert | Stanford AIMI account approval required |
| CT | Thoracic lung-nodule detection/segmentation | LIDC-IDRI | Open, CC BY 3.0, approximately 133 GB |
| MRI | Not selected | Not selected | Body region and clinical task required |
| Ultrasound | Not selected | Not selected | Organ and clinical task required |
| Skin | Lesion malignancy research classification | ISIC 2024 Permissive | Open, CC BY 4.0; local download prepared |

## Official sources

- MURA: <https://stanfordmlgroup.github.io/competitions/mura/>
- CheXpert: <https://stanfordmlgroup.github.io/competitions/chexpert/>
- LIDC-IDRI: <https://www.cancerimagingarchive.net/collection/lidc-idri/>
- ISIC challenge data: <https://challenge.isic-archive.com/data/>

## ISIC local preparation

The permissive archive and metadata are stored outside the repository. After
extracting the archive, build the joined manifest and train a research model:

```powershell
python prepare_isic.py "C:\Users\parth\Documents\ProbStrip-datasets\ISIC2024-Permissive"
python train_classifier.py `
  "C:\Users\parth\Documents\ProbStrip-datasets\ISIC2024-Permissive\train-image\image" `
  --labels-csv "C:\Users\parth\Documents\ProbStrip-datasets\ISIC2024-Permissive\training_manifest.csv" `
  --id-column isic_id --label-column malignant --group-column lesion_id `
  --task-id skin_malignancy_isic2024 --dataset-name ISIC_2024_Permissive
```

The exact extracted image directory can differ by archive release. Confirm it
before training. Use a capped pilot run before a full run on local hardware.

## Release gate

A checkpoint remains research-only until it has all of the following:

- documented license, provenance, exclusions, and subject-level split;
- a locked external test set from a different institution or acquisition source;
- AUROC, sensitivity, specificity, calibration, subgroup, and failure-case reports;
- a model card defining intended use and known contraindications;
- clinician review of the human-computer workflow;
- prospective validation and applicable regulatory review before clinical use.

No research checkpoint may prescribe treatment or replace the original image,
clinical history, physical examination, laboratory results, or specialist review.
