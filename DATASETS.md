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
| Skin | Lesion malignancy research classification | ISIC 2024 Permissive | Downloaded locally; 1,000-sample CPU pilot trained |

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
  --task-id skin_malignancy_isic2024 --dataset-name ISIC_2024_Permissive `
  --dataset-license CC-BY-4.0 `
  --dataset-source-url https://challenge.isic-archive.com/data/ `
  --intended-population "Research images matching the ISIC 2024 permissive lesion-crop protocol"
```

The exact extracted image directory can differ by archive release. Confirm it
before training. Use a capped pilot run before a full run on local hardware.

## Stanford dataset preparation

After downloading MURA through your Stanford AIMI account:

```powershell
python prepare_medical_dataset.py mura "C:\path\to\MURA-v1.1" --output mura_manifest.csv
python train_classifier.py "C:\path\to\MURA-v1.1" `
  --labels-csv mura_manifest.csv --id-column sample_id --path-column relative_path `
  --label-column abnormal --group-column patient_id `
  --task-id mura_abnormality --dataset-name MURA `
  --dataset-license "Stanford AIMI terms" `
  --dataset-source-url https://stanfordmlgroup.github.io/competitions/mura/ `
  --intended-population "Musculoskeletal radiograph studies matching MURA acquisition and labeling"
```

After downloading CheXpert through your Stanford AIMI account, create one
explicit finding task at a time. This example uses pleural effusion and records
the uncertain-label policy:

```powershell
python prepare_medical_dataset.py chexpert "C:\path\to\CheXpert-v1.0-small" `
  --label "Pleural Effusion" --uncertain positive --output chexpert_effusion.csv
python train_classifier.py "C:\path\to\CheXpert-v1.0-small" `
  --labels-csv chexpert_effusion.csv --id-column sample_id --path-column relative_path `
  --label-column target --group-column patient_id `
  --task-id chexpert_pleural_effusion --dataset-name CheXpert `
  --dataset-license "Stanford AIMI terms" `
  --dataset-source-url https://stanfordmlgroup.github.io/competitions/chexpert/ `
  --intended-population "Chest radiographs matching CheXpert acquisition and labeling"
```

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

## Completed skin pilot

The first end-to-end pipeline check used all 294 available malignant examples
and a reproducible negative subset, for 1,000 samples total. It used 850 training
and 150 validation samples at 64 px because the installed PyTorch build is
CPU-only. Internal validation produced AUROC 0.893225, sensitivity 0.795455,
specificity 0.858491, balanced accuracy 0.826973, and ECE 0.126641.

These are development-set results, not external clinical performance. The
checkpoint remains `research_only` and patient inference is disabled. The full
217,477-image dataset is stored outside Git at
`C:\Users\parth\Documents\ProbStrip-datasets\ISIC2024-Permissive`.
