# ISIC 2024 permissive skin classifier model card

## Status

- Release status: research only
- Patient inference: disabled
- Task: binary skin-lesion malignancy research classification
- Input domain: 15 mm by 15 mm lesion crops from the ISIC 2024 protocol
- Not valid for: arbitrary phone photos, diagnosis, triage, or treatment

## Data and provenance

- Source: <https://challenge.isic-archive.com/data/>
- Dataset: ISIC 2024 Permissive Training Input and Ground Truth
- License recorded by the source: CC BY 4.0
- Local manifest SHA-256: `8e86a4281adaa2743d3d51037394826f09fc73eae09ee05430118a2fca895ce2`
- Pilot samples: 1,000, including all 294 positive examples and 706 seeded negative examples
- Split: 850 training and 150 validation samples, grouped by lesion identifier where available

## Pilot performance

| Metric | Internal validation result |
| --- | ---: |
| AUROC | 0.893225 |
| Sensitivity | 0.795455 |
| Specificity | 0.858491 |
| Balanced accuracy | 0.826973 |
| Brier score | 0.129711 |
| Expected calibration error | 0.126641 |
| Validation-derived threshold | 0.46 |

The threshold and performance were measured on the same small validation set.
They must not be presented as independent or clinical performance.

## Release blockers

- Train with the full dataset using a CUDA-enabled environment and stronger architecture.
- Evaluate on a locked external dataset with patient-level separation.
- Measure performance across skin tone, age, sex, anatomical site, and acquisition source.
- Calibrate on a separate calibration set and freeze the operating threshold.
- Validate the intended human-computer workflow prospectively with dermatologists.
- Complete privacy, security, usability, and applicable regulatory review.
