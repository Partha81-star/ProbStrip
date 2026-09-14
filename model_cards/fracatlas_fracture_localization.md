# FracAtlas fracture-localization model

## Intended function

The model localizes fracture candidates in hand, leg, hip, and shoulder X-rays and
returns a confidence score for each detected box. It is the official YOLOv8
localization checkpoint published with FracAtlas.

## Data provenance

- Dataset: FracAtlas
- Images: 4,083 musculoskeletal radiographs from three hospitals in Bangladesh
- Fracture images: 717, containing 922 annotated fracture instances
- Annotation: two radiologists with medical-officer confirmation
- License: CC BY 4.0
- Source: https://doi.org/10.6084/m9.figshare.22363012

## Published validation performance

- Box precision: 80.7%
- Box recall: 47.3%
- mAP50: 56.2%

This published checkpoint is the model selected by the application. ProbStrip
uses a 25% box-confidence threshold and reports the highest detected box score as
model confidence, not as probability of disease or diagnostic certainty.

## Local training verification

On 2026-09-14, the repository pipeline trained YOLOv8n for five CPU epochs from
generic pretrained weights using a deterministic 70/15/15 image-level split of
FracAtlas v7. The untouched local test split produced precision 0.107659, recall
0.086957, mAP50 0.030377, and mAP50-95 0.009054. That short run was rejected for
deployment because it was materially worse than the published checkpoint. The
exact run record is in `fracatlas_local_training_metrics.json`.

Because recall is limited, a negative result cannot rule out fracture. The model is not
validated for chest, skull, or spine X-rays, or for clinical deployment outside the
source setting.
