# ProbStrip datasets and checkpoints

Raw medical datasets remain outside Git. The repository contains deployable
checkpoints, compact evaluation records, training code, and model cards.

| App workflow | Model task | Dataset | Integrated status |
| --- | --- | --- | --- |
| Retinal fundus | Vessel segmentation | CHASE_DB1-style local set | Checkpoint integrated |
| Bone or joint X-ray | Fracture localization | FracAtlas v7 | Published checkpoint integrated |
| Chest X-ray | Pneumonia vs normal classification | PneumoniaMNIST | Locally trained checkpoint integrated |

## Sources

- FracAtlas: <https://doi.org/10.6084/m9.figshare.22363012>
- PneumoniaMNIST: <https://zenodo.org/records/10519652>
- CHASE_DB1: <https://blogs.kingston.ac.uk/retinal/chasedb1/>

## Recorded performance

The FracAtlas checkpoint's published evaluation reports precision 0.807, recall
0.473, and mAP50 0.562. The integrated PneumoniaMNIST checkpoint's held-out test
report records AUROC 0.95549, sensitivity 0.984615, and specificity 0.65812 at
the configured 0.29 threshold. These values describe their respective test data
and do not guarantee the same performance on a new hospital, device, age group,
or photographed display.

The retinal model was trained for vessel segmentation. It has no labels for
retinal infection, diabetic retinopathy, glaucoma, or other eye disease, so the
application does not claim those findings.

## Reproducibility and release checks

Before changing a deployed checkpoint, retain:

- dataset source, license, exclusions, and split method;
- locked test-set metrics, calibration, and failure examples;
- intended population and contraindications;
- checkpoint hash and preprocessing configuration;
- privacy, security, accessibility, and subgroup evaluation.

No model score should be interpreted as a guarantee, and negative screens do
not rule out disease or injury when symptoms remain concerning.
