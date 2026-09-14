# PneumoniaMNIST pneumonia screening model

## Intended use

This compact classifier is an offline prototype for distinguishing pneumonia-labelled
from normal images in the PneumoniaMNIST pediatric chest X-ray benchmark. It is not a
general chest X-ray interpreter and cannot detect fractures, tumors, effusions, heart
disease, tuberculosis, or other conditions.

## Data and evaluation

- Dataset: PneumoniaMNIST, derived from 5,856 pediatric chest X-rays
- Official split: 4,708 train, 524 validation, 624 test
- Input: 28 x 28 grayscale image
- License: CC BY 4.0
- Source: https://zenodo.org/records/10519652

The training script selects its checkpoint and operating threshold on the validation
split, then evaluates once on the untouched test split. See `training_report.json` next
to the checkpoint for the measured results.

## Safety limitations

The source population, low image resolution, label process, and single-site dataset do
not represent general clinical use. The model has no prospective, external, subgroup,
device, or workflow validation. It must not be used to diagnose, rule out, or recommend
treatment for pneumonia.
