# ProbStrip Model Card

## Model

- Architecture: residual StripConv U-Net
- Inputs: one 256 x 256 channel derived from a retinal image
- Outputs: pixel-level vessel logits
- Uncertainty method: Monte Carlo dropout variance
- Included artifact: `checkpoints/latest_model.pth`

## Intended use

The checkpoint is provided for research, education, interface development, and
retinal-vessel segmentation experiments. ProbStrip may help a researcher or
qualified clinician inspect a proposed vessel map and identify regions where
the model is unstable across stochastic passes.

The application also contains a deterministic image-review workspace for
X-ray, CT, MRI, ultrasound, and external-photo exports. That workspace performs
technical quality checks, local contrast enhancement, and intensity-edge
visualization only. It does not use this retinal checkpoint and is not an
anatomy, fracture, lesion, or disease detector.

## Prohibited use

Do not use this checkpoint to diagnose, exclude, triage, or treat disease. Do
not use it autonomously, for emergency decisions, or as a replacement for the
original retinal image and qualified clinical examination.

## Training-data provenance

Repository history indicates development with CHASE_DB1-style image and mask
names. Complete immutable training provenance, subject split records,
demographics, device metadata, and the exact final training run configuration
are not available for the included checkpoint. This is a major limitation.

## Known limitations

- The source dataset is small and does not represent general ophthalmic care.
- The model does not classify disease, lesions, arteries, or veins.
- Color information is reduced to a single model input channel.
- The fixed thresholds shipped in the demo are uncalibrated research defaults.
- Camera, population, acquisition, and pathology shifts may cause silent error.
- Monte Carlo dropout variance is not a guarantee of correctness.
- Morphology measurements can change with threshold, image quality, scale, and
  registration error.

## Evaluation requirements

Before any prospective study, evaluate at the subject level on independent,
representative, multi-device data. Report Dice, IoU, sensitivity, specificity,
calibration error, selective risk, image-quality rejection, topology errors,
subgroup performance, and failure cases. Keep a locked test set and publish the
intended-use population and all exclusion criteria.

Run the evidence pipeline with:

```powershell
python evaluate_dataset.py C:\path\to\dataset --output-dir evaluation_results
```

This writes per-image metrics and a validation-derived calibration profile. A
profile remains dataset-specific and does not establish clinical validity.

## Human oversight

The interface preserves the original prediction and lets a qualified reviewer
save a threshold-refined or externally corrected mask. Reviewed output must
remain attributable, auditable, and linked to the original image and model
version in any future clinical system.
