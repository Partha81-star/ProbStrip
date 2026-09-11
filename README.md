# ProbStrip

ProbStrip is a research medical-image review application. Its retinal workflow combines a
StripConv U-Net with Monte Carlo dropout to create a vessel map and highlight
areas where repeated model passes disagree.

The same Review image screen includes a non-AI workflow for bone and chest
X-rays, CT, MRI, ultrasound, and external photographs. It provides technical
quality feedback, contrast enhancement, structure-edge views, and downloadable
patient and clinical reports. A qualified clinician can record an impression
in the report. It does not reuse the retinal model or invent a diagnosis.

The public interface is designed for patients and clinicians to review the same
result at different levels of detail. It deliberately does **not** generate a
medical diagnosis.

## Current workflow

1. Select retinal, X-ray, CT, MRI, ultrasound, or external imaging on **Review image**.
2. Upload an image, capture one with the device camera, or use the retinal demonstration.
3. Check modality-appropriate technical image quality.
4. For retinal images, produce a teal vessel overlay and mark uncertain areas in amber.
5. For other images, create contrast-enhanced and structure-edge review views.
6. Explain the result in patient-friendly language.
7. Download an easy-to-read HTML report, technical JSON, or preliminary
   FHIR-shaped JSON.
8. Compare retinal research measurements from two usable images in the current session.
9. Capture a still image on a phone or preview a live retinal overlay with WebRTC.
10. Refine or replace a vessel mask in the clinician review workspace.
11. Add a qualified-clinician impression to a general-imaging report.
12. Register two retinal visits before showing a guarded vessel-map change view.

## Safety and intended use

ProbStrip is an educational and research prototype. It has not been validated
or authorized as a medical device and must not be used to diagnose, rule out,
or treat disease. The current checkpoint was developed from a small retinal
vessel dataset. A qualified healthcare professional must review the original
image and any generated map.

Images are processed in memory and are not intentionally written to disk by the
Streamlit application. A public demo is not an appropriate place for
identifiable patient data.

## Run locally

Use Python 3.11 or 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The default checkpoint is `checkpoints/latest_model.pth`. Override it with the
`PROBSTRIP_CHECKPOINT` environment variable. The application fails visibly and
does not return a prediction if the checkpoint cannot be loaded.

### Camera modes

- **Take a photo** uses Streamlit's native camera control and is the recommended
  path on phones. A captured still receives the same quality gate and full
  uncertainty report as an uploaded image.
- **Live vessel preview** uses WebRTC and a throttled 128px single model pass.
  It is intended for positioning feedback only and does not calculate the
  uncertainty values shown in a saved report.

Browser camera access requires HTTPS except on `localhost`. Streamlit Community
Cloud supplies HTTPS automatically. The app includes a public STUN server for
connection setup; restrictive institutional or mobile networks may also require
a separately configured TURN service.

## Test

```powershell
python -m pip install -r requirements-dev.txt
pytest -q
```

## Train and evaluate

Datasets use a simple `Images/` and `Masks/` directory layout. Common CHASE_DB1
mask names such as `Image_01L_1stHO.png` are discovered automatically. Training
uses inferred subject groups so paired eyes are not split across training and
validation.

```powershell
python run_dataset.py C:\path\to\dataset --epochs 20
python evaluate_dataset.py C:\path\to\independent-validation-data --output-dir evaluation_results
```

Evaluation writes per-image Dice, IoU, sensitivity, specificity, calibration,
uncertainty, and topology-error measurements. It also writes a dataset-specific
`calibration_profile.json`. Review that artifact before copying it to
`calibration/profile.json`; the app will then disclose and use those thresholds.
This does not make the model clinically calibrated.

## Free deployment on Streamlit Community Cloud

1. Sign in at <https://share.streamlit.io> using the GitHub account that owns
   this repository.
2. Select **Create app** and choose `Partha81-star/ProbStrip`.
3. Use branch `main` and entrypoint `app.py`.
4. In advanced settings, select Python 3.11.
5. Deploy. Updates pushed to `main` are redeployed automatically.

The app is optimized for CPU use by limiting stochastic passes. Training is kept
in offline scripts and is intentionally absent from the public interface.

## Project structure

```text
app.py                         Patient and clinician Streamlit experience
clinical/modalities.py         Multi-modality quality and structure views
clinical/quality.py            Acquisition-quality gate
clinical/analysis.py           Visuals, research measurements, review outcome
clinical/reporting.py          HTML, JSON, and preliminary FHIR exports
clinical/live.py               Throttled real-time WebRTC frame processor
clinical/biomarkers.py         Vessel morphology and mask refinement
clinical/registration.py       Guarded affine visit registration and change maps
clinical/calibration.py        Validation-derived threshold analysis
models/                        Probabilistic StripConv U-Net
inference/                     Monte Carlo dropout inference
training/                      Offline training utilities
evaluate_dataset.py            External evaluation and calibration profile CLI
tests/                         Safety and report behavior tests
```

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, provenance gaps, known
limitations, and the evidence required before a prospective study.

## Next validation milestones

- Acquire approved FIVES and other adult/pathology-inclusive datasets; data is
  not redistributed by this repository.
- Run and publish independent multi-device and multi-site evaluation.
- Add artery/vein and lesion models only after obtaining suitable labels and
  defining a clinician-approved intended use.
- Complete subgroup, privacy, security, accessibility, and usability studies.
- Conduct prospective clinical studies and applicable regulatory review.
