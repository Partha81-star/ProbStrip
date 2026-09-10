# ProbStrip

ProbStrip is a research retinal-vessel review application. It combines a
StripConv U-Net with Monte Carlo dropout to create a vessel map and highlight
areas where repeated model passes disagree.

The public interface is designed for patients and clinicians to review the same
result at different levels of detail. It deliberately does **not** generate a
medical diagnosis.

## Current workflow

1. Upload a fundus-camera retinal photograph or use the included demonstration.
2. Check brightness, contrast, sharpness, glare, and retinal framing.
3. Stop and recommend a retake when capture quality is inadequate.
4. Produce a teal vessel overlay and mark uncertain areas in amber.
5. Explain the result in patient-friendly English or Hindi.
6. Download an easy-to-read HTML report, technical JSON, or preliminary
   FHIR-shaped JSON.
7. Compare research measurements from two usable images in the current session.
8. Capture a still image on a phone or preview a live vessel overlay with WebRTC.

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
clinical/quality.py            Acquisition-quality gate
clinical/analysis.py           Visuals, research measurements, review outcome
clinical/reporting.py          HTML, JSON, and preliminary FHIR exports
clinical/live.py               Throttled real-time WebRTC frame processor
models/                        Probabilistic StripConv U-Net
inference/                     Monte Carlo dropout inference
training/                      Offline training utilities
tests/                         Safety and report behavior tests
```

## Next validation milestones

- Patient-level splits and external evaluation on multiple devices and sites.
- FIVES and other adult/pathology-inclusive retinal datasets.
- Calibration and selective-risk validation for every abstention threshold.
- Clinician correction tools and geometrically registered longitudinal images.
- Subgroup, privacy, security, usability, and prospective clinical studies.
