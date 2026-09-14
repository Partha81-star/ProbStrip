# ProbStrip

ProbStrip is a responsive Streamlit medical-image screening application with
three supported workflows:

- retinal fundus vessel segmentation with uncertainty visualization;
- bone and joint X-ray fracture-candidate localization;
- pediatric chest X-ray pneumonia-pattern classification.

Each completed screening produces a patient-friendly result, model score or
screening status, suggested next step, and downloadable PDF. An optional
language service converts structured model output into simpler wording; images
and patient identifiers are not sent to that service.

## Model scope

The fracture detector uses the published FracAtlas YOLOv8 checkpoint. The chest
classifier was trained locally on PneumoniaMNIST and evaluated on its held-out
test split. The retinal checkpoint maps vessels from fundus images; it does not
classify retinal infection or other eye diseases. These limitations are shown
in the application because a model score is not the same as diagnostic certainty.

ProbStrip does not prescribe medication. Urgent or worsening symptoms require
medical care even when a screening result is negative.

## Run locally

Use Python 3.11 or 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The default retinal checkpoint is `checkpoints/latest_model.pth`. The fracture
checkpoint is `checkpoints/fracture/yolov8_localization_fractureAtlas.pt`; the
pneumonia checkpoint is
`checkpoints/classifiers/pneumoniamnist_pneumonia/latest_model.pth`.

## Patient explanations

The app works without an external API key and falls back to deterministic text.
To enable API-assisted plain-language explanations, create a new restricted API
key and configure one of these secrets:

```toml
GOOGLE_API_KEY = "your-new-restricted-key"
```

Use `.streamlit/secrets.toml.example` as the template. Never commit
`.streamlit/secrets.toml`; it is ignored by Git.

## Camera modes

The upload page accepts a device-camera still for all supported image types.
The camera page also provides a low-latency retinal vessel preview using WebRTC.
Camera access requires HTTPS outside `localhost`; Streamlit Community Cloud
provides HTTPS.

## Test

```powershell
python -m pip install -r requirements-dev.txt
pytest -q
```

## Train and evaluate

Retinal segmentation:

```powershell
python run_dataset.py C:\path\to\dataset --epochs 50 --patience 8 --dataset-name CHASE_DB1
python evaluate_dataset.py C:\path\to\independent-validation-data --output-dir evaluation_results
```

Fracture localization:

```powershell
python prepare_fracatlas.py "C:\path\to\FracAtlas" "C:\path\to\FracAtlas-yolo"
python train_fracatlas.py "C:\path\to\FracAtlas-yolo\dataset.yaml" --epochs 50
```

The checked-in pneumonia checkpoint was trained with the generic classification
pipeline using PneumoniaMNIST. Provenance and measured test performance are in
`checkpoints/classifiers/pneumoniamnist_pneumonia/training_report.json`.

## Deploy free on Streamlit Community Cloud

1. Sign in at <https://share.streamlit.io> with the GitHub account that owns the repository.
2. Create an app from `Partha81-star/ProbStrip`, branch `main`, entrypoint `app.py`.
3. Select Python 3.11 in advanced settings.
4. Add the replacement API key to the app's Secrets field using the TOML name above.
5. Deploy. New commits to `main` are redeployed automatically.

Images are processed in memory and are not intentionally saved by the app. Do
not upload identifiable medical images to a public demonstration deployment.

## Project structure

```text
app.py                              Responsive Streamlit interface
clinical/chest_detection.py         Pneumonia classifier inference
clinical/fracture_detection.py      Fracture detector inference and overlays
clinical/patient_explanation.py     Plain-language structured explanation layer
clinical/reporting.py               Retinal report exports
clinical/general_reporting.py       X-ray report exports
clinical/pdf_reporting.py           Patient PDF generation
clinical/live.py                    Real-time retinal camera preview
models/                             Segmentation and classifier architectures
training/                           Reproducible offline training utilities
tests/                              Inference, safety, UI, and report tests
```

See `MODEL_CARD.md`, `model_cards/`, and `DATASETS.md` for dataset provenance,
measured performance, intended populations, and known limitations.
