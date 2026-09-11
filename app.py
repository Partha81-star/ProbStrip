import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from clinical.analysis import (
    build_visuals,
    calculate_research_measures,
    model_input_channel,
    resize_scan,
    review_outcome,
)
from clinical.biomarkers import calculate_vascular_biomarkers, reviewed_mask
from clinical.modalities import MODALITIES, analyze_general_image
from clinical.quality import assess_image_quality
from clinical.registration import register_followup, vessel_change_map
from clinical.live import LiveVesselProcessor
from clinical.reporting import (
    make_case_id,
    make_report_payload,
    report_as_fhir,
    report_as_html,
    report_as_json,
)
from data.preprocessing import MedicalImagePreprocessor
from inference.mc_dropout_inference import StochasticInferenceEngine
from models.probabilistic_unet import ProbabilisticUNet


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = APP_ROOT / "checkpoints" / "latest_model.pth"
DEMO_IMAGE = APP_ROOT / "test_results" / "Image_14L_input.png"
DEFAULT_CALIBRATION = APP_ROOT / "calibration" / "profile.json"

st.set_page_config(
    page_title="ProbStrip Retinal Review",
    page_icon="PS",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root { --ink:#17202a; --muted:#5f6b76; --teal:#0f766e; --amber:#b45309; }
    .block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem; }
    h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
    .brand { font-size: 1.55rem; font-weight: 750; color: var(--ink); margin-bottom: .15rem; }
    .brand-sub { color: var(--muted); margin-bottom: 1.3rem; }
    .safety-bar { border-left: 5px solid var(--amber); background:#fff7ed; padding: .8rem 1rem; margin: .4rem 0 1.3rem; }
    .result-ready { border-left: 5px solid #0f766e; background:#ecfdf5; padding: 1rem; }
    .result-review { border-left: 5px solid #d97706; background:#fffbeb; padding: 1rem; }
    .result-retake { border-left: 5px solid #b91c1c; background:#fef2f2; padding: 1rem; }
    .plain-note { color:var(--muted); font-size:.92rem; }
    [data-testid="stMetric"] { border-top: 2px solid #d7e3e1; padding-top: .65rem; }
    [data-testid="stSidebar"] { background:#f7faf9; }
    [data-testid="stFileUploaderDropzone"] { min-height: 7rem; }
    video { width:100% !important; height:auto !important; border-radius:6px; }
    iframe { max-width:100%; }
    .stButton > button, .stDownloadButton > button { border-radius:6px; min-height:2.65rem; }
    @media (prefers-color-scheme: dark) {
      h1, h2, h3, .brand { color:#f4f7f6; }
      .brand-sub, .plain-note { color:#b4c0bc; }
      .safety-bar { background:#422006; }
      .result-ready { background:#052e2b; }
      .result-review { background:#422006; }
      .result-retake { background:#450a0a; }
      [data-testid="stSidebar"] { background:#111917; }
    }
    @media (max-width: 720px) {
      .block-container { padding: .8rem .9rem 2rem; }
      h1 { font-size:1.55rem !important; line-height:1.25; }
      h2 { font-size:1.25rem !important; }
      [data-testid="stHorizontalBlock"] { gap:.7rem; }
      [data-testid="stMetric"] { min-width:0; }
      [data-testid="stMetricValue"] { font-size:1.35rem; }
      [data-testid="stFileUploaderDropzoneInstructions"] span { font-size:.84rem; }
      .stButton > button, .stDownloadButton > button { width:100%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PATIENT_TEXT = {
    "English": {
        "no_diagnosis": "Research vessel analysis only",
        "no_diagnosis_body": (
            "ProbStrip maps visible blood vessels and marks places where the software "
            "is unsure. It cannot tell whether you have an eye disease."
        ),
        "ready": "Vessel map ready for clinician review",
        "review": "Clinician review is especially important",
        "retake": "A clearer retinal image is needed",
        "next": "What to do next",
        "ready_body": (
            "The software produced a vessel map with limited flagged uncertainty. "
            "Only a qualified clinician can interpret what it means for your health."
        ),
        "ready_next": "Discuss the image during your normal eye-care appointment.",
        "review_body": (
            "The model was unsure in a noticeable part of the image. Amber areas "
            "show where its vessel map needs closer review."
        ),
        "review_next": "Share the original image and this report with an eye-care professional.",
        "retake_body": (
            "The capture-quality check found issues that can make the vessel map "
            "unreliable. This result should not be interpreted clinically."
        ),
        "retake_next": "Ask the imaging professional to repeat the retinal photograph.",
    },
    "Hindi": {
        "no_diagnosis": "कोई निदान तैयार नहीं किया गया",
        "no_diagnosis_body": (
            "ProbStrip दिखाई देने वाली रक्त वाहिकाओं का नक्शा बनाता है और उन स्थानों "
            "को चिन्हित करता है जहां सॉफ्टवेयर अनिश्चित है। यह आंख की बीमारी का निदान नहीं करता।"
        ),
        "ready": "रक्त वाहिका मानचित्र डॉक्टर की समीक्षा के लिए तैयार है",
        "review": "डॉक्टर द्वारा समीक्षा विशेष रूप से जरूरी है",
        "retake": "रेटिना की अधिक स्पष्ट तस्वीर की जरूरत है",
        "next": "अब क्या करें",
        "ready_body": (
            "सॉफ्टवेयर ने सीमित अनिश्चितता के साथ रक्त वाहिकाओं का नक्शा बनाया है। "
            "केवल योग्य डॉक्टर ही आपके स्वास्थ्य के लिए इसका अर्थ बता सकते हैं।"
        ),
        "ready_next": "अपनी नियमित आंखों की जांच में डॉक्टर से इस तस्वीर पर चर्चा करें।",
        "review_body": (
            "मॉडल तस्वीर के एक महत्वपूर्ण हिस्से में अनिश्चित था। नारंगी भाग दिखाते हैं "
            "कि डॉक्टर को कहां अधिक ध्यान से समीक्षा करनी चाहिए।"
        ),
        "review_next": "मूल तस्वीर और यह रिपोर्ट आंखों के डॉक्टर को दिखाएं।",
        "retake_body": (
            "तस्वीर की गुणवत्ता में ऐसी समस्याएं मिलीं जो रक्त वाहिका मानचित्र को "
            "अविश्वसनीय बना सकती हैं। इस परिणाम की चिकित्सकीय व्याख्या न करें।"
        ),
        "retake_next": "इमेजिंग पेशेवर से रेटिना की तस्वीर दोबारा लेने को कहें।",
    },
}


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, device: str, variant="report"):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"The trained model was not found at {path}. No prediction was made."
        )

    model = ProbabilisticUNet(
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256],
        strip_kernel_size=7,
        dropout_prob=0.2,
    ).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def get_live_processor(checkpoint_path, device, decision_threshold):
    model = load_model(checkpoint_path, device, variant="live")
    return LiveVesselProcessor(
        model=model,
        device=device,
        decision_threshold=decision_threshold,
        process_every=2,
        input_size=128,
    )


def decode_image(image_bytes: bytes):
    encoded = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("This file could not be read as an image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_png(image: np.ndarray) -> bytes:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("The processed image could not be exported.")
    return encoded.tobytes()


def resize_for_review(image: np.ndarray, maximum_side: int = 1400) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, maximum_side / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


@st.cache_data(show_spinner=False)
def file_fingerprint(path_string: str):
    digest = hashlib.sha256()
    with Path(path_string).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@st.cache_data(show_spinner=False)
def load_calibration_profile(profile_path: str):
    path = Path(profile_path)
    if not path.is_file():
        return {
            "status": "uncalibrated research defaults",
            "decision_threshold": 0.5,
            "uncertainty_threshold": 0.02,
            "dataset": "none",
        }
    with path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    required = {"status", "decision_threshold", "uncertainty_threshold"}
    if not required.issubset(profile):
        raise ValueError(f"Calibration profile is missing: {sorted(required - profile.keys())}")
    return profile


def run_analysis(rgb_image, settings):
    display = resize_scan(rgb_image)
    quality = assess_image_quality(display)
    empty_measures = {
        "visible_vessel_coverage_percent": 0.0,
        "central_vessel_coverage_percent": 0.0,
        "low_confidence_area_percent": 100.0,
        "mean_model_variance": 0.0,
        "maximum_model_variance": 0.0,
        "fractal_dimension": 0.0,
        "skeleton_length_pixels": 0,
        "estimated_mean_width_pixels": 0.0,
        "endpoint_region_count": 0,
        "branch_region_count": 0,
        "vessel_component_count": 0,
        "skeleton_density_percent": 0.0,
    }

    if quality.status == "Retake recommended":
        outcome = review_outcome(quality.status, 100.0)
        return {
            "quality": quality,
            "outcome": outcome,
            "measures": empty_measures,
            "display": display,
            "prediction": None,
        }

    channel = model_input_channel(display)
    tensor = MedicalImagePreprocessor(use_clahe=True, norm_mode="minmax").process(
        channel
    )
    model = load_model(settings["checkpoint"], settings["device"])
    engine = StochasticInferenceEngine(
        model=model,
        num_samples=settings["mc_samples"],
        decision_threshold=settings["decision_threshold"],
        uncertainty_threshold=settings["uncertainty_threshold"],
        device=settings["device"],
    )
    result = engine.predict_stochastic(tensor)
    mean_prediction = result["mean_prediction"].squeeze().cpu().numpy()
    variance_map = result["variance_map"].squeeze().cpu().numpy()
    binary, uncertain, overlay, heat = build_visuals(
        display,
        mean_prediction,
        variance_map,
        settings["decision_threshold"],
        settings["uncertainty_threshold"],
    )
    measures = calculate_research_measures(display, binary, uncertain, variance_map)
    outcome = review_outcome(
        quality.status, measures["low_confidence_area_percent"]
    )
    return {
        "quality": quality,
        "outcome": outcome,
        "measures": measures,
        "display": display,
        "prediction": mean_prediction,
        "variance": variance_map,
        "binary": binary,
        "uncertain": uncertain,
        "overlay": overlay,
        "heat": heat,
    }


def quality_table(quality):
    rows = []
    for check in quality.checks:
        rows.append(
            {
                "Capture check": check.name,
                "Measured value": f"{check.value:.1f} {check.unit}",
                "Result": check.status,
                "How to improve": check.guidance,
            }
        )
    return pd.DataFrame(rows)


def effective_mask(case):
    return case.get("reviewed_binary", case.get("binary"))


def effective_measures(case):
    return case.get("reviewed_measures", case["measures"])


def create_and_store_case(image_bytes, settings, language):
    rgb = decode_image(image_bytes)
    image_hash = hashlib.sha256(image_bytes).hexdigest()[:12]
    case = run_analysis(rgb, settings)
    case_id = make_case_id()
    report_text = PATIENT_TEXT[language]
    report_outcome = {
        **case["outcome"],
        "title": report_text[case["outcome"]["level"]],
        "explanation": report_text[f'{case["outcome"]["level"]}_body'],
        "next_step": report_text[f'{case["outcome"]["level"]}_next'],
    }
    payload = make_report_payload(
        case_id,
        case["quality"],
        case["measures"],
        report_outcome,
        {
            "model": "ProbStrip StripConv U-Net",
            "mc_samples": settings["mc_samples"],
            "decision_threshold": settings["decision_threshold"],
            "uncertainty_threshold": settings["uncertainty_threshold"],
            "image_fingerprint": image_hash,
            "calibration_status": settings["calibration_status"],
            "calibration_dataset": settings["calibration_dataset"],
            "checkpoint_sha256": settings["checkpoint_sha256"],
        },
        language=language,
    )
    case.update({"case_id": case_id, "payload": payload})
    st.session_state.cases.append(case)
    st.session_state.active_case = len(st.session_state.cases) - 1
    return case


def render_patient_result(case, language):
    text = PATIENT_TEXT[language]
    outcome = case["outcome"]
    display_measures = effective_measures(case)
    clinician_reviewed = (
        case["payload"].get("clinician_review", {}).get("status") == "reviewed"
    )
    translated_title = text.get(outcome["level"], outcome["title"])
    translated_body = text[f'{outcome["level"]}_body']
    translated_next = text[f'{outcome["level"]}_next']
    st.markdown(
        f'<div class="safety-bar"><strong>{text["no_diagnosis"]}</strong><br>'
        f'{text["no_diagnosis_body"]}</div>',
        unsafe_allow_html=True,
    )
    if clinician_reviewed:
        st.success(
            "A qualified clinician marked this report as reviewed. The displayed map "
            "and measurements include the saved correction; the original AI output "
            "remains in the technical record."
        )
    st.markdown(
        f'<div class="result-{outcome["level"]}"><strong>{translated_title}</strong>'
        f'<br>{translated_body}<br><br><strong>{text["next"]}:</strong> '
        f'{translated_next}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Your report", anchor=False)
    m1, m2, m3 = st.columns(3)
    m1.metric("Image quality", f"{case['quality'].score}/100")
    if case["prediction"] is None:
        m2.metric("Vessel map", "Not created")
        m3.metric("Software confidence", "Not available")
    else:
        m2.metric(
            "Visible vessel coverage",
            f"{display_measures['visible_vessel_coverage_percent']:.1f}%",
            help="The share of the visible retinal field marked as vessel by the model. This is not a disease score.",
        )
        m3.metric(
            "Area needing review",
            f"{display_measures['low_confidence_area_percent']:.1f}%",
            help="The share of the image where repeated model passes disagreed.",
        )

    report_tab, image_tab, quality_tab = st.tabs(
        ["What this means", "Your images", "Image-quality details"]
    )
    with report_tab:
        st.markdown("**What the numbers mean**")
        st.write(
            "Visible vessel coverage describes the software's vessel map. Area needing "
            "review describes the software's uncertainty. Neither number says whether "
            "your eye is healthy or unhealthy."
        )
        st.markdown("**When to seek care**")
        st.write(
            "Do not wait for this report if you have sudden vision loss, severe eye pain, "
            "new flashes or many new floaters. Contact a healthcare professional promptly."
        )
    with image_tab:
        if case["prediction"] is None:
            st.image(case["display"], caption="Uploaded retinal image", width="stretch")
            st.info("A vessel map was not created because the image-quality gate stopped analysis.")
        else:
            left, right = st.columns(2)
            left.image(
                case["display"], caption="Original retinal image", width="stretch"
            )
            right.image(
                case.get("reviewed_overlay", case["overlay"]),
                caption="Teal: mapped vessels. Amber: areas needing review.",
                width="stretch",
            )
    with quality_tab:
        st.write(case["quality"].summary)
        st.dataframe(quality_table(case["quality"]), hide_index=True, width="stretch")

    st.subheader("Take this report to your clinician", anchor=False)
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download easy-to-read report",
        report_as_html(case["payload"]),
        file_name=f"{case['case_id']}-patient-report.html",
        mime="text/html",
        width="stretch",
    )
    d2.download_button(
        "Download clinical data",
        report_as_json(case["payload"]),
        file_name=f"{case['case_id']}-clinical-data.json",
        mime="application/json",
        width="stretch",
    )
    st.caption(
        f"Report reference {case['case_id']}. The image and report remain only in this browser session."
    )


def analyze_page(settings, language):
    st.header("Review a retinal image", anchor=False)
    st.write(
        "Upload a retinal photograph to check its capture quality and create a vessel map "
        "for review by an eye-care professional."
    )
    st.markdown(
        '<div class="safety-bar"><strong>Research use only.</strong> This tool does not '
        "diagnose diabetic retinopathy, glaucoma, hypertension, or any other condition.</div>",
        unsafe_allow_html=True,
    )

    with st.form("scan-form", clear_on_submit=False):
        source = st.radio(
            "Choose an image",
            ["Upload my retinal image", "Use the demonstration image"],
            horizontal=True,
        )
        uploaded = None
        if source == "Upload my retinal image":
            uploaded = st.file_uploader(
                "Retinal photograph",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                help="Use a fundus-camera image with the circular retina centered and in focus.",
            )
        consent = st.checkbox(
            "I understand this is a research vessel map, not a medical diagnosis."
        )
        submitted = st.form_submit_button(
            "Check image and create report", type="primary", width="stretch"
        )

    if submitted:
        if not consent:
            st.warning("Please confirm that you understand the intended use before continuing.")
        elif source == "Upload my retinal image" and uploaded is None:
            st.warning("Choose a retinal image first.")
        else:
            try:
                if source == "Use the demonstration image":
                    image_bytes = DEMO_IMAGE.read_bytes()
                else:
                    image_bytes = uploaded.getvalue()
                with st.spinner("Checking image quality and mapping visible vessels..."):
                    create_and_store_case(image_bytes, settings, language)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                st.error(str(exc))

    if st.session_state.active_case is not None:
        render_patient_result(
            st.session_state.cases[st.session_state.active_case], language
        )


def camera_page(settings, language):
    st.header("Camera and live preview", anchor=False)
    st.write(
        "Use a phone camera to capture a retinal photograph, or preview the vessel "
        "overlay continuously with a webcam. A fundus-camera image is still required."
    )
    capture_tab, live_tab = st.tabs(["Take a photo", "Live vessel preview"])

    with capture_tab:
        st.markdown("**Best for phones and tablets**")
        st.caption(
            "Use the rear camera when available. Center the circular retinal image, "
            "avoid screen glare, and hold the device steady."
        )
        camera_image = st.camera_input(
            "Take a retinal photograph",
            help="This opens the device camera. It is not a substitute for a fundus camera.",
        )
        camera_consent = st.checkbox(
            "I understand the captured image creates a research map, not a diagnosis.",
            key="camera-consent",
        )
        analyze_capture = st.button(
            "Check captured image and create report",
            type="primary",
            width="stretch",
            disabled=camera_image is None,
        )
        if analyze_capture:
            if not camera_consent:
                st.warning("Confirm the intended use before creating the report.")
            else:
                try:
                    with st.spinner(
                        "Checking capture quality and running uncertainty analysis..."
                    ):
                        case = create_and_store_case(
                            camera_image.getvalue(), settings, language
                        )
                    render_patient_result(case, language)
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    st.error(str(exc))

    with live_tab:
        st.markdown("**Low-latency research preview**")
        st.info(
            "The live overlay uses one smaller model pass for speed and does not calculate "
            "clinical uncertainty. Teal pixels are the latest vessel estimate. Use the "
            "photo tab to create a full report."
        )
        live_enabled = st.toggle(
            "Enable live webcam",
            help="Your browser will ask for camera permission after this is enabled.",
        )
        if live_enabled:
            try:
                processor = get_live_processor(
                    settings["checkpoint"],
                    settings["device"],
                    settings["decision_threshold"],
                )
                ice_configuration = {
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]}
                    ]
                }
                webrtc_streamer(
                    key="probstrip-live-vessel-preview",
                    mode=WebRtcMode.SENDRECV,
                    frontend_rtc_configuration=ice_configuration,
                    server_rtc_configuration=ice_configuration,
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 640},
                            "height": {"ideal": 480},
                            "facingMode": {"ideal": "environment"},
                        },
                        "audio": False,
                    },
                    video_frame_callback=processor.process,
                    async_processing=True,
                    media_toggle_controls=False,
                )
                st.caption(
                    "Camera access requires HTTPS after deployment. Streamlit Community "
                    "Cloud provides HTTPS; some restricted networks may additionally "
                    "require a TURN server."
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                st.error(str(exc))
        else:
            st.caption("Enable live webcam when you are ready to grant camera access.")


def general_imaging_page():
    st.header("X-ray and other imaging", anchor=False)
    st.write(
        "Improve visibility and review image structure from common medical images. "
        "This workspace does not detect fractures, tumors, infections, or other disease."
    )
    st.markdown(
        '<div class="safety-bar"><strong>Clinician interpretation is required.</strong> '
        "The enhanced and teal edge views are visual aids, not diagnostic findings.</div>",
        unsafe_allow_html=True,
    )

    modality = st.selectbox("Imaging type", list(MODALITIES))
    modality_info = MODALITIES[modality]
    st.caption(modality_info["guidance"])
    source = st.segmented_control(
        "Image source",
        ["Upload image", "Use camera"],
        default="Upload image",
        width="stretch",
    )
    media = None
    if source == "Use camera":
        st.info(
            "Photographing an X-ray film or screen can add glare and distortion. "
            "An original de-identified digital export is more reliable."
        )
        media = st.camera_input(f"Photograph the {modality_info['short']}")
    else:
        media = st.file_uploader(
            f"Upload {modality_info['short']}",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="general-imaging-upload",
            help="Use a de-identified PNG, JPEG, or TIFF export. DICOM is not supported in this release.",
        )

    with st.expander("Image display controls"):
        control_a, control_b = st.columns(2)
        clip_limit = control_a.slider(
            "Contrast enhancement", 1.0, 5.0, 2.5, 0.5
        )
        edge_sensitivity = control_b.slider(
            "Structure-edge sensitivity", 0.5, 2.0, 1.0, 0.1
        )
        invert = st.toggle(
            "Invert black and white",
            help="Useful when the exported radiograph uses the opposite display convention.",
        )

    consent = st.checkbox(
        "I understand these are image-review aids and not a medical diagnosis.",
        key="general-imaging-consent",
    )
    process = st.button(
        "Create image review",
        type="primary",
        width="stretch",
        disabled=media is None,
    )
    if process:
        if not consent:
            st.warning("Confirm the intended use before creating the image review.")
        else:
            try:
                original = resize_for_review(decode_image(media.getvalue()))
                result = analyze_general_image(
                    original,
                    modality,
                    clip_limit=clip_limit,
                    edge_sensitivity=edge_sensitivity,
                    invert=invert,
                )
                result["original"] = original
                st.session_state.general_image_review = result
            except ValueError as exc:
                st.error(str(exc))

    result = st.session_state.get("general_image_review")
    if result is None:
        return
    if result["modality"] != modality:
        st.info("Create a new review to apply the selected imaging type.")
        return

    quality = result["quality"]
    st.subheader(f"{result['modality']} review", anchor=False)
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Technical quality", f"{quality['score']}/100")
    metric_b.metric("Quality status", quality["status"])
    metric_c.metric(
        "Visible edge area",
        f"{result['edge_area_percent']:.1f}%",
        help="The amount of edge contrast in this image, not an abnormality score.",
    )
    original_col, enhanced_col = st.columns(2)
    original_col.image(result["original"], caption="Original image", width="stretch")
    enhanced_col.image(
        result["enhanced"],
        caption="Contrast-enhanced grayscale view",
        width="stretch",
        clamp=True,
    )
    st.image(
        result["overlay"],
        caption="Teal marks visible intensity edges. It does not mark fractures or disease.",
        width="stretch",
    )
    with st.expander("Technical quality details"):
        quality_rows = [
            {
                "Check": check["name"],
                "Measured value": f"{check['value']:.1f} {check['unit']}",
                "Result": "Pass" if check["ok"] else "Review",
            }
            for check in quality["checks"]
        ]
        st.dataframe(pd.DataFrame(quality_rows), hide_index=True, width="stretch")

    summary = {
        "imaging_type": result["modality"],
        "technical_quality": quality,
        "visible_edge_area_percent": result["edge_area_percent"],
        "interpretation": "No diagnosis generated; clinician review required.",
    }
    download_a, download_b = st.columns(2)
    download_a.download_button(
        "Download enhanced image",
        encode_png(result["enhanced"]),
        file_name="probstrip-enhanced-image.png",
        mime="image/png",
        width="stretch",
    )
    download_b.download_button(
        "Download review summary",
        json.dumps(summary, indent=2),
        file_name="probstrip-image-review.json",
        mime="application/json",
        width="stretch",
    )


def compare_page():
    st.header("Compare visits", anchor=False)
    st.write(
        "Align and compare two vessel maps from this session. Measurements are shown "
        "only as research morphology and are not a diagnosis."
    )
    mapped = [case for case in st.session_state.cases if case["prediction"] is not None]
    if len(mapped) < 2:
        st.info("Analyze at least two usable retinal images in this session to compare them.")
        return
    labels = [case["case_id"] for case in mapped]
    c1, c2 = st.columns(2)
    first_label = c1.selectbox("Earlier visit", labels, index=0)
    second_label = c2.selectbox("Later visit", labels, index=len(labels) - 1)
    first = mapped[labels.index(first_label)]
    second = mapped[labels.index(second_label)]
    if first_label == second_label:
        st.warning("Choose two different reports.")
        return

    first_mask = effective_mask(first)
    second_mask = effective_mask(second)
    with st.spinner("Aligning the later image to the earlier visit..."):
        registration = register_followup(
            first["display"], second["display"], second_mask
        )

    if registration.success:
        st.success(
            f"Registration passed ({registration.method}, score {registration.score:.3f})."
        )
    else:
        st.warning(registration.message)

    rows = []
    fields = [
        ("Visible vessel coverage", "visible_vessel_coverage_percent", "%"),
        ("Central vessel coverage", "central_vessel_coverage_percent", "%"),
        ("Area needing review", "low_confidence_area_percent", "%"),
        ("Fractal dimension", "fractal_dimension", ""),
        ("Estimated mean width", "estimated_mean_width_pixels", " px"),
        ("Branch regions", "branch_region_count", ""),
    ]
    for label, key, unit in fields:
        earlier = effective_measures(first)[key]
        later = effective_measures(second)[key]
        rows.append(
            {
                "Research measurement": label,
                "Earlier": f"{earlier:.2f}{unit}",
                "Later": f"{later:.2f}{unit}",
                "Difference": f"{later - earlier:+.2f}{unit}",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    before, after = st.columns(2)
    before.image(first["overlay"], caption=f"Earlier: {first_label}", width="stretch")
    after.image(
        registration.aligned_image if registration.success else second["display"],
        caption=(
            f"Later image aligned: {second_label}"
            if registration.success
            else f"Later image, not aligned: {second_label}"
        ),
        width="stretch",
    )

    if registration.success and registration.aligned_mask is not None:
        change = vessel_change_map(first_mask, registration.aligned_mask)
        st.metric(
            "Changed vessel-map pixels",
            f"{change['changed_vessel_percent']:.1f}%",
            help=(
                "Pixel disagreement after automated alignment. Capture differences and "
                "model errors can cause change; this is not biological progression."
            ),
        )
        st.image(
            change["visualization"],
            caption=(
                "Teal: present in both maps. Blue: only later map. Red: only earlier map."
            ),
            width="stretch",
        )
    st.caption(
        "A clinician must confirm comparable field of view, image quality, and alignment "
        "before interpreting any difference."
    )


def clinician_page():
    st.header("Clinician details", anchor=False)
    st.write(
        "Technical output for review and research documentation. Every report remains preliminary."
    )
    if not st.session_state.cases:
        st.info("Analyze a retinal image to view its technical record.")
        return
    labels = [case["case_id"] for case in st.session_state.cases]
    selected = st.selectbox("Report reference", labels, index=len(labels) - 1)
    case = st.session_state.cases[labels.index(selected)]
    st.dataframe(quality_table(case["quality"]), hide_index=True, width="stretch")
    st.markdown("**Original AI measurements**")
    st.json(case["payload"]["research_measures"], expanded=False)

    note_key = f"note-{selected}"
    sign_key = f"sign-{selected}"
    candidate_mask = None
    if case["prediction"] is not None:
        p1, p2, p3 = st.columns(3)
        p1.image(case["display"], caption="Input", width="stretch")
        p2.image(
            (case["prediction"] * 255).astype(np.uint8),
            caption="Mean vessel probability",
            width="stretch",
        )
        p3.image(case["heat"], caption="MC-dropout variance", width="stretch")

        st.subheader("Review and correct the vessel map", anchor=False)
        correction_source = st.radio(
            "Correction method",
            ["Refine model threshold", "Upload corrected binary mask"],
            horizontal=True,
            key=f"correction-source-{selected}",
        )
        if correction_source == "Refine model threshold":
            controls_a, controls_b = st.columns(2)
            threshold = controls_a.slider(
                "Reviewed vessel threshold",
                0.20,
                0.90,
                float(case["payload"]["model_settings"]["decision_threshold"]),
                0.01,
                key=f"review-threshold-{selected}",
            )
            minimum_area = controls_b.slider(
                "Remove regions smaller than",
                1,
                100,
                4,
                1,
                key=f"review-area-{selected}",
                help="Pixel area at the 256 x 256 analysis resolution.",
            )
            candidate_mask = reviewed_mask(
                case["prediction"], threshold, minimum_area
            )
            correction_details = {
                "method": "threshold refinement",
                "threshold": threshold,
                "minimum_component_area": minimum_area,
            }
        else:
            corrected_upload = st.file_uploader(
                "Corrected mask",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key=f"corrected-mask-{selected}",
                help="Upload a black-background mask with reviewed vessels in white.",
            )
            if corrected_upload is not None:
                corrected_rgb = decode_image(corrected_upload.getvalue())
                corrected_gray = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2GRAY)
                candidate_mask = cv2.resize(
                    corrected_gray,
                    (case["display"].shape[1], case["display"].shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) >= 128
                correction_details = {"method": "uploaded clinician mask"}
            else:
                correction_details = {"method": "uploaded clinician mask"}
                st.info("Upload a corrected mask to preview and approve it.")

        if candidate_mask is not None:
            candidate_measures = calculate_research_measures(
                case["display"],
                candidate_mask,
                case["uncertain"],
                case["variance"],
            )
            _, _, candidate_overlay, _ = build_visuals(
                case["display"],
                candidate_mask.astype(np.float32),
                case["variance"],
                0.5,
                float(case["payload"]["model_settings"]["uncertainty_threshold"]),
            )
            original_col, candidate_col = st.columns(2)
            original_col.image(
                case.get("reviewed_overlay", case["overlay"]),
                caption="Current saved map",
                width="stretch",
            )
            candidate_col.image(
                candidate_overlay,
                caption="Candidate reviewed map",
                width="stretch",
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Research measurement": key.replace("_", " ").title(),
                            "Original AI": case["measures"].get(key),
                            "Candidate review": candidate_measures.get(key),
                        }
                        for key in (
                            "visible_vessel_coverage_percent",
                            "fractal_dimension",
                            "estimated_mean_width_pixels",
                            "branch_region_count",
                            "vessel_component_count",
                        )
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

    st.text_area(
        "Clinician note",
        key=note_key,
        placeholder="Document image limitations, corrections, or follow-up here.",
    )
    st.checkbox("Reviewed by a qualified clinician", key=sign_key)
    save_review = st.button(
        "Save clinician review",
        type="primary",
        width="stretch",
        disabled=(case["prediction"] is not None and candidate_mask is None),
    )
    if save_review and not st.session_state.get(sign_key):
        st.warning("Confirm qualified-clinician review before saving.")
    elif save_review:
        if candidate_mask is not None:
            case["reviewed_binary"] = candidate_mask
            case["reviewed_overlay"] = candidate_overlay
            case["reviewed_measures"] = candidate_measures
        case["payload"]["clinician_review"] = {
            "status": "reviewed",
            "note": st.session_state.get(note_key, ""),
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "correction": correction_details if candidate_mask is not None else None,
            "reviewed_research_measures": (
                candidate_measures if candidate_mask is not None else None
            ),
        }
        st.success("Review status is stored for this session and included in new downloads.")

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download technical JSON",
        report_as_json(case["payload"]),
        file_name=f"{selected}-technical.json",
        mime="application/json",
        width="stretch",
    )
    d2.download_button(
        "Download preliminary FHIR JSON",
        report_as_fhir(case["payload"]),
        file_name=f"{selected}-fhir.json",
        mime="application/fhir+json",
        width="stretch",
    )
    st.caption(
        "The FHIR-shaped export is an interoperability prototype and requires local profiling and validation before EHR use."
    )
    if st.button("Delete selected session report", width="stretch"):
        index = labels.index(selected)
        st.session_state.cases.pop(index)
        st.session_state.active_case = (
            len(st.session_state.cases) - 1 if st.session_state.cases else None
        )
        st.rerun()


def safety_page():
    st.header("Safety and privacy", anchor=False)
    st.subheader("What ProbStrip does", anchor=False)
    st.write(
        "It checks basic capture quality, creates a retinal blood-vessel segmentation, "
        "and highlights pixels where repeated stochastic model passes disagree. The "
        "general imaging workspace can enhance contrast and show intensity edges in "
        "X-rays, CT, MRI, ultrasound, and external photographs."
    )
    st.subheader("What ProbStrip does not do", anchor=False)
    st.write(
        "It does not detect or rule out a disease, fracture, or lesion; prescribe "
        "treatment; replace a clinical examination; or provide emergency advice. The "
        "current AI checkpoint is retinal-only, was developed from a small research "
        "dataset, and has not been prospectively validated."
    )
    st.subheader("Privacy in this release", anchor=False)
    st.write(
        "Images are processed in memory and are not intentionally saved by the application. "
        "Reports stay in the current Streamlit session. Do not upload identifying clinical "
        "images to a public demonstration deployment."
    )
    st.write(
        "Use the sidebar control to clear every image and report reference held by the "
        "current browser session. Closing or expiring the session also releases them."
    )
    st.subheader("For researchers", anchor=False)
    st.write(
        "Clinical use requires representative multi-site evaluation, subgroup analysis, "
        "calibration and abstention validation, cybersecurity controls, quality management, "
        "and the regulatory review applicable in the deployment country."
    )


def main():
    if "cases" not in st.session_state:
        st.session_state.cases = []
    if "active_case" not in st.session_state:
        st.session_state.active_case = None
    if "general_image_review" not in st.session_state:
        st.session_state.general_image_review = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = os.getenv("PROBSTRIP_CHECKPOINT", str(DEFAULT_CHECKPOINT))
    calibration_path = os.getenv(
        "PROBSTRIP_CALIBRATION", str(DEFAULT_CALIBRATION)
    )
    try:
        calibration = load_calibration_profile(calibration_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Calibration profile error: {exc}")
        calibration = {
            "status": "invalid profile; research defaults active",
            "decision_threshold": 0.5,
            "uncertainty_threshold": 0.02,
            "dataset": "none",
        }
    try:
        checkpoint_sha256 = file_fingerprint(checkpoint)
    except OSError:
        checkpoint_sha256 = "unavailable"

    with st.sidebar:
        st.markdown('<div class="brand">ProbStrip</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="brand-sub">Medical image review</div>', unsafe_allow_html=True
        )
        page = st.radio(
            "Navigation",
            [
                "Review image",
                "Camera and live",
                "X-ray and other imaging",
                "Compare visits",
                "Clinician details",
                "Safety and privacy",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        language = st.selectbox("Patient report language", ["English", "Hindi"])
        with st.expander("Advanced model settings"):
            mc_samples = st.slider("Repeated model passes", 5, 20, 10, 5)
            decision_threshold = st.slider(
                "Vessel decision threshold",
                0.20,
                0.90,
                float(np.clip(calibration["decision_threshold"], 0.20, 0.90)),
                0.01,
            )
            uncertainty_threshold = st.slider(
                "Uncertainty flag threshold",
                0.0001,
                0.1000,
                float(np.clip(calibration["uncertainty_threshold"], 0.0001, 0.1)),
                0.0001,
                format="%.4f",
            )
            st.caption(
                f"Calibration: {calibration['status']} | Dataset: "
                f"{calibration.get('dataset', 'unspecified')}"
            )
        st.caption(f"Compute: {device.upper()} | Session reports: {len(st.session_state.cases)}")
        st.caption(f"Model: {checkpoint_sha256[:12]}")
        if (st.session_state.cases or st.session_state.general_image_review) and st.button(
            "Clear all session images and reports", width="stretch"
        ):
            st.session_state.cases = []
            st.session_state.active_case = None
            st.session_state.general_image_review = None
            st.rerun()

    settings = {
        "device": device,
        "checkpoint": checkpoint,
        "mc_samples": mc_samples,
        "decision_threshold": decision_threshold,
        "uncertainty_threshold": uncertainty_threshold,
        "calibration_status": calibration["status"],
        "calibration_dataset": calibration.get("dataset", "unspecified"),
        "checkpoint_sha256": checkpoint_sha256,
    }

    if page == "Review image":
        analyze_page(settings, language)
    elif page == "Camera and live":
        camera_page(settings, language)
    elif page == "X-ray and other imaging":
        general_imaging_page()
    elif page == "Compare visits":
        compare_page()
    elif page == "Clinician details":
        clinician_page()
    else:
        safety_page()


if __name__ == "__main__":
    main()
