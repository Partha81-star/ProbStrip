import glob
import io
import os
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader

from data.dataset import ElongatedStructureDataset
from data.preprocessing import MedicalImagePreprocessor
from inference.mc_dropout_inference import StochasticInferenceEngine
from models.probabilistic_unet import ProbabilisticUNet
from training.losses import BCEDiceLoss
from training.metrics import (
    dice_similarity_coefficient,
    expected_calibration_error,
    intersection_over_union,
)


st.set_page_config(
    page_title="ProbStrip Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: clamp(1.8rem, 4vw, 2.5rem);
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: clamp(0.85rem, 2vw, 1.05rem);
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.4;
        font-weight: 400;
    }
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #f8fafc;
        }
        .sub-header {
            color: #94a3b8;
        }
    }
    @media (max-width: 768px) {
        .stButton>button {
            width: 100% !important;
        }
        .stSelectbox, .stSlider {
            margin-bottom: 8px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cached_model(checkpoint_path, device, in_channels=1, out_channels=1, strip_kernel_size=7):
    model = ProbabilisticUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        features=[32, 64, 128, 256],
        strip_kernel_size=strip_kernel_size,
        dropout_prob=0.2,
    ).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
    model.eval()
    return model


@st.cache_data
def get_default_dataset_paths():
    default_base = r"C:\Users\parth\Documents\Prob-strip-dataset"
    img_dir = os.path.join(default_base, "Images")
    mask_dir = os.path.join(default_base, "Masks")

    if os.path.exists(img_dir):
        images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        return images, mask_dir
    return [], ""


def main():
    st.markdown('<div class="main-header">ProbStrip Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Interpretable Medical Image Segmentation with Probabilistic Strip-CNNs & Monte Carlo Uncertainty Quantification</div>',
        unsafe_allow_html=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with st.sidebar:
        st.subheader("Navigation")
        app_mode = st.radio(
            "Select Section",
            [
                "Single-Scan Diagnostics",
                "Live Camera Analysis",
                "Training Manager",
                "Batch Evaluation",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown(f"**Compute Device**: `{device.upper()}`")
        st.markdown(f"**Model Checkpoint**: `checkpoints/latest_model.pth`")

    if app_mode == "Single-Scan Diagnostics":
        st.subheader("Single-Scan Diagnostics & Uncertainty Quantification")

        default_images, default_mask_dir = get_default_dataset_paths()

        with st.expander("Inference Parameters", expanded=False):
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            with exp_col1:
                num_samples = st.slider("MC Passes (N)", 1, 50, 15, step=1)
            with exp_col2:
                uncertainty_thresh = st.slider("Uncertainty Threshold", 0.005, 0.08, 0.02, step=0.005, format="%.3f")
            with exp_col3:
                decision_thresh = st.slider("Decision Threshold (tau)", 0.1, 0.9, 0.50, step=0.01, format="%.2f")
                checkpoint_path = st.text_input("Checkpoint Path", value="checkpoints/latest_model.pth")

        if "num_samples" not in locals():
            num_samples = 15
            uncertainty_thresh = 0.02
            decision_thresh = 0.50
            checkpoint_path = "checkpoints/latest_model.pth"

        col_src1, col_src2 = st.columns([1, 1])

        input_bytes = None
        mask_bytes = None
        selected_img_path = None
        matched_mask_path = None

        with col_src1:
            input_source = st.radio(
                "Image Source",
                ["Preset Dataset Sample", "Upload Custom Image"],
                horizontal=True,
            )

        with col_src2:
            if input_source == "Preset Dataset Sample":
                if default_images:
                    selected_img_path = st.selectbox(
                        "Choose Sample Image",
                        default_images,
                        format_func=lambda x: os.path.basename(x),
                    )
                    base_name = os.path.splitext(os.path.basename(selected_img_path))[0]
                    potential_mask = os.path.join(default_mask_dir, f"{base_name}_1stHO.png")
                    if os.path.exists(potential_mask):
                        matched_mask_path = potential_mask
                else:
                    st.info("No preset dataset found at local path. Use 'Upload Custom Image' option.")
            else:
                uploaded_img = st.file_uploader("Upload Scan (JPG/PNG/TIF)", type=["png", "jpg", "jpeg", "tif"])
                if uploaded_img is not None:
                    input_bytes = uploaded_img.read()

                uploaded_mask = st.file_uploader("Upload Ground Truth Mask (Optional)", type=["png", "jpg", "jpeg", "tif"])
                if uploaded_mask is not None:
                    mask_bytes = uploaded_mask.read()

        run_diag = st.button("Run Stochastic Inference", type="primary", use_container_width=True)

        if "current_image_key" not in st.session_state:
            st.session_state["current_image_key"] = None
        if "inference_cache" not in st.session_state:
            st.session_state["inference_cache"] = None

        image_identifier = (
            selected_img_path
            if input_source == "Preset Dataset Sample"
            else (hash(input_bytes) if input_bytes else None)
        )

        need_recompute = run_diag or (
            st.session_state["inference_cache"] is None and image_identifier is not None
        ) or (
            st.session_state["current_image_key"] != (image_identifier, num_samples, checkpoint_path)
            and run_diag
        )

        if need_recompute and image_identifier is not None:
            raw_img = None
            gt_mask = None

            if selected_img_path is not None and input_source == "Preset Dataset Sample":
                raw_img = cv2.imread(selected_img_path, cv2.IMREAD_GRAYSCALE)
                if matched_mask_path and os.path.exists(matched_mask_path):
                    gt_mask = cv2.imread(matched_mask_path, cv2.IMREAD_GRAYSCALE)
            elif input_bytes is not None:
                nparr = np.frombuffer(input_bytes, np.uint8)
                raw_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if mask_bytes is not None:
                    mask_arr = np.frombuffer(mask_bytes, np.uint8)
                    gt_mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)

            if raw_img is not None:
                raw_img = cv2.resize(raw_img, (256, 256))
                preprocessor = MedicalImagePreprocessor(use_clahe=True, norm_mode="minmax")
                img_tensor = preprocessor.process(raw_img)

                model = get_cached_model(checkpoint_path, device)
                engine = StochasticInferenceEngine(
                    model=model,
                    num_samples=num_samples,
                    decision_threshold=decision_thresh,
                    uncertainty_threshold=uncertainty_thresh,
                    device=device,
                )

                results = engine.predict_stochastic(img_tensor)

                st.session_state["inference_cache"] = {
                    "raw_img": raw_img,
                    "gt_mask": gt_mask,
                    "mean_pred": results["mean_prediction"].squeeze().cpu().numpy(),
                    "variance_map": results["variance_map"].squeeze().cpu().numpy(),
                }
                st.session_state["current_image_key"] = (image_identifier, num_samples, checkpoint_path)

        if st.session_state["inference_cache"] is not None:
            cached = st.session_state["inference_cache"]
            raw_img = cached["raw_img"]
            gt_mask = cached["gt_mask"]
            mean_pred = cached["mean_pred"]
            variance_map = cached["variance_map"]

            binary_mask = (mean_pred >= decision_thresh).astype(np.float32)
            low_conf = (variance_map >= uncertainty_thresh).astype(np.float32)

            var_norm = (
                (variance_map - variance_map.min())
                / (variance_map.max() - variance_map.min() + 1e-8)
                * 255
            ).astype(np.uint8)
            var_color = cv2.applyColorMap(var_norm, cv2.COLORMAP_JET)
            var_color_rgb = cv2.cvtColor(var_color, cv2.COLOR_BGR2RGB)

            flag_overlay = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
            flag_overlay[low_conf > 0] = [255, 0, 0]

            st.divider()

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)

            if gt_mask is not None:
                gt_mask_resized = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                gt_bin = (gt_mask_resized > 127).astype(np.float32)

                dsc = dice_similarity_coefficient(mean_pred, gt_bin, threshold=decision_thresh)
                iou = intersection_over_union(mean_pred, gt_bin, threshold=decision_thresh)
                ece = expected_calibration_error(mean_pred, gt_bin)

                col_m1.metric("Dice Similarity (DSC)", f"{dsc:.4f}")
                col_m2.metric("IoU (Jaccard)", f"{iou:.4f}")
                col_m3.metric("Calibration Error (ECE)", f"{ece:.4f}")
                col_m4.metric("Mean Epistemic Variance", f"{np.mean(variance_map):.6f}")
            else:
                col_m1.metric("Dice Similarity (DSC)", "N/A - No Mask")
                col_m2.metric("IoU (Jaccard)", "N/A - No Mask")
                col_m3.metric("Calibration Error (ECE)", "N/A - No Mask")
                col_m4.metric("Mean Epistemic Variance", f"{np.mean(variance_map):.6f}")

            st.divider()
            st.subheader("Visual Interpretability & Uncertainty Decomposition")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.image(raw_img, caption="1. Input Medical Scan", use_container_width=True)
                if gt_mask is not None:
                    st.image((gt_bin * 255).astype(np.uint8), caption="Ground Truth Mask", use_container_width=True)

            with c2:
                st.image((mean_pred * 255).astype(np.uint8), caption="2. Probabilistic Mean Mask", use_container_width=True)
                st.image((binary_mask * 255).astype(np.uint8), caption="Binary Decision Mask", use_container_width=True)

            with c3:
                st.image(var_color_rgb, caption="3. Epistemic Uncertainty (Variance)", use_container_width=True)

            with c4:
                st.image(flag_overlay, caption="4. Low-Confidence Warning (Red)", use_container_width=True)

    elif app_mode == "Live Camera Analysis":
        st.subheader("Live Camera Capture & Analysis")

        if "webcam_history" not in st.session_state:
            st.session_state["webcam_history"] = []

        with st.expander("Camera Parameters", expanded=False):
            cam_s1, cam_s2, cam_s3, cam_s4 = st.columns(4)
            with cam_s1:
                cam_mc_passes = st.slider("MC Passes", 5, 30, 10, step=5, key="cam_mc")
            with cam_s2:
                cam_decision = st.slider("Decision Threshold", 0.1, 0.9, 0.5, step=0.05, key="cam_dec")
            with cam_s3:
                cam_unc_thresh = st.slider("Uncertainty Threshold", 0.005, 0.08, 0.02, step=0.005, key="cam_unc")
            with cam_s4:
                cam_checkpoint = st.text_input("Checkpoint Path", value="checkpoints/latest_model.pth", key="cam_ckpt")

        if "cam_mc_passes" not in locals():
            cam_mc_passes = 10
            cam_decision = 0.5
            cam_unc_thresh = 0.02
            cam_checkpoint = "checkpoints/latest_model.pth"

        camera_image = st.camera_input("Capture Scan via Camera")

        if camera_image is not None:
            cam_bytes = camera_image.getvalue()
            nparr = np.frombuffer(cam_bytes, np.uint8)
            captured_frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if captured_frame is not None:
                captured_frame = cv2.resize(captured_frame, (256, 256))
                preprocessor = MedicalImagePreprocessor(use_clahe=True, norm_mode="minmax")
                cam_tensor = preprocessor.process(captured_frame)

                cam_model = get_cached_model(cam_checkpoint, device)
                cam_engine = StochasticInferenceEngine(
                    model=cam_model,
                    num_samples=cam_mc_passes,
                    decision_threshold=cam_decision,
                    uncertainty_threshold=cam_unc_thresh,
                    device=device,
                )

                cam_results = cam_engine.predict_stochastic(cam_tensor)

                cam_mean = cam_results["mean_prediction"].squeeze().cpu().numpy()
                cam_var = cam_results["variance_map"].squeeze().cpu().numpy()
                cam_binary = (cam_mean >= cam_decision).astype(np.float32)
                cam_low_conf = (cam_var >= cam_unc_thresh).astype(np.float32)

                cam_var_norm = (
                    (cam_var - cam_var.min())
                    / (cam_var.max() - cam_var.min() + 1e-8)
                    * 255
                ).astype(np.uint8)
                cam_var_color = cv2.applyColorMap(cam_var_norm, cv2.COLORMAP_JET)
                cam_var_rgb = cv2.cvtColor(cam_var_color, cv2.COLOR_BGR2RGB)

                cam_overlay = cv2.cvtColor(captured_frame, cv2.COLOR_GRAY2RGB)
                cam_overlay[cam_low_conf > 0] = [255, 0, 0]

                foreground_ratio = float(np.mean(cam_binary))
                mean_variance = float(np.mean(cam_var))
                max_variance = float(np.max(cam_var))
                low_conf_ratio = float(np.mean(cam_low_conf))

                st.divider()
                st.subheader("Analysis Metrics")

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Foreground Coverage", f"{foreground_ratio * 100:.1f}%")
                mc2.metric("Mean Variance", f"{mean_variance:.6f}")
                mc3.metric("Max Variance", f"{max_variance:.6f}")
                mc4.metric("Low-Confidence Ratio", f"{low_conf_ratio * 100:.1f}%")

                vc1, vc2, vc3, vc4 = st.columns(4)
                with vc1:
                    st.image(captured_frame, caption="1. Captured Scan", use_container_width=True)
                with vc2:
                    st.image((cam_mean * 255).astype(np.uint8), caption="2. Probabilistic Mask", use_container_width=True)
                with vc3:
                    st.image(cam_var_rgb, caption="3. Uncertainty Heatmap", use_container_width=True)
                with vc4:
                    st.image(cam_overlay, caption="4. Low-Confidence Warning (Red)", use_container_width=True)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["webcam_history"].append({
                    "Timestamp": timestamp,
                    "Foreground Coverage (%)": round(foreground_ratio * 100, 2),
                    "Mean Variance": round(mean_variance, 6),
                    "Max Variance": round(max_variance, 6),
                    "Low-Confidence Ratio (%)": round(low_conf_ratio * 100, 2),
                    "MC Passes": cam_mc_passes,
                    "Decision Threshold": cam_decision,
                    "Uncertainty Threshold": cam_unc_thresh,
                })

        if st.session_state["webcam_history"]:
            st.divider()
            st.subheader("Session History")
            hist_df = pd.DataFrame(st.session_state["webcam_history"])
            st.dataframe(hist_df, use_container_width=True)

            hist_csv = io.StringIO()
            hist_df.to_csv(hist_csv, index=False)
            st.download_button(
                label="Download Session Report (CSV)",
                data=hist_csv.getvalue(),
                file_name=f"probstrip_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            if st.button("Clear History", use_container_width=True):
                st.session_state["webcam_history"] = []
                st.rerun()

    elif app_mode == "Training Manager":
        st.subheader("Model Training Manager")
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            dataset_source = st.selectbox(
                "Training Dataset",
                ["Preset Local CHASE_DB1", "Synthetic Generator"],
            )
            epochs = st.slider("Number of Epochs", 5, 50, 15, step=5)
            batch_size = st.select_slider("Batch Size", options=[2, 4, 8, 16], value=4)

        with col_t2:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-4, 5e-4, 1e-3, 5e-3],
                value=1e-3,
                format_func=lambda x: f"{x:.0e}",
            )
            save_checkpoint_dir = st.text_input("Checkpoint Directory", value="checkpoints")

        start_training = st.button("Start Training", type="primary", use_container_width=True)

        if start_training:
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            st.info("Initializing dataset and dataloaders...")

            default_images, default_mask_dir = get_default_dataset_paths()

            if dataset_source == "Preset Local CHASE_DB1" and default_images:
                matched_pairs = []
                for impath in default_images:
                    bname = os.path.splitext(os.path.basename(impath))[0]
                    mpath = os.path.join(default_mask_dir, f"{bname}_1stHO.png")
                    if os.path.exists(mpath):
                        matched_pairs.append((impath, mpath))

                if matched_pairs:
                    train_imgs = [p[0] for p in matched_pairs]
                    train_masks = [p[1] for p in matched_pairs]
                    dataset = ElongatedStructureDataset(
                        image_paths=train_imgs,
                        mask_paths=train_masks,
                        image_size=(256, 256),
                        augment=True,
                    )
                else:
                    dataset = ElongatedStructureDataset(
                        length=40, image_size=(256, 256), augment=True
                    )
            else:
                dataset = ElongatedStructureDataset(
                    length=40, image_size=(256, 256), augment=True
                )

            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            model = ProbabilisticUNet(
                in_channels=1,
                out_channels=1,
                features=[32, 64, 128, 256],
                strip_kernel_size=7,
                dropout_prob=0.2,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = BCEDiceLoss()

            prog_bar = st.progress(0)
            status_text = st.empty()
            plot_holder = st.empty()

            history_losses = []
            history_dsc = []

            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                epoch_dsc = 0.0

                for images, masks in loader:
                    images = images.to(device)
                    masks = masks.to(device)

                    optimizer.zero_grad()
                    preds = model(images)
                    loss = criterion(preds, masks)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * images.size(0)
                    epoch_dsc += dice_similarity_coefficient(preds, masks) * images.size(0)

                total_samples = len(dataset)
                avg_loss = epoch_loss / total_samples
                avg_dsc = epoch_dsc / total_samples

                history_losses.append(avg_loss)
                history_dsc.append(avg_dsc)

                prog_bar.progress((epoch + 1) / epochs)
                status_text.markdown(
                    f"**Epoch {epoch + 1}/{epochs}** — Loss: `{avg_loss:.4f}` | DSC: `{avg_dsc:.4f}`"
                )

                df_metrics = pd.DataFrame({"BCEDice Loss": history_losses, "Train DSC": history_dsc})
                plot_holder.line_chart(df_metrics)

            save_path = os.path.join(save_checkpoint_dir, "latest_model.pth")
            torch.save({"model_state_dict": model.state_dict()}, save_path)
            st.success(f"Training completed. Model saved to `{save_path}`")
            get_cached_model.clear()

    elif app_mode == "Batch Evaluation":
        st.subheader("Batch Evaluation & Reporting")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            batch_dir = st.text_input(
                "Dataset Directory",
                value=r"C:\Users\parth\Documents\Prob-strip-dataset",
            )
        with col_b2:
            batch_samples = st.slider("MC Passes", 5, 30, 10, step=5)

        run_batch = st.button("Run Batch Evaluation", type="primary", use_container_width=True)

        if run_batch:
            img_dir = os.path.join(batch_dir, "Images")
            mask_dir = os.path.join(batch_dir, "Masks")

            if not os.path.exists(img_dir):
                st.error(f"Images directory not found at: {img_dir}")
            else:
                image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
                st.info(f"Found {len(image_files)} scans. Executing stochastic inference...")

                model = get_cached_model("checkpoints/latest_model.pth", device)
                engine = StochasticInferenceEngine(
                    model=model,
                    num_samples=batch_samples,
                    decision_threshold=0.5,
                    uncertainty_threshold=0.02,
                    device=device,
                )

                records = []
                b_prog = st.progress(0)

                for idx, impath in enumerate(image_files):
                    bname = os.path.basename(impath)
                    name_no_ext = os.path.splitext(bname)[0]
                    maskpath = os.path.join(mask_dir, f"{name_no_ext}_1stHO.png")

                    raw_img = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
                    raw_img = cv2.resize(raw_img, (256, 256))
                    preprocessor = MedicalImagePreprocessor(use_clahe=True, norm_mode="minmax")
                    img_tensor = preprocessor.process(raw_img)

                    results = engine.predict_stochastic(img_tensor)
                    mean_pred = results["mean_prediction"].squeeze().cpu().numpy()
                    variance_map = results["variance_map"].squeeze().cpu().numpy()
                    mean_var = float(np.mean(variance_map))

                    dsc, iou, ece = np.nan, np.nan, np.nan
                    if os.path.exists(maskpath):
                        gt_mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
                        gt_mask = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                        gt_bin = (gt_mask > 127).astype(np.float32)

                        dsc = dice_similarity_coefficient(mean_pred, gt_bin)
                        iou = intersection_over_union(mean_pred, gt_bin)
                        ece = expected_calibration_error(mean_pred, gt_bin)

                    records.append({
                        "Image Name": bname,
                        "Dice Similarity (DSC)": dsc,
                        "IoU (Jaccard)": iou,
                        "Calibration Error (ECE)": ece,
                        "Mean Epistemic Variance": mean_var,
                    })

                    b_prog.progress((idx + 1) / len(image_files))

                df_results = pd.DataFrame(records)

                st.subheader("Summary Statistics")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Average DSC", f"{df_results['Dice Similarity (DSC)'].mean():.4f}")
                sc2.metric("Average IoU", f"{df_results['IoU (Jaccard)'].mean():.4f}")
                sc3.metric("Average ECE", f"{df_results['Calibration Error (ECE)'].mean():.4f}")
                sc4.metric("Average Epistemic Variance", f"{df_results['Mean Epistemic Variance'].mean():.6f}")

                st.subheader("Per-Image Results")
                st.dataframe(df_results, use_container_width=True)

                csv_buffer = io.StringIO()
                df_results.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Full Evaluation Report (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="probstrip_evaluation_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
