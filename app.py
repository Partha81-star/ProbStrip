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
    page_title="ProbStrip - Interpretable Medical Image Segmentation",
    page_icon="🔬",
    layout="wide",
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
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
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
    st.title("🔬 ProbStrip Studio")
    st.markdown(
        "**Interpretable Medical Image Segmentation using Probabilistic Strip-CNNs & Monte Carlo Dropout Uncertainty Quantification**"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Single-Scan UQ Diagnostics",
        "🚀 Extended Training Manager",
        "📊 Batch Evaluation & Report Export",
        "📹 Live Webcam Analysis",
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with tab1:
        st.subheader("Interactive Single-Scan Diagnostics & Uncertainty Quantification")

        default_images, default_mask_dir = get_default_dataset_paths()

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.8, 1.2, 1.2])

        input_bytes = None
        mask_bytes = None
        selected_img_path = None
        matched_mask_path = None

        with col_ctrl1:
            input_source = st.radio(
                "Image Source",
                ["Preset Dataset Sample", "Upload Custom Image"],
                horizontal=True,
            )

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
                    st.info("No preset images found. Set the dataset path in loader_utils.py.")
            else:
                uploaded_img = st.file_uploader(
                    "Upload Medical Image (JPG/PNG/TIF)",
                    type=["png", "jpg", "jpeg", "tif"],
                )
                if uploaded_img is not None:
                    input_bytes = uploaded_img.read()

                uploaded_mask = st.file_uploader(
                    "Upload Ground Truth Mask (Optional)",
                    type=["png", "jpg", "jpeg", "tif"],
                )
                if uploaded_mask is not None:
                    mask_bytes = uploaded_mask.read()

            st.markdown("")
            run_diag = st.button(
                "⚡ Run Stochastic Inference",
                type="primary",
                use_container_width=True,
            )

        with col_ctrl2:
            num_samples = st.slider("MC Stochastic Passes (N)", 1, 50, 15, step=1)
            uncertainty_thresh = st.slider(
                "Uncertainty Flag Threshold", 0.005, 0.08, 0.02, step=0.005, format="%.3f"
            )

        with col_ctrl3:
            decision_thresh = st.slider("Decision Threshold (tau)", 0.1, 0.9, 0.50, step=0.01, format="%.2f")
            checkpoint_path = st.text_input(
                "Model Checkpoint Path",
                value="checkpoints/latest_model.pth",
            )

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
                st.image(flag_overlay, caption="4. Low-Confidence Flagged Warning (Red)", use_container_width=True)

    with tab2:
        st.subheader("Extended Training Manager")
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            dataset_source = st.selectbox(
                "Training Dataset",
                ["Preset Local CHASE_DB1", "Synthetic Generator"],
            )
            epochs = st.slider("Number of Epochs", 5, 50, 15, step=5)
            batch_size = st.select_slider("Batch Size", options=[2, 4, 8, 16], value=4)

        with col_t2:
            lr = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 2e-3], value=1e-3)
            strip_k = st.select_slider("Strip Conv Kernel Size (K)", options=[5, 7, 9, 11], value=7)
            dropout_val = st.slider("Dropout Probability", 0.1, 0.4, 0.2, step=0.05)

        start_train = st.button("🚀 Start Extended Training", type="primary")

        if start_train:
            st.info(f"Initializing Probabilistic Strip-UNet on {device.upper()}...")

            if dataset_source == "Preset Local CHASE_DB1" and default_images:
                img_paths = default_images
                mask_paths = []
                for p in img_paths:
                    bname = os.path.splitext(os.path.basename(p))[0]
                    mp = os.path.join(default_mask_dir, f"{bname}_1stHO.png")
                    mask_paths.append(mp)

                s_idx = int(0.8 * len(img_paths))
                train_ds = ElongatedStructureDataset(
                    image_paths=img_paths[:s_idx],
                    mask_paths=mask_paths[:s_idx],
                    image_size=(256, 256),
                    use_clahe=True,
                    augment=True,
                )
                val_ds = ElongatedStructureDataset(
                    image_paths=img_paths[s_idx:],
                    mask_paths=mask_paths[s_idx:],
                    image_size=(256, 256),
                    use_clahe=True,
                    augment=False,
                )
            else:
                train_ds = ElongatedStructureDataset(length=64, image_size=(128, 128))
                val_ds = ElongatedStructureDataset(length=16, image_size=(128, 128))

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            train_model = ProbabilisticUNet(
                in_channels=1,
                out_channels=1,
                features=[32, 64, 128, 256],
                strip_kernel_size=strip_k,
                dropout_prob=dropout_val,
            ).to(device)

            optimizer = torch.optim.AdamW(train_model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = BCEDiceLoss()

            progress_bar = st.progress(0)
            status_text = st.empty()
            chart_col1, chart_col2 = st.columns(2)

            train_losses = []
            val_losses = []
            dsc_scores = []
            ece_scores = []

            loss_chart = chart_col1.line_chart()
            metric_chart = chart_col2.line_chart()

            for ep in range(epochs):
                train_model.train()
                r_loss = 0.0
                for imgs, tgts in train_loader:
                    imgs, tgts = imgs.to(device), tgts.to(device)
                    optimizer.zero_grad()
                    outs = train_model(imgs)
                    loss = criterion(outs, tgts)
                    loss.backward()
                    optimizer.step()
                    r_loss += loss.item() * imgs.size(0)

                t_loss = r_loss / len(train_loader.dataset)

                train_model.eval()
                v_loss = 0.0
                all_probs, all_tgts = [], []
                with torch.no_grad():
                    for imgs, tgts in val_loader:
                        imgs, tgts = imgs.to(device), tgts.to(device)
                        outs = train_model(imgs)
                        loss = criterion(outs, tgts)
                        v_loss += loss.item() * imgs.size(0)
                        all_probs.append(torch.sigmoid(outs).cpu())
                        all_tgts.append(tgts.cpu())

                v_loss = v_loss / len(val_loader.dataset)
                all_probs = torch.cat(all_probs, dim=0)
                all_tgts = torch.cat(all_tgts, dim=0)

                dsc = dice_similarity_coefficient(all_probs, all_tgts)
                ece = expected_calibration_error(all_probs, all_tgts)

                train_losses.append(t_loss)
                val_losses.append(v_loss)
                dsc_scores.append(dsc)
                ece_scores.append(ece)

                loss_df = pd.DataFrame({"Train Loss": train_losses, "Val Loss": val_losses})
                metric_df = pd.DataFrame({"DSC": dsc_scores, "ECE": ece_scores})

                loss_chart.line_chart(loss_df)
                metric_chart.line_chart(metric_df)

                progress_bar.progress((ep + 1) / epochs)
                status_text.text(
                    f"Epoch {ep + 1}/{epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | DSC: {dsc:.4f} | ECE: {ece:.4f}"
                )

            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model_state_dict": train_model.state_dict()}, "checkpoints/latest_model.pth")
            st.success("Training successfully finished! Checkpoint saved to checkpoints/latest_model.pth")

    with tab3:
        st.subheader("Automated Batch Evaluation & CSV Report Generator")
        col_b1, col_b2 = st.columns([2, 1])

        with col_b1:
            batch_dir = st.text_input(
                "Dataset Root Directory for Batch Testing",
                value=r"C:\Users\parth\Documents\Prob-strip-dataset",
            )
        with col_b2:
            batch_samples = st.slider("MC Passes for Batch Testing", 5, 30, 10, step=5)

        run_batch = st.button("📊 Run Batch Evaluation", type="primary")

        if run_batch:
            img_dir = os.path.join(batch_dir, "Images")
            mask_dir = os.path.join(batch_dir, "Masks")

            if not os.path.exists(img_dir):
                st.error(f"Images folder not found at: {img_dir}")
            else:
                image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
                st.info(f"Found {len(image_files)} test scans. Executing multi-pass stochastic inference...")

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

                st.markdown("### Summary Statistics")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Average DSC", f"{df_results['Dice Similarity (DSC)'].mean():.4f}")
                sc2.metric("Average IoU", f"{df_results['IoU (Jaccard)'].mean():.4f}")
                sc3.metric("Average ECE", f"{df_results['Calibration Error (ECE)'].mean():.4f}")
                sc4.metric("Average Epistemic Variance", f"{df_results['Mean Epistemic Variance'].mean():.6f}")

                st.markdown("### Per-Image Benchmark Results")
                st.dataframe(df_results, use_container_width=True)

                csv_buffer = io.StringIO()
                df_results.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Full Evaluation CSV Report",
                    data=csv_buffer.getvalue(),
                    file_name="probstrip_evaluation_report.csv",
                    mime="text/csv",
                )


    with tab4:
        st.subheader("Live Webcam Capture & Real-Time Segmentation Analysis")

        if "webcam_history" not in st.session_state:
            st.session_state["webcam_history"] = []

        cam_col1, cam_col2 = st.columns([1, 2])

        with cam_col1:
            cam_mc_passes = st.slider("MC Passes (Webcam)", 5, 30, 10, step=5, key="cam_mc")
            cam_decision = st.slider("Decision Threshold (Webcam)", 0.1, 0.9, 0.5, step=0.05, key="cam_dec")
            cam_unc_thresh = st.slider("Uncertainty Threshold (Webcam)", 0.005, 0.08, 0.02, step=0.005, key="cam_unc")
            cam_checkpoint = st.text_input(
                "Checkpoint (Webcam)",
                value="checkpoints/latest_model.pth",
                key="cam_ckpt",
            )

        with cam_col2:
            camera_image = st.camera_input("Capture Medical Scan via Webcam")

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

                st.markdown("---")
                st.markdown("### Live Capture Analysis Results")

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Foreground Coverage", f"{foreground_ratio * 100:.1f}%")
                mc2.metric("Mean Epistemic Variance", f"{mean_variance:.6f}")
                mc3.metric("Max Epistemic Variance", f"{max_variance:.6f}")
                mc4.metric("Low-Confidence Pixel Ratio", f"{low_conf_ratio * 100:.1f}%")

                vc1, vc2, vc3, vc4 = st.columns(4)
                with vc1:
                    st.image(captured_frame, caption="Captured Input", use_container_width=True)
                with vc2:
                    st.image((cam_mean * 255).astype(np.uint8), caption="Probabilistic Mean Mask", use_container_width=True)
                with vc3:
                    st.image(cam_var_rgb, caption="Epistemic Uncertainty Heatmap", use_container_width=True)
                with vc4:
                    st.image(cam_overlay, caption="Low-Confidence Flagged (Red)", use_container_width=True)

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
            st.markdown("---")
            st.markdown("### Webcam Session Capture History")
            hist_df = pd.DataFrame(st.session_state["webcam_history"])
            st.dataframe(hist_df, use_container_width=True)

            hist_csv = io.StringIO()
            hist_df.to_csv(hist_csv, index=False)
            st.download_button(
                label="📥 Download Webcam Session Report CSV",
                data=hist_csv.getvalue(),
                file_name=f"probstrip_webcam_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

            if st.button("🗑️ Clear Session History"):
                st.session_state["webcam_history"] = []
                st.rerun()


if __name__ == "__main__":
    main()
