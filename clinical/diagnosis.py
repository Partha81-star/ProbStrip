import numpy as np

# Retained for research comparison only. Production reports must not expose
# untrained or unvalidated heuristic disease scores.
AUTOMATED_DIAGNOSIS_ENABLED = False


def assess_retinal_diagnosis(measures: dict) -> dict:
    if not AUTOMATED_DIAGNOSIS_ENABLED:
        return {
            "status": "disabled",
            "primary_condition": "No automated diagnosis",
            "risk_level": "Not assessed",
            "risk_score": None,
            "summary": "No automated diagnosis was generated. Vessel measurements are research outputs only.",
            "differential_diagnoses": [],
            "recommendation": "A qualified clinician must interpret the original image and clinical context.",
        }
    if not measures:
        return {
            "status": "inconclusive",
            "primary_condition": "Insufficient Data",
            "risk_level": "Undetermined",
            "risk_score": 0,
            "summary": "Measurements are not available to generate a diagnostic assessment.",
            "differential_diagnoses": [],
            "recommendation": "Ensure a clear, in-focus scan is provided for analysis.",
        }

    vessel_cov = float(measures.get("visible_vessel_coverage_percent", 10.0))
    central_cov = float(measures.get("central_vessel_coverage_percent", 10.0))
    fractal_dim = float(measures.get("fractal_dimension", 1.45))
    branch_pts = int(measures.get("branch_points", 50))
    endpoints = int(measures.get("endpoints", 50))
    uncertainty = float(measures.get("low_confidence_area_percent", 0.0))

    dr_score = 0
    dr_evidence = []
    if central_cov < 6.5:
        dr_score += 40
        dr_evidence.append("Marked central macular capillary dropout detected (<6.5%)")
    elif central_cov < 8.0:
        dr_score += 25
        dr_evidence.append("Mild central microvascular rarefaction observed (<8.0%)")
    else:
        dr_evidence.append("Central macular vessel density is within expected range")

    if vessel_cov < 7.5:
        dr_score += 30
        dr_evidence.append("Diffuse vascular attenuation across retinal field (<7.5%)")
    elif vessel_cov < 9.5:
        dr_score += 15
        dr_evidence.append("Mild generalized vessel density reduction (<9.5%)")

    if fractal_dim < 1.36 and fractal_dim > 0.1:
        dr_score += 25
        dr_evidence.append(f"Significantly simplified branching fractal dimension ({fractal_dim:.2f})")
    elif fractal_dim < 1.41 and fractal_dim > 0.1:
        dr_score += 15
        dr_evidence.append(f"Borderline vascular fractal complexity ({fractal_dim:.2f})")

    dr_score = min(100, max(5, dr_score))
    if dr_score >= 60:
        dr_risk = "High"
    elif dr_score >= 35:
        dr_risk = "Moderate"
    else:
        dr_risk = "Low"

    hr_score = 0
    hr_evidence = []
    if vessel_cov < 8.5 and central_cov >= 7.0:
        hr_score += 35
        hr_evidence.append("Focal arteriolar narrowing with relative central preservation")
    elif vessel_cov < 10.0:
        hr_score += 20
        hr_evidence.append("Generalized arteriolar caliber constriction")

    if fractal_dim < 1.39 and fractal_dim > 0.1:
        hr_score += 30
        hr_evidence.append("Reduced branching density consistent with hypertensive remodeling")

    ratio = branch_pts / max(endpoints, 1)
    if ratio < 0.6:
        hr_score += 25
        hr_evidence.append("Reduced branch-to-endpoint ratio indicating vascular rarefaction")

    hr_score = min(100, max(5, hr_score))
    if hr_score >= 60:
        hr_risk = "High"
    elif hr_score >= 35:
        hr_risk = "Moderate"
    else:
        hr_risk = "Low"

    is_score = 0
    is_evidence = []
    if central_cov < 6.0:
        is_score += 55
        is_evidence.append("Foveal avascular zone expansion suspected from central dropout")
    elif central_cov < 7.5:
        is_score += 30
        is_evidence.append("Mild macular capillary non-perfusion signs")

    if vessel_cov < 7.0:
        is_score += 35
        is_evidence.append("Severe regional hypoperfusion detected")

    is_score = min(100, max(5, is_score))
    if is_score >= 60:
        is_risk = "High"
    elif is_score >= 35:
        is_risk = "Moderate"
    else:
        is_risk = "Low"

    differentials = [
        {
            "condition": "Diabetic Retinopathy (Microvascular Dropout)",
            "risk_level": dr_risk,
            "probability_percent": dr_score,
            "evidence": "; ".join(dr_evidence),
        },
        {
            "condition": "Hypertensive Retinopathy (Vascular Narrowing)",
            "risk_level": hr_risk,
            "probability_percent": hr_score,
            "evidence": "; ".join(hr_evidence),
        },
        {
            "condition": "Macular Ischemia / Capillary Rarefaction",
            "risk_level": is_risk,
            "probability_percent": is_score,
            "evidence": "; ".join(is_evidence),
        },
    ]

    max_score = max(dr_score, hr_score, is_score)
    if max_score == dr_score and dr_score >= 35:
        primary = "Diabetic Retinopathy Microvascular Pattern"
        primary_risk = dr_risk
    elif max_score == hr_score and hr_score >= 35:
        primary = "Hypertensive Retinopathy Caliber Pattern"
        primary_risk = hr_risk
    elif max_score == is_score and is_score >= 35:
        primary = "Macular Microvascular Ischemia Pattern"
        primary_risk = is_risk
    else:
        primary = "No Apparent Microvascular Abnormality"
        primary_risk = "Low"

    if primary_risk == "High":
        recommendation = "Urgent comprehensive dilated retinal examination by an ophthalmologist or retina specialist is recommended."
        summary = f"High probability of {primary} identified based on quantitative microvascular rarefaction and reduced branching complexity."
    elif primary_risk == "Moderate":
        recommendation = "Consult an optometrist or ophthalmologist for routine dilated retinal evaluation within 4-6 weeks."
        summary = f"Moderate indicators consistent with {primary} identified; clinical correlation with patient glycemic and blood pressure history is advised."
    else:
        recommendation = "Continue regular annual eye examinations as recommended by healthcare provider."
        summary = "Vascular density, branching complexity, and foveal capillary network are within normal clinical thresholds."

    if uncertainty > 15.0:
        summary += " Note: High model uncertainty was flagged in parts of the image, so clinical confirmation is essential."

    return {
        "status": "evaluated",
        "primary_condition": primary,
        "risk_level": primary_risk,
        "risk_score": max_score,
        "summary": summary,
        "differential_diagnoses": differentials,
        "recommendation": recommendation,
    }


def _extract_image_features(image: "np.ndarray") -> dict:
    """Extract a rich set of structural features from the raw image for diagnosis."""
    import cv2
    import numpy as np

    if image is None:
        return {}

    # Normalise to grayscale uint8
    if image.ndim == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)

    h, w = gray.shape
    total_pixels = h * w

    # ── Global intensity stats ────────────────────────────────────────────────
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))

    # ── CLAHE-enhanced image for better feature detection ─────────────────────
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # ── Multi-scale edge detection (Canny + Sobel) ────────────────────────────
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    canny = cv2.Canny(blurred, 40, 120)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    norm_sobel = (sobel_mag / (sobel_mag.max() + 1e-8) * 255).astype(np.uint8)

    # ── Local variance map (reveals heterogeneous regions) ───────────────────
    local_mean = cv2.blur(gray.astype(np.float32), (15, 15))
    local_sq_mean = cv2.blur((gray.astype(np.float32)) ** 2, (15, 15))
    local_var = local_sq_mean - local_mean**2
    local_var = np.clip(local_var, 0, None)
    high_var_mask = local_var > float(np.percentile(local_var, 85))

    # ── Bright / dark region segmentation ────────────────────────────────────
    _, bright_mask = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(enhanced, 40, 255, cv2.THRESH_BINARY_INV)
    bright_ratio = float(np.mean(bright_mask > 0))
    dark_ratio = float(np.mean(dark_mask > 0))

    # ── Line detection (Hough) for cortical fracture lines ───────────────────
    lines = cv2.HoughLinesP(
        canny, rho=1, theta=np.pi / 180,
        threshold=30, minLineLength=max(20, min(h, w) // 15),
        maxLineGap=10
    )
    num_lines = len(lines) if lines is not None else 0
    # compute average line length
    avg_line_length = 0.0
    if lines is not None:
        lengths = [
            np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            for x1, y1, x2, y2 in lines[:, 0]
        ]
        avg_line_length = float(np.mean(lengths))

    # ── Isolated blob / lesion detection ─────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morphed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    blob_count, _, blob_stats, _ = cv2.connectedComponentsWithStats(
        morphed, connectivity=8
    )
    # meaningful blobs: area between 0.02% and 5% of image
    min_blob = max(total_pixels * 0.0002, 4)
    max_blob = total_pixels * 0.05
    meaningful_blobs = int(sum(
        min_blob <= blob_stats[i, cv2.CC_STAT_AREA] <= max_blob
        for i in range(1, blob_count)
    ))

    # ── Histogram entropy ─────────────────────────────────────────────────────
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_prob = hist / (hist.sum() + 1e-8)
    entropy = float(-np.sum(hist_prob * np.log2(hist_prob + 1e-8)))

    # ── Asymmetry (left vs right halves) ─────────────────────────────────────
    left = enhanced[:, : w // 2].astype(float)
    right = np.fliplr(enhanced[:, w // 2:]).astype(float)
    min_w = min(left.shape[1], right.shape[1])
    asymmetry = float(np.mean(np.abs(left[:, :min_w] - right[:, :min_w])))

    # ── Thoracic & lung field analysis (Chest X-ray specific) ────────────────
    # Anatomical right lung corresponds to image-left (x: 8% to 48%, y: 18% to 85%)
    # Anatomical left lung corresponds to image-right (x: 52% to 92%, y: 18% to 85%)
    r_lung = gray[int(h * 0.18) : int(h * 0.85), int(w * 0.08) : int(w * 0.48)]
    l_lung = gray[int(h * 0.18) : int(h * 0.85), int(w * 0.52) : int(w * 0.92)]
    r_lung_mean = float(np.mean(r_lung)) if r_lung.size > 0 else 0.0
    l_lung_mean = float(np.mean(l_lung)) if l_lung.size > 0 else 0.0
    lung_diff = abs(r_lung_mean - l_lung_mean)

    r_dense = float(np.mean(r_lung > 150)) if r_lung.size > 0 else 0.0
    l_dense = float(np.mean(l_lung > 150)) if l_lung.size > 0 else 0.0

    # Zones: mid (40-65%), lower basilar (65-88%)
    rz_mid = gray[int(h * 0.40) : int(h * 0.65), int(w * 0.08) : int(w * 0.48)]
    lz_mid = gray[int(h * 0.40) : int(h * 0.65), int(w * 0.52) : int(w * 0.92)]
    mid_diff = abs(float(np.mean(rz_mid)) - float(np.mean(lz_mid))) if rz_mid.size > 0 and lz_mid.size > 0 else 0.0

    rz_lower = gray[int(h * 0.65) : int(h * 0.88), int(w * 0.08) : int(w * 0.48)]
    lz_lower = gray[int(h * 0.65) : int(h * 0.88), int(w * 0.52) : int(w * 0.92)]
    lower_diff = abs(float(np.mean(rz_lower)) - float(np.mean(lz_lower))) if rz_lower.size > 0 and lz_lower.size > 0 else 0.0
    r_lower_mean = float(np.mean(rz_lower)) if rz_lower.size > 0 else 0.0
    l_lower_mean = float(np.mean(lz_lower)) if lz_lower.size > 0 else 0.0

    # Cardiac transverse diameter proxy
    cardiac_slice = gray[int(h * 0.55) : int(h * 0.75), :]
    cardiac_row = np.mean(cardiac_slice, axis=0) if cardiac_slice.size > 0 else np.zeros(w)
    cardiac_dense_width = float(np.sum(cardiac_row > 140)) / max(w, 1)

    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "bright_ratio": bright_ratio,
        "dark_ratio": dark_ratio,
        "num_lines": num_lines,
        "avg_line_length": avg_line_length,
        "meaningful_blobs": meaningful_blobs,
        "entropy": entropy,
        "asymmetry": asymmetry,
        "high_var_ratio": float(np.mean(high_var_mask)),
        "r_lung_mean": r_lung_mean,
        "l_lung_mean": l_lung_mean,
        "lung_diff": lung_diff,
        "r_dense": r_dense,
        "l_dense": l_dense,
        "mid_diff": mid_diff,
        "lower_diff": lower_diff,
        "r_lower_mean": r_lower_mean,
        "l_lower_mean": l_lower_mean,
        "cardiac_dense_width": cardiac_dense_width,
    }


def _diagnose_bone_xray(feats: dict, quality: int) -> tuple[str, str, int, str, list]:
    """Bone / joint X-ray: fracture & osteoporosis detection."""
    evidence = []
    differentials = []

    lines = feats.get("num_lines", 0)
    avg_ll = feats.get("avg_line_length", 0)
    blobs = feats.get("meaningful_blobs", 0)
    high_var = feats.get("high_var_ratio", 0)
    entropy = feats.get("entropy", 5.0)
    bright = feats.get("bright_ratio", 0)
    edge = feats.get("edge_area_percent", 0)

    # ── Fracture scoring (graduated, additive) ────────────────────────────────
    fx_score = 0

    # Line count — primary fracture indicator (Hough lines across cortex)
    if lines >= 150:
        fx_score += 55
        evidence.append(f"Very high cortical line count ({lines} lines, avg {avg_ll:.0f}px) — strong fracture pattern")
    elif lines >= 50:
        fx_score += 45
        evidence.append(f"High cortical line density ({lines} lines, avg {avg_ll:.0f}px) — probable discontinuity")
    elif lines >= 15:
        fx_score += 35
        evidence.append(f"Multiple linear discontinuities ({lines} lines, avg {avg_ll:.0f}px)")
    elif lines >= 6:
        fx_score += 22
        evidence.append(f"Several cortical linear patterns ({lines} lines)")
    elif lines >= 2:
        fx_score += 10
        evidence.append(f"Sparse cortical line pattern ({lines} lines)")

    # Average line length bonus — longer lines = more likely a real fracture line
    if avg_ll >= 60:
        fx_score += 15
        evidence.append(f"Long discontinuity lines (avg {avg_ll:.0f}px) — consistent with cortical fracture trajectory")
    elif avg_ll >= 35:
        fx_score += 8
        evidence.append(f"Moderate line length (avg {avg_ll:.0f}px)")

    # Local variance — cortical disruption creates high local heterogeneity
    if high_var > 0.18:
        fx_score += 20
        evidence.append(f"High structural heterogeneity ({high_var*100:.1f}%) — consistent with cortical disruption")
    elif high_var > 0.10:
        fx_score += 10
        evidence.append(f"Moderate structural heterogeneity ({high_var*100:.1f}%)")

    # Edge density
    if edge > 12.0:
        fx_score += 15
        evidence.append(f"Dense edge network ({edge:.1f}%) — complex cortical surface or fracture fragments")
    elif edge > 7.0:
        fx_score += 8
        evidence.append(f"Elevated edge density ({edge:.1f}%)")

    # Focal blobs — possible bone fragments or callus
    if blobs >= 4:
        fx_score += 10
        evidence.append(f"Multiple focal anomalous regions ({blobs}) — possible fragments or periosteal reaction")
    elif blobs >= 2:
        fx_score += 5

    fx_score = min(100, fx_score)

    # ── Osteoporosis proxy ────────────────────────────────────────────────────
    osteo_score = 0
    if bright < 0.08 and entropy < 5.5:
        osteo_score = 50
    elif bright < 0.15 and entropy < 6.0:
        osteo_score = 30
    elif bright < 0.20:
        osteo_score = 18

    differentials.append({
        "condition": "Cortical Fracture / Bone Discontinuity",
        "risk_level": "High" if fx_score >= 50 else ("Moderate" if fx_score >= 28 else "Low"),
        "probability_percent": fx_score,
        "evidence": "; ".join(evidence) if evidence else "Insufficient structural line evidence",
    })
    differentials.append({
        "condition": "Osteoporosis / Reduced Bone Density",
        "risk_level": "Moderate" if osteo_score >= 30 else "Low",
        "probability_percent": osteo_score,
        "evidence": f"Bright-pixel density ({bright*100:.1f}%), image entropy {entropy:.2f}",
    })

    primary_score = max(fx_score, osteo_score)
    if fx_score >= 50:
        cond = "Cortical Bone Fracture or Discontinuity Detected"
        level = "High"
        rec = "Orthopedic evaluation recommended. Multi-view radiographic correlation and CT confirmation advised."
    elif fx_score >= 28:
        cond = "Possible Cortical Irregularity or Hairline Fracture"
        level = "Moderate"
        rec = "Secondary radiological review advised. Consider CT or MRI to rule out occult fracture."
    elif osteo_score >= 30:
        cond = "Possible Reduced Bone Density (Osteoporosis Pattern)"
        level = "Moderate"
        rec = "DEXA bone density scan advised. Clinical correlation with patient risk factors."
    else:
        cond = "No Definite Fracture or Major Abnormality Detected"
        level = "Low"
        rec = "Correlate with clinical tenderness, mechanism of injury, and physical examination."

    return cond, level, primary_score, rec, differentials


def _diagnose_chest_xray(feats: dict, quality: int) -> tuple[str, str, int, str, list]:
    """Chest X-ray: consolidation, pneumonia, effusion, cardiomegaly patterns."""
    evidence = []
    differentials = []

    bright = feats.get("bright_ratio", 0)
    dark = feats.get("dark_ratio", 0)
    asymmetry = feats.get("asymmetry", 0)
    blobs = feats.get("meaningful_blobs", 0)
    r_mean = feats.get("r_lung_mean", 0)
    l_mean = feats.get("l_lung_mean", 0)
    lung_diff = feats.get("lung_diff", asymmetry)
    r_dense = feats.get("r_dense", 0)
    l_dense = feats.get("l_dense", 0)
    max_dense = max(r_dense, l_dense)
    mid_diff = feats.get("mid_diff", asymmetry)
    lower_diff = feats.get("lower_diff", asymmetry)
    r_lower_mean = feats.get("r_lower_mean", 0)
    l_lower_mean = feats.get("l_lower_mean", 0)
    cardiac_dense_width = feats.get("cardiac_dense_width", 0.40)

    # Determine laterality (in standard radiologic projection, image-left is patient's right)
    if r_mean > l_mean + 12 or (r_dense > l_dense + 0.25):
        side = "Right lung"
    elif l_mean > r_mean + 12 or (l_dense > r_dense + 0.25):
        side = "Left lung"
    elif r_dense > 0.60 and l_dense > 0.60:
        side = "Bilateral"
    else:
        side = "Unilateral"

    # ── 1. Pulmonary Consolidation / Pneumonia Assessment ─────────────────────
    consol_score = 0
    consol_evidence = []

    # Significant unilateral opacity or zone asymmetry
    max_asym = max(mid_diff, lung_diff, asymmetry)
    if max_asym >= 35:
        consol_score += 55
        consol_evidence.append(
            f"Dense parenchymal opacification in {side} (asymmetry index: {max_asym:.1f})"
        )
    elif max_asym >= 20:
        consol_score += 35
        consol_evidence.append(
            f"Moderate density asymmetry in {side} (asymmetry index: {max_asym:.1f})"
        )
    elif max_asym >= 12:
        consol_score += 18
        consol_evidence.append(f"Mild parenchymal asymmetry in {side}")

    # Confluent parenchymal dense consolidation
    if max_dense >= 0.80:
        consol_score += 30
        consol_evidence.append(
            f"Confluent alveolar consolidation replacing normal aerated lung ({max_dense*100:.1f}% dense opacification)"
        )
    elif max_dense >= 0.60:
        consol_score += 20
        consol_evidence.append(
            f"Extensive alveolar infiltration ({max_dense*100:.1f}% dense opacification)"
        )
    elif max_dense >= 0.40:
        consol_score += 10
        consol_evidence.append(
            f"Patchy parenchymal infiltrates ({max_dense*100:.1f}% opacity)"
        )

    # Obscuration of cardiac or diaphragmatic border (silhouette sign)
    if lower_diff >= 25 or (max(r_lower_mean, l_lower_mean) > 190 and abs(r_lower_mean - l_lower_mean) > 20):
        consol_score += 12
        consol_evidence.append(
            "Loss of distinct hemidiaphragmatic/cardiac silhouette (positive silhouette sign)"
        )

    # Diffuse bilateral consolidation check
    if r_dense > 0.65 and l_dense > 0.65:
        consol_score = max(consol_score, 80)
        consol_evidence.append(
            "Extensive bilateral alveolar infiltrates consistent with diffuse pneumonitis / bronchopneumonia"
        )

    consol_score = min(100, max(0, consol_score))

    # ── 2. Pleural Effusion Pattern ───────────────────────────────────────────
    effusion_score = 0
    effusion_evidence = []
    if lower_diff >= 28 and max(r_lower_mean, l_lower_mean) > 180:
        effusion_score += 50
        effusion_evidence.append(
            f"Marked dependent basilar opacification in {side} with costophrenic recess obliteration"
        )
    elif lower_diff >= 16 and max(r_lower_mean, l_lower_mean) > 160:
        effusion_score += 30
        effusion_evidence.append(
            f"Moderate basilar blunting / haziness in {side}"
        )

    if max_dense >= 0.70 and (mid_diff > 25 or lower_diff > 25):
        effusion_score += 20
        effusion_evidence.append("Associated parapneumonic fluid or dense hemithorax opacification")
    effusion_score = min(100, max(0, effusion_score))

    # ── 3. Cardiomegaly Pattern ───────────────────────────────────────────────
    cardio_score = 0
    cardio_evidence = []
    if cardiac_dense_width > 0.52:
        cardio_score += 45
        cardio_evidence.append(
            f"Cardiothoracic ratio exceeds normal limit ({cardiac_dense_width*100:.1f}% > 50%)"
        )
    elif cardiac_dense_width > 0.46:
        cardio_score += 25
        cardio_evidence.append(
            f"Borderline cardiac silhouette enlargement ({cardiac_dense_width*100:.1f}%)"
        )
    else:
        cardio_evidence.append(f"Cardiac transverse dimension appears within normal limits")

    if consol_score >= 50 and "Right" in side:
        cardio_evidence.append("Note: Right heart border obscured by overlying parenchymal consolidation")

    cardio_score = min(100, max(0, cardio_score))

    differentials.extend([
        {
            "condition": "Pulmonary Consolidation / Pneumonia Pattern",
            "risk_level": "High" if consol_score >= 50 else ("Moderate" if consol_score >= 25 else "Low"),
            "probability_percent": consol_score,
            "evidence": "; ".join(consol_evidence) if consol_evidence else "No significant consolidation or infiltrate identified",
        },
        {
            "condition": "Pleural Effusion Pattern",
            "risk_level": "High" if effusion_score >= 50 else ("Moderate" if effusion_score >= 25 else "Low"),
            "probability_percent": effusion_score,
            "evidence": "; ".join(effusion_evidence) if effusion_evidence else "Costophrenic angles and basilar margins appear clear",
        },
        {
            "condition": "Cardiomegaly Pattern",
            "risk_level": "Moderate" if cardio_score >= 35 else "Low",
            "probability_percent": cardio_score,
            "evidence": "; ".join(cardio_evidence) if cardio_evidence else "Cardiac silhouette within normal parameters",
        },
    ])

    primary_score = max(consol_score, effusion_score, cardio_score)
    if consol_score >= 50:
        cond = f"{side} Pulmonary Consolidation / Pneumonia Pattern"
        level = "High"
        rec = (
            "Urgent clinical and respiratory correlation. Sputum Gram stain and culture, "
            "blood cultures, and inflammatory markers (CBC with differential, CRP/procalcitonin) advised. "
            "Initiate targeted antimicrobial protocol if bacterial etiology suspected. "
            "Repeat radiograph or thoracic CT recommended to monitor resolution."
        )
    elif effusion_score >= 50:
        cond = f"{side} Pleural Effusion Pattern"
        level = "High"
        rec = (
            "Pulmonology consultation recommended. Thoracic ultrasound to confirm effusion volume "
            "and assess for loculation/septation; consider diagnostic thoracentesis."
        )
    elif consol_score >= 25:
        cond = f"Possible {side} Infiltrate or Early Consolidation"
        level = "Moderate"
        rec = "Correlate with respiratory symptoms (fever, cough, dyspnea). Follow-up radiograph in 48-72 hours."
    elif effusion_score >= 25:
        cond = f"Possible {side} Basilar Opacity or Blunting"
        level = "Moderate"
        rec = "Consider lateral decubitus chest radiograph or thoracic ultrasound for evaluation."
    elif cardio_score >= 35:
        cond = "Possible Cardiomegaly Pattern"
        level = "Moderate"
        rec = "Cardiology evaluation and echocardiogram to assess cardiac chamber dimensions and function."
    else:
        cond = "No Major Chest Pathology Pattern Detected"
        level = "Low"
        rec = "Lung fields appear well-aerated without focal consolidation. Correlate with clinical symptoms."

    return cond, level, primary_score, rec, differentials


def _diagnose_ct_mri(feats: dict, quality: int, modality: str) -> tuple[str, str, int, str, list]:
    """CT / MRI: mass lesion, hemorrhage, structural anomaly."""
    evidence = []
    blobs = feats.get("meaningful_blobs", 0)
    high_var = feats.get("high_var_ratio", 0)
    bright = feats.get("bright_ratio", 0)
    entropy = feats.get("entropy", 5.0)
    asymmetry = feats.get("asymmetry", 0)

    lesion_score = 0
    if blobs >= 6:
        lesion_score += 35
        evidence.append(f"Multiple focal structural anomalies detected ({blobs} regions)")
    elif blobs >= 3:
        lesion_score += 20
        evidence.append(f"Focal structural heterogeneity ({blobs} anomalous regions)")

    if high_var > 0.22:
        lesion_score += 30
        evidence.append(f"High local intensity variance ({high_var*100:.1f}%) — mass effect or edema pattern")
    elif high_var > 0.12:
        lesion_score += 15
        evidence.append("Mild structural heterogeneity")

    if bright > 0.25 and "CT" in modality:
        lesion_score += 20
        evidence.append(f"Hyperdense regions ({bright*100:.1f}%) — possible hemorrhage or calcification")

    if asymmetry > 25:
        lesion_score += 15
        evidence.append(f"Structural asymmetry ({asymmetry:.0f}) — mass effect suspected")

    lesion_score = min(100, lesion_score)
    differentials = [{
        "condition": f"Focal {'Hyperdense Lesion / Hemorrhage' if 'CT' in modality else 'Signal Abnormality / Lesion'}",
        "risk_level": "High" if lesion_score >= 60 else ("Moderate" if lesion_score >= 30 else "Low"),
        "probability_percent": lesion_score,
        "evidence": "; ".join(evidence) if evidence else "No focal anomaly signals detected",
    }]

    if lesion_score >= 60:
        cond = f"Focal Structural Abnormality Detected on {modality}"
        level = "High"
        rec = "Immediate radiological review. Contrast-enhanced study and specialist referral recommended."
    elif lesion_score >= 30:
        cond = f"Possible Structural Anomaly on {modality}"
        level = "Moderate"
        rec = "Follow-up imaging with radiologist correlation advised."
    else:
        cond = "No Major Structural Abnormality Detected"
        level = "Low"
        rec = "Clinical correlation and routine follow-up."

    return cond, level, lesion_score, rec, differentials


def _diagnose_ultrasound(feats: dict, quality: int) -> tuple[str, str, int, str, list]:
    """Ultrasound: echogenic lesions, cysts, structural discontinuities."""
    blobs = feats.get("meaningful_blobs", 0)
    high_var = feats.get("high_var_ratio", 0)
    edge = feats.get("edge_area_percent", 0)
    dark = feats.get("dark_ratio", 0)
    evidence = []

    lesion_score = 0
    if blobs >= 4:
        lesion_score += 35
        evidence.append(f"Multiple focal echogenic anomalies ({blobs} regions)")
    elif blobs >= 2:
        lesion_score += 18
        evidence.append(f"Focal echogenic / hypoechoic regions ({blobs})")

    if dark > 0.40:
        lesion_score += 25
        evidence.append(f"Significant hypoechoic (anechoic) area ({dark*100:.1f}%) — possible cystic structure or fluid")

    if high_var > 0.15:
        lesion_score += 20
        evidence.append("High local variance — heterogeneous tissue architecture")

    lesion_score = min(100, lesion_score)
    differentials = [
        {
            "condition": "Focal Echogenic or Hypoechoic Lesion",
            "risk_level": "Moderate" if lesion_score >= 30 else "Low",
            "probability_percent": lesion_score,
            "evidence": "; ".join(evidence) if evidence else "No definite focal lesion echo pattern",
        },
        {
            "condition": "Cystic Structure or Fluid Collection",
            "risk_level": "Low" if dark < 0.45 else "Moderate",
            "probability_percent": min(100, int(dark * 100)),
            "evidence": f"Anechoic area ratio: {dark*100:.1f}%",
        },
    ]

    if lesion_score >= 50:
        cond = "Focal Echogenic Lesion or Structural Discontinuity Detected"
        level = "Moderate"
        rec = "Targeted ultrasound follow-up, Doppler assessment, or cross-sectional imaging recommended."
    elif lesion_score >= 25:
        cond = "Possible Focal Ultrasound Abnormality"
        level = "Moderate"
        rec = "Clinical correlation with symptoms. Repeat ultrasound or further imaging if indicated."
    else:
        cond = "Preserved Fascicular and Soft-Tissue Architecture"
        level = "Low"
        rec = "Routine clinical follow-up."

    return cond, level, lesion_score, rec, differentials


def _diagnose_skin(feats: dict, quality: int) -> tuple[str, str, int, str, list]:
    """Skin / external: lesion asymmetry, border irregularity, colour variance."""
    asymmetry = feats.get("asymmetry", 0)
    blobs = feats.get("meaningful_blobs", 0)
    high_var = feats.get("high_var_ratio", 0)
    bright = feats.get("bright_ratio", 0)
    edge = feats.get("edge_area_percent", 0)
    evidence = []

    lesion_score = 0
    if asymmetry > 35:
        lesion_score += 30
        evidence.append(f"High lesion asymmetry index ({asymmetry:.0f}) — irregular shape")
    elif asymmetry > 20:
        lesion_score += 15
        evidence.append(f"Moderate shape asymmetry ({asymmetry:.0f})")

    if high_var > 0.25:
        lesion_score += 25
        evidence.append(f"High colour/texture heterogeneity ({high_var*100:.1f}%) — irregular pigmentation")

    if blobs >= 3:
        lesion_score += 20
        evidence.append(f"Multiple focal regions of interest ({blobs}) — satellite lesions or multi-focal pattern")

    if edge > 10:
        lesion_score += 15
        evidence.append(f"Irregular border density ({edge:.1f}%)")

    lesion_score = min(100, lesion_score)
    differentials = [
        {
            "condition": "Atypical Pigmented Lesion / Possible Melanoma Pattern",
            "risk_level": "High" if lesion_score >= 60 else ("Moderate" if lesion_score >= 35 else "Low"),
            "probability_percent": lesion_score,
            "evidence": "; ".join(evidence) if evidence else "No significant ABCDE criteria detected",
        },
        {
            "condition": "Benign Seborrheic or Inflammatory Lesion",
            "risk_level": "Low",
            "probability_percent": max(0, 50 - lesion_score),
            "evidence": "Low asymmetry / heterogeneity pattern",
        },
    ]

    if lesion_score >= 60:
        cond = "Atypical Pigmented Lesion — Malignancy Pattern"
        level = "High"
        rec = "Urgent dermatology review. Dermoscopy and possible biopsy recommended."
    elif lesion_score >= 35:
        cond = "Irregular Skin Lesion — Monitoring Advised"
        level = "Moderate"
        rec = "Dermatology referral for clinical assessment. Serial photography for monitoring."
    else:
        cond = "No High-Risk Lesion Pattern Detected"
        level = "Low"
        rec = "Routine skin examination at next scheduled appointment."

    return cond, level, lesion_score, rec, differentials


def assess_general_diagnosis(
    modality: str,
    edge_area_percent: float,
    quality_score: int,
    image: "np.ndarray | None" = None,
) -> dict:
    """
    Comprehensive image-driven diagnostic engine.

    Parameters
    ----------
    modality        : One of the MODALITIES keys from clinical/modalities.py
    edge_area_percent: Pre-computed edge area from the structure view (fallback feature)
    quality_score   : Technical quality 0-100
    image           : Raw numpy image array (RGB or grayscale) for deep analysis
    """
    if not AUTOMATED_DIAGNOSIS_ENABLED:
        return {
            "status": "disabled",
            "primary_condition": "No automated diagnosis",
            "risk_level": "Not assessed",
            "risk_score": None,
            "summary": "No automated diagnosis was generated. This workflow provides technical image review only.",
            "differential_diagnoses": [],
            "recommendation": "A qualified clinician must interpret the original image and clinical context.",
        }

    import numpy as np  # local guard so module stays importable without numpy top-level

    # Extract rich features if image is provided
    feats = _extract_image_features(image) if image is not None else {
        "edge_area_percent": edge_area_percent,
        "num_lines": 0,
        "meaningful_blobs": 0,
        "high_var_ratio": 0.10,
        "bright_ratio": 0.20,
        "dark_ratio": 0.20,
        "asymmetry": 10.0,
        "entropy": 5.0,
        "r_lung_mean": 100.0,
        "l_lung_mean": 100.0,
        "lung_diff": 0.0,
        "r_dense": 0.10,
        "l_dense": 0.10,
        "mid_diff": 0.0,
        "lower_diff": 0.0,
        "r_lower_mean": 100.0,
        "l_lower_mean": 100.0,
        "cardiac_dense_width": 0.35,
    }
    # Always inject the pre-computed edge area (it's already computed upstream)
    feats["edge_area_percent"] = edge_area_percent

    # ── Route to modality-specific engine ────────────────────────────────────
    m = modality.lower()
    if "bone" in m or ("x-ray" in m and "chest" not in m):
        cond, level, score, rec, differentials = _diagnose_bone_xray(feats, quality_score)
    elif "chest" in m:
        cond, level, score, rec, differentials = _diagnose_chest_xray(feats, quality_score)
    elif "ct" in m:
        cond, level, score, rec, differentials = _diagnose_ct_mri(feats, quality_score, "CT")
    elif "mri" in m:
        cond, level, score, rec, differentials = _diagnose_ct_mri(feats, quality_score, "MRI")
    elif "ultrasound" in m:
        cond, level, score, rec, differentials = _diagnose_ultrasound(feats, quality_score)
    elif "skin" in m or "external" in m or "photo" in m:
        cond, level, score, rec, differentials = _diagnose_skin(feats, quality_score)
    else:
        # Generic fallback
        cond = "Evaluated Structural Morphology"
        level = "Low"
        score = 20
        rec = "Review scan with corresponding clinical specialist."
        differentials = [{
            "condition": cond,
            "risk_level": level,
            "probability_percent": score,
            "evidence": f"Edge coverage {edge_area_percent:.1f}%, quality {quality_score}/100",
        }]

    # Quality caveat
    quality_note = ""
    if quality_score < 60:
        quality_note = " Note: Image quality is below recommended threshold — results may be less reliable."

    return {
        "status": "evaluated",
        "primary_condition": cond,
        "risk_level": level,
        "risk_score": score,
        "summary": (
            f"Automated multi-feature analysis indicates {level.lower()} probability of {cond}."
            + quality_note
        ),
        "differential_diagnoses": differentials,
        "recommendation": rec,
    }
