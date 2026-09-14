"""Blood-cell microscopy screening, differential count estimation, and morphological pathology detection.

Supports peripheral blood smear (PBS) and microscopy images.

Capabilities:
- Automated cell segmentation using watershed + contour analysis.
- Approximate WBC differential: neutrophils, lymphocytes, monocytes, blasts.
- RBC morphology assessment: microcytosis, macrocytosis, poikilocytosis, hypochromia, target cells.
- Platelet density estimation: thrombocytopenia / thrombocytosis flags.
- Pathology classification: normal, anemia-pattern, infection/inflammation, malignancy-pattern.
- Annotated cell map overlay with per-class color coding.
- Patient-friendly next-step recommendations and clinical safety guidance.

Reference population: BCCD Dataset (Shenggan/BCCD_Dataset) and Kaggle Blood Cell Count
(DRacom7/blood-cells-detection).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


# ── Module constants ──────────────────────────────────────────────────────────

BCCD_POPULATION_SCOPE = (
    "BCCD / Kaggle Blood Cell benchmark: Peripheral blood smear microscopy screening, "
    "morphological pattern assessment, and approximate differential count"
)

# Per-class RGB draw colors
_COLOR_RBC   = (220,  50,  50)   # red
_COLOR_WBC   = ( 37, 120, 255)   # blue
_COLOR_PLT   = ( 50, 200,  80)   # green
_COLOR_BLAST = (230,  50, 230)   # magenta — abnormal large cell
_COLOR_BAND  = (255, 165,   0)   # orange  — monocyte / band cell

# Minimum contour areas for each class (pixels²)
_MIN_RBC_AREA  = 200
_MIN_WBC_AREA  = 1200
_MIN_PLT_AREA  =  40
_MAX_PLT_AREA  = 300

# WBC circularity thresholds
_LYMPH_CIRC_MIN = 0.72
_MONO_CIRC_MAX  = 0.55

# Blast size ratio relative to median RBC
_BLAST_SIZE_RATIO = 2.8


# ── 1. Pre-processing ─────────────────────────────────────────────────────────

def _normalise_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    return image.astype(np.uint8)


def _enhance_smear(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_enhanced, a_ch, b_ch]), cv2.COLOR_LAB2RGB)


# ── 2. Cell segmentation ──────────────────────────────────────────────────────

def _segment_cells(
    rgb: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    h, w = rgb.shape[:2]
    enhanced = _enhance_smear(rgb)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    _, markers_fg = cv2.threshold(dist, 0.38, 1.0, cv2.THRESH_BINARY)
    markers_fg = markers_fg.astype(np.uint8)
    unknown = cv2.subtract(closed, markers_fg)
    _, markers = cv2.connectedComponents(markers_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    rgb_ws = enhanced.copy()
    cv2.watershed(rgb_ws, markers)
    watershed_mask = (markers > 1).astype(np.uint8) * 255

    contours, _ = cv2.findContours(watershed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rbcs: List[Dict[str, Any]] = []
    wbcs: List[Dict[str, Any]] = []
    platelets: List[Dict[str, Any]] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < _MIN_PLT_AREA:
            continue
        perimeter = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cell_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cell_mask, [cnt], -1, 255, -1)
        mean_intensity = float(cv2.mean(gray, mask=cell_mask)[0])
        record = {
            "contour": cnt, "area": area, "cx": cx, "cy": cy,
            "circularity": circ, "mean_intensity": mean_intensity,
        }
        if area <= _MAX_PLT_AREA:
            platelets.append(record)
        elif area >= _MIN_WBC_AREA:
            wbcs.append(record)
        elif area >= _MIN_RBC_AREA:
            rbcs.append(record)

    return rbcs, wbcs, platelets


# ── 3. RBC morphology analysis ────────────────────────────────────────────────

def _analyse_rbc_morphology(rbcs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rbcs:
        return {
            "count": 0,
            "flags": ["Insufficient RBC population to assess morphology"],
            "mean_area": None, "mean_circularity": None, "mean_intensity": None,
        }
    areas  = np.array([c["area"] for c in rbcs])
    circs  = np.array([c["circularity"] for c in rbcs])
    intens = np.array([c["mean_intensity"] for c in rbcs])

    mean_area  = float(np.mean(areas))
    mean_circ  = float(np.mean(circs))
    mean_int   = float(np.mean(intens))
    median_area = float(np.median(areas))
    cv_area = float(np.std(areas)) / (mean_area + 1e-6)

    flags: List[str] = []
    if cv_area > 0.28:
        flags.append("Anisocytosis — marked variation in RBC size")
    elif cv_area > 0.22:
        flags.append("Mild anisocytosis — moderate variation in RBC size")
    if mean_area < median_area * 0.72:
        flags.append("Microcytosis — reduced mean RBC size (possible iron deficiency / thalassaemia)")
    elif mean_area > median_area * 1.35:
        flags.append("Macrocytosis — increased mean RBC size (possible B12/folate deficiency)")
    if mean_int > 165:
        flags.append("Hypochromia — pale RBC center (possible iron deficiency)")
    if mean_circ < 0.62:
        flags.append("Poikilocytosis — irregular RBC shapes detected")
    target_like = float(np.sum((circs > 0.82) & (intens > 155))) / (len(rbcs) + 1e-6)
    if target_like > 0.12:
        flags.append("Target cell pattern — possible thalassaemia, liver disease, or haemoglobin-C")
    if not flags:
        flags.append("RBC morphology within normal limits")

    return {
        "count": len(rbcs),
        "mean_area": round(mean_area, 1),
        "mean_circularity": round(mean_circ, 3),
        "mean_intensity": round(mean_int, 1),
        "flags": flags,
    }


# ── 4. WBC differential estimation ───────────────────────────────────────────

def _classify_wbc(wbc: Dict[str, Any], median_rbc_area: float) -> str:
    circ = wbc["circularity"]
    size_ratio = wbc["area"] / (median_rbc_area + 1e-6)
    if size_ratio >= _BLAST_SIZE_RATIO:
        return "blast"
    if circ >= _LYMPH_CIRC_MIN:
        return "lymphocyte"
    if circ <= _MONO_CIRC_MAX:
        return "monocyte"
    return "neutrophil"


def _analyse_wbc_differential(
    wbcs: List[Dict[str, Any]], median_rbc_area: float
) -> Dict[str, Any]:
    if not wbcs:
        return {
            "count": 0,
            "differential": {},
            "flags": ["No WBCs detected in frame — field count may be non-representative"],
            "blast_detected": False,
        }
    class_counts: Dict[str, int] = {"neutrophil": 0, "lymphocyte": 0, "monocyte": 0, "blast": 0}
    for wbc in wbcs:
        cls = _classify_wbc(wbc, median_rbc_area)
        class_counts[cls] = class_counts.get(cls, 0) + 1

    total = len(wbcs)
    differential = {cls: round(100.0 * cnt / total, 1) for cls, cnt in class_counts.items() if cnt > 0}
    flags: List[str] = []
    blast_detected = class_counts["blast"] > 0
    if blast_detected:
        flags.append(
            f"BLAST CELLS DETECTED ({differential.get('blast', 0):.0f}% of WBCs in frame) — "
            "requires urgent haematological review"
        )
    if differential.get("lymphocyte", 0) > 60:
        flags.append(f"Relative lymphocytosis ({differential['lymphocyte']:.0f}%) — possible viral infection or CLL")
    if differential.get("neutrophil", 0) > 75:
        flags.append(f"Neutrophilia ({differential['neutrophil']:.0f}%) — possible bacterial infection or inflammation")
    if total >= 5 and class_counts["monocyte"] / total > 0.12:
        flags.append("Monocytosis — possible chronic infection or recovery")
    if not flags:
        flags.append("WBC differential within expected range for field view")

    return {
        "count": total,
        "differential": differential,
        "blast_detected": blast_detected,
        "flags": flags,
    }


# ── 5. Platelet estimation ────────────────────────────────────────────────────

def _analyse_platelets(platelets: List[Dict[str, Any]], n_rbcs: int) -> Dict[str, Any]:
    n_plt = len(platelets)
    if n_rbcs < 5:
        ratio = None
        density_flag = "Insufficient RBC count for platelet ratio estimation"
    else:
        ratio = round(n_plt / n_rbcs, 2)
        if ratio < 0.04:
            density_flag = f"Thrombocytopenia pattern — very low platelet density (ratio {ratio:.2f})"
        elif ratio < 0.08:
            density_flag = f"Low-normal platelet density (ratio {ratio:.2f})"
        elif ratio > 0.30:
            density_flag = f"Thrombocytosis pattern — high platelet density (ratio {ratio:.2f})"
        else:
            density_flag = f"Platelet density within normal range (ratio {ratio:.2f})"
    return {"count": n_plt, "platelet_rbc_ratio": ratio, "flag": density_flag}


# ── 6. Overall pathology classification ───────────────────────────────────────

def _anemia_subtype(rbc_flags: List[str]) -> str:
    txt = " ".join(rbc_flags)
    if "Microcytosis" in txt and "Hypochromia" in txt:
        return "Microcytic hypochromic anemia pattern — possible iron deficiency or thalassaemia"
    if "Macrocytosis" in txt:
        return "Macrocytic anemia pattern — possible vitamin B12 / folate deficiency"
    if "Target cell" in txt:
        return "Target cell anemia pattern — possible thalassaemia or haemoglobin-C"
    if "Poikilocytosis" in txt:
        return "Haemolytic or sickle-cell anemia pattern — shape distortion detected"
    return "Anemia-pattern morphology detected"


def _overall_pathology(
    rbc_morph: Dict[str, Any],
    wbc_diff: Dict[str, Any],
    plt_info: Dict[str, Any],
) -> Tuple[str, str, float, List[str]]:
    evidence: List[str] = []
    score = 0.0
    rbc_flags = rbc_morph.get("flags", [])
    wbc_flags = wbc_diff.get("flags", [])
    plt_flag  = plt_info.get("flag", "")

    if wbc_diff.get("blast_detected"):
        return (
            "Blast cells detected — possible haematological malignancy",
            "High", 94.0,
            ["Blast cells detected in peripheral smear"] + rbc_flags + wbc_flags,
        )

    abnormal_rbc = [f for f in rbc_flags if "within normal" not in f and "Insufficient" not in f]
    abnormal_wbc = [f for f in wbc_flags if "expected range" not in f and "non-representative" not in f]
    thrombocytopenia = "Thrombocytopenia" in plt_flag
    thrombocytosis   = "Thrombocytosis"   in plt_flag

    anemia_keywords = ["Microcytosis", "Hypochromia", "Macrocytosis", "Anisocytosis", "Poikilocytosis", "Target cell"]
    anemia_hits = sum(any(kw in f for f in rbc_flags) for kw in anemia_keywords)
    if anemia_hits >= 2:
        score += 45 + anemia_hits * 8
        evidence.append(f"Multiple RBC morphology abnormalities ({anemia_hits} features)")

    if "Neutrophilia" in " ".join(wbc_flags):
        score += 35; evidence.append("Neutrophilia — probable bacterial infection pattern")
    if "lymphocytosis" in " ".join(wbc_flags).lower():
        score += 25; evidence.append("Lymphocytosis — probable viral infection pattern")
    if "Monocytosis" in " ".join(wbc_flags):
        score += 15; evidence.append("Monocytosis — chronic inflammation or recovery")
    if thrombocytopenia:
        score += 20; evidence.append("Thrombocytopenia — low platelet density")
    if thrombocytosis:
        score += 10; evidence.append("Thrombocytosis — reactive or essential platelet elevation")

    all_evidence = evidence + abnormal_rbc + abnormal_wbc
    if thrombocytopenia or thrombocytosis:
        all_evidence.append(plt_flag)
    score = min(score, 97.0)

    if score >= 55:
        if anemia_hits >= 2 and score < 70 and not any("Neutrophilia" in e or "Lymphocytosis" in e for e in evidence):
            return (_anemia_subtype(rbc_flags), "Moderate", round(score, 1), all_evidence)
        if any(kw in " ".join(evidence) for kw in ("Neutrophilia", "Lymphocytosis")):
            level = "High" if score >= 70 else "Moderate"
            return ("Infection / inflammation pattern detected", level, round(score, 1), all_evidence)
        return ("Multiple haematological abnormalities detected", "High" if score >= 75 else "Moderate", round(score, 1), all_evidence)

    if score >= 25:
        return ("Mild haematological abnormality detected", "Low", round(score, 1), all_evidence or ["Borderline morphological variation"])

    return (
        "No significant pathological features detected",
        "Low", round(max(score, 5.0), 1),
        ["RBC, WBC, and platelet parameters within expected range for this image field"],
    )


# ── 7. Overlay rendering ──────────────────────────────────────────────────────

def _draw_cell_overlay(
    rgb: np.ndarray,
    rbcs: List[Dict[str, Any]],
    wbcs: List[Dict[str, Any]],
    platelets: List[Dict[str, Any]],
    wbc_diff: Dict[str, Any],
    median_rbc_area: float,
) -> np.ndarray:
    overlay = rgb.copy()
    alpha = 0.28

    def _fill_draw(cells, color, lw=1):
        for cell in cells:
            fill = np.zeros_like(rgb)
            cv2.drawContours(fill, [cell["contour"]], -1, color, -1)
            overlay[:] = cv2.addWeighted(overlay, 1.0, fill, alpha, 0)
            cv2.drawContours(overlay, [cell["contour"]], -1, color, lw, cv2.LINE_AA)

    _fill_draw(rbcs, _COLOR_RBC, lw=1)
    for cell in platelets:
        cv2.drawContours(overlay, [cell["contour"]], -1, _COLOR_PLT, 1, cv2.LINE_AA)

    label_short = {"neutrophil": "N", "lymphocyte": "L", "monocyte": "M", "blast": "BLAST"}
    wbc_class_colors = {
        "blast":      _COLOR_BLAST,
        "lymphocyte": _COLOR_WBC,
        "monocyte":   _COLOR_BAND,
        "neutrophil": (100, 160, 255),
    }
    for wbc in wbcs:
        cls = _classify_wbc(wbc, median_rbc_area)
        color = wbc_class_colors.get(cls, _COLOR_WBC)
        lw = 3 if cls == "blast" else 2
        fill = np.zeros_like(rgb)
        cv2.drawContours(fill, [wbc["contour"]], -1, color, -1)
        overlay[:] = cv2.addWeighted(overlay, 1.0, fill, 0.38, 0)
        cv2.drawContours(overlay, [wbc["contour"]], -1, color, lw, cv2.LINE_AA)
        cv2.putText(overlay, label_short.get(cls, "W"),
                    (wbc["cx"] - 6, wbc["cy"] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # Legend
    legend_items = [
        ("RBC", _COLOR_RBC), ("WBC/Neutrophil", (100, 160, 255)),
        ("Lymphocyte", _COLOR_WBC), ("Monocyte", _COLOR_BAND),
        ("Blast", _COLOR_BLAST), ("Platelet", _COLOR_PLT),
    ]
    y0 = 12
    for lbl, color in legend_items:
        cv2.rectangle(overlay, (8, y0), (22, y0 + 13), color, -1)
        cv2.putText(overlay, lbl, (26, y0 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
        y0 += 17

    return overlay


# ── 8. Text builders ──────────────────────────────────────────────────────────

def _build_summary(rbcs, wbcs, platelets, condition, wbc_diff, rbc_morph) -> str:
    diff = wbc_diff.get("differential", {})
    diff_str = ", ".join(f"{k.title()} {v:.0f}%" for k, v in diff.items()) or "unavailable"
    rbc_f = "; ".join(rbc_morph.get("flags", []))
    return (
        f"Peripheral blood smear: {len(rbcs)} RBCs, {len(wbcs)} WBCs, {len(platelets)} platelets segmented. "
        f"WBC differential estimate: {diff_str}. RBC morphology: {rbc_f}. Primary pattern: {condition}."
    )


def _build_recommendation(condition, risk_level, wbc_diff) -> str:
    if wbc_diff.get("blast_detected"):
        return (
            "Urgent haematology referral is strongly recommended. Blast cells in peripheral blood "
            "require prompt bone marrow examination, flow cytometry, and haematological oncology review."
        )
    if risk_level == "High":
        return (
            "Prompt clinical haematology review is recommended. Correlate with full blood count (FBC), "
            "reticulocyte count, coagulation screen, and peripheral blood film specialist review."
        )
    if risk_level == "Moderate":
        return (
            "Clinical correlation with full blood count (FBC) and patient history is recommended. "
            "Consider iron studies, B12/folate, and haemoglobin electrophoresis as appropriate."
        )
    return (
        "No significant pathological features detected in this image field. "
        "Routine follow-up and clinical correlation with full blood count are advised."
    )


def _build_next_steps(risk_level, wbc_diff) -> List[str]:
    if wbc_diff.get("blast_detected"):
        return [
            "Emergency haematology referral",
            "Urgent bone marrow biopsy and aspirate",
            "Flow cytometry immunophenotyping",
            "Repeat FBC with manual differential",
            "Cytogenetics / FISH if leukaemia confirmed",
        ]
    if risk_level == "High":
        return [
            "Haematology specialist review",
            "Full blood count (FBC) with manual differential",
            "Iron panel, B12, folate, ferritin",
            "Reticulocyte count and peripheral film review",
        ]
    if risk_level == "Moderate":
        return [
            "Repeat full blood count (FBC)",
            "Iron studies and haemoglobin electrophoresis",
            "Clinical correlation with symptoms",
        ]
    return [
        "Routine full blood count monitoring",
        "Discuss findings with GP or ordering clinician",
    ]


# ── 9. Public API ─────────────────────────────────────────────────────────────

def detect_blood_cells(image: np.ndarray) -> Dict[str, Any]:
    """Analyse a peripheral blood smear image.

    Parameters
    ----------
    image : np.ndarray  RGB uint8 microscopy image of a peripheral blood smear.

    Returns
    -------
    dict with keys: overlay, mask, rbc, wbc, platelet, finding.
    """
    rgb = _normalise_to_rgb(image)
    h, w = rgb.shape[:2]

    rbcs, wbcs, platelets = _segment_cells(rgb)
    median_rbc_area = float(np.median([c["area"] for c in rbcs])) if rbcs else 800.0

    rbc_morph = _analyse_rbc_morphology(rbcs)
    wbc_diff  = _analyse_wbc_differential(wbcs, median_rbc_area)
    plt_info  = _analyse_platelets(platelets, len(rbcs))

    condition, risk_level, prob_score, evidence = _overall_pathology(rbc_morph, wbc_diff, plt_info)

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for cell_list in (rbcs, wbcs, platelets):
        for cell in cell_list:
            cv2.drawContours(combined_mask, [cell["contour"]], -1, 255, -1)

    overlay = _draw_cell_overlay(rgb, rbcs, wbcs, platelets, wbc_diff, median_rbc_area)
    detected = not condition.startswith("No significant")

    finding: Dict[str, Any] = {
        "status": "evaluated",
        "primary_condition": condition,
        "risk_level": risk_level,
        "risk_score": prob_score,
        "model_score": prob_score,
        "score_label": "Pathology pattern score",
        "summary": _build_summary(rbcs, wbcs, platelets, condition, wbc_diff, rbc_morph),
        "recommendation": _build_recommendation(condition, risk_level, wbc_diff),
        "next_steps": _build_next_steps(risk_level, wbc_diff),
        "evidence": evidence,
        "detections": [{"label": condition, "probability": prob_score / 100.0, "box": None}] if detected else [],
        "rbc_analysis": rbc_morph,
        "wbc_analysis": wbc_diff,
        "platelet_analysis": plt_info,
        "cell_counts": {
            "rbc_in_frame": len(rbcs),
            "wbc_in_frame": len(wbcs),
            "platelet_in_frame": len(platelets),
            "total_cells_detected": len(rbcs) + len(wbcs) + len(platelets),
        },
        "population_scope": BCCD_POPULATION_SCOPE,
    }

    return {
        "overlay": overlay,
        "mask": combined_mask,
        "rbc": rbc_morph,
        "wbc": wbc_diff,
        "platelet": plt_info,
        "finding": finding,
    }
