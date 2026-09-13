import numpy as np


def assess_retinal_diagnosis(measures: dict) -> dict:
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


def assess_general_diagnosis(modality: str, edge_area_percent: float, quality_score: int) -> dict:
    if "X-ray" in modality or "bone" in modality.lower():
        if edge_area_percent > 16.0:
            cond = "High-Density Bone Cortical Discontinuity (Fracture Suspected)"
            level = "High"
            score = 78
            rec = "Orthopedic evaluation and multi-view radiographic correlation recommended to confirm cortical fracture line."
        elif edge_area_percent > 11.0:
            cond = "Subtle Cortical Irregularity or Fissure"
            level = "Moderate"
            score = 48
            rec = "Secondary radiological review advised to rule out occult hairline fracture."
        else:
            cond = "No Definite Fracture Discontinuity"
            level = "Low"
            score = 15
            rec = "Correlate with local tenderness and physical examination."
    elif "Ultrasound" in modality:
        if edge_area_percent > 14.0:
            cond = "Focal Echogenic Discontinuity or Hypoechoic Lesion"
            level = "Moderate"
            score = 52
            rec = "Targeted ultrasound follow-up or Doppler assessment suggested."
        else:
            cond = "Preserved Fascicular / Soft-Tissue Architecture"
            level = "Low"
            score = 18
            rec = "Routine clinical follow-up."
    else:
        cond = "Evaluated Structural Morphology"
        level = "Low"
        score = 20
        rec = "Review scan with corresponding clinical specialist."

    return {
        "status": "evaluated",
        "primary_condition": cond,
        "risk_level": level,
        "risk_score": score,
        "summary": f"Automated structure analysis indicates {level.lower()} probability of {cond}.",
        "differential_diagnoses": [
            {
                "condition": cond,
                "risk_level": level,
                "probability_percent": score,
                "evidence": f"Calculated structural edge coverage: {edge_area_percent:.1f}%, Image quality: {quality_score}/100",
            }
        ],
        "recommendation": rec,
    }
