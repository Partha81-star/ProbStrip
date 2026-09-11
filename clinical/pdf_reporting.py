from io import BytesIO
from xml.sax.saxutils import escape

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    PageBreak,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


TEAL = colors.HexColor("#0F766E")
INK = colors.HexColor("#17202A")
MUTED = colors.HexColor("#5F6B76")
PALE_TEAL = colors.HexColor("#ECFDF5")
PALE_AMBER = colors.HexColor("#FFF7ED")
LINE = colors.HexColor("#D7E3E1")


def _safe(value):
    text = str(value or "")
    ascii_text = text.encode("ascii", "ignore").decode("ascii").strip()
    if text and len(ascii_text) < len(text) * 0.55:
        return "Non-English text is available in the accompanying JSON report."
    return escape(ascii_text or text)


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=5 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            "Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=INK,
            spaceBefore=5 * mm,
            spaceAfter=2.5 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodySmall",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            textColor=INK,
        )
    )
    styles.add(
        ParagraphStyle(
            "Muted",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=MUTED,
        )
    )
    return styles


def _page_footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "ProbStrip research image review")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def _image(array, max_width=78 * mm, max_height=60 * mm):
    image = np.asarray(array)
    if image.ndim == 3:
        encoded_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        encoded_image = image.astype(np.uint8)
    ok, encoded = cv2.imencode(".png", encoded_image)
    if not ok:
        return None
    buffer = BytesIO(encoded.tobytes())
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    return Image(buffer, width=width * scale, height=height * scale)


def _document(story):
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title="ProbStrip patient report",
        author="ProbStrip",
    )
    document.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return output.getvalue()


def _notice(text, styles):
    table = Table([[Paragraph(f"<b>Important:</b> {_safe(text)}", styles["BodySmall"])]], colWidths=[170 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_AMBER),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#D97706")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def _quality_table(checks, styles, general=False):
    rows = [["Capture check", "Measured value", "Result"]]
    for check in checks:
        status = ("Pass" if check.get("ok") else "Review") if general else check["status"]
        rows.append(
            [
                Paragraph(_safe(check["name"]), styles["BodySmall"]),
                Paragraph(f"{check['value']:.1f} {_safe(check['unit'])}", styles["BodySmall"]),
                Paragraph(_safe(status), styles["BodySmall"]),
            ]
        )
    table = Table(rows, colWidths=[70 * mm, 55 * mm, 45 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAF9")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _clinician_section(payload, styles):
    review = payload.get("clinician_review") or {}
    if review.get("status") != "reviewed":
        return [
            Paragraph("Clinical interpretation", styles["Section"]),
            Paragraph(
                "Awaiting review by a qualified healthcare professional. No automated diagnosis was generated.",
                styles["BodySmall"],
            ),
        ]
    return [
        Paragraph("Clinician review", styles["Section"]),
        Paragraph(
            f"<b>Impression or diagnosis:</b> {_safe(review.get('impression') or 'No diagnosis entered.')}",
            styles["BodySmall"],
        ),
        Spacer(1, 1.5 * mm),
        Paragraph(f"<b>Notes:</b> {_safe(review.get('note') or review.get('observations') or 'None entered.')}", styles["BodySmall"]),
        Spacer(1, 1.5 * mm),
        Paragraph(f"<b>Recommended next step:</b> {_safe(review.get('recommendation') or 'Follow local clinical guidance.')}", styles["BodySmall"]),
    ]


def retinal_report_as_pdf(payload, original_image=None, result_image=None):
    styles = _styles()
    quality = payload["quality"]
    review = payload.get("clinician_review") or {}
    measures = review.get("reviewed_research_measures") or payload["research_measures"]
    outcome = payload["review_outcome"]
    english_outcomes = {
        "ready": "Vessel map ready for clinician review.",
        "review": "Clinician review is especially important because the model flagged uncertainty.",
        "retake": "A clearer retinal image is needed before vessel mapping.",
    }
    outcome_text = english_outcomes.get(outcome.get("level"), outcome.get("title", "Review required."))

    story = [
        Paragraph("ProbStrip Retinal Vessel Report", styles["ReportTitle"]),
        Paragraph(f"Report reference: {_safe(payload['case_id'])}<br/>Created: {_safe(payload['created_at'])}", styles["Muted"]),
        Spacer(1, 4 * mm),
        _notice(payload["safety_notice"], styles),
        Paragraph("Result summary", styles["Section"]),
        Paragraph(f"<b>{_safe(outcome_text)}</b>", styles["BodySmall"]),
        Spacer(1, 2 * mm),
        Paragraph(
            "The values below describe the software output. They do not show whether the eye is healthy or unhealthy.",
            styles["BodySmall"],
        ),
    ]

    metric_rows = [
        ["Image quality", "Visible vessel coverage", "Area needing review"],
        [
            f"{quality['score']}/100",
            f"{measures['visible_vessel_coverage_percent']:.2f}%",
            f"{measures['low_confidence_area_percent']:.2f}%",
        ],
    ]
    metrics = Table(metric_rows, colWidths=[56 * mm] * 3)
    metrics.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_TEAL),
                ("BOX", (0, 0), (-1, -1), 0.5, TEAL),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, LINE),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("FONTSIZE", (0, 1), (-1, 1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([Spacer(1, 4 * mm), metrics])
    story.extend(_clinician_section(payload, styles))

    original = _image(original_image) if original_image is not None else None
    result = _image(result_image) if result_image is not None else None
    if original is not None and result is not None:
        image_table = Table(
            [
                [original, result],
                ["Original image", "Reviewed vessel overlay"],
            ],
            colWidths=[85 * mm, 85 * mm],
        )
        image_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 1), (-1, 1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(KeepTogether([Paragraph("Images", styles["Section"]), image_table]))

    story.extend(
        [
            PageBreak(),
            Paragraph("Image-quality checks", styles["Section"]),
            Paragraph(_safe(quality["summary"]), styles["BodySmall"]),
            Spacer(1, 2 * mm),
            _quality_table(quality["checks"], styles),
            Paragraph("Technical provenance", styles["Section"]),
            Paragraph(
                f"Model: {_safe(payload.get('model_settings', {}).get('model', 'ProbStrip'))}<br/>"
                f"Checkpoint: {_safe(payload.get('model_settings', {}).get('checkpoint_sha256', 'Not recorded'))}<br/>"
                f"Calibration: {_safe(payload.get('model_settings', {}).get('calibration_status', 'Research profile'))}<br/>"
                f"Quality policy: {_safe(quality.get('policy_version', 'legacy'))}",
                styles["BodySmall"],
            ),
            Paragraph("Safety reminder", styles["Section"]),
            Paragraph(
                "Seek prompt professional care for sudden vision loss, severe eye pain, new flashes, or many new floaters. Do not wait for this report.",
                styles["BodySmall"],
            ),
        ]
    )
    return _document(story)


def general_report_as_pdf(payload, original_image=None, result_image=None):
    styles = _styles()
    quality = payload["technical_quality"]
    story = [
        Paragraph(f"ProbStrip {_safe(payload['imaging_type'])} Report", styles["ReportTitle"]),
        Paragraph(f"Report reference: {_safe(payload['case_id'])}<br/>Created: {_safe(payload['created_at'])}", styles["Muted"]),
        Spacer(1, 4 * mm),
        _notice(payload["safety_notice"], styles),
        Paragraph("Technical observations", styles["Section"]),
    ]
    for item in payload["technical_observations"]:
        story.append(Paragraph(f"- {_safe(item)}", styles["BodySmall"]))
    story.extend(_clinician_section(payload, styles))

    original = _image(original_image) if original_image is not None else None
    result = _image(result_image) if result_image is not None else None
    if original is not None and result is not None:
        images = Table(
            [[original, result], ["Original image", "Enhanced review image"]],
            colWidths=[85 * mm, 85 * mm],
        )
        images.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"), ("FONTSIZE", (0, 1), (-1, 1), 8)]))
        story.append(KeepTogether([Paragraph("Images", styles["Section"]), images]))
    story.extend(
        [
            Paragraph("Image-quality checks", styles["Section"]),
            Paragraph(f"Quality score: {quality['score']}/100. {_safe(quality['status'])}", styles["BodySmall"]),
            Spacer(1, 2 * mm),
            _quality_table(quality["checks"], styles, general=True),
            Paragraph("Report limitation", styles["Section"]),
            Paragraph(
                "This technical enhancement is not a validated disease detector. A qualified clinician must review the original image.",
                styles["BodySmall"],
            ),
        ]
    )
    return _document(story)
