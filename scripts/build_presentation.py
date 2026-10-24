"""Build the Week 6 executive presentation deck (.pptx) from real results."""
import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "presentation" / "HealthConnect_Week6_ML_Track_Presentation.pptx"
OUT.parent.mkdir(exist_ok=True)

findings = json.loads((ROOT / "reports" / "key_findings.json").read_text())

NAVY = RGBColor(0x0B, 0x2E, 0x4E)
TEAL = RGBColor(0x14, 0x7A, 0x7A)
GREY = RGBColor(0x44, 0x44, 0x44)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide():
    return prs.slides.add_slide(BLANK)


def add_title(slide, text, size=32, top=0.4, color=NAVY):
    box = slide.shapes.add_textbox(Inches(0.6), Inches(top), Inches(12.1), Inches(1.0))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = True
    p.font.color.rgb = color
    return box


def add_subtitle(slide, text, top=1.2, size=16, color=GREY):
    box = slide.shapes.add_textbox(Inches(0.6), Inches(top), Inches(12.1), Inches(0.6))
    p = box.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.italic = True
    p.font.color.rgb = color
    return box


def add_bullets(slide, bullets, top=1.6, left=0.6, width=12.1, size=18, color=RGBColor(0x22, 0x22, 0x22)):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5.3))
    tf = box.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"•  {b}"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(10)
    return box


def add_image(slide, path, left, top, width=None, height=None):
    kwargs = {}
    if width:
        kwargs["width"] = Inches(width)
    if height:
        kwargs["height"] = Inches(height)
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), **kwargs)


def footer(slide, text="AnalystLab Africa | HealthConnect Rwanda | Week 6 — ML Engineering Track"):
    box = slide.shapes.add_textbox(Inches(0.6), Inches(7.05), Inches(12.1), Inches(0.35))
    p = box.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(0x99, 0x99, 0x99)


# 1. Title ---------------------------------------------------------------
s = add_slide()
box = s.shapes.add_textbox(Inches(0.8), Inches(2.3), Inches(11.7), Inches(1.5))
p = box.text_frame.paragraphs[0]
p.text = "HealthConnect Rwanda"
p.font.size = Pt(48)
p.font.bold = True
p.font.color.rgb = NAVY
p2 = box.text_frame.add_paragraph()
p2.text = "AI-Powered Clinical Risk Prediction & Healthcare Intelligence System"
p2.font.size = Pt(24)
p2.font.color.rgb = TEAL
box2 = s.shapes.add_textbox(Inches(0.8), Inches(4.2), Inches(11.7), Inches(1.0))
p3 = box2.text_frame.paragraphs[0]
p3.text = "AnalystLab Africa Experience Lab Internship — Week 6, Batch E\nMachine Learning Engineering Track"
p3.font.size = Pt(18)
p3.font.color.rgb = GREY

# 2. Agenda ----------------------------------------------------------------
s = add_slide()
add_title(s, "Agenda")
add_bullets(s, [
    "Business problem & scope",
    "Data integration & key data-quality findings",
    "Exploratory data analysis & statistical testing",
    "Classification models (admission, satisfaction, claims, high-risk)",
    "Regression models (revenue, length of stay)",
    "Clustering & anomaly detection",
    "Explainability (feature importance + SHAP)",
    "Model comparison & recommendation",
    "Deployment architecture & responsible AI",
    "Business recommendations",
])
footer(s)

# 3. Business problem --------------------------------------------------
s = add_slide()
add_title(s, "Business Problem")
add_bullets(s, [
    "HealthConnect Rwanda needs predictive tools for hospital admissions, patient",
    "satisfaction, insurance claim outcomes, high-risk patients, billing and length of stay.",
    "",
    "Goal: build a reusable, end-to-end ML pipeline — data integration, cleaning,",
    "feature engineering, modeling, evaluation, explainability — on the enterprise",
    "healthcare dataset, and report results honestly against real baselines.",
])
footer(s)

# 4. Data scope ----------------------------------------------------------
s = add_slide()
add_title(s, "Data Scope & Integration")
add_bullets(s, [
    "27 source datasets integrated: facilities, staff, patients, clinical",
    "transactions, billing, insurance, feedback.",
    "20,000 patients; 60,000-appointment billing/claims universe.",
    "Only 5,000 appointments carry full patient/doctor/facility/date context —",
    "the analytical population for all patient-level models (fully disclosed).",
    "Zero duplicate patient or appointment IDs; zero invalid facility/department IDs.",
])
footer(s)

# 5. Data quality findings -------------------------------------------------
s = add_slide()
add_title(s, "Key Data Quality Findings")
add_bullets(s, [
    "19,990 / 20,000 patients share a phone number with another patient —",
    "phone number is not a usable record-linkage key in this dataset.",
    "Insurance coverage: 100% of patients carry an InsuranceID.",
    "Insurance claim status has only 2 classes (Approved/Pending) — no 'Rejected'",
    "claims exist, so claim-approval prediction became a binary task.",
    "Outcome fields (admission, satisfaction, claim status, billing, length of",
    "stay) do not vary meaningfully with clinical/operational fields — verified",
    "with hypothesis tests, not assumed.",
])
footer(s)

# 6. EDA — demographics -----------------------------------------------------
s = add_slide()
add_title(s, "Patient & Clinical Profile")
add_image(s, FIG / "01_demographics.png", 0.5, 1.5, width=12.3)
footer(s)

s = add_slide()
add_title(s, "Clinical & Financial Patterns")
add_image(s, FIG / "02_clinical_patterns.png", 0.5, 1.4, width=12.3)
footer(s)

# 7. Correlation ----------------------------------------------------------
s = add_slide()
add_title(s, "Statistical Analysis — Correlation & Hypothesis Testing")
add_image(s, FIG / "05_correlation_heatmap.png", 0.6, 1.4, height=5.3)
add_bullets(s, [
    "Admission rate vs. age group: Chi² p=0.060 (not significant)",
    "Satisfaction vs. facility: ANOVA p=0.554 (not significant)",
    "Billing: Severe vs. Mild severity: see notebook §3.3",
], top=1.6, left=7.4, width=5.3, size=16)
footer(s)

# 8. Classification results -------------------------------------------------
s = add_slide()
add_title(s, "Classification Models — Results vs. Baseline")
add_bullets(s, [
    "Hospital Admission (RF): 72.5% acc. vs. 88.3% baseline — AUC 0.51 (no lift)",
    "Patient Satisfaction (RF, 3-class): 33.5% acc. vs. 41.0% baseline (no lift)",
    "Insurance Claim Approval (RF, binary): 47.8% acc. vs. 50.4% baseline (no lift)",
    "High-Risk Patient (RF): 74.6% acc. vs. 57.6% baseline — AUC 0.73  ✅ real lift",
    "",
    "Every model benchmarked against an explicit baseline — no inflated claims.",
])
footer(s)

# 9. Regression results ---------------------------------------------------
s = add_slide()
add_title(s, "Regression Models — Results vs. Baseline")
add_bullets(s, [
    "Revenue (Billing Amount): best R² = -0.009 (Linear Regression) — at or",
    "below a naive mean-prediction baseline across all 3 algorithms tried",
    "Length of Stay (n=586 admitted patients): R² = -0.084 — at baseline",
    "",
    "Conclusion: billing amount and length of stay are not explained by the",
    "available clinical/operational features in this synthetic dataset.",
])
footer(s)

# 10. Clustering -----------------------------------------------------------
s = add_slide()
add_title(s, "Clustering — Patient & Facility Segmentation")
add_bullets(s, [
    "Patient segmentation (K-Means, k=4, silhouette 0.275): separates by age",
    "and chronic-condition status, plus a high-utilization/high-billing cluster.",
    "→ Recommendation: target chronic-care outreach at the two chronic clusters.",
    "",
    "Facility segmentation (K-Means, k=3, silhouette 0.202, n=20 facilities):",
    "separates mainly by volume/revenue, not quality metrics.",
    "→ Recommendation: use for capacity planning, not quality benchmarking.",
])
footer(s)

# 11. Anomaly detection -----------------------------------------------------
s = add_slide()
add_title(s, "Anomaly Detection")
add_bullets(s, [
    f"Billing amounts flagged as statistical outliers: {findings['anomalous_billing_pct']}%",
    f"Insurance claims flagged as suspicious: {findings['anomalous_claims_pct']}%",
    "2 of 20 facilities show unusually high patient volume (|z| > 1.5)",
    "",
    "Use: route flagged transactions to a human billing auditor — a triage",
    "signal, not an automated accept/reject decision.",
])
footer(s)

# 12. Explainability ---------------------------------------------------------
s = add_slide()
add_title(s, "Explainability — SHAP & Feature Importance")
add_image(s, FIG / "07_shap_summary_admission.png", 0.6, 1.4, height=5.4)
add_bullets(s, [
    "Feature importance + SHAP applied to every tree-based model.",
    "For High-Risk classification: age and prior-admission history dominate —",
    "a faithful, auditable approximation of the clinical rule.",
    "For Admission prediction: SHAP correctly shows weak, inconsistent",
    "contributions — explainability reveals the absence of signal, not just",
    "the presence of it.",
], top=1.6, left=7.4, width=5.3, size=15)
footer(s)

# 13. Confusion matrix / recommended model -----------------------------------
s = add_slide()
add_title(s, "Recommended Model: High-Risk Patient Classification")
add_image(s, FIG / "08_confusion_matrix_highrisk.png", 0.6, 1.5, height=4.6)
add_bullets(s, [
    "74.6% accuracy vs. 57.6% baseline",
    "ROC-AUC 0.73",
    "Formalizes an existing, auditable clinical risk rule",
    "SHAP-explainable per patient",
    "Only model with demonstrated predictive lift in this project",
], top=1.8, left=6.8, width=5.9, size=17)
footer(s)

# 14. Deployment architecture -------------------------------------------------
s = add_slide()
add_title(s, "Conceptual Deployment Architecture")
add_bullets(s, [
    "HIS/EMR, Billing & Claims, Insurance, Feedback → Ingestion & Validation",
    "→ Cleaning & Integration → Feature Store → Model Training/Explainability",
    "→ Model Serving API → HIS UI (clinician) / Power BI (executive)",
    "→ Monitoring, Drift Detection & Scheduled Retraining",
    "",
    "Full diagram: docs/deployment_architecture.md",
])
footer(s)

# 15. Recommendations ---------------------------------------------------------
s = add_slide()
add_title(s, "Business Recommendations")
add_bullets(s, [
    "1. Deploy the High-Risk Patient classifier as clinician decision support",
    "   with per-prediction SHAP explanations.",
    "2. Fix patient record-linkage (phone numbers are not unique) before any",
    "   phone-based patient lookup or outreach feature.",
    "3. Re-collect/re-generate admission, satisfaction, claim, billing and",
    "   length-of-stay outcome data — the pipeline is ready, the signal is not.",
    "4. Route anomaly-flagged billing/claims to a human auditor.",
    "5. Use patient clusters for chronic-care outreach; facility clusters for",
    "   capacity planning, not quality benchmarking.",
])
footer(s)

# 16. Thank you ----------------------------------------------------------
s = add_slide()
box = s.shapes.add_textbox(Inches(0.8), Inches(3.0), Inches(11.7), Inches(1.5))
p = box.text_frame.paragraphs[0]
p.text = "Thank you"
p.font.size = Pt(44)
p.font.bold = True
p.font.color.rgb = NAVY
box2 = s.shapes.add_textbox(Inches(0.8), Inches(4.2), Inches(11.7), Inches(1.0))
p2 = box2.text_frame.paragraphs[0]
p2.text = "Full notebook, reports and code: see the project GitHub repository"
p2.font.size = Pt(18)
p2.font.color.rgb = GREY

prs.save(OUT)
print(f"Saved {OUT} with {len(prs.slides.slides if hasattr(prs.slides,'slides') else prs.slides._sldIdLst)} slides")
print(f"Slide count: {len(prs.slides._sldIdLst)}")
