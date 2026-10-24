# Executive Summary

## HealthConnect Rwanda — AI-Powered Clinical Risk Prediction & Healthcare Intelligence System
**AnalystLab Africa Experience Lab Internship — Week 6, Machine Learning Engineering Track**

## 1. Business Problem

HealthConnect Rwanda needs predictive tools to anticipate hospital admissions, patient
satisfaction, insurance claim outcomes, high-risk patients, billing amounts and length
of stay, so clinicians and administrators can act proactively. This project builds and
honestly evaluates that predictive layer end to end on the provided synthetic
enterprise dataset.

## 2. Approach

- Integrated 27 source datasets (facilities, staff, patients, clinical transactions,
  billing, insurance, feedback) into a validated, appointment-level analytical
  dataset of 5,000 appointments (the population for which full patient/doctor/
  facility/date context exists — see `reports/data_quality_report.md`).
- Engineered 17+ predictive features (age group, prior-visit/admission/lab/
  prescription counts, department and doctor workload, facility utilization,
  revenue category, a constructed clinical risk score, etc.).
- Built and compared 4 classification models (hospital admission, patient
  satisfaction, insurance claim approval, high-risk patient) and 2 regression models
  (billing amount, length of stay), each benchmarked against an explicit baseline.
- Segmented patients (4 clusters) and facilities (3 clusters) via K-Means; flagged
  statistical anomalies in billing and insurance claims via Isolation Forest.
- Interpreted models with feature importance and SHAP.

## 3. Key Findings

- **Admission rate**: 11.7% of the 5,000 appointments resulted in admission.
- **Average patient satisfaction**: 2.93 / 5 (across the 35.1% of appointments with
  feedback) — flat across severity levels and facilities (ANOVA p = 0.554), i.e. no
  facility or severity group stands out as a satisfaction driver in this data.
- **Insurance coverage**: 100% of the 20,000 patients carry an insurance ID — far
  higher than the brief's assumed partial-coverage scenario.
- **Duplicate phone numbers**: 19,990 of 20,000 patients share a phone number with
  another patient — phone numbers are not usable as a record-linkage key in this
  dataset (see Data Quality Report).
- **42.4% of the appointment population** meets the constructed high-risk criteria
  (age ≥ 65, chronic condition, severe diagnosis, or prior admission) — the one model
  that reliably reproduces this flag is the recommended deployment candidate
  (74.6% accuracy vs. 57.6% baseline, ROC-AUC 0.73).
- **No other target variable** (admission, satisfaction, claim approval, billing
  amount, length of stay) shows a statistically meaningful relationship to the
  clinical/operational features in this synthetic dataset — every model built for
  them performs at or below its baseline, and this is evidenced with explicit
  hypothesis tests and baseline comparisons rather than asserted.
- **Anomaly detection** flags ~3% of billing amounts and ~3% of insurance claims as
  statistical outliers warranting human review; 2 of 20 facilities show unusually
  high patient volume relative to the rest.

## 4. Recommendations

1. **Deploy the High-Risk Patient classifier** as a decision-support flag for
   clinicians, paired with SHAP explanations per prediction (see
   `reports/explainable_ai_report.md`) — it is the only model here with demonstrated
   lift, and it formalizes a consistent, auditable clinical rule.
2. **Fix patient record-linkage data quality** before any phone-based patient lookup
   or SMS outreach feature is built — the current phone-number field cannot
   distinguish patients.
3. **Re-collect or re-generate the admission/satisfaction/claim/billing/length-of-stay
   outcome data** before relying on predictive models for those use cases — the
   reusable pipeline built here (data integration → cleaning → feature engineering →
   training → evaluation → explainability) needs no structural changes once real
   signal is present in the target variables.
4. **Route Isolation-Forest-flagged billing and claims to a human billing auditor**,
   not an automated accept/reject — consistent with the Responsible AI commitment
   to human oversight.
5. **Use the patient segmentation** (age/chronic-condition clusters) to target
   chronic-disease outreach programmes, and the **facility segmentation** for
   capacity/staffing planning rather than quality benchmarking (facility clusters do
   not separate meaningfully on admission rate or satisfaction in this data).

## 5. Limitations

- All data is synthetic/educational; findings describe what the *provided data*
  supports, not real HealthConnect Rwanda operations.
- Patient/appointment-level models are trained on the 5,000-appointment subset with
  full dimensional context, not the full 60,000-appointment universe referenced by
  the billing/diagnosis/payment tables — fully disclosed in
  `reports/data_quality_report.md`.
- Model performance is honestly bounded by the signal available in the synthetic
  data, not by the modeling approach; every metric in `reports/model_evaluation_report.md`
  is reported against a naive baseline so this is never obscured.
