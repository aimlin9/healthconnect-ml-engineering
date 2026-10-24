# Executive Summary (Draft)

## HealthConnect Rwanda — AI-Powered Clinical Risk Prediction & Healthcare Intelligence System
**AnalystLab Africa Experience Lab Internship — Week 6, Machine Learning Engineering Track**

> This summary will be finalized with real figures once models are trained. Structure
> follows the brief's required sections (business problem, approach, key findings,
> recommendations, limitations).

## 1. Business Problem

HealthConnect Rwanda operates hospitals and health facilities across Rwanda and needs
predictive tools to anticipate hospital admissions, patient satisfaction, insurance claim
outcomes, high-risk patients, billing amounts and length of stay — so clinicians and
administrators can act proactively rather than reactively.

## 2. Approach

- Integrated 27 enterprise datasets (facilities, staff, patients, clinical transactions,
  billing, insurance, feedback) into a machine-learning-ready dataset.
- Engineered predictive features (age group, prior-visit counts, chronic-condition
  counts, department/doctor workload, facility utilization, revenue category, etc.).
- Built and compared classification models (admission, satisfaction, claim approval,
  high-risk flag) and regression models (billing amount, length of stay).
- Segmented patients and facilities via clustering; flagged anomalies (billing, claims,
  lab results, duplicate records).
- Interpreted the best-performing models with feature importance and SHAP.

## 3. Key Findings

_TBD after modeling — to be filled with actual metrics and the 3-5 most decision-relevant
findings (e.g. top admission-risk drivers, which facilities cluster as under-resourced,
anomaly rate in billing)._

## 4. Recommendations

_TBD — tied directly to findings above (e.g. staffing reallocation, proactive outreach
to patients flagged low-satisfaction, claims-review process for flagged anomalies)._

## 5. Limitations

- All data is synthetic/educational; findings are illustrative of the *method*, not a
  claim about real HealthConnect Rwanda operations.
- Model performance is bounded by the signal available in the synthetic data; see
  `reports/model_evaluation_report.md` for honest per-model limitations.
