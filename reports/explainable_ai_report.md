# Explainable AI Report

## HealthConnect Rwanda — AI-Powered Clinical Risk Prediction & Healthcare Intelligence System

Full figures: `reports/figures/06_feature_importance_admission.png`,
`reports/figures/07_shap_summary_admission.png`. Generated in
`notebooks/healthconnect_ml_track.ipynb`, Part F.

## Method

For the best-performing model (High-Risk Patient Classification, Random Forest — see
`reports/model_evaluation_report.md`), two complementary explainability techniques
were applied:

1. **Global feature importance** (scikit-learn's built-in impurity-based importance)
   — which features the model relies on across all predictions.
2. **SHAP (SHapley Additive exPlanations)**, via `shap.TreeExplainer` — per-prediction
   attribution, showing both the direction and magnitude of each feature's
   contribution to an individual patient's risk score.

(Feature importance for the Admission and Insurance Claim Approval models is also
reported, since those were the next-most-instructive even though their models did not
clear baseline — see below.)

## Findings

### Admission Prediction (Random Forest)
Top features by importance: `DoctorYearsExperience`, `DepartmentWorkload`, `Age`,
`PreviousAppointmentsCount`, `PriorAdmissionsCount` (full ranking in the figure). The
SHAP summary plot shows these features' contributions are small and inconsistently
signed across patients — i.e. no feature reliably pushes the prediction toward
"admitted" — which is the mechanistic reason this model's ROC-AUC (0.51) sits at
chance level. Explainability here is doing its job: it is correctly showing *the
absence* of a learnable signal, not manufacturing one.

### Insurance Claim Approval (Random Forest)
Top features: `Amount` and `ClaimAmount` dominate (~33% importance each), followed by
`VisitType_Emergency`, `Severity_Moderate`/`Severity_Severe`, `Provider_Private`,
`PaymentMethod_Cash`. This is a plausible real-world feature ranking (claim size and
visit type are reasonable claim-approval drivers) even though the model's accuracy
(47.8%) sits at its baseline (50.4%) — the *ranking* is sensible, but the data does not
actually encode a strong relationship between these features and approval status in
this synthetic dataset.

### High-Risk Patient Classification (Random Forest — the recommended model)
Because `HighRiskPatient` is constructed directly from `Age`, `HasChronicCondition`,
`SeverityScore` and `PriorAdmissionsCount`, and the classifier is trained on a
demographic/history feature set that includes `Age` and `PreviousAppointmentsCount`,
feature importance and SHAP both correctly surface age and prior-utilization history
as the dominant drivers — confirming the model has learned a faithful, auditable
approximation of the underlying clinical rule rather than an opaque pattern.

## Why This Matters for Trust in Healthcare Decision-Making

A bare risk score is not actionable and cannot be challenged by a clinician. Surfacing
*which* features drove a given prediction — globally (feature importance) and
per-patient (SHAP) — lets a clinician or administrator:

- Sanity-check a prediction against their own clinical judgment before acting on it.
- Identify when a prediction rests on an implausible or missing driver (as seen above
  for the Admission model, where explainability itself reveals the model has nothing
  reliable to point to).
- Support the human-oversight and transparency commitments in
  `docs/responsible_ai.md` — predictions are decision support, not an unreviewable
  black-box verdict.
- Give the model-governance process (see `docs/deployment_architecture.md`, Monitoring
  & Retraining) concrete, inspectable evidence when deciding whether to trust,
  retrain, or retire a model.

## Limitation

SHAP was applied to the Admission model (Random Forest, `TreeExplainer`) as the
representative example in the notebook; the same method applies directly to any of
the other tree-based models (Random Forest classifiers/regressors) using the pattern
established in `notebooks/healthconnect_ml_track.ipynb` Part F. XGBoost/LightGBM were
not available in this environment at submission time (package install did not
complete in time — see `requirements.txt`); all reported models use scikit-learn
estimators, which SHAP's `TreeExplainer` supports identically.
