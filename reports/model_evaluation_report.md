# Model Evaluation Report

## HealthConnect Rwanda — AI-Powered Clinical Risk Prediction & Healthcare Intelligence System
**AnalystLab Africa Experience Lab Internship — Week 6, Machine Learning Engineering Track**

Full code, confusion matrices and metric output: `notebooks/healthconnect_ml_track.ipynb`.
Data-reality context (why these numbers look the way they do): `reports/data_quality_report.md`.

Every model below is reported **against an explicit baseline** (majority-class accuracy
for classification, mean-prediction RMSE for regression) rather than in isolation —
this is the only honest way to judge whether a model actually learned something.

## Summary Table

| Algorithm | Business Problem | Accuracy | Baseline Acc. | Precision | Recall | F1 | ROC-AUC | Strengths | Limitations | Recommended Use Case |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Hospital Admission Prediction | 0.522 | 0.883 | 0.125 | 0.514 | 0.201 | 0.527 | Interpretable coefficients | Below baseline on accuracy; AUC ≈ coin flip | Not recommended as-is |
| Random Forest | Hospital Admission Prediction | 0.725 | 0.883 | 0.107 | 0.185 | 0.136 | 0.512 | Handles mixed feature types | AUC ≈ 0.51 — no real separation of admitted vs. not | Re-train once real outcome data available |
| Random Forest | Patient Satisfaction Prediction (3-class) | 0.335 | 0.410 | 0.326 | 0.322 | 0.322 | — | Multi-class support | Below majority-class baseline | Not recommended as-is |
| Random Forest | Insurance Claim Approval Prediction (binary) | 0.478 | 0.504 | 0.483 | 0.498 | 0.490 | 0.482 | Surfaces plausible top features (ClaimAmount, Amount) | At baseline — status is effectively independent of billing/claim fields in this data | Not recommended as-is |
| Random Forest | High-Risk Patient Classification | **0.746** | 0.576 | 0.782 | 0.555 | 0.649 | **0.727** | Real lift over baseline; interpretable, formalizes a clinical rule | Target is a constructed composite of the same input features (not an independent outcome) | **Recommended** — codifies a consistent, auditable risk-flagging rule |
| Linear Regression | Revenue (Billing Amount) Prediction | R²=-0.009, RMSE=83,292 vs. baseline 83,053 | — | — | — | — | — | Simple, fast | Worse than predicting the mean | Not recommended as-is |
| Random Forest Regressor | Revenue Prediction | R²=-0.033, RMSE=84,276 | — | — | — | — | — | — | Worse than baseline | Not recommended as-is |
| Gradient Boosting Regressor | Revenue Prediction | R²=-0.036, RMSE=84,407 | — | — | — | — | — | — | Worse than baseline | Not recommended as-is |
| Random Forest Regressor | Length of Stay Prediction | R²=-0.084, RMSE=2.96 vs. baseline 2.88 (n=586 admitted) | — | — | — | — | — | — | Worse than baseline; small admitted-patient sample (586) | Re-train once real outcome data available |
| K-Means (k=4) | Patient Segmentation | Silhouette 0.275 | — | — | — | — | — | Clear age/chronic-condition structure | Small per-cluster behavioral differences in outcomes | Use for chronic-care outreach targeting |
| K-Means (k=3) | Facility Segmentation | Silhouette 0.202 | — | — | — | — | — | Separates by volume/revenue | Only 20 facilities — directional, not statistically robust | Use for capacity planning, not quality benchmarking |
| Isolation Forest | Anomaly Detection (billing, claims) | 3.0% flag rate (both) | — | — | — | — | — | Catches statistical outliers for human review | Not a fraud verdict — needs human triage | Billing-audit triage queue |

## Final Model Selection

**High-Risk Patient Classification (Random Forest)** is the only model in this project
that shows real predictive lift over its baseline (74.6% vs. 57.6% accuracy, ROC-AUC
0.73). It is recommended for the conceptual deployment described in
`docs/deployment_architecture.md`, with the explicit caveat that its target label is a
constructed composite of age, chronic-condition status, diagnosis severity and
admission history — the model formalizes an existing, auditable clinical rule rather
than discovering a new relationship, which is itself a legitimate and useful ML
application (consistent, scalable rule application with SHAP-explainable output).

Every other model (admission, satisfaction, claim approval, revenue, length of stay)
performs at or below its baseline. This is **not a pipeline failure** — the same
ingestion, cleaning, feature-engineering, training and evaluation harness is reused
across all six models and runs end to end with zero errors (see the executed
notebook). It reflects a genuine property of the provided synthetic dataset: outcome
fields (`WasAdmitted`, `Rating`, `ClaimStatus`, `Amount`, `LengthOfStayDays`) do not
vary meaningfully with the clinical/operational fields that would plausibly predict
them (quantified in `reports/data_quality_report.md` under "Signal Check"). The
recommendation for these five models is to re-run this same pipeline once real
HealthConnect Rwanda operational data is available — the code requires no structural
changes, only real signal in the training data.
