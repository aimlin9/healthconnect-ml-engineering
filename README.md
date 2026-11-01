# HealthConnect Rwanda — AI-Powered Clinical Risk Prediction & Healthcare Intelligence System

**AnalystLab Africa Experience Lab Internship — Weeks 6–7, Batch E**
**Track:** Machine Learning Engineering

> **Week 7 (Testing → Refinement → Validation):** the Week 6 High-Risk Patient model is now a
> tested, validated, reproducible pipeline. Week 6 inference passed 2/14 reliability probes; the
> refined pipeline passes 14/14, with 61 automated tests. See [Week 7](#week-7--ml-pipeline-testing--reliability).

## Project

HealthConnect Rwanda operates a network of hospitals and health facilities across Rwanda's
provinces, districts, sectors and villages. This project builds a machine-learning-ready
healthcare analytics pipeline over 27 interconnected enterprise datasets (facilities, staff,
patients, clinical transactions, billing, insurance claims, feedback) and delivers:

- Hospital admission prediction (classification)
- Patient satisfaction prediction (classification)
- Insurance claim approval prediction (classification)
- High-risk patient classification
- Revenue (billing) prediction (regression)
- Length-of-stay prediction (regression)
- Patient and facility segmentation (clustering)
- Anomaly detection (billing fraud, duplicate records, abnormal lab results)
- Model explainability (feature importance, SHAP)
- A conceptual deployment architecture for a hospital information system

See `docs/AnalystLab_Africa_Week6_Brief.pdf` for the full assignment brief.

## Repository Structure

```
AnalystLab_Week6/
├── data/
│   ├── dataset_manifest.json      # source -> Google Drive/Sheets mapping for all 27 datasets
│   ├── raw/                       # original downloaded datasets (27 files)
│   ├── processed/                 # cleaned, validated, integrated dataset
│   └── feature_engineered/        # ML-ready feature set
├── notebooks/                     # end-to-end ML notebook(s)
├── models/                        # trained model artifacts, comparison table, metrics
│   └── high_risk_pipeline.joblib  # [Week 7] saved pipeline (+ _meta.json)
├── reports/                       # Data Quality Report, Model Evaluation, Explainable AI, Executive Summary
│   ├── figures/                   # exported chart images used in reports/slides
│   └── week7/                     # [Week 7] testing report, summary, issue/decision logs, evidence/
├── src/hc_pipeline/               # [Week 7] pipeline package: config, validation, features, pipeline
├── tests/                         # [Week 7] pytest suite (61 tests)
├── presentation/                  # executive presentation slides
├── docs/                          # deployment architecture, responsible AI, assignment brief
├── references/                    # data dictionary / source notes
├── scripts/                       # data download + reusable pipeline scripts
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py              # pulls the 27 source datasets
python scripts/01_build_core_dataset.py       # validate, clean, integrate -> data/processed/
python scripts/02_feature_engineering.py      # -> data/feature_engineered/healthconnect_features.csv
python scripts/py_to_ipynb.py notebooks/healthconnect_ml_track.py   # build + execute the notebook
python scripts/build_presentation.py          # -> presentation/*.pptx
```

## Data

27 datasets are sourced from the AnalystLab Africa Week 6 brief (Google Drive / Google Sheets
links). `data/dataset_manifest.json` records the dataset name, source URL and file ID used to
fetch each one. Two multi-tab workbooks (`Health Facilities`, `Clinical Transactions`) are
exported as `.xlsx` and split per sheet during cleaning, since each tab represents a distinct
entity (e.g. Clinical Transactions -> Appointments, Diagnoses, Laboratory Tests, Prescriptions,
Admissions, Referrals). See `docs/table_relationships.md` for how all 27 tables relate, and
`reports/data_quality_report.md` for the key scope/coverage finding that drives everything
downstream (only 5,000 of 60,000 appointments carry full patient/doctor/facility context).

## Results (headline numbers)

- 5,000-appointment integrated analytical dataset, 75 engineered features.
- 4 classification models + 2 regression models, every one benchmarked against an explicit
  baseline (majority-class / mean-prediction) — not reported in isolation.
- **Only the High-Risk Patient classifier shows real predictive lift** (74.6% accuracy vs.
  57.6% baseline, ROC-AUC 0.73) — recommended for deployment. The other five models perform
  at or below baseline because the outcome fields do not vary meaningfully with the available
  features in this synthetic dataset (verified with hypothesis tests, not assumed). Full
  honest breakdown: `reports/model_evaluation_report.md`.
- Patient segmentation (4 clusters) and facility segmentation (3 clusters) via K-Means;
  anomaly detection on billing/claims via Isolation Forest.
- Model explainability via feature importance + SHAP (`reports/explainable_ai_report.md`).

## Deliverables Checklist (ML Engineering Track)

| Deliverable | Location |
|---|---|
| Machine Learning Notebook (.ipynb) | `notebooks/healthconnect_ml_track.ipynb` |
| Cleaned and feature-engineered dataset | `data/feature_engineered/healthconnect_features.csv` |
| Model Evaluation Report | `reports/model_evaluation_report.md` |
| Explainable AI Report | `reports/explainable_ai_report.md` |
| Confusion Matrices and Evaluation Metrics | notebook Part G + `models/*_comparison.csv` + `reports/figures/08_confusion_matrix_highrisk.png` |
| Model Comparison Table | `reports/model_evaluation_report.md` |
| Deployment Architecture Diagram | `docs/deployment_architecture.md` |
| Executive Summary (3–5 pages) | `reports/executive_summary.md` |
| Presentation Slides (10–15 slides) | `presentation/HealthConnect_Week6_ML_Track_Presentation.pptx` |
| Responsible / Ethical AI | `docs/responsible_ai.md` |
| Data Quality Report | `reports/data_quality_report.md` |

## Week 7 — ML Pipeline Testing & Reliability

Week 7 tested the Week 6 recommended model as a pipeline, fixed what failed, and retested.

| | Week 6 | Week 7 |
|---|---|---|
| Reliability probes passed | 2 / 14 | 14 / 14 |
| Score a single patient | Crashed (batch-dependent `get_dummies`) | Works; score identical alone or in a batch |
| Impossible / missing input (Age 250, NaN) | Silently scored | Rejected with row and reason |
| Saved, loadable model | None | `models/high_risk_pipeline.joblib` + metadata |
| Feature using future visits | 11.1% of rows | 0% (point-in-time count) |
| Logging | `print()` | Loads, rejects, extrapolation warnings, scores, latency |
| Automated tests | None | 61, mutation-checked (5/5 injected bugs caught) |
| ROC-AUC (same test split) | 0.727 | 0.721 |
| Recall | 0.555 @ 0.50 | 0.785 @ 0.33 — but precision 0.49; threshold choice open with Data Science |

**Run it**

```bash
source .venv/bin/activate
python scripts/03_train_pipeline.py                          # train, evaluate, save the pipeline
pytest                                                       # 61 tests
python scripts/predict.py IN.csv OUT.csv --skip-invalid --log-file run.log
python scripts/week7_audit_week6_pipeline.py                 # before: probes vs. Week 6 code
python scripts/week7_retest_probes.py                        # after: same probes vs. refined pipeline
python scripts/week7_mutation_check.py                       # does the suite catch injected bugs?
```

Input columns: `Gender`, `VisitType`, `Age`, `DoctorYearsExperience`, `PriorAppointmentsCount`,
`DepartmentWorkload` (contract in `src/hc_pipeline/config.py`). Output adds `risk_probability`,
`high_risk_flag`, `threshold`, `model_version`.

| Week 7 Deliverable | Location |
|---|---|
| Testing & Reliability Package (main report) | `reports/week7/ML_Pipeline_Testing_Reliability_Report.md` |
| Week 7 Project Summary | `reports/week7/Week7_Project_Summary.md` |
| Integrated pipeline | `src/hc_pipeline/`, `scripts/03_train_pipeline.py`, `scripts/predict.py` |
| Test plan, cases and results | Report §2–§5, `tests/`, `reports/week7/evidence/pytest_output.txt` |
| Before / after evidence | `reports/week7/evidence/week6_audit_results.json`, `week7_retest_results.json` |
| Model re-evaluation | `reports/week7/evidence/model_refinement_results.json`, `threshold_operating_points.json` |
| End-to-end demo | `reports/week7/evidence/cli_demo/` |
| Issue log / decision log | `reports/week7/issue_log.csv`, `reports/week7/decision_log.md` |
| Configuration / dependencies | `src/hc_pipeline/config.py`, `requirements.txt`, report §7–§8 |

## Status

Week 6 submission — complete.
Week 7 — pipeline tested, refined and retested; cross-track test with Data Science and the
threshold decision are open (report §10, ISS-09).
