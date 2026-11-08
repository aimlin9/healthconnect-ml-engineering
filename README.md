# HealthConnect Rwanda — AI-Powered Clinical Risk Prediction & Healthcare Intelligence System

**AnalystLab Africa Experience Lab Internship — Weeks 6–8, Batch E**
**Track:** Machine Learning Engineering

> **Week 8 (Final Integration → Presentation):** the pipeline reproduces identically on a second
> OS and Python version, handles 12/12 error cases as specified, and was integrated with the Data
> Science final model package: their 15 contract tests pass 15/15 here. After agreeing one shared
> contract with Data Science (per-row `status`, empty in → empty out), 12/15 pass against this pipeline
> (C3–C5 await their strict-mode harness). The High-Risk model passes the DS release gate but `Age ≥ 65`
> alone scores the same AUC, so it is labelled **DEMO ONLY**. See [Week 8](#week-8--final-integration--presentation).

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
| LinkedIn / X post drafts + images | `presentation/week7/` (images built by `scripts/week7_post_images.py`) |

## Week 8 — Final Integration & Presentation

| Check | Result | Evidence |
|---|---|---|
| Reproducibility (fresh clone, Windows 11, Python 3.12.10) | Identical to the committed model: same fingerprint, threshold and metrics, 0 flags differ | `reports/week8/evidence/reproducibility_check.json` |
| Error handling (CLI exit codes and messages) | 12/12 cases as specified | `reports/week8/evidence/error_handling_matrix.csv` |
| End-to-end (1,250 test rows through the CLI) | CLI output identical to in-process `predict()`; ROC-AUC 0.721 | `reports/week8/evidence/end_to_end_check.json` |
| Data Science package, their 15 contract tests | 15/15 (Py 3.12.10 / sklearn 1.9.1 and Py 3.14.6 / 1.9.0) | `reports/week8/evidence/ds_contract/` |
| Same 15 tests against this pipeline | 11/15 → 12/15 after the agreed changes; C3–C5 wait on the DS strict harness (DS agreed to reject, not impute) |
| DS release gate on the High-Risk model | AUC 0.721 (CI 0.691–0.752), shuffle p = 0.0099, lift 1.82 @ 20%: passes, **but** `Age ≥ 65` alone also gives 0.721 → DEMO ONLY | `reports/week8/evidence/release_gate_high_risk.json` | `ds_contract/B_mle_pipeline_vs_ds_contract.csv` |
| This pipeline's data into the DS scorer | 5,000/5,000 scored, 0 rejected | `ds_contract/summary.json` |

The Data Science model is a validated null result (hold-out AUC 0.467). It is integrated as an
**interface only** and is not for decisions. The shared-contract changes proposed to Data
Science were agreed on 8 Nov and implemented (`reports/week8/decision_log.md`, D-08 to D-13;
reply: `reports/week8/DS_Reply_2.md`). Every output row now carries `status` and `model_status`.

```bash
python scripts/week8_readiness_check.py                              # reproducibility, error handling, end-to-end
python scripts/week8_release_gate.py                                 # DS release gate + rule-restatement check
python scripts/week8_ds_contract_crosscheck.py PATH/TO/handoff_ml_engineering   # DS contract tests (DS package kept outside the repo)
python presentation/week8/build_week8_slides.py                      # slides (PNG + PDF), rendered with headless Edge
python presentation/week8/make_video.py                              # slides + your recordings -> MP4 (needs ffmpeg)
```

| Week 8 Deliverable | Location |
|---|---|
| Final ML Pipeline & Engineering Package (transition, readiness, pipeline, config, testing, error handling, reproducibility, limitations, HC-POD integration, demo workflow) | `reports/week8/ML_Engineering_Final_Package.md` |
| Response to the Data Science hand-off | `reports/week8/DS_Handoff_Response.md`, `reports/week8/DS_Reply_2.md` |
| Week 8 decision & issue log | `reports/week8/decision_log.md` |
| Integration and readiness evidence | `reports/week8/evidence/` |
| Presentation slides | `presentation/week8/HealthConnect_Week8_MLE_Slides_Ramsey.pdf`, `presentation/week8/slides/` |
| Video narration script | `presentation/week8/video_script.md` |
| Individual presentation video (9:27, own narration) | `presentation/week8/HealthConnect_Week8_MLE_Presentation_Ramsey.mp4` |

## Status

Week 6 submission — complete.
Week 7 — pipeline tested, refined and retested.
Week 8 — reproducibility, error handling and end-to-end checks pass; Data Science package received and tested
(15/15 on their package); shared contract agreed and implemented (12/15, C3–C5 waiting on the DS strict harness);
High-Risk model gated and labelled demo-only; threshold replaced by a clinic-chosen flag budget.
