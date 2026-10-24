# HealthConnect Rwanda — AI-Powered Clinical Risk Prediction & Healthcare Intelligence System

**AnalystLab Africa Experience Lab Internship — Week 6, Batch E**
**Track:** Machine Learning Engineering

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
├── reports/                       # Data Quality Report, Model Evaluation, Explainable AI, Executive Summary
│   └── figures/                   # exported chart images used in reports/slides
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
python scripts/download_data.py
```

## Data

27 datasets are sourced from the AnalystLab Africa Week 6 brief (Google Drive / Google Sheets
links). `data/dataset_manifest.json` records the dataset name, source URL and file ID used to
fetch each one. Two multi-tab workbooks (`Health Facilities`, `Clinical Transactions`) are
exported as `.xlsx` and split per sheet during cleaning, since each tab represents a distinct
entity (e.g. Clinical Transactions -> Appointments, Diagnoses, Laboratory Tests, Prescriptions,
Admissions, Referrals).

## Status

Work in progress — Week 6 submission, deadline Sunday 11:59 PM WAT.
