# Part H — Conceptual Deployment Architecture

## HealthConnect Rwanda Clinical Risk Prediction & Healthcare Intelligence System

### 1. Data Sources

- **Hospital Information System (HIS)** — appointments, admissions, referrals
- **Electronic Medical Records (EMR)** — diagnoses, laboratory tests, prescriptions
- **Billing & Insurance Systems** — billing, insurance claims, payments
- **Patient Feedback Platform** — satisfaction ratings and free-text feedback
- **Facility Master Data** — health facilities, departments, staff, beds, equipment
- **Reference/Geo Data** — countries, provinces, districts, sectors, villages

These correspond directly to the 27 source tables in `data/dataset_manifest.json`.

### 2. Data Preprocessing Pipeline

```
Raw source tables (27)
   -> Ingestion layer (scheduled batch pull from HIS/EMR/Billing exports)
   -> Validation (schema checks, FK integrity, duplicate IDs, impossible ages/amounts)
   -> Cleaning (missing values, type coercion, category standardization)
   -> Integration (join on patient_id / facility_id / department_id / doctor_id)
   -> Feature engineering (age group, prior visit counts, workload indices, etc.)
   -> Feature store (versioned, reproducible feature set per model)
```

### 3. Model Serving Layer

- Trained models (admission risk, satisfaction, claim approval, high-risk flag, billing
  amount, length-of-stay) are serialized (`models/*.pkl`) and served behind a thin
  **prediction API** (FastAPI-style service), one endpoint per use case.
- Each endpoint accepts a patient/appointment payload, applies the same preprocessing +
  feature engineering pipeline used in training (no train/serve skew), and returns a
  prediction plus a confidence score and top contributing features (SHAP values).

### 4. API Integration with the Hospital Information System (HIS)

- The HIS calls the prediction API synchronously at key workflow points:
  - **At appointment check-in** -> admission-risk score surfaced to the triage nurse.
  - **At appointment close-out** -> satisfaction-risk score flags patients for proactive
    follow-up.
  - **At claim submission** -> claim-approval likelihood shown to the billing officer
    before submission to the insurer.
- Authentication via service-to-service API keys/OAuth2 client-credentials; all calls
  logged for audit.

### 5. Dashboard Integration

- Predictions and model-performance metrics feed the existing Power BI / executive
  dashboards (Week 6 Data Analytics track) via a scheduled export or a shared
  predictions table, so executives see risk distributions, not just historical KPIs.
- Clinicians see per-patient risk flags inside the HIS UI; administrators see
  facility/department-level aggregates (e.g. "% of today's admissions flagged high-risk").

### 6. Monitoring & Retraining Strategy

- **Data drift monitoring**: track feature distributions (age mix, diagnosis mix,
  facility mix) weekly; alert if drift exceeds threshold.
- **Performance monitoring**: track rolling precision/recall/AUC against realized
  outcomes (e.g. did the patient actually get admitted) with a feedback-label delay
  window appropriate to each target.
- **Retraining cadence**: scheduled monthly retrain on the latest 12 months of data,
  plus an ad-hoc retrain trigger if monitored performance drops below an agreed floor.
- **Champion/challenger**: new model versions are shadow-scored against the live
  model before promotion; promotion requires sign-off and is logged.
- **Rollback**: previous model version kept available for immediate rollback if a
  regression is detected post-deployment.

### 7. Architecture Diagram

```
 ┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌───────────────┐
 │  HIS / EMR  │   │   Billing   │   │  Insurance    │   │   Feedback    │
 │  Appointments│  │  & Claims   │   │  Systems      │   │   Platform    │
 └──────┬──────┘   └──────┬──────┘   └──────┬───────┘   └──────┬────────┘
        │                 │                  │                  │
        └─────────────────┴────────┬─────────┴──────────────────┘
                                    ▼
                         ┌───────────────────────┐
                         │  Ingestion & Validation │
                         └──────────┬─────────────┘
                                    ▼
                         ┌───────────────────────┐
                         │ Cleaning & Integration  │
                         └──────────┬─────────────┘
                                    ▼
                         ┌───────────────────────┐
                         │  Feature Engineering    │
                         │     (Feature Store)     │
                         └──────────┬─────────────┘
                                    ▼
                 ┌──────────────────┴───────────────────┐
                 ▼                                       ▼
      ┌─────────────────────┐                ┌───────────────────────┐
      │  Model Training &    │                │   Model Serving API    │
      │  Evaluation (offline)│ ───deploy────> │ (admission / satisfaction│
      │  + SHAP explainability│               │  / claim / risk / LOS /  │
      └──────────┬────────────┘               │  billing amount)        │
                 │                             └───────────┬─────────────┘
                 ▼                                          ▼
      ┌─────────────────────┐                ┌───────────────────────┐
      │ Monitoring & Drift   │ <──feedback──  │  HIS UI (clinician) /  │
      │ Detection, Retraining│                │  Power BI (executive)  │
      └─────────────────────┘                └───────────────────────┘
```
