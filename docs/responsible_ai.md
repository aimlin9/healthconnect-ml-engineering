# Ethical and Responsible AI

## HealthConnect Rwanda Clinical Risk Prediction & Healthcare Intelligence System

All datasets used in this project are **synthetic, educational data** generated for the
AnalystLab Africa internship programme. No real patient-identifiable information is
processed. The safeguards below describe how this system *would* need to be governed if
deployed against real HealthConnect Rwanda data.

### Patient Privacy and Confidentiality

- Minimum-necessary access: clinicians see patient-level predictions only for patients
  under their active care; executives see aggregated, de-identified metrics only.
- Patient identifiers (name, phone, national ID) are excluded from model features; a
  pseudonymous `patient_id` is used for joins and never surfaced in dashboards or logs.
- Any model inputs/outputs logged for monitoring are access-controlled and retained
  under a defined policy, not indefinitely.

### Data Security

- Data in transit (API calls to/from the HIS) is encrypted (TLS); data at rest is
  encrypted in the feature store and model registry.
- Role-based access control (RBAC) separates data-engineering, model-serving and
  clinical-consumer roles; service-to-service calls are authenticated.
- Regular access audits and a documented incident-response process for any suspected
  breach.

### Bias and Fairness

- Model performance (precision/recall/AUC) must be evaluated **by subgroup** (province,
  facility type, age group, gender, insurance status), not only in aggregate, so that a
  model that works well overall but poorly for a specific province or facility type is
  caught before deployment.
- **Done in this project:** the recommended high-risk classifier was evaluated per
  facility and with leave-one-facility-out validation (ROC-AUC 0.70–0.79 on unseen
  facilities; recall is the weak point) — see `reports/model_evaluation_report.md`.
  **Not yet done:** province and facility-type slices (the source data does not link
  facilities to a district or ownership type) and age group / gender / insurance
  slices. These remain open items before any real deployment.
- Chronic-disease and risk features are checked for proxies that could encode
  socioeconomic or geographic bias (e.g. facility type standing in for ability to pay).
- Any observed fairness gap is documented and mitigated (reweighting, threshold
  adjustment per subgroup, or feature removal) before promotion to production.

### Explainability

- Every prediction surfaced to a clinician or administrator is accompanied by its top
  contributing features (SHAP values), not a bare score — supporting, not replacing,
  clinical judgment, consistent with the brief's prompt-engineering guidance.
- Global feature importance and model-comparison documentation (see
  `reports/model_evaluation_report.md`) are kept current for every deployed model.

### Human Oversight

- Model outputs are **decision-support**, not automated decisions: a clinician or
  billing officer always makes the final call (e.g. admission, claim submission).
- High-stakes flags (e.g. high-risk patient classification) route to a human review
  step rather than triggering an automated action.

### Regulatory Compliance

- Alignment with Rwanda's data protection law and Ministry of Health governance
  requirements for health data is a prerequisite for any real deployment; a Data
  Protection Impact Assessment (DPIA) would be required before processing real patient
  data.

### AI Governance

- A named model owner and a lightweight model-risk committee (clinical + technical +
  compliance representation) approve new models and material retrains before
  production promotion.
- Model cards documenting training data, intended use, known limitations and
  performance-by-subgroup are maintained per model.

### Risk Management

- See `reports/model_evaluation_report.md` for per-model strengths/limitations and
  recommended use case.
- Key risks: data drift (patient mix changes over time), label leakage (e.g. using
  post-admission data to predict admission), and over-reliance on model output by
  time-pressured clinicians. Each is mitigated via monitoring, feature-timing audits,
  and UI design that frames predictions as advisory.
