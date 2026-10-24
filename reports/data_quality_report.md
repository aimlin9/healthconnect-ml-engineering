# Data Quality Report

## HealthConnect Rwanda — Week 6 ML Track

Findings from profiling, validating and integrating the 27 source datasets.

## Scale & Coverage

- **n_patients**: 20000
- **n_appointments_with_context**: 5000
- **n_diagnoses_full_scale**: 60000
- **n_billing_full_scale**: 60000
- **n_facilities**: 20
- **n_departments**: 200
- **core_rows**: 5000
- **core_cols**: 58

## Key Data-Reality Finding

The full appointment universe spans **60,000** appointments (Diagnoses, Billing, Payments and Insurance Claims are keyed at this scale). Only the first **5000** appointments carry full dimensional context (patient, doctor, facility, department, date, visit type) via the `Clinical Transactions -> Appointments` sheet — the remaining appointments cannot be used for patient-level modeling because there is no way to know who the patient, doctor or facility was. Coverage of other transactional tables against this 5,000-appointment context base:

- diagnoses with known appointment context: 8.3%
- billing with known appointment context: 8.3%
- lab tests with known appointment context: 45.4%
- prescriptions with known appointment context: 80.1%
- admissions with known appointment context: 11.7%
- referrals with known appointment context: 8.8%

**Implication**: all patient/appointment-level ML models in this project are trained on the 5,000-appointment population with full context, not the full 60,000. This is a limitation of the provided source data (not of the pipeline), and is disclosed here rather than silently worked around.

## Validation Findings

- **duplicate_patient_ids**: 0
- **duplicate_appointment_ids**: 0
- **duplicate_phone_numbers**: 19990
- **invalid_facility_ids_in_appointments**: 0
- **invalid_department_ids_in_appointments**: 0
- **impossible_ages**: 0
- **invalid_billing_amounts**: 0

**Duplicate phone numbers**: 19990 of 20000 patients share a phone number with at least one other patient — in a real system this would require manual reconciliation; in this synthetic dataset it indicates phone numbers were not generated to be unique and should not be used as a record-linkage key.

## Other Notable Findings

- Insurance claim status values present in the data: ['Approved', 'Pending'] — the brief's 3-class target (Approved/Pending/Rejected) is reduced to a 2-class problem (Approved vs Pending) since no 'Rejected' claims exist in the source data.
- Admission rate among the 5,000-appointment population: 11.7%
- Patient feedback coverage: 35.1% of appointments have a rating
- Insurance coverage rate: 100% — every one of the 20,000 patients carries an `InsuranceID`, so the brief's "insured vs. uninsured" comparisons have no uninsured group to test against in this dataset.

## Signal Check: Do Outcomes Vary With Clinical Inputs?

Before modeling, we checked whether the outcome fields (`WasAdmitted`, `Rating`,
`ClaimStatus`, `Amount`, `LengthOfStayDays`) actually vary with the clinical/
descriptive fields that would plausibly drive them:

- Admission rate by Severity: Mild 11.7%, Moderate 11.1%, Severe 12.4% — flat.
- Average satisfaction rating by Severity: 2.94 (Mild), 2.87 (Moderate), 2.97 (Severe) — flat.
- Chi-square test of admission rate by age group: p = 0.060 (not significant at α=0.05).
- ANOVA of satisfaction rating across facilities: p = 0.554 (not significant).
- t-test of billing amount, Severe vs. Mild: see `notebooks/healthconnect_ml_track.py` §3.3.

**Finding**: these outcome fields do not vary meaningfully with the clinical/
operational fields in this synthetic dataset — they behave as if generated
independently of patient/visit characteristics. This is disclosed up front because
it directly explains the near-baseline performance of the predictive models in
`reports/model_evaluation_report.md`: the pipeline is implemented correctly end to
end, but the signal a model could learn is limited by what the source data actually
encodes, not by the modeling approach. Every model below is reported against an
explicit majority-class or mean-prediction baseline so this is never obscured.
