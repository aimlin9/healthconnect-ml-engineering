"""
Part A (Data Preparation) — load, validate, clean and integrate the 27 HealthConnect
Rwanda source tables into one ML-ready, appointment-level core dataset.

Key data-reality discovered during profiling (documented in the Data Quality Report):
  - The full "appointment universe" is 60,000 appointments (Diagnoses/Billing/Payments/
    Insurance Claims are all keyed at that scale).
  - Only the first 5,000 of those appointments have full dimensional context
    (PatientID/DoctorID/FacilityID/DepartmentID/Date/VisitType), via the
    "Clinical Transactions -> Appointments" sheet. Appointments beyond that have no
    known patient/doctor/facility, so they cannot be used for patient-level ML.
  - Laboratory Tests / Prescriptions / Admissions / Referrals are each a *partial*
    sample over the full 60,000-appointment range (not every appointment gets a lab
    test, admission, etc. — clinically realistic sparsity), with coverage against the
    5,000-appointment context base of ~45%, ~80%, ~12%, ~9% respectively.

Given this, the analytical population for all patient/appointment-level ML models is
the 5,000 appointments with full context, left-joined to whichever sparse tables match.
This is documented, not hidden, in reports/data_quality_report.md.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS = ROOT / "reports"

dq = {}  # data-quality findings, written out at the end


def log(key, value):
    dq[key] = value
    print(f"[dq] {key}: {value}")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
patients = pd.read_csv(RAW / "patients.csv")
doctors = pd.read_csv(RAW / "doctors.csv")
nurses = pd.read_csv(RAW / "nurses.csv")
pharmacists = pd.read_csv(RAW / "pharmacists.csv")
lab_scientists = pd.read_csv(RAW / "laboratory_scientists.csv")
radiographers = pd.read_csv(RAW / "radiographers.csv")
admin_staff = pd.read_csv(RAW / "administrative_staff.csv")
all_staff = pd.concat(
    [doctors, nurses, pharmacists, lab_scientists, radiographers, admin_staff],
    ignore_index=True,
)

departments = pd.read_csv(RAW / "departments.csv")
medical_specialties = pd.read_csv(RAW / "medical_specialties.csv")
healthcare_services = pd.read_csv(RAW / "healthcare_services.csv")
medical_equipment = pd.read_csv(RAW / "medical_equipment.csv")
rooms = pd.read_csv(RAW / "rooms.csv")
beds = pd.read_csv(RAW / "beds.csv")
ownership_types = pd.read_csv(RAW / "ownership_types.csv")

countries = pd.read_csv(RAW / "countries.csv")
provinces = pd.read_csv(RAW / "provinces.csv")
districts = pd.read_csv(RAW / "districts.csv")
sectors = pd.read_csv(RAW / "sectors.csv")
villages = pd.read_csv(RAW / "villages.csv")

diagnoses_full = pd.read_csv(RAW / "diagnoses.csv")  # 60,000-scale
billing = pd.read_csv(RAW / "billing.csv")
payments = pd.read_csv(RAW / "payments.csv")
insurance_claims = pd.read_csv(RAW / "insurance_claims.csv")
patient_feedback = pd.read_csv(RAW / "patient_feedback.csv")
revenue_summary = pd.read_csv(RAW / "revenue_summary.csv")

xl = pd.ExcelFile(RAW / "clinical_transactions.xlsx")
appointments = xl.parse("Appointments")
lab_tests = xl.parse("Laboratory_Tests")
prescriptions = xl.parse("Prescriptions")
admissions = xl.parse("Admissions")
referrals = xl.parse("Referrals")

log("n_patients", len(patients))
log("n_appointments_with_context", len(appointments))
log("n_diagnoses_full_scale", len(diagnoses_full))
log("n_billing_full_scale", len(billing))

# ---------------------------------------------------------------------------
# 2. Validation (Task 3)
# ---------------------------------------------------------------------------
log("duplicate_patient_ids", int(patients["PatientID"].duplicated().sum()))
log("duplicate_appointment_ids", int(appointments["AppointmentID"].duplicated().sum()))
log("duplicate_phone_numbers", int(patients["Phone"].duplicated().sum()))

known_facility_ids = set(departments["FacilityID"].dropna().unique()) | set(
    all_staff["FacilityID"].dropna().unique()
)
known_department_ids = set(departments["DepartmentID"].dropna().unique())

invalid_facility_in_appts = (~appointments["FacilityID"].isin(known_facility_ids)).sum()
invalid_department_in_appts = (~appointments["DepartmentID"].isin(known_department_ids)).sum()
log("invalid_facility_ids_in_appointments", int(invalid_facility_in_appts))
log("invalid_department_ids_in_appointments", int(invalid_department_in_appts))

impossible_ages = int(((patients["Age"] < 0) | (patients["Age"] > 120)).sum())
log("impossible_ages", impossible_ages)

invalid_billing_amounts = int((billing["Amount"] <= 0).sum())
log("invalid_billing_amounts", invalid_billing_amounts)

appt_ids_full_context = set(appointments["AppointmentID"])
pct_diag_matched = diagnoses_full["AppointmentID"].isin(appt_ids_full_context).mean()
pct_billing_matched = billing["AppointmentID"].isin(appt_ids_full_context).mean()
log("pct_diagnoses_with_known_appointment_context", round(float(pct_diag_matched) * 100, 1))
log("pct_billing_with_known_appointment_context", round(float(pct_billing_matched) * 100, 1))
for name, df in [("lab_tests", lab_tests), ("prescriptions", prescriptions),
                  ("admissions", admissions), ("referrals", referrals)]:
    pct = df["AppointmentID"].isin(appt_ids_full_context).mean()
    log(f"pct_{name}_with_known_appointment_context", round(float(pct) * 100, 1))

# ---------------------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------------------
patients = patients.copy()
patients["DateOfBirth"] = pd.to_datetime(patients["DateOfBirth"], errors="coerce")
patients["ChronicCondition"] = patients["ChronicCondition"].fillna("None")
patients["HasChronicCondition"] = (patients["ChronicCondition"] != "None").astype(int)

bins = [0, 17, 35, 50, 65, 200]
labels = ["0-17", "18-35", "36-50", "51-65", "65+"]
patients["AgeGroup"] = pd.cut(patients["Age"], bins=bins, labels=labels, right=True)

appointments = appointments.copy()
appointments["Date"] = pd.to_datetime(appointments["Date"], errors="coerce")

billing_clean = billing.copy()
billing_clean["Status"] = billing_clean["Status"].str.strip().str.title()

insurance_claims_clean = insurance_claims.copy()
insurance_claims_clean["Status"] = insurance_claims_clean["Status"].str.strip().str.title()
log("insurance_claim_status_values", sorted(insurance_claims_clean["Status"].unique().tolist()))

# ---------------------------------------------------------------------------
# 4. Integration -> one row per appointment (the 5,000 with full context)
# ---------------------------------------------------------------------------
core = appointments.merge(
    patients, on="PatientID", how="left", suffixes=("", "_patient")
)
core = core.merge(
    doctors[["StaffID", "FirstName", "LastName", "Specialty", "YearsExperience"]]
    .rename(columns={"StaffID": "DoctorID", "FirstName": "DoctorFirstName",
                      "LastName": "DoctorLastName", "Specialty": "DoctorSpecialty",
                      "YearsExperience": "DoctorYearsExperience"}),
    on="DoctorID", how="left",
)
core = core.merge(
    departments.rename(columns={"FacilityID": "DeptFacilityID"}),
    on="DepartmentID", how="left",
)

diag_slim = diagnoses_full.drop_duplicates("AppointmentID")[
    ["AppointmentID", "Diagnosis", "Severity"]
]
core = core.merge(diag_slim, on="AppointmentID", how="left")

core = core.merge(
    billing_clean.rename(columns={"Status": "BillingStatus"}),
    on="AppointmentID", how="left",
)
core = core.merge(
    payments.groupby("BillingID", as_index=False).agg(
        AmountPaid=("AmountPaid", "sum"), PaymentMethod=("Method", "first")
    ),
    on="BillingID", how="left",
)
core = core.merge(
    insurance_claims_clean.rename(columns={"Status": "ClaimStatus"}),
    on="BillingID", how="left",
)
core = core.merge(
    patient_feedback[["AppointmentID", "Rating", "Comment"]], on="AppointmentID", how="left"
)
core = core.merge(
    lab_tests[["AppointmentID", "Test", "Result"]].rename(
        columns={"Test": "LabTest", "Result": "LabResult"}
    ),
    on="AppointmentID", how="left",
)
core = core.merge(
    prescriptions[["AppointmentID", "Medication", "Dosage"]], on="AppointmentID", how="left"
)
core = core.merge(
    admissions[["AppointmentID", "LengthOfStayDays"]], on="AppointmentID", how="left"
)
core = core.merge(
    referrals[["AppointmentID", "ReferredFacilityID"]], on="AppointmentID", how="left"
)

core["WasAdmitted"] = core["LengthOfStayDays"].notna().astype(int)
core["WasReferred"] = core["ReferredFacilityID"].notna().astype(int)
core["HadLabTest"] = core["LabTest"].notna().astype(int)
core["HadPrescription"] = core["Medication"].notna().astype(int)
core["GaveFeedback"] = core["Rating"].notna().astype(int)
core["HasInsuranceClaim"] = core["ClaimStatus"].notna().astype(int)

log("core_rows", len(core))
log("core_cols", len(core.columns))
log("admission_rate_pct", round(float(core["WasAdmitted"].mean()) * 100, 1))
log("feedback_coverage_pct", round(float(core["GaveFeedback"].mean()) * 100, 1))

core.to_csv(PROCESSED / "core_appointments.csv", index=False)
patients.to_csv(PROCESSED / "patients_clean.csv", index=False)

# Facility / department-level operational aggregates (used for clustering + KPIs)
dept_agg = (
    core.groupby("DepartmentID")
    .agg(
        appointment_count=("AppointmentID", "count"),
        admission_rate=("WasAdmitted", "mean"),
        avg_satisfaction=("Rating", "mean"),
        avg_billing=("Amount", "mean"),
    )
    .reset_index()
)
dept_agg.to_csv(PROCESSED / "department_aggregates.csv", index=False)

facility_agg = (
    core.groupby("DeptFacilityID")
    .agg(
        appointment_count=("AppointmentID", "count"),
        admission_rate=("WasAdmitted", "mean"),
        avg_satisfaction=("Rating", "mean"),
        total_revenue=("Amount", "sum"),
        unique_doctors=("DoctorID", "nunique"),
    )
    .reset_index()
    .rename(columns={"DeptFacilityID": "FacilityID"})
)
staff_per_facility = all_staff.groupby("FacilityID").size().rename("staff_count")
facility_agg = facility_agg.merge(staff_per_facility, on="FacilityID", how="left")
beds_per_facility = (
    rooms.merge(beds, on="RoomID", how="left")
    .groupby("FacilityID")
    .size()
    .rename("bed_count")
)
facility_agg = facility_agg.merge(beds_per_facility, on="FacilityID", how="left")
facility_agg.to_csv(PROCESSED / "facility_aggregates.csv", index=False)

log("n_facilities", int(facility_agg["FacilityID"].nunique()))
log("n_departments", int(dept_agg["DepartmentID"].nunique()))

# ---------------------------------------------------------------------------
# 5. Write Data Quality Report
# ---------------------------------------------------------------------------
REPORTS.mkdir(exist_ok=True)
(REPORTS / "data_quality_findings.json").write_text(json.dumps(dq, indent=2))

md = ["# Data Quality Report\n",
      "## HealthConnect Rwanda — Week 6 ML Track\n",
      "Findings from profiling, validating and integrating the 27 source datasets.\n",
      "## Scale & Coverage\n"]
for k in ["n_patients", "n_appointments_with_context", "n_diagnoses_full_scale",
          "n_billing_full_scale", "n_facilities", "n_departments", "core_rows", "core_cols"]:
    md.append(f"- **{k}**: {dq[k]}")

md.append("\n## Key Data-Reality Finding\n")
md.append(
    "The full appointment universe spans **60,000** appointments (Diagnoses, Billing, "
    "Payments and Insurance Claims are keyed at this scale). Only the first "
    f"**{dq['n_appointments_with_context']}** appointments carry full dimensional "
    "context (patient, doctor, facility, department, date, visit type) via the "
    "`Clinical Transactions -> Appointments` sheet — the remaining appointments cannot "
    "be used for patient-level modeling because there is no way to know who the "
    "patient, doctor or facility was. Coverage of other transactional tables against "
    "this 5,000-appointment context base:\n"
)
for k in ["pct_diagnoses_with_known_appointment_context", "pct_billing_with_known_appointment_context",
          "pct_lab_tests_with_known_appointment_context", "pct_prescriptions_with_known_appointment_context",
          "pct_admissions_with_known_appointment_context", "pct_referrals_with_known_appointment_context"]:
    md.append(f"- {k.replace('pct_', '').replace('_', ' ')}: {dq[k]}%")

md.append(
    "\n**Implication**: all patient/appointment-level ML models in this project are "
    "trained on the 5,000-appointment population with full context, not the full "
    "60,000. This is a limitation of the provided source data (not of the pipeline), "
    "and is disclosed here rather than silently worked around."
)

md.append("\n## Validation Findings\n")
for k in ["duplicate_patient_ids", "duplicate_appointment_ids", "duplicate_phone_numbers",
          "invalid_facility_ids_in_appointments", "invalid_department_ids_in_appointments",
          "impossible_ages", "invalid_billing_amounts"]:
    md.append(f"- **{k}**: {dq[k]}")

md.append(
    f"\n**Duplicate phone numbers**: {dq['duplicate_phone_numbers']} of {dq['n_patients']} "
    "patients share a phone number with at least one other patient — in a real system "
    "this would require manual reconciliation; in this synthetic dataset it indicates "
    "phone numbers were not generated to be unique and should not be used as a "
    "record-linkage key."
)

md.append("\n## Other Notable Findings\n")
md.append(f"- Insurance claim status values present in the data: {dq['insurance_claim_status_values']} "
           "— the brief's 3-class target (Approved/Pending/Rejected) is reduced to a 2-class "
           "problem (Approved vs Pending) since no 'Rejected' claims exist in the source data.")
md.append(f"- Admission rate among the 5,000-appointment population: {dq['admission_rate_pct']}%")
md.append(f"- Patient feedback coverage: {dq['feedback_coverage_pct']}% of appointments have a rating")

(REPORTS / "data_quality_report.md").write_text("\n".join(md) + "\n")
print("\nWrote reports/data_quality_report.md and reports/data_quality_findings.json")
print("Wrote data/processed/{core_appointments,patients_clean,department_aggregates,facility_aggregates}.csv")
