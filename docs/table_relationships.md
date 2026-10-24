# Part A, Task 1 — Table Relationships

How the 27 source tables (see `references/data_dictionary.md`) relate to each other,
and the keys used to integrate them in `scripts/01_build_core_dataset.py`.

## Geographic hierarchy (patient home address)
```
Countries -> Provinces -> Districts -> Sectors -> Villages -> Patients.VillageID
```
Note: this hierarchy attaches to **patients**, not to facilities — the source data
does not include a facility-level address/geography table (see
`reports/data_quality_report.md`).

## Facility operational structure
```
Health Facilities (FacilityID, referenced — no standalone facility-master table)
  -> Departments.FacilityID
       -> Medical_Specialties.DepartmentID
       -> Healthcare_Services.DepartmentID
       -> Medical_Equipment.{DepartmentID, FacilityID}
       -> Rooms.{DepartmentID, FacilityID}
            -> Beds.RoomID
Staff tables (Doctors, Nurses, Pharmacists, Laboratory Scientists, Radiographers,
Administrative Staff) each carry FacilityID + DepartmentID directly.
Ownership_Types is a lookup table (not currently joined to a facility-master table,
for the same reason above).
```

## Clinical / financial core (the appointment-centric fact table)
```
Appointments (5,000 rows, full context: PatientID, DoctorID, FacilityID,
DepartmentID, Date, VisitType)
  -> Patients.PatientID
  -> Doctors.StaffID (DoctorID)
  -> Departments.DepartmentID
  -> Diagnoses.AppointmentID            (standalone table, 60,000-row scale)
  -> Billing.AppointmentID              (60,000-row scale)
       -> Payments.BillingID            (60,000-row scale)
       -> Insurance_Claims.BillingID    (42,036-row scale)
  -> Patient_Feedback.AppointmentID     (21,064-row scale)
  -> Laboratory_Tests.AppointmentID     (bundle sheet, 5,000-row sample)
  -> Prescriptions.AppointmentID        (bundle sheet, 5,000-row sample)
  -> Admissions.AppointmentID           (bundle sheet, 5,000-row sample)
  -> Referrals.AppointmentID            (bundle sheet, 4,708-row sample)
  -> Revenue_Summary (independent monthly rollup, not joined row-level)
```

**Key scale finding**: Diagnoses, Billing, Payments and Insurance Claims are keyed
at a 60,000-appointment scale, but only the first 5,000 of those appointments have
known Patient/Doctor/Facility/Department/Date context (via the Appointments sheet).
Laboratory_Tests, Prescriptions, Admissions and Referrals are each a *different*,
independently sampled subset of the 60,000-appointment range (coverage against the
5,000-appointment context base: ~45%, ~80%, ~12%, ~9% respectively — realistic
clinical sparsity, not every appointment gets a lab test, prescription, admission or
referral). Full quantification: `reports/data_quality_report.md`.

All relationships above were validated for referential integrity (duplicate IDs,
orphaned foreign keys, invalid Facility/Department IDs) in
`scripts/01_build_core_dataset.py` — see the Data Quality Report for results.
