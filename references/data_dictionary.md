# Data Dictionary — Source Tables

27 source tables from the AnalystLab Africa Week 6 HealthConnect Rwanda brief. See
`data/dataset_manifest.json` for the download source per table and
`data/raw/_download_results.json` for the resolved local file per table once downloaded.

| # | Table | Description |
|---|-------|-------------|
| 1 | Countries | Country master information |
| 2 | Provinces | Province information |
| 3 | Districts | District information |
| 4 | Sectors | Sector information |
| 5 | Villages | Village information |
| 6 | Ownership Types | Government, Private, Faith-Based, NGO, Military, etc. |
| 7 | Health Facilities | Healthcare facility master information (multi-tab workbook) |
| 8 | Departments | Clinical and administrative departments |
| 9 | Medical Specialties | Doctor specialization information |
| 10 | Healthcare Services | Medical and healthcare services offered |
| 11 | Medical Equipment | Medical equipment inventory |
| 12 | Rooms | Hospital rooms and wards |
| 13 | Beds | Bed inventory and allocation |
| 14 | Doctors | Doctor master information |
| 15 | Nurses | Nurse master information |
| 16 | Pharmacists | Pharmacist master information |
| 17 | Laboratory Scientists | Laboratory scientist information |
| 18 | Radiographers | Radiographer information |
| 19 | Administrative Staff | Hospital administrative staff |
| 20 | Patients | Patient master records |
| 21 | Diagnoses | Patient diagnosis records |
| 22 | Clinical Transactions | Multi-tab workbook: Appointments, Diagnoses, Laboratory Tests, Prescriptions, Admissions, Referrals |
| 23 | Billing | Billing and invoicing records |
| 24 | Insurance Claims | Insurance claim transactions |
| 25 | Payments | Patient payment transactions |
| 26 | Patient Feedback | Patient satisfaction and feedback records |
| 27 | Revenue Summary | Monthly and facility revenue summary |

Column-level schema for each table is appended after profiling (see
`notebooks/01_data_profiling.ipynb` / Data Quality Report in `reports/`).
