# Reply 3 to Data Science: Retest with Interface v1.1

**From:** Gyimah Ramsey Opoku (ML Engineering) · **To:** Emmanuella Esi Arhin (Data Science) · **Date:** 8 November 2026
**Evidence:** `reports/week8/evidence/ds_contract/` · **Run:** `python scripts/week8_ds_contract_crosscheck.py <package dir>`

Environment: Windows 11, Python 3.12.10, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4.
Package v1.1 reviewed before running; `model.joblib` and `sample_input.csv` are byte-identical to v1.0.

## Results

| Run | Result |
|---|---|
| Your package v1.1, unchanged (`policy: strict`) | **15/15** |
| `run_contract_tests(my_score_fn, sample, strict=True)` against my pipeline | **15/15** (was 12/15 with v1.0) |
| My data pipeline (5,000 rows) into your v1.1 scorer | 5,000 scored, 0 rejected, top-20% budget flags 20.0% |

## Per-test result, my pipeline

| ID | Scenario | Result | My pipeline's row status |
|---|---|---|---|
| C1 | Valid batch of 5 | PASS | 5 scored, 0.318–0.848 |
| C2 | Single row | PASS | 1 scored |
| C3 | Missing numeric values | PASS | `REJECTED: Age: missing value` / `REJECTED: PriorAppointmentsCount: missing value`; rows 3–5 scored |
| C4 | Missing categorical value | PASS | `REJECTED: Gender: missing value`; rows 2–5 scored |
| C5 | Unseen category 'Zanzibar' | PASS | `REJECTED: Gender: unknown category 'Zanzibar'; allowed ['Female', 'Male']`; rows 2–5 scored |
| C6 | Required column missing | PASS | `InputValidationError: row all, Age: required column missing` |
| C7 | Extra column | PASS | ignored, scores unchanged |
| C8 | Column order | PASS | same scores |
| C9 | Numbers as text | PASS | coerced, same scores |
| C10 | 'abc' in Age | PASS | row 1 rejected, rows 2–5 scored |
| C11 | Age 250, PriorVisits −3, Age 2.5 | PASS | `Age: 250.0 outside valid range [0, 120]`; `PriorAppointmentsCount: -3.0 outside valid range [0, 1000]`; `Age: 2.5 is not a whole number`; rows 4–5 scored |
| C12 | Text case | PASS | normalised, same scores |
| C13 | Empty input | PASS | empty output |
| C14 | Repeat call | PASS | identical |
| C15 | Duplicate index | PASS | scored |

No test needs a different expectation for my pipeline. The Age 2.5 row is rejected by the same
whole-number rule as yours, only the wording differs ("2.5 is not a whole number"). My sample has
no Diagnosis column, so your harness uses Gender for C4, C5 and C12, as designed.

## Exchange record (your §9)

| Item | Detail |
|---|---|
| Changes made as a result | ML Eng: per-row `status` + `model_status`, empty→empty, rejected rows kept in output. DS: strict policy (default), Age 0–120, whole numbers, allowed values in reasons |
| Retest result | **15/15** against my pipeline with the v1.1 strict harness; your package 15/15 in my environment |
| Open items | None on the interface. Both models stay labelled demo only (D-12) |
