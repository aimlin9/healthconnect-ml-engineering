# Reply 2 to Data Science: Changes Made, Retest and High-Risk Model Gate Results

**From:** Gyimah Ramsey Opoku (ML Engineering) · **To:** Emmanuella Esi Arhin (Data Science) · **Date:** 8 November 2026
**Evidence:** `reports/week8/evidence/` · **Decisions:** `reports/week8/decision_log.md` (D-08 to D-13)

## 1. Changes made on my side

| Agreed | Done |
|---|---|
| Reject missing values and unknown categories | Already the behaviour; kept (D-10) |
| Use your per-row status column | Every output row now has `status` (SCORED / SCORED_WITH_WARNINGS: reason / REJECTED: reason) and `model_status`. With `--skip-invalid`, rejected rows stay in the output in input order (D-09) |
| Empty input → empty output (C13) | Done: empty in, empty out, exit 0 (D-08) |
| Age limit 120, whole numbers | Already my limits; thanks for aligning yours (D-11) |
| PriorVisits point-in-time | Confirmed. It's the same definition as my `PriorAppointmentsCount` (strictly earlier visits, same-day ordered by appointment ID), so that question is closed |

Tests: **65/65** (3 updated for the agreed behaviour, 4 added). Error-handling matrix: **12/12**.

## 2. Retest with your harness

| Run | Result |
|---|---|
| Your package, unchanged, Python 3.12.10 / sklearn 1.9.1 | 15/15 |
| Your 15 tests against my pipeline (your current harness) | **12/15** (was 11/15). C13 now passes. C3, C4 and C5 fail only because your current harness still expects imputation |

Once you send the strict option and updated expectations for C3–C5, I'll re-run and we can fill in the retest row.

## 3. My High-Risk model: target, hold-out metrics and your release gate

**Target:** `HighRiskPatient` = [(Age ≥ 65) + HasChronicCondition + (SeverityScore ≥ 3) + (PriorAdmissionsCount ≥ 1)] ≥ 2.
It's a constructed rule. **Inputs:** Gender, VisitType, Age, DoctorYearsExperience,
PriorAppointmentsCount, DepartmentWorkload.

**Locked hold-out:** 25% stratified split (seed 42), n = 1,250, positive rate 42.4%. It was never
used for training or for tuning the threshold.

| Gate check | Result | Pass? |
|---|---|---|
| AUC ≥ 0.60 | 0.721 | Yes |
| Lower 95% bootstrap CI > 0.50 | 0.691 (CI 0.691–0.752, 2,000 resamples) | Yes |
| Label-shuffle p < 0.05 | p = 0.0099 (100 refits on shuffled labels; null AUC mean 0.496, max 0.619) | Yes |
| Lift > 1.3 at the flag budget | 1.77 @ 10%, **1.82 @ 20%**, 1.84 @ 30% | Yes |

**But you were right about the rule.** Age is both a model input and part of the label's rule:

| Hold-out AUC | |
|---|---|
| My model, all 6 inputs | 0.721 |
| Random forest on Age only | 0.723 |
| The single rule component `Age ≥ 65` | **0.721** |

The model passes the statistical gate only by re-deriving the age part of its own label. The other
five inputs add nothing. So it isn't usable either: it now runs with
`model_status = "DEMO ONLY - passes the statistical release gate but restates its constructed label…"`.
Following your suggestion, the demo runs both models, each with its own status label, and neither is
described as usable (D-12).

**Flag budget:** agreed. I'll report lift at 10/20/30% and leave the percentage to the clinic's
review capacity, so the 0.33 threshold is no longer the operating rule (D-13).

## Exchange record update (your §9)

| Item | Detail |
|---|---|
| Changes made as a result | ML Eng: per-row status + model_status, empty→empty, rejected rows kept in output. DS: strict option, Age 120, whole numbers (pending) |
| Partner's requirements received | Yes (your 8 Nov reply) |
| Retest result | 12/15 with the current harness; 15/15 expected once the strict option and updated C3–C5 expectations arrive |
