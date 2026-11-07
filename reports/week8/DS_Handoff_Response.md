# Response to Data Science Hand-off — Admission-Proxy Scorer (w7-1.0)

**From:** Gyimah Ramsey Opoku (ML Engineering) · **To:** Emmanuella Esi Arhin (Data Science)
**Date:** 7 November 2026 · **Package received:** `handoff_ml_engineering.zip`, 7 Nov 2026
**Evidence:** `reports/week8/evidence/ds_contract/` · re-run: `python scripts/week8_ds_contract_crosscheck.py <package folder>`

I've understood that the model is a validated null result and is not to be used for decisions.
Everything below tests the **interface** only.

## 1. `python hc_contract_tests.py`: 15/15 passed

| Environment | Python | scikit-learn | pandas | numpy | Result |
|---|---|---|---|---|---|
| ML Engineering pipeline env (Windows 11) | 3.12.10 | 1.9.1 | 3.1.0 | 2.5.4 | **15/15** |
| Matches your training env | 3.14.6 | 1.9.0 | 3.0.3 | 2.4.6 | **15/15** |

- I did not run `pip install -r requirements.txt`, because it would downgrade the pipeline's pinned
  scikit-learn 1.9.1. I ran the tests in the two existing environments above instead.
- Under 1.9.1 the model loads with six `InconsistentVersionWarning`s (one per estimator) plus your
  `hc_scorer` version warning. Loading works, and **scores are identical** between 1.9.0 and 1.9.1
  (max difference on `sample_input.csv` = 0.0; same 8/20 flags).
- Before loading, I checked the pickle's contents: it contains only numpy, joblib and sklearn
  classes (Pipeline, ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder, LogisticRegression).

## 2. `run_contract_tests(my_score_fn, sample_df)` with my pipeline: 11/15 passed, failing C3, C4, C5, C13

`my_score_fn` is my pipeline's `predict()` unchanged. The adapter only renames
`risk_probability` → `risk_score`, writes rejected rows as `REJECTED: <reason>`, and maps
`PriorVisits` → `PriorAppointmentsCount`.

| ID | Your contract | My pipeline | Why |
|---|---|---|---|
| C3 | Missing number → imputed and scored | Row **rejected** ("missing value") | Week 7 decision D-02: a risk score on a missing age is worse than no score |
| C4 | Missing category → scored, flagged | Row **rejected** | Same as C3 |
| C5 | Unseen category → scored with warning | Row **rejected** with the allowed list | Same as C3 |
| C13 | Empty input → empty output | Raises `InputValidationError: input has no rows` | Treated as a caller error |

The other 11 behave the same in both: single row, missing column names the column, extra columns
ignored, column order, numeric text coerced, bad rows rejected while the batch still scores,
case normalised, deterministic, duplicate index.

The two contracts differ on **soft problems**: you score them and flag them, while I reject them.
Neither approach is wrong, but the final HealthConnect interface should pick one (see §5).

## 3. Does my pipeline send the columns and types in `schema.json`?

**My model's inputs, no; my data pipeline, yes.**

- My High-Risk model uses Gender, VisitType, Age, DoctorYearsExperience, PriorAppointmentsCount
  and DepartmentWorkload. It does not use Diagnosis or HasChronic.
- My feature dataset does carry all five of your inputs. With two renames
  (`PriorAppointmentsCount` → `PriorVisits`, `HasChronicCondition` → `HasChronic`), I sent all
  **5,000 rows** through your scorer: **5,000 SCORED, 0 rejected, 0 warnings**. Your top-20%
  threshold flagged exactly **20.0%** of them, and scores ranged from 0.068 to 0.128.

| Column | Yours | Mine | Difference |
|---|---|---|---|
| Age | number, 0–110 | whole number, 0–120 | Ages 111–120 pass my check and are rejected by yours. We should agree on one limit |
| PriorVisits / PriorAppointmentsCount | number, 0–50 | whole number, 0–1000 | Name and range. **Question below** |
| HasChronic | number, 0–1 | (HasChronicCondition, not a model input) | Name only |
| Gender | Female/Male | Female/Male | Same |
| Diagnosis | 5 categories | (not a model input) | All 5 values in my data match your list |
| Numbers | floats accepted | whole numbers required | 2.5 visits passes yours and is rejected by mine |

**Question:** is your `PriorVisits` a point-in-time count (only visits *before* this one)? In
Week 7 (ISS-04) I found that a "total visits − 1" count includes future visits in 11.1% of rows.
My `PriorAppointmentsCount` is point-in-time. If yours is too, the two should match exactly.

## 4. Prediction time-point

Because your model needs **Diagnosis**, it can only run **after consultation**. I will not
connect it to a booking-time trigger while I10 is open. My own pipeline's inputs don't include
diagnosis. The final trigger point is a decision for PM and DS together.

## 5. What I'd ask Data Science to change or decide

1. **Soft-problem policy (C3/C4/C5):** keep your score-and-flag approach, or adopt my reject
   approach for the final interface? My proposal: since your model isn't being used for
   decisions, I adopt your **per-row `status` column** in my pipeline output, which is clearer than
   my separate `.rejected.csv`, and keep rejecting missing values. I'll make that change once you agree.
2. **Empty input (C13):** I'm happy to switch to your behaviour (return empty output).
3. **Shared limits:** agree on Age max (110 vs 120) and whether counts must be whole numbers.
4. **PriorVisits definition:** see the question in §3.
5. **Optional:** re-save `model.joblib` under scikit-learn 1.9.1 to remove the warnings. Scores
   don't change, so this isn't blocking.
6. **Threshold (my ISS-09):** you replaced a probability threshold with a top-20% flag budget
   because F1 tuning flagged 95%. My High-Risk model's threshold 0.33 flags 67.6% of patients.
   Would you endorse the same flag-budget rule for it?

## 6. My pipeline requirements (for your compatibility check)

- Python 3.12 or 3.13; pinned `requirements.txt` (scikit-learn **1.9.1**, pandas 3.1.0, numpy
  2.5.4, joblib 1.6.0).
- Input contract: `src/hc_pipeline/config.py` (columns, categories, valid ranges).
- The model must be a fitted sklearn estimator with `predict_proba`, saved with joblib, plus a
  metadata JSON (features, threshold, sklearn version). Your package already meets this.

## Exchange record (your §9)

| Item | Detail |
|---|---|
| Package sent | 7 Nov 2026, zip |
| Partner's environment | Python 3.12.10 / scikit-learn 1.9.1 (also 3.14.6 / 1.9.0) |
| Contract test result in partner pipeline | Your package: 15/15 in both envs. My pipeline behind your contract: 11/15, failing C3, C4, C5, C13 (policy differences, §2) |
| Issues found | Soft-problem policy; empty input; Age range; numeric types; PriorVisits definition; version warnings (harmless) |
| Changes made as a result | Pending your reply on §5 |
| Partner's requirements received | §6 above |
| Retest result | After the changes are agreed |
