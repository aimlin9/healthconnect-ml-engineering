# HealthConnect ML Pipeline Testing & Reliability Package

**AnalystLab Africa Experience Lab — Week 7, Batch E · Machine Learning Engineering Track**
**Component under test:** High-Risk Patient Classification pipeline (Week 6 recommended model)

Every number below comes from a script in this repository and the evidence files it writes to
`reports/week7/evidence/`. To reproduce everything: see [Reproducing this report](#reproducing-this-report).

---

## 1. Week 6 → Week 7 Transition

| | |
|---|---|
| **Main Week 6 output** | End-to-end ML notebook over 27 HealthConnect datasets; 6 models benchmarked against explicit baselines |
| **Most important result** | High-Risk Patient classifier — the only model with real lift (accuracy 0.746 vs. 0.576 baseline, ROC-AUC 0.727); held up across all 20 facilities (leave-one-facility-out AUC 0.70–0.79) |
| **Main limitation / dependency** | The model existed only inside the notebook: no saved artifact, no inference entry point, no input checks. Recall was weak (0.555). Deployment architecture (`docs/deployment_architecture.md`) assumed a service that could score patients. |
| **Component needing testing** | The inference path: data → features → preprocessing → model → risk flag |
| **Relevant tracks** | Data Science (model, target, threshold), Project Management (issue/decision tracking), Data Analytics (validated findings behind the features) |
| **What I intended to test** | Whether the Week 6 pipeline can score real inputs, reject bad inputs, reproduce its results, and avoid using information unavailable at prediction time |
| **What I expected to improve or validate** | A loadable, validated, logged, reproducible pipeline that passes the same probes the Week 6 version failed — without losing predictive performance |

## 2. Testing Readiness (Part 1)

**A. Component.** The High-Risk Patient pipeline. Purpose: flag patients likely to need proactive
follow-up so HealthConnect can prioritise reminders and support (supports the project goal of
reducing missed appointments and improving patient support). Requirement: score a patient or a
batch reliably and safely, with output a downstream system (dashboard, reminder workflow, assistant
escalation) can consume.

**B. Testing objective.** Does the pipeline correctly process inputs and produce the expected
output — for valid, unusual and invalid inputs — and are its results reproducible from the
documented setup?

**C/D. Method and evidence.**

| Stage | What | Evidence |
|---|---|---|
| 1. Baseline audit | 14 probes run against the Week 6 inference code, reproduced verbatim from notebook §4.4 | `scripts/week7_audit_week6_pipeline.py` → `evidence/week6_audit_results.json` |
| 2. Refinement | New `src/hc_pipeline` package + training script + CLI | `src/hc_pipeline/`, `scripts/03_train_pipeline.py`, `scripts/predict.py` |
| 3. Retest | Same 14 probes against the refined pipeline | `scripts/week7_retest_probes.py` → `evidence/week7_retest_results.json` |
| 4. Automated test suite | 61 pytest tests (data processing, validation, model integration, CLI) | `tests/` → `evidence/pytest_output.txt`, `evidence/pytest_results.xml` |
| 5. Test-suite quality | Mutation testing: inject 5 realistic bugs, confirm tests catch each | `scripts/week7_mutation_check.py` → `evidence/mutation_check_results.json` |
| 6. Model re-evaluation | Week 6 vs. refined model, same test split + leave-one-facility-out | `evidence/model_refinement_results.json`, `evidence/threshold_operating_points.json` |
| 7. Reproducibility | Fresh `git clone` + fresh venv from `requirements.txt` → train → test | Section 8 |
| 8. End-to-end demo | CSV in → scored CSV + rejects file + log | `evidence/cli_demo/` |

## 3. Baseline Audit — Week 6 Pipeline (Before)

**Result: 2 / 14 passed.**

| ID | Area | Scenario | Expected | Week 6 actual | Result |
|---|---|---|---|---|---|
| W6-01 | Persistence | A trained model artifact exists | Serialized model + metadata | None — model lived only in notebook memory | FAIL |
| W6-02 | Integration | Score a large mixed batch | One probability per row | 1,250 probabilities | PASS |
| W6-03 | Preprocessing | **Score a single patient** | One probability | `ValueError: feature names should match…` | FAIL |
| W6-04 | Preprocessing | Score a batch of only Emergency visits | One probability per row | Same `ValueError` | FAIL |
| W6-05 | Validation | Unseen category (`Telehealth`) | Clear rejection | Same `ValueError` (misleading message) | FAIL |
| W6-06 | Validation | Required column missing | Error naming the column | Raw `KeyError` | FAIL |
| W6-07 | Validation | Age supplied as text (`"45"`) | Coerce or clear type error | Same `ValueError` | FAIL |
| W6-08 | Validation | Age = −5 and 250 | Rejected | **Silently scored** (0.348, 0.764) | FAIL |
| W6-09 | Validation | Age missing (NaN) | Rejected | **Silently scored** (0.325, 0.279) | FAIL |
| W6-10 | Validation | Empty input | Clear "no rows" error | Same `ValueError` | FAIL |
| W6-11 | Reproducibility | Train twice, same seed | Identical predictions | Identical | PASS |
| W6-12 | Data processing | All features informative | No constant features | `HasInsurance` = 1 in all 5,000 rows | FAIL |
| W6-13 | Data processing | Features use only past information | Point-in-time counts | `PreviousAppointmentsCount` counts **future** visits in 11.1% of rows | FAIL |
| W6-14 | Observability | Inference is logged | Run info + errors logged | `print()` only | FAIL |

**Root causes**

- **W6-03/04/05/07/10 — one bug.** Week 6 re-ran `pd.get_dummies(drop_first=True)` on whatever
  rows arrived. Dummy columns depend on which categories are present in *that batch*, so any input
  that isn't a large mixed batch produces columns the model was never trained on. The notebook only
  ever scored its own 1,250-row test split, so this never surfaced. A real-time service scores one
  patient at a time — the core use case was broken.
- **W6-08/09 — no input contract.** Nothing defined what valid input looks like; scikit-learn's
  Random Forest accepts NaN and any number, so impossible values produced confident-looking scores.
- **W6-13 — training/serving skew.** `PreviousAppointmentsCount` was computed as *total visits − 1*
  across the whole dataset, so a patient's first appointment "knew" about their later ones. A live
  system only knows past visits.
- **W6-12 — constant feature.** `HasInsurance` never varies in this dataset, so it carried no
  information but was part of the model's required input.

## 4. Refinements Made

| Issue | Refinement | Where |
|---|---|---|
| No artifact (W6-01) | Model saved with joblib alongside metadata: version, scikit-learn version, feature list, category list, training ranges, threshold, training-data fingerprint, test metrics. `load()` refuses an artifact whose feature list doesn't match the code and warns on a scikit-learn version mismatch. | `pipeline.py: save/load`, `models/high_risk_pipeline*.{joblib,json}` |
| Batch-dependent encoding (W6-03/04/05/07/10) | Preprocessing moved **inside** a scikit-learn `Pipeline`: `OneHotEncoder` with a fixed category list from `config.py`. Encoding no longer depends on the batch. | `pipeline.py: build_pipeline` |
| No input contract (W6-05–10) | `validate_input()`: required columns, category whitelist (case/whitespace-insensitive), numeric coercion, valid ranges, whole numbers, missing values, empty input. Reports **every** problem with row and column in one pass. Two modes: `raise` (default) or `skip` (score valid rows, return rejects). | `validation.py`, `config.py` |
| Future-visit leakage (W6-13) | `PriorAppointmentsCount` = appointments strictly before this one, ordered by date then ID. | `features.py` |
| Constant feature (W6-12) | `HasInsurance` dropped from the model input. | `config.py: FEATURES` |
| No logging (W6-14) | `hc_pipeline` logger: model loaded, rows read, rows rejected (with reasons), values outside the training range (extrapolation warning), rows scored/flagged, latency. CLI `--log-file`. | `pipeline.py`, `scripts/predict.py` |
| Weak recall (Week 6 known issue) | Decision threshold tuned on a validation split carved from training data (never the test set) to reach ≥ 0.80 recall; stored in metadata; overridable with `--threshold`. **See §6 — this trade-off is not settled.** | `pipeline.py: tune_threshold/train` |
| No deployable entry point | CLI: CSV in → scored CSV out, rejects file, exit codes (0 ok, 2 invalid input, 3 model missing). | `scripts/predict.py` |

## 5. Retest — Refined Pipeline (After)

**Result: 14 / 14 passed** (`evidence/week7_retest_results.json`).

| ID | Week 6 | Week 7 | Week 7 actual |
|---|---|---|---|
| W6-01 | FAIL | PASS | Loaded `models/high_risk_pipeline.joblib` v1.1.0-week7 |
| W6-02 | PASS | PASS | 1,250 probabilities |
| W6-03 | FAIL | PASS | 1 probability for a single patient |
| W6-04 | FAIL | PASS | 20 probabilities for an all-Emergency batch |
| W6-05 | FAIL | PASS | Rejected: `unknown category 'Telehealth'; allowed ['Consultation', 'Emergency', 'Follow-up']` |
| W6-06 | FAIL | PASS | Rejected: `DepartmentWorkload: required column missing` |
| W6-07 | FAIL | PASS | `"45"` coerced to 45 and scored |
| W6-08 | FAIL | PASS | Rejected: `Age: -5 outside valid range [0, 120]`, `Age: 250 outside valid range` |
| W6-09 | FAIL | PASS | Rejected: `Age: missing value` |
| W6-10 | FAIL | PASS | Rejected: `input has no rows` |
| W6-11 | PASS | PASS | Identical probabilities across two training runs |
| W6-12 | FAIL | PASS | No constant features |
| W6-13 | FAIL | PASS | 0.0% of rows use future visits |
| W6-14 | FAIL | PASS | Structured logging throughout |

### 5.1 Automated test suite — 61 / 61 passed

| File | Tests | Covers (brief task #) |
|---|---|---|
| `test_data_processing.py` | 11 | Feature file schema, unique IDs, no missing inputs, categories and ranges match the contract, target reproduces its documented rule, no constant features, point-in-time feature logic incl. same-day visits, no input mutation (#4, #5) |
| `test_validation.py` | 26 | Valid rows pass unchanged; dict input; extra columns ignored; column order irrelevant; case/whitespace normalisation; numeric strings coerced; non-tabular, empty, missing-column inputs; 10 parametrised invalid values each rejected with row + reason; all problems reported at once; skip mode (#7, #9, #13) |
| `test_model_integration.py` | 16 | Single-patient and single-category scoring; **row score identical whether scored alone or in a batch**; output contract; invalid input never reaches the model; extrapolation and scoring logged; held-out AUC ≥ 0.70; deterministic training; saved artifact == fresh retrain; save/load round-trip; clear errors for missing or mismatched artifacts (#6, #8, #14, #16, #19) |
| `test_cli.py` | 8 | End-to-end CSV scoring; exit code 2 + no output on invalid input; `--skip-invalid` rejects file; CSV line numbers in errors; normalised values in output; missing file; threshold override; log file (#13, #14, #16) |

### 5.2 Does the test suite actually catch bugs? — Mutation testing, 5 / 5 caught

A suite that always passes proves nothing on its own. Each mutation re-introduces a realistic
defect into a temporary copy of the package; a control run of the unmutated copy must pass first.

| ID | Injected defect | Caught by |
|---|---|---|
| M1 | Range check removed | 8 tests (validation, skip mode) |
| M2 | Unknown categories passed through | 4 tests (incl. `test_invalid_input_never_reaches_model`) |
| M3 | Point-in-time feature ignores visit date | `test_prior_appointments_counts_only_earlier_visits`, `test_saved_artifact_reproduces_fresh_training` |
| M4 | Model seed removed | `test_training_is_deterministic`, `test_saved_artifact_reproduces_fresh_training` |
| M5 | Flag ignores stored threshold | `test_output_contract` |

**Two problems found in my own testing, and fixed:**

1. *The first mutation run reported 5/5 caught — falsely.* The temporary copy couldn't find the
   data, so every run errored before testing anything. Fixed by mirroring the repo layout and
   adding the control run plus a rule that setup errors don't count as catches.
2. *M3 was initially caught only indirectly.* The unit test written for it used appointment IDs
   whose order matched the dates, so sorting by ID alone also passed. Fixed the fixture so ID
   order and date order disagree; M3 is now caught by its dedicated test.

### 5.3 End-to-end demo (`evidence/cli_demo/`)

8 sample appointments, with an impossible age, an unknown visit type and a messy `" female "` planted:

```
$ python scripts/predict.py sample_input.csv scored.csv
ERROR hc_pipeline: CSV line 7, VisitType: unknown category 'Telehealth'; allowed [...]
ERROR hc_pipeline: CSV line 4, Age: 250 outside valid range [0, 120]
ERROR hc_pipeline: no output written (2 problem(s)); fix the input or use --skip-invalid
[exit code 2]

$ python scripts/predict.py sample_input.csv scored.csv --skip-invalid --log-file run.log
INFO  hc_pipeline: loaded model 1.1.0-week7 (threshold 0.33)
WARNING hc_pipeline: rejected 2 of 8 row(s) failing validation
INFO  hc_pipeline: scored 6 row(s), flagged 4 high-risk, rejected 2, 84.0 ms
[exit code 0]
```

The demo itself surfaced two more issues, both fixed and covered by new tests: errors quoted
0-based DataFrame rows instead of CSV line numbers, and the output file echoed raw input
(`" female "`) rather than the normalised value the model scored.

## 6. Model Integration Testing — Did the Refinements Cost Performance?

Same held-out test split as Week 6 (n = 1,250, 42.4% high-risk).

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Week 6 (get_dummies, original features) | 0.50 | 0.746 | 0.782 | 0.555 | 0.649 | 0.727 |
| Refined (feature fixes only) | 0.50 | 0.745 | 0.781 | 0.553 | 0.648 | 0.721 |
| Refined (feature fixes + tuned threshold) | 0.33 | 0.566 | 0.492 | **0.785** | 0.605 | 0.721 |
| Majority-class baseline | — | 0.576 | — | — | — | — |

Leave-one-facility-out (train on 19 facilities, test on the 20th; threshold re-tuned inside each fold):

| | Recall (min / median / max) | Precision | ROC-AUC |
|---|---|---|---|
| Week 6 @ 0.50 | 0.477 / 0.581 / 0.645 | 0.744 / 0.806 / 0.884 | 0.697 / 0.752 / 0.790 |
| Refined @ tuned (0.32–0.34) | 0.730 / **0.817** / 0.952 | 0.392 / 0.476 / 0.566 | 0.675 / 0.731 / 0.781 |

**Interpretation**

- **Removing the leaked feature and the constant feature cost almost nothing** (AUC 0.727 → 0.721).
  The Week 6 result was not driven by leakage — that is itself a validation of the Week 6 model.
- **Threshold tuning buys recall at a steep price.** Recall rises to 0.785 (0.817 median on unseen
  facilities), but precision falls to 0.49 and accuracy drops *below* the majority baseline. At
  0.33 the model flags **67.6%** of patients when 42.4% are actually high-risk.
- **The threshold is fragile.** Model probabilities are bunched around 0.30–0.36:

| Threshold | Recall | Precision | Accuracy | % of patients flagged |
|---|---|---|---|---|
| 0.30 | 0.960 | 0.440 | 0.464 | 92.6 |
| **0.33** (shipped) | 0.785 | 0.492 | 0.566 | 67.6 |
| 0.36 | 0.638 | 0.661 | 0.708 | 40.9 |
| 0.40 | 0.577 | 0.739 | 0.734 | 33.1 |
| 0.50 | 0.553 | 0.781 | 0.745 | 30.0 |

  (Computed on the test split for information only; the shipped threshold was chosen on validation data.)

**Engineering position:** the pipeline ships with the threshold produced by the documented rule
(0.33, recall ≥ 0.80) because a missed high-risk patient is the costlier error for a support
tool — but **the operating point is a Data Science / clinical decision, not an engineering one**.
It is stored in metadata and overridable per run, so changing it needs no code change. Logged as
decision D-03 and open issue ISS-09.

## 7. Configuration & Dependency Management

- **Single source of truth:** `src/hc_pipeline/config.py` — feature list, category whitelist,
  valid ranges, model hyperparameters, split seed, recall target, artifact paths, model version.
- **Pinned dependencies:** `requirements.txt` (full freeze; Week 7 added `pytest`, `pluggy`,
  `iniconfig`). Key pins: Python 3.13, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, joblib 1.6.0.
- **Artifact versioning:** metadata records `model_version`, `sklearn_version` and a
  `train_data_fingerprint` (hash of the training rows), so an artifact can be traced to the code
  and data that produced it.
- **Artifact in git:** `models/high_risk_pipeline.joblib` (2.5 MB) is committed (a `.gitignore`
  exception) so Week 8 integration can load it without retraining; it is fully reproducible (§8).

## 8. Reproducibility Check

Fresh `git clone` into an empty directory → `python3 -m venv .venv` →
`pip install -r requirements.txt` → `python scripts/03_train_pipeline.py` → `pytest`:

| Step | Result |
|---|---|
| Dependency install | Succeeded (exit 0) |
| Training | Succeeded (exit 0) |
| Test suite | All passed |
| Retrained metadata vs. committed | Identical threshold, metrics, feature list, training ranges and **training-data fingerprint**; only `trained_at` differs |

## 9. Testing, Refinement & Validation Record

| Component | Objective | Test / Scenario | Expected | Actual (before) | P/F | Issue | Action | Retest | Collab. track | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| Inference | Real-time scoring works | Score 1 patient | 1 probability | ValueError | F | Batch-dependent get_dummies | Encoder inside Pipeline, fixed categories | PASS | DS | W6-03, `test_single_patient_scored` |
| Inference | Batch-independent | Same rows alone vs. in batch | Identical scores | n/a (couldn't score) | F | As above | As above | PASS | — | `test_row_scores_independent_of_batch` |
| Validation | Bad input rejected | Age 250 / −5 / NaN | Rejected with reason | Silently scored | F | No input contract | `validate_input()` | PASS | DA (valid ranges) | W6-08/09, `test_validation.py` |
| Validation | Bad input rejected | Unknown category | Rejected, allowed values listed | Misleading error | F | No whitelist | Category whitelist | PASS | — | W6-05 |
| Validation | Missing column | Drop a column | Error names column | Raw KeyError | F | No contract | Column check | PASS | — | W6-06 |
| Features | No future info | Point-in-time check | 0% leakage | 11.1% rows leak | F | total−1 count | `add_prior_appointments` | PASS | DS | W6-13 |
| Features | Informative inputs | Constant-feature check | None | `HasInsurance` | F | No variance | Dropped | PASS | DS | W6-12 |
| Persistence | Loadable model | Load artifact | Model + metadata | None existed | F | Notebook only | `save/load` + metadata checks | PASS | — | W6-01 |
| Persistence | Safe loading | Mismatched feature list | Refuse to load | n/a | — | — | Feature-list check | PASS | — | `test_feature_mismatch_refuses_to_load` |
| Observability | Runs are traceable | Inspect log | Load/reject/score logged | print() only | F | No logging | Logger + `--log-file` | PASS | PM | W6-14, `cli_demo/run.log` |
| Model | Performance kept | Same test split | AUC ≈ Week 6 | 0.727 | — | — | Feature fixes | 0.721 | DS | `model_refinement_results.json` |
| Model | Recall | Tuned threshold | Recall ≥ 0.80 target | 0.555 | F | Misses high-risk patients | Validation-tuned threshold | 0.785 test / 0.817 LOFO median; precision 0.49 | DS | §6 |
| Reproducibility | Same result twice | Retrain | Identical | Identical | P | — | — | PASS | — | W6-11 |
| Reproducibility | Documented setup works | Fresh clone + venv | Tests pass, same model | n/a | — | — | — | PASS, identical fingerprint | — | §8 |
| Test suite | Tests catch bugs | 5 mutations | All caught | False 5/5 (harness bug) | F | Harness couldn't find data | Mirror layout + control run | 5/5 genuine | — | `mutation_check_results.json` |
| CLI | Usable errors | Invalid CSV row | CSV line number | 0-based index | F | Off-by-two | `csv_line()` | PASS | — | `test_cli_reports_csv_line_numbers` |
| CLI | Output matches scoring | `" female "` input | `Female` in output | Raw value echoed | F | Raw passthrough | Write normalised values | PASS | — | `test_cli_output_uses_normalised_values` |

Full issue log with owners and status: `reports/week7/issue_log.csv`.

## 10. HC-POD Cross-Track Testing

**Status at submission: requested, no response yet — cross-track test not completed.**
I asked the Data Science track to collaborate on testing the pipeline. No feedback had arrived
by the Week 7 deadline, so the joint Test → Finding → Action → Retest cycle has not happened. The
ML Engineering side is ready, and the steps below will be completed and recorded as soon as Data
Science responds. Nothing in this section claims a collaboration that has not taken place.

| # | Field | Data Science ↔ ML Engineering (primary) |
|---|---|---|
| 1 | Track collaborated with | Data Science — collaboration requested; awaiting response |
| 2 | Project dependency | The pipeline serves the Data Science candidate model; DS owns the target definition and operating threshold |
| 3 | Component to be tested | High-Risk Patient pipeline: input contract, feature definitions, threshold |
| 4 | Received | Nothing yet |
| 5 | Ready to provide | Input contract (`config.py`), model artifact + metadata, threshold operating-point table (§6), findings ISS-04 (feature leakage), ISS-05 (constant feature), ISS-08 (label leakage), ISS-09 (threshold) |
| 6 | Planned testing activity | DS exports 20–50 of their validation rows to CSV → `python scripts/predict.py rows.csv out.csv` → compare `risk_probability` with the DS notebook; any mismatch is a contract issue to fix and retest. DS chooses the operating threshold from the §6 table (closes ISS-09) and decides on ISS-08. |
| 7–12 | Finding, action, retest, what changed, evidence, impact | Pending the DS response; will be added to this report and the issue log |

Because no DS feedback arrived, the threshold (ISS-09) and label-leakage (ISS-08) decisions stay
open and are carried into Week 8.

Other dependencies to report: **Project Management** — issue log and decision log in this package
feed their Integration & Issue Log. **Data Analytics** — confirm the valid ranges in `config.py`
(e.g. Age 0–120) match their validated data profile.

## 11. Assumptions, Limitations, Risks & Dependencies

**Resolved this week:** ISS-01 to ISS-07, ISS-10, ISS-11 and ISS-16 (see issue log) — no artifact,
batch-dependent encoding, no input validation, future-visit leakage in the model input, constant
feature, no logging, no entry point, CLI usability issues, mutation-harness fault.

**Unresolved / new**

| Issue | Type | Impact | Mitigation / owner |
|---|---|---|---|
| ISS-08: The **target label** uses `PriorAdmissionsCount`, which also counts future admissions (same total−1 pattern) | Modelling / data | Labels for some patients reflect admissions after the scored visit; reported metrics may be optimistic | Recompute label point-in-time and retrain — **Data Science** decision, as it changes the target |
| ISS-09: Threshold trade-off (recall 0.785 vs. 67.6% of patients flagged) and steep sensitivity between 0.30–0.36 | Modelling / operational | Too many flags can overwhelm follow-up capacity; too few misses patients | DS + clinic stakeholders choose operating point from §6; no code change needed |
| ISS-12: Target is a constructed rule of age/chronic/severity/admissions (carried from Week 6) | Modelling | Model formalises a rule rather than predicting an independent outcome | Retrain on a real outcome when available |
| ISS-13: Synthetic dataset; only 5,000 of 60,000 appointments carry full context | Data | Results may not transfer to real clinic data | Re-run pipeline on real data; no code change required |
| ISS-14: Province / facility-type fairness slices untestable (no facility→district link) | Data / responsible AI | Possible subgroup gaps undetected | Obtain facility master data; slice tests ready to add |
| ISS-15: CLI/batch only — no HTTP service, authentication or monitoring | Technical | Week 8 integration needs an agreed interface | Week 8: thin API wrapper over `predict()` if required by the integration plan |
| Valid ranges in `config.py` are engineering judgement (e.g. DoctorYearsExperience ≤ 60) | Assumption | Could reject an unusual but real value | Confirm with Data Analytics |

**Testing limitations:** tests run against the provided synthetic data only; no load/latency
testing beyond the logged ~80 ms for a small batch; no test of concurrent use.

## 12. Week 8 Readiness

| Area | Status |
|---|---|
| Pipeline loads, validates, scores, logs | Ready |
| Reproducible from documented setup | Ready (verified in clean environment) |
| Automated tests | Ready — 61 tests, mutation-checked |
| Threshold / operating point | **Blocked on DS decision (ISS-09)** |
| Target label leakage | **Needs DS decision (ISS-08)** |
| Cross-track test with DS | **Requested, no response by deadline (§10)** |
| Serving interface for final integration | CLI + Python API available; HTTP wrapper only if the Week 8 plan needs it |

**Before Week 8:** (1) follow up with Data Science, run the cross-track test in §10 and record the real outcome;
(2) get the threshold decision; (3) agree with DS whether to fix ISS-08 now or document it;
(4) agree the integration interface with Project Management.

## Reproducing this report

```bash
source .venv/bin/activate
python scripts/week7_audit_week6_pipeline.py   # before: 14 probes vs. Week 6 code
python scripts/03_train_pipeline.py            # train + evaluate + save refined pipeline
python scripts/week7_retest_probes.py          # after: same 14 probes vs. refined pipeline
pytest                                         # 61 tests
python scripts/week7_mutation_check.py         # does the suite catch injected bugs?
python scripts/predict.py IN.csv OUT.csv [--skip-invalid] [--threshold T] [--log-file F]
```
