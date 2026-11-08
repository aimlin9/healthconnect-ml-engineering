# HealthConnect Final ML Pipeline & Engineering Package — Week 8

**AnalystLab Africa Experience Lab · Batch E · ML Engineering track** · Gyimah Ramsey Opoku
Stage: Final Integration → Project Readiness → Presentation → Showcase

> Data Science contract tests ran on 7 Nov 2026; Data Science replied on 8 Nov and the agreed
> changes are implemented on both sides and retested: Data Science's v1.1 strict harness passes
> 15/15 against this pipeline (§7, `reports/week8/decision_log.md`). Nothing here claims an
> integration result that has not happened.

---

## 1. Week 7 → Week 8 Transition

| # | Item | Response |
|---|---|---|
| 1 | Main Week 7 output | The Week 6 High-Risk Patient model rebuilt as a tested, reproducible pipeline: `src/hc_pipeline` + `scripts/predict.py` CLI, saved artifact + metadata, 61 automated tests |
| 2 | Most important testing result | The Week 6 inference code passed **2/14** reliability probes; the refined pipeline passes **14/14**, at almost no cost in ranking quality (ROC-AUC 0.727 → 0.721) |
| 3 | Major issue discovered | Week 6 could not score a single patient (`get_dummies` re-run per batch), and silently scored impossible or missing values |
| 4 | Refined | Fixed-category encoding inside the sklearn Pipeline, input validation, point-in-time feature (no future visits), logging, saved artifact, recall-targeted threshold |
| 5 | Ready | The scoring pipeline (load → validate → score → log), CLI and Python API |
| 6 | Still to integrate | The Data Science final model package and its contract tests (run in Week 8, §7); the threshold (ISS-09) and label (ISS-08) decisions |
| 7 | Tracks in final integration | Data Science (model + requirements ↔ pipeline constraints); Project Management (issue/decision logs, demo slot); Data Analytics (valid input ranges) |
| 8 | Will demonstrate | One patient scored end-to-end through the CLI, bad input rejected with line and reason, the reproducibility check, and the DS contract-test result |

## 2. Final Integration Readiness (Part 1)

| Requirement | Response |
|---|---|
| What is my final component? | The HealthConnect ML scoring pipeline: input contract + validation, preprocessing and model in one saved sklearn Pipeline, CLI (`scripts/predict.py`) and Python API (`hc_pipeline.load/predict`), logging, tests |
| What problem does it address? | A model in a notebook cannot be used by anyone else. The pipeline lets other tracks and a clinic system score patients the same way every time, and refuses input it cannot score safely |
| Current status | Ready for demonstration (not production). Reproduced on a second OS and Python version this week (§6) |
| What did I improve in Week 7? | See §1, row 4 |
| Depends on | **Data Science:** final model, target definition, operating threshold. **Data Analytics:** confirmation of valid input ranges |
| Who depends on my output | **Data Science:** a pipeline to run their model through. **PM:** issue/decision logs, readiness status, demo. **GenAI:** none directly (the assistant does not call the model; see §9) |
| Output I provide | Input/output contract (`config.py`), artifact + metadata, CLI, error-handling matrix, reproducibility evidence, issue and decision logs |
| Output I need | DS final model package + requirements note (received); DS decisions on contract, threshold and label (received 8 Nov, D-08 to D-13); DS strict-mode harness (pending) |
| Evidence of readiness | `reports/week8/evidence/` (reproducibility, error handling, end-to-end); 65/65 tests; Week 7 mutation check 5/5; DS release gate applied (§7a) |
| Limitations remaining | §8 |

## 3. Final Integrated Pipeline

```
input CSV / DataFrame / dict
   │
   ▼  validate_input()            columns present · categories whitelisted (case/space-insensitive)
   │                              numbers coerced · hard ranges · whole numbers · no missing values
   │── rejected rows ─► exit 2, or OUTPUT.rejected.csv with --skip-invalid (CSV line + reason)
   ▼
 training-range check ──► WARNING logged (extrapolation), row still scored
   ▼
 saved sklearn Pipeline      OneHotEncoder(fixed categories) + RandomForest  (models/high_risk_pipeline.joblib)
   ▼
 risk_probability · high_risk_flag (≥ threshold from metadata) · threshold · model_version
   ▼
 OUTPUT.csv (input columns carried through, normalised values written) + log
```

**Input contract** (`src/hc_pipeline/config.py`)

| Column | Type | Valid | Seen in training |
|---|---|---|---|
| Gender | category | Female, Male | — |
| VisitType | category | Consultation, Emergency, Follow-up | — |
| Age | int | 0–120 | 0–90 |
| DoctorYearsExperience | int | 0–60 | 1–25 |
| PriorAppointmentsCount | int | 0–1000 | 0–3 |
| DepartmentWorkload | int | 0–100,000 | 13–38 |

**Output:** all input columns + `risk_probability` (0–1, 4 dp), `high_risk_flag` (0/1),
`threshold`, `model_version`. **Exit codes:** 0 ok · 2 invalid input / bad arguments · 3 model missing.

**Model:** `1.1.0-week7`, `model_status` DEMO ONLY (§7a). Threshold 0.33 kept for compatibility; operating rule is now a flag budget chosen by the clinic (D-13). Test split: ROC-AUC 0.721,
recall 0.785, precision 0.492; flags 67.6% of patients.

## 4. Repository Structure, Dependencies & Configuration

- Structure: see `README.md` → *Repository Structure*. Week 8 adds `reports/week8/` and
  `scripts/week8_readiness_check.py`.
- Dependencies: `requirements.txt` (fully pinned). Key pins: scikit-learn 1.9.1, pandas 3.1.0,
  numpy 2.5.4, joblib 1.6.0. Python 3.12 or 3.13 verified.
- Configuration: one file, `src/hc_pipeline/config.py` — features, categories, valid ranges,
  model hyper-parameters, split seed, recall target. The decision threshold lives in the model
  metadata and can be overridden per run with `--threshold` without retraining.
- `load()` refuses an artifact whose feature list differs from the code, and warns if the
  scikit-learn minor version differs from the one used for training.

## 5. Testing Summary

| Check | Result | Evidence |
|---|---|---|
| Automated test suite | 65/65 pass (Windows 11, Python 3.12.10); 3 updated for D-08/D-09, 4 added | `pytest` |
| Mutation check (Week 7) | 5/5 injected bugs caught | `reports/week7/evidence/mutation_check_results.json` |
| Reliability probes (Week 7) | 14/14 (Week 6 code: 2/14) | `reports/week7/evidence/week7_retest_results.json` |
| Error-handling matrix (Week 8) | 12/12 cases give the specified exit code and output (adds empty input) | `reports/week8/evidence/error_handling_matrix.csv` |
| End-to-end CLI run (Week 8) | 1,250/1,250 test rows scored; CLI output identical to in-process `predict()`; IDs carried through; ROC-AUC 0.721 | `reports/week8/evidence/end_to_end_check.json` |
| Reproducibility (Week 8) | Identical — see §6 | `reports/week8/evidence/reproducibility_check.json` |
| DS contract tests, DS package unchanged | 15/15 (v1.0: Py 3.12.10 / sklearn 1.9.1 and Py 3.14.6 / 1.9.0; v1.1 strict: Py 3.12.10 / 1.9.1) | `reports/week8/evidence/ds_contract/` |
| DS contract tests against my pipeline | 11/15 on 7 Nov → 12/15 after the agreed changes → **15/15** with the DS v1.1 strict harness (8 Nov, ISS-17 closed) | `ds_contract/B_mle_pipeline_vs_ds_contract.csv` |
| My data → DS scorer | 5,000/5,000 scored, 0 rejected | `ds_contract/summary.json` |

## 5a. Error-Handling Evidence

| Case | Exit | Behaviour |
|---|---|---|
| Valid single patient | 0 | Scored |
| Age 250 | 2 | `CSV line 2, Age: 250 outside valid range [0, 120]`; no output written |
| Age blank | 2 | `Age: missing value` |
| VisitType = Surgery | 2 | `unknown category 'Surgery'; allowed [...]` |
| Required column missing | 2 | `DepartmentWorkload: required column missing` |
| Age = "fifty" | 2 | `Age: not a number` |
| Mixed batch with `--skip-invalid` | 0 | 2 rows scored (" male " normalised to Male), 1 written to `.rejected.csv` |
| Age 110 (valid but outside training range) | 0 | Scored, with an extrapolation WARNING |
| `--threshold 1.5` | 2 | Rejected before scoring |
| Input file missing | 2 | `cannot read …` |
| Model artifact missing | 3 | `model artifact not found …; run scripts/03_train_pipeline.py first` |

## 6. Reproducibility

Week 7 reproduced the model in a fresh Linux environment (Python 3.13). In Week 8 I repeated it
on a different setup: **fresh `git clone` on Windows 11, new Python 3.12.10 venv from
`requirements.txt`**, then `pytest` and a retrain.

| | Committed artifact | Fresh retrain (Windows, Py 3.12) |
|---|---|---|
| Training-data fingerprint | 64e9964b632a1cff | 64e9964b632a1cff |
| Threshold | 0.33 | 0.33 |
| Test metrics | AUC 0.721, recall 0.785, precision 0.492 | identical |
| Max per-patient probability difference (5,000 rows) | — | 0.0 |
| Flags that disagree | — | 0 |

Re-run: `python scripts/week8_readiness_check.py`.

## 7. Data Science Integration — Final Model Package

Data Science (Emmanuella Esi Arhin) sent the final model package on 7 Nov 2026: an
admission-proxy scorer (ridge logistic regression; Age, PriorVisits, HasChronic, Gender,
Diagnosis), `hc_scorer.py`, `schema.json`, a requirements note and `hc_contract_tests.py`
(C1–C15). Their position: the model is a **validated null result** (hold-out AUC 0.467, release
gate failed) and **must not be used for decisions**. Only the **scoring interface** is being
integrated. Full reply: `reports/week8/DS_Handoff_Response.md`.

| Item | Result |
|---|---|
| Package safety | Code reviewed before running; pickle checked to contain only numpy, joblib and sklearn classes |
| Versions | Python 3.12.10 / scikit-learn 1.9.1 (pipeline env), and 3.14.6 / 1.9.0 (DS training env) |
| DS package, unchanged | **15/15** in both environments; version warnings under 1.9.1 but scores identical (max diff 0.0) |
| DS tests against my pipeline | **11/15** on 7 Nov (C3, C4, C5, C13) → **12/15** on 8 Nov (C13 fixed) → **15/15** with the DS v1.1 strict harness (C3–C5 now expect rejection) |
| Why they failed | Policy difference, not a bug. DS scored soft problems and flagged them; my pipeline rejects them (decision D-02). My pipeline raised on empty input; theirs returned empty output. Both resolved by the shared contract (D-08 to D-11) |
| My data pipeline → DS scorer | All 5,000 rows scored after two column renames; 0 rejected; top-20% budget flagged exactly 20.0% |
| Schema mismatches (7 Nov) | Age max 110 vs 120; whole numbers required (mine) vs floats accepted (theirs); `PriorVisits` vs `PriorAppointmentsCount` (point-in-time definition to confirm). All resolved: DS v1.1 uses Age 0–120 and whole numbers; PriorVisits confirmed point-in-time |
| Time-point constraint | DS model needs Diagnosis, so it can only run after consultation; it will not be wired to a booking trigger (DS issue I10) |
| What changed as a result | DS agreed (8 Nov). **ML Eng:** per-row `status` and `model_status` on every output row, rejected rows kept in output, empty input → empty output (D-08, D-09). **DS:** strict reject option, Age 120, whole numbers (D-10, D-11). PriorVisits confirmed point-in-time |

### 7a. Release gate applied to the High-Risk model (DS request)

| Gate (locked hold-out, n = 1,250) | Result |
|---|---|
| AUC ≥ 0.60 | 0.721 ✔ |
| Lower 95% CI > 0.50 | 0.691 ✔ |
| Label-shuffle p < 0.05 | 0.0099 ✔ |
| Lift > 1.3 at 20% budget | 1.82 ✔ (1.77 @ 10%, 1.84 @ 30%) |
| **Rule-restatement check** | `Age ≥ 65` alone: AUC **0.721**; Age-only model 0.723 → the model restates its constructed label |

Verdict (D-12): passes the statistical gate, but **not usable**: labelled DEMO ONLY in `model_status`.
Evidence: `reports/week8/evidence/release_gate_high_risk.json`; re-run `python scripts/week8_release_gate.py`.

## 8. Known Limitations

| Limitation | Impact | Owner |
|---|---|---|
| ISS-09 → D-13: threshold 0.33 flags 67.6%; replaced by a flag budget with lift, percentage set by the clinic | Needs a clinic decision before any use | Clinic |
| ISS-08: target label counts future admissions (true in only 3.1% of rows) | Minor; model is demo-only anyway | Documented |
| ISS-12: target is a constructed rule; **confirmed**: `Age ≥ 65` alone matches the model's AUC | No model in the solution predicts a real outcome | Retrain on real outcomes |
| ISS-13: synthetic data; 5,000 of 60,000 appointments have full context | Results may not transfer to real data | PM |
| ISS-14: province / facility-type fairness untestable | Subgroup gaps could go undetected | Data Analytics / PM |
| ISS-15: CLI and Python API only — no HTTP service, authentication or monitoring | Not deployable as a live service | ML Engineering + PM |
| Valid ranges are engineering judgement | An unusual real value could be rejected | Data Analytics to confirm |
| Not tested: load, concurrency, real clinic data | Demonstration-ready, not production-ready | — |

**What the pipeline can be used for:** demonstrating a safe, repeatable scoring workflow; batch
scoring of synthetic data for prioritisation research. **What it cannot be used for:** clinical
or appointment decisions about real patients.

## 9. HC-POD Final Integration

| # | Field | Data Science ↔ ML Engineering | Project Management ← ML Engineering |
|---|---|---|---|
| 1 | Track | Data Science | Project Management |
| 2 | Dependency | DS owns the final model, target and threshold; ML Eng owns the scoring interface | PM needs component status, issues and demo requirements |
| 3 | Output received | Final model package, requirements note and `hc_contract_tests.py` (7 Nov) | PENDING (presentation slot / sequence) |
| 4 | Output provided | `DS_Handoff_Response.md` (7 Nov), `DS_Reply_2.md` and `DS_Reply_3.md` (8 Nov, v1.1 retest 15/15): versions, test results, schema comparison, requirements, High-Risk model target, hold-out metrics and gate result | This package; issue log; decision log; readiness status (§2) |
| 5 | Integration activity | Ran the DS tests on their package and on my pipeline; fed all 5,000 pipeline rows into the DS scorer | Hand over §1, §2, §8 for the project status report and risk register |
| 6 | What changed | One shared row-level contract (status column, rejects, empty→empty, limits); High-Risk model relabelled DEMO ONLY after the DS gate; threshold replaced by a flag budget | PENDING |
| 7 | How it improves the solution | One agreed interface contract instead of two that silently disagree; confirmed the pipeline's data can feed the DS model | — |
| 8 | Evidence | `reports/week8/evidence/ds_contract/`, `scripts/week8_ds_contract_crosscheck.py` | — |
| 9 | Contribution to final presentation | ML Engineering section: pipeline, evidence, DS contract-test result | Demo steps (§10) |

## 10. Demonstration Workflow (≈2 minutes in the walkthrough)

1. `pytest -q` → 65 passed.
2. Score one patient: `python scripts/predict.py demo_patient.csv out.csv` → show `risk_probability`, flag, model version.
3. Show a bad row (Age 250) rejected with CSV line and reason; rerun with `--skip-invalid`.
4. `python scripts/week8_readiness_check.py` → reproducibility identical, 12/12 error cases, end-to-end match.
5. DS contract tests: `python scripts/week8_ds_contract_crosscheck.py <DS package>` → DS package 15/15; my pipeline 15/15 (DS v1.1 strict harness).
5a. `python scripts/week8_release_gate.py` → gate passed, but `Age ≥ 65` alone gives the same AUC, so DEMO ONLY.
6. Close on limitations: demonstration-ready, not for real patient decisions.
