# HealthConnect Final ML Pipeline & Engineering Package — Week 8

**AnalystLab Africa Experience Lab · Batch E · ML Engineering track** · Gyimah Ramsey Opoku
Stage: Final Integration → Project Readiness → Presentation → Showcase

> Data Science contract tests ran on 7 Nov 2026 (§7). Items still marked **PENDING-DS** are
> waiting on Data Science's reply to `reports/week8/DS_Handoff_Response.md`. Nothing here claims
> an integration result that has not happened.

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
| Output I need | DS final model package + requirements note; DS threshold and ISS-08 decisions |
| Evidence of readiness | `reports/week8/evidence/` (reproducibility, error handling, end-to-end); 61/61 tests; Week 7 mutation check 5/5 |
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

**Model:** `1.1.0-week7`, threshold 0.33 (provisional, ISS-09). Test split: ROC-AUC 0.721,
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
| Automated test suite | 61/61 pass (Windows 11, Python 3.12.10) | `pytest` |
| Mutation check (Week 7) | 5/5 injected bugs caught | `reports/week7/evidence/mutation_check_results.json` |
| Reliability probes (Week 7) | 14/14 (Week 6 code: 2/14) | `reports/week7/evidence/week7_retest_results.json` |
| Error-handling matrix (Week 8) | 11/11 cases give the specified exit code and output | `reports/week8/evidence/error_handling_matrix.csv` |
| End-to-end CLI run (Week 8) | 1,250/1,250 test rows scored; CLI output identical to in-process `predict()`; IDs carried through; ROC-AUC 0.721 | `reports/week8/evidence/end_to_end_check.json` |
| Reproducibility (Week 8) | Identical — see §6 | `reports/week8/evidence/reproducibility_check.json` |
| DS contract tests, DS package unchanged | 15/15 (Py 3.12.10 / sklearn 1.9.1 and Py 3.14.6 / 1.9.0) | `reports/week8/evidence/ds_contract/` |
| DS contract tests against my pipeline | 11/15. C3, C4, C5, C13 fail because of policy differences (§7) | `ds_contract/B_mle_pipeline_vs_ds_contract.csv` |
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
| DS tests against my pipeline | **11/15**. Failing: C3, C4 (missing values), C5 (unseen category), C13 (empty input) |
| Why they fail | Policy difference, not a bug. DS scores soft problems and flags them; my pipeline rejects them (decision D-02). My pipeline raises on empty input; theirs returns empty output |
| My data pipeline → DS scorer | All 5,000 rows scored after two column renames; 0 rejected; top-20% budget flagged exactly 20.0% |
| Schema mismatches | Age max 110 vs 120; whole numbers required (mine) vs floats accepted (theirs); `PriorVisits` vs `PriorAppointmentsCount` (point-in-time definition to confirm) |
| Time-point constraint | DS model needs Diagnosis, so it can only run after consultation; it will not be wired to a booking trigger (DS issue I10) |
| What changed as a result | **PENDING-DS.** Proposed: add a per-row `status` column to my output (adopted from the DS contract), return empty output on empty input, and align limits once DS agrees |

## 8. Known Limitations

| Limitation | Impact | Owner |
|---|---|---|
| ISS-09: threshold 0.33 flags 67.6% of patients at precision 0.49 | Too many flags for follow-up capacity | Data Science + clinic |
| ISS-08: target label counts future admissions | Reported metrics may be optimistic | Data Science |
| ISS-12: target is a constructed rule, not an observed outcome | The model reproduces a rule; it does not predict a real event | Data Science |
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
| 4 | Output provided | `DS_Handoff_Response.md`: versions, 15/15, 11/15 with reasons, schema comparison, time-point answer, my pipeline requirements | This package; issue log; decision log; readiness status (§2) |
| 5 | Integration activity | Ran the DS tests on their package and on my pipeline; fed all 5,000 pipeline rows into the DS scorer | Hand over §1, §2, §8 for the project status report and risk register |
| 6 | What changed | Found that the two scoring contracts disagree on soft problems and empty input, plus three schema differences. Code changes are proposed and PENDING-DS | PENDING |
| 7 | How it improves the solution | One agreed interface contract instead of two that silently disagree; confirmed the pipeline's data can feed the DS model | — |
| 8 | Evidence | `reports/week8/evidence/ds_contract/`, `scripts/week8_ds_contract_crosscheck.py` | — |
| 9 | Contribution to final presentation | ML Engineering section: pipeline, evidence, DS contract-test result | Demo steps (§10) |

## 10. Demonstration Workflow (≈2 minutes in the walkthrough)

1. `pytest -q` → 61 passed.
2. Score one patient: `python scripts/predict.py demo_patient.csv out.csv` → show `risk_probability`, flag, model version.
3. Show a bad row (Age 250) rejected with CSV line and reason; rerun with `--skip-invalid`.
4. `python scripts/week8_readiness_check.py` → reproducibility identical, 11/11 error cases, end-to-end match.
5. DS contract tests: `python scripts/week8_ds_contract_crosscheck.py <DS package>` → DS package 15/15; my pipeline 11/15, with the reasons.
6. Close on limitations: demonstration-ready, not for real patient decisions.
