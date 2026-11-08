# Week 8 Decision & Issue Log: ML Pipeline ↔ Data Science

Continues `reports/week7/decision_log.md` and `issue_log.csv`. Decisions D-08 to D-12 were agreed
with Data Science (E. E. Arhin) on 8 Nov 2026 in reply to `DS_Handoff_Response.md`.

## Decisions

| ID | Decision | Alternatives | Reason | Owner |
|---|---|---|---|---|
| D-08 | Empty input returns empty output (exit 0, header-only CSV) instead of an error | Keep the Week 7 "no rows" error | Shared contract with the DS scorer (C13); an empty batch is not a fault. Supersedes the Week 7 W6-10 pass condition | ML Eng + DS |
| D-09 | Every output row carries `status` (SCORED / SCORED_WITH_WARNINGS: reason / REJECTED: reason) and `model_status`; with `--skip-invalid`, rejected rows stay in the output in input order | Separate `.rejected.csv` only (Week 7) | Adopted from the DS contract: one row per input, problems visible where the score is read. `.rejected.csv` kept for existing users | ML Eng (adopted from DS) |
| D-10 | Missing values and unknown categories stay **rejected**, not imputed | DS score-and-flag | Agreed by DS: a risk score on a missing age is worse than no score. DS adds a strict option to `hc_scorer.py` | ML Eng + DS |
| D-11 | Shared limits: Age 0–120; Age, prior-visit counts and flags must be whole numbers; prior visits are point-in-time (strictly earlier visits, same-day ordered by appointment ID) | DS Age 110; DS accepting decimals | DS moves Age to 120 and whole numbers; DS confirmed PriorVisits uses the same point-in-time definition as `PriorAppointmentsCount` (ISS-04 fix). DS keeps its own 50 cap, ML Eng keeps 1000 | ML Eng + DS |
| D-12 | A model is called usable only if it passes the DS release gate on a locked hold-out **and** its target is a real outcome. Both models run in the demo, each with its own `model_status` | Call the High-Risk model usable because it passes the statistical gate | The High-Risk model passes the gate (AUC 0.721, CI 0.691–0.752, shuffle p = 0.0099, lift 1.82 @ 20%), but `Age>=65` alone scores the same AUC (0.721): it restates its constructed label. Labelled **DEMO ONLY** | ML Eng + DS |
| D-13 | Report flag budgets with lift (10/20/30%) instead of a probability threshold; the clinic chooses the percentage from review capacity | Keep threshold 0.33 (flags 67.6%) | DS: a threshold that flags two-thirds of patients is not triage. The 0.33 threshold stays in metadata for backwards compatibility | DS + clinic |

## Issue updates

| ID | Week 7 status | Week 8 status | Evidence |
|---|---|---|---|
| ISS-08 Label counts future admissions | Open (DS) | **Documented, not fixed.** `PriorAdmissionsCount>=1` is true in only 3.1% of rows, and the model is labelled demo-only (D-12), so retraining on a corrected label would not change its use | `release_gate_high_risk.json` → `share_of_rows_with_component` |
| ISS-09 Threshold flags 67.6% | Open (DS + clinic) | **Superseded by D-13**: flag budget + lift; percentage to be set by the clinic | `release_gate_high_risk.json` → `lift` |
| ISS-12 Constructed target | Open, documented | **Confirmed and quantified**: the model's AUC equals that of `Age>=65` alone | `release_gate_high_risk.json` → `rule_restatement_check` |
| ISS-15 Integration interface | Open (Week 8) | **Resolved for the demo**: shared row-level contract (D-08 to D-11); CLI + Python API | `ds_contract/`, tests |
| ISS-17 DS harness expects imputation for C3–C5 | New | **Open (DS)**: DS ships `hc_scorer.py` strict option and updated expectations; then re-run | `ds_contract/B_mle_pipeline_vs_ds_contract.csv` (12/15) |

## Retest after the changes (8 Nov 2026)

| Check | Before (7 Nov) | After (8 Nov) |
|---|---|---|
| Automated tests | 61/61 | 65/65 (3 updated for D-08/D-09, 4 added) |
| Error-handling matrix | 11/11 | 12/12 (adds empty input) |
| DS contract tests against this pipeline (current DS harness) | 11/15 (C3, C4, C5, C13) | 12/15 (C3, C4, C5: DS harness still expects imputation, ISS-17) |
| DS package in this environment | 15/15 | 15/15 |
| Reproducibility | Identical | Identical |
