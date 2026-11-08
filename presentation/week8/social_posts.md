# LinkedIn & X Post — Week 8

Images are in `post_images/`. Images 01, 04 and 05 are real VS Code screenshots taken on 8 Nov 2026
(actual source code and live terminal runs). Images 06 and 07 are charts drawn from the evidence files.

**LinkedIn images (attach in this order):**
1. `06_ds_contract_tests.png`: the 15 Data Science contract tests, 11/15 → 12/15 → 15/15
2. `01_vscode_predict.png`: `predict()` in VS Code; every output row gets a status
3. `04_vscode_terminal_tests_readiness.png`: live run, 65 tests passed and the readiness check
4. `07_release_gate_chart.png`: the model and the single rule Age ≥ 65 score the same AUC

**X images:** `06_ds_contract_tests.png` + `04_vscode_terminal_tests_readiness.png`

Spare images if you want to swap one in: `02_vscode_validation.png`, `03_vscode_config_contract.png`,
`05_vscode_terminal_release_gate.png`.

---

## LinkedIn

Week 8 of the @AnalystLab Africa Experience Lab, the final week: connecting my work to another team's, and finding out we'd built two different rulebooks.

My ML Engineering piece is the scoring pipeline for HealthConnect. It checks the input, scores each patient, and refuses data it can't score safely. This week the Data Science track sent me their final model with 15 contract tests.

My pipeline passed 11 of them.

None of the 4 failures was a crash. We'd simply made different choices. Their scorer accepted questionable rows and flagged them, while mine rejected them with a reason. Their scorer returned nothing for empty input, and mine raised an error. We also disagreed on the maximum age (110 vs 120) and on whether decimals were allowed.

So we agreed on one shared contract, and both sides changed their code:
→ Bad rows are rejected with a reason, and every output row has a status
→ Empty input gives empty output
→ Same limits on both sides: age 0–120, whole numbers only
→ Confirmed that "prior visits" only counts visits that have already happened

After their v1.1 update: 15/15.

Then Data Science ran their release gate on my Week 7 High-Risk model. It passed every statistical check (AUC 0.721, lift 1.82 at a 20% flag budget). The extra check is what caught it: the single rule "Age ≥ 65" scores the same AUC. My model was mostly restating how its own label was built, so it's now marked DEMO ONLY. I also replaced the fixed threshold, which flagged two-thirds of patients, with a flag budget that the clinic chooses.

Where it ends: reproducible on a fresh machine, 65/65 tests, and 12/12 error cases handled. It's ready to demonstrate, but not ready to make decisions about real patients, and I'd rather say that clearly than oversell it.

Biggest lesson: two components that each "work" can still disagree. That only shows up when you test them against each other.

Thanks to Emmanuella Esi Arhin on the Data Science track for the contract tests and quick replies.

Code, evidence and the final package: https://github.com/aimlin9/healthconnect-ml-engineering

#AnalystLabAfrica #MachineLearning #MLOps #DataScience #HealthTech

---

## X (Twitter)

Week 8 @analystlabafric: integrated my pipeline with Data Science. Contract tests 11/15 → 15/15 once we agreed one shared contract. Their release gate then showed my model just restates Age ≥ 65. Demo-ready, not for patients.

https://github.com/aimlin9/healthconnect-ml-engineering
#AnalystLabAfrica
