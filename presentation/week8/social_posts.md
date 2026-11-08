# LinkedIn & X Posts — Week 8

Images are in `post_images/`. Images 01–05 are real VS Code screenshots taken on 8 Nov 2026: the actual
source files, and live runs in the VS Code terminal. Charts 06–07 are drawn from the evidence files by
`scripts/week8_post_images.py`.

| File | Shows |
|---|---|
| `01_vscode_predict.png` | `predict()` in `pipeline.py`: every output row gets a `status` and `model_status` (line 140) |
| `02_vscode_validation.png` | `validate_input()`: missing values, unknown categories, out-of-range and non-whole numbers are each rejected with a reason |
| `03_vscode_config_contract.png` | `config.py`: the input contract (Age 0–120, whole numbers) and the DEMO ONLY model status |
| `04_vscode_terminal_tests_readiness.png` | Live run: `pytest` 65 passed, readiness check (reproducible, 12/12 error cases, 1,250/1,250 scored, AUC 0.721) |
| `05_vscode_terminal_release_gate.png` | Live run: release gate passed, but `Age>=65` alone scores the same hold-out AUC (0.721) |
| `06_ds_contract_tests.png` | Chart: the 15 Data Science contract tests across three runs, 11/15 → 12/15 → 15/15 |
| `07_release_gate_chart.png` | Chart: the model vs one rule, against the AUC ≥ 0.60 gate |

Which images go with which post:
- **Post 1 (main LinkedIn):** 06, 04, 05, 07
- **Post 2 (code walkthrough):** 01, 02, 03
- **Post 3 (release gate lesson):** 05, 07
- **X post / thread:** 06 + 04 on the first tweet; 05 on the release-gate tweet

Optional: post `HealthConnect_Week8_MLE_Presentation_Ramsey.mp4` as a video instead.

---

## Post 1 — LinkedIn (main Week 8 post)

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

## Post 1 — X (single post)

Week 8 @analystlabafric: integrated my pipeline with Data Science. Contract tests 11/15 → 15/15 once we agreed one shared contract. Their release gate then showed my model just restates Age ≥ 65. Demo-ready, not for patients.

https://github.com/aimlin9/healthconnect-ml-engineering
#AnalystLabAfrica

### X thread version (each tweet under 280 characters)

1/ Week 8 @analystlabafric: integrated my pipeline with Data Science. Contract tests 11/15 → 15/15 once we agreed one shared contract. Their release gate then showed my model just restates Age ≥ 65. Demo-ready, not for patients. 🧵

2/ The 4 failures weren't crashes. Their scorer flagged bad rows; mine rejected them. Theirs returned nothing on empty input; mine raised an error. We agreed one contract: reject with a reason, a status on every row, empty in → empty out. Both sides changed code.

3/ Then their release gate on my model: AUC 0.721, lift 1.82 at a 20% flag budget, every check passed. But the single rule "Age ≥ 65" also scores 0.721. The label was built partly from age, so the model mostly restates it. It's now marked DEMO ONLY.

4/ Where it ends: 65/65 tests, 12/12 error cases, reproducible on a fresh machine. Ready to demonstrate, not to make decisions about real patients.

Code + evidence: https://github.com/aimlin9/healthconnect-ml-engineering
#AnalystLabAfrica

---

## Post 2 — LinkedIn (code walkthrough, a few days after Post 1)

Images: `01_vscode_predict.png`, `02_vscode_validation.png`, `03_vscode_config_contract.png`

Three files hold my Week 8 HealthConnect scoring pipeline together. These screenshots come straight from VS Code.

1️⃣ config.py: the input contract. Two categories with fixed values, four numbers with hard limits (age 0–120, whole numbers only). This is now the single reference that both Data Science and I work from. It also stores the model's status in plain words: DEMO ONLY.

2️⃣ validation.py: every problem becomes a row, a column and a reason, whether that's a missing value, an unknown category, something that isn't a number, a value out of range, or a decimal where a whole number belongs. Nothing gets scored silently.

3️⃣ pipeline.py: predict(). Every output row carries its own status (scored, scored with a warning, or rejected with the reason) plus the model's status. Anyone reading the output can see straight away that this model isn't meant for real decisions.

None of this is clever. It's the boring part that decides whether someone else can actually use your model.

Code: https://github.com/aimlin9/healthconnect-ml-engineering

#AnalystLabAfrica #MLOps #Python #DataValidation #MachineLearning

---

## Post 3 — LinkedIn (release gate lesson)

Images: `05_vscode_terminal_release_gate.png`, `07_release_gate_chart.png`

The most useful check in Week 8 wasn't one of the statistical tests.

The Data Science track ran their release gate on my High-Risk Patient model, using a locked hold-out of 1,250 patients:
→ AUC 0.721 (95% CI 0.691–0.752)
→ Label-shuffle test p = 0.0099
→ Lift 1.82 when flagging the top 20%

All four gates passed.

Then came one more question: what does the label actually reward?

"High-risk" was defined as at least two of: age 65 or over, a chronic condition, severity of 3 or more, a prior admission. Only age is one of the model's inputs.

The single rule "Age ≥ 65" scores an AUC of 0.721, and the same model trained on age alone scores 0.723. So the model learned age, which is how the label was built, not anything about real outcomes.

It's now labelled DEMO ONLY on every row it scores.

Lesson: before celebrating a metric, check what a one-line rule scores on the same data.

Evidence and the gate script: https://github.com/aimlin9/healthconnect-ml-engineering

#AnalystLabAfrica #MachineLearning #ModelEvaluation #ResponsibleAI #HealthTech
