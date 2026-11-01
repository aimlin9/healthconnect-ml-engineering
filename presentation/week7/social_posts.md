# LinkedIn & X Posts — Week 7

Images (attach in this order): `01_before_after_probes.png`, `02_terminal_tests_cli.png`,
`03_threshold_tradeoff.png`

---

## LinkedIn

Week 7 of the @AnalystLab Africa Experience Lab: testing my own work, and finding out how much of it was broken.

In Week 6 I built a high-risk patient classifier for HealthConnect. It beat its baseline and held up across all 20 facilities, and I was pretty happy with it.

This week the brief was testing, refinement and validation, so I ran 14 reliability checks against the exact code from my notebook.

It passed 2.

The biggest one: it couldn't score a single patient. I'd been re-running one-hot encoding on whatever data came in, so the columns depended on the batch. It worked fine on my 1,250-row test set and crashed the moment you gave it one real patient, which is the whole point of the thing. It also happily scored a patient aged 250, and one feature was quietly counting visits that hadn't happened yet.

What I changed:
→ Moved preprocessing inside the model pipeline so encoding can't drift between training and prediction
→ Added input validation that rejects impossible or missing values and says exactly which row and why
→ Replaced the leaky feature with one that only counts earlier visits
→ Saved the model with its settings and added logging and a command-line tool
→ Wrote 61 automated tests

Retest: 14/14 checks passed. I also broke the code on purpose in 5 ways to check the tests would notice. My first attempt said "5/5 caught", and that was wrong too: the test copy couldn't find the data. Fixed it, and now all 5 are genuinely caught.

The good news: fixing the leak barely moved performance (AUC 0.727 → 0.721), so the Week 6 result was real.

The part that isn't solved: tuning the threshold raised recall from 55% to 78.5%, but the model then flags 68% of patients when only 42% are actually high-risk (last image). That trade-off is a clinical call, not an engineering one, so it stays open going into Week 8.

Biggest lesson: "it works in my notebook" and "it works" are very different claims.

Code, tests and the full testing report: https://github.com/aimlin9/healthconnect-ml-engineering

#AnalystLabAfrica #MachineLearning #MLOps #Testing #HealthTech

---

## X (Twitter)

Week 7 @analystlabafric: tested my Week 6 model as a real pipeline. It passed 2/14 checks and couldn't even score one patient. Fixed + 61 tests → 14/14. The open question: higher recall means flagging 68% of patients.

https://github.com/aimlin9/healthconnect-ml-engineering
#AnalystLabAfrica
