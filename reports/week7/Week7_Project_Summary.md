# Week 7 Project Summary — ML Engineering Track

**HealthConnect Experience Lab · Batch E** · Main deliverable: `reports/week7/ML_Pipeline_Testing_Reliability_Report.md`

1. **Planned to test:** whether the Week 6 High-Risk Patient model could be used as a pipeline —
   score real inputs, reject bad ones, and reproduce its results.

2. **Actually tested:** 14 probes against the Week 6 inference code (verbatim from the notebook),
   then the same probes, a 61-test automated suite, mutation testing, a clean-environment
   reproduction and a re-evaluation of the refined model on the original test split and across
   all 20 facilities.

3. **Most important results:** Week 6 passed **2/14** probes; the refined pipeline passes **14/14**.
   61/61 tests pass and the suite catches 5/5 injected bugs. Removing leaked and constant features
   cost almost nothing (AUC 0.727 → 0.721).

4. **Issues found:** the Week 6 model could not score a single patient (batch-dependent encoding);
   impossible and missing values were silently scored; a feature counted future visits (11.1% of
   rows); a feature was constant; no saved model, no logging. Also found problems in my own
   testing — a mutation harness that falsely reported success, and a unit test too weak to catch
   the bug it was written for.

5. **Refinements:** `src/hc_pipeline` package (fixed-category encoding inside an sklearn Pipeline,
   input validation, point-in-time feature, saved model + metadata, logging), CLI
   `scripts/predict.py`, recall-targeted threshold.

6. **Retest results:** 14/14 probes, 61/61 tests, 5/5 mutations caught; recall 0.555 → 0.785
   (0.817 median on unseen facilities); fresh clone reproduced an identical model.

7. **Tracks collaborated with:** I asked Data Science to collaborate; no response had arrived by
   the deadline. The issue and decision logs are ready for Project Management.

8. **Tested collaboratively:** not yet. The planned test (DS scores their validation rows through
   `predict.py` and compares the results with their notebook) is set out in report §10 and will
   run once Data Science responds.

9. **What changed as a result:** nothing yet from cross-track work. The threshold (ISS-09) and
   label-leakage (ISS-08) decisions are waiting on Data Science.

10. **Key validation outcome:** the Week 6 model's performance was genuine (not driven by leakage);
    the pipeline around it was not deployable and now is.

11. **Major challenges:** recall vs. alert volume — at the tuned threshold the model flags 67.6% of
    patients (42.4% are high-risk) and the threshold is fragile between 0.30 and 0.36.

12. **Important decisions:** reject invalid input by default; ship threshold 0.33 as configurable
    and hand the final choice to Data Science; fix the model input's leakage but leave the label
    change to Data Science; commit the model artifact (`reports/week7/decision_log.md`).

13. **Remaining limitations:** constructed target label; synthetic data; province / facility-type
    fairness untestable; CLI only, no HTTP service or monitoring.

14. **Remaining issues / dependencies:** ISS-08 label also counts future admissions (DS);
    ISS-09 threshold decision (DS + clinic); Week 8 integration interface (PM).

15. **My contribution:** turned the Week 6 notebook model into a tested, validated, reproducible
    pipeline other tracks can call, and documented every failure found along the way, including
    my own.

16. **Before Week 8:** complete the DS cross-track test and record it; get the threshold decision;
    agree on ISS-08; agree the integration interface with PM.
