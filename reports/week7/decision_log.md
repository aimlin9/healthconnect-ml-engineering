# Week 7 Decision Log — ML Pipeline

| ID | Decision | Alternatives considered | Reason | Owner |
|---|---|---|---|---|
| D-01 | Move one-hot encoding inside an sklearn `Pipeline` with a fixed category list | Save the Week 6 dummy column list and reindex at inference | One object holds preprocessing and model, so they cannot drift apart; the encoder errors on unknown categories instead of silently zero-filling | ML Engineering |
| D-02 | Reject invalid rows by default; offer `--skip-invalid` | Silently drop or impute bad rows | A clinical risk score on an impossible or missing age is worse than no score; skip mode still lets batch jobs proceed while recording every reject | ML Engineering |
| D-03 | Ship threshold 0.33 from the documented rule (recall ≥ 0.80 on validation), stored in metadata and overridable | Keep Week 6's 0.50; pick 0.36 | Missing a high-risk patient is the costlier error for a support tool; but the cost (67.6% flagged, precision 0.49) is a clinical/DS call, so the value is configurable and logged as open ISS-09 | ML Engineering (provisional) → Data Science |
| D-04 | Replace `PreviousAppointmentsCount` with a point-in-time count; leave the label's `PriorAdmissionsCount` unchanged | Fix both now | The model input is an engineering concern; changing the label changes what the model predicts, which is a Data Science decision (ISS-08) | ML Engineering / Data Science |
| D-05 | Drop `HasInsurance` | Keep for future data | Constant in all 5,000 rows; requiring it from callers adds a failure point with no benefit. Re-add if real data varies | ML Engineering |
| D-06 | Commit the 2.5 MB model artifact (`.gitignore` exception) | Require retraining after clone | Week 8 teammates can load it immediately; reproducibility is proven separately (identical fingerprint in a clean environment) | ML Engineering |
| D-07 | Keep the Week 6 dataset rather than the `HealthConnect_Appointment_Data.csv` named in the Week 7 brief | Switch datasets | Brief requires continuing from Week 6 work and not selecting a new dataset | ML Engineering |
