# LinkedIn & X Posts — Week 6

Fill in the GitHub repo link once pushed: `https://github.com/aimlin9/healthconnect-ml-engineering`

Suggested screenshots to attach (both platforms): the SHAP summary plot
(`reports/figures/07_shap_summary_admission.png`), the confusion matrix
(`reports/figures/08_confusion_matrix_highrisk.png`), and a VS Code screenshot of the
executed notebook (`notebooks/healthconnect_ml_track.ipynb`) open with outputs
visible.

---

## LinkedIn

Week 6 of the AnalystLab Africa Experience Lab Internship (Machine Learning
Engineering track) is done: an end-to-end ML pipeline for HealthConnect Rwanda, a
simulated multi-facility healthcare network.

What I built:
→ Integrated 27 source datasets (patients, appointments, diagnoses, billing,
insurance claims, staff, facilities) into one validated, ML-ready dataset
→ Engineered 75 features (prior-visit history, department/doctor workload,
clinical risk scoring, revenue categories, and more)
→ Built and evaluated 4 classification models and 2 regression models —
hospital admission, patient satisfaction, insurance claim approval, high-risk
patient flagging, billing prediction, and length-of-stay prediction
→ Added K-Means clustering for patient and facility segmentation, Isolation
Forest for anomaly detection, and SHAP for model explainability
→ Wrote up a conceptual deployment architecture and a Responsible AI section
covering privacy, bias, human oversight and governance

The part I'm most glad I did: every model is reported against an honest baseline,
not just a raw accuracy number. Most of the targets in this dataset turned out to
carry very little real signal relative to the clinical/operational features — and
rather than overstate the results, I measured that, documented exactly why (with
hypothesis tests), and built the one model that *did* show real lift (a high-risk
patient classifier, 74.6% accuracy vs. a 57.6% baseline) into the recommended
deployment path. The pipeline itself is fully reusable the moment real signal is
available.

Full notebook, reports and code: [repo link]

#AnalystLabAfrica #MachineLearning #HealthTech #DataScience #Rwanda

cc @AnalystLab Africa

---

## X (Twitter)

Week 6 @analystlabafric ML track: pipeline for a simulated Rwandan hospital network
— 27 datasets, 6 models, clustering, SHAP. Benchmarked every model vs. baseline —
only 1 showed real lift, and that's the one I'd deploy.

[repo link]
#AnalystLabAfrica
