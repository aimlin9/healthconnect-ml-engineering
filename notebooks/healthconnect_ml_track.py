# %% [markdown]
# # HealthConnect Rwanda — AI-Powered Clinical Risk Prediction & Healthcare Intelligence System
# **AnalystLab Africa Experience Lab Internship — Week 6, Machine Learning Engineering Track**
#
# This notebook implements the full ML track brief: data integration, EDA, statistical
# analysis, feature engineering review, classification models (admission, satisfaction,
# claim approval, high-risk), regression models (billing amount, length of stay),
# clustering (patient + facility segmentation), anomaly detection, explainability
# (feature importance + SHAP), model comparison, and a conceptual deployment write-up
# (see `docs/deployment_architecture.md` and `docs/responsible_ai.md`).
#
# All data is the synthetic, educational HealthConnect Rwanda dataset provided in the
# Week 6 brief. See `reports/data_quality_report.md` for a full account of data-quality
# findings, including the key scope finding that patient/appointment-level modeling is
# based on the 5,000 appointments with full dimensional context (of 60,000 total).

# %%
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import (GradientBoostingRegressor, IsolationForest,
                               RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, mean_absolute_error,
                              mean_squared_error, precision_score,
                              r2_score, recall_score, roc_auc_score,
                              silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
FIG_DIR = "../reports/figures"
import os
os.makedirs(FIG_DIR, exist_ok=True)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

print("XGBoost available:", HAS_XGB)
print("SHAP available:", HAS_SHAP)

# %% [markdown]
# ## 1. Load the feature-engineered dataset

# %%
df = pd.read_csv("../data/feature_engineered/healthconnect_features.csv",
                  parse_dates=["Date", "DateOfBirth"])
print(df.shape)
df.head(3)

# %% [markdown]
# ## 2. Exploratory Data Analysis
# ### 2.1 Patient demographics

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df["Gender"].value_counts().plot.bar(ax=axes[0], title="Gender Distribution")
axes[0].set_ylabel("Patients (appointments)")
df["AgeGroup"].value_counts().reindex(["0-17", "18-35", "36-50", "51-65", "65+"]).plot.bar(
    ax=axes[1], title="Age Group Distribution", color="teal")
df["BloodGroup"].value_counts().plot.bar(ax=axes[2], title="Blood Group Distribution", color="indianred")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_demographics.png", dpi=120)
plt.show()

# %%
print("Insurance coverage rate: {:.1f}%".format(df["HasInsurance"].mean() * 100))
print("Chronic condition prevalence: {:.1f}%".format(df["HasChronicCondition"].mean() * 100))
print("\nChronic condition breakdown:")
print(df["ChronicCondition"].value_counts())

# %% [markdown]
# ### 2.2 Clinical patterns

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df["Diagnosis"].value_counts().plot.barh(ax=axes[0], title="Diagnosis Frequency")
df["Severity"].value_counts().plot.bar(ax=axes[1], title="Severity Distribution", color="darkorange")
df["VisitType"].value_counts().plot.bar(ax=axes[2], title="Visit Type", color="slateblue")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_clinical_patterns.png", dpi=120)
plt.show()

# %%
print("Admission rate: {:.1f}%".format(df["WasAdmitted"].mean() * 100))
print("Referral rate: {:.1f}%".format(df["WasReferred"].mean() * 100))
print("Lab test rate: {:.1f}%".format(df["HadLabTest"].mean() * 100))
print("Prescription rate: {:.1f}%".format(df["HadPrescription"].mean() * 100))

# %% [markdown]
# ### 2.3 Financial patterns

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["Amount"], bins=40, ax=axes[0], color="seagreen")
axes[0].set_title("Billing Amount Distribution")
df["BillingStatus"].value_counts().plot.bar(ax=axes[1], title="Billing Status", color="salmon")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_financial.png", dpi=120)
plt.show()

print("Average billing amount: {:.2f}".format(df["Amount"].mean()))
print("\nPayment method distribution:")
print(df["PaymentMethod"].value_counts(dropna=False))
print("\nInsurance claim status distribution:")
print(df["ClaimStatus"].value_counts(dropna=False))

# %% [markdown]
# ### 2.4 Patient experience

# %%
fb = df.loc[df["GaveFeedback"] == 1]
fig, ax = plt.subplots(figsize=(6, 4))
fb["Rating"].value_counts().sort_index().plot.bar(ax=ax, color="goldenrod")
ax.set_title(f"Satisfaction Rating Distribution (n={len(fb)})")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_satisfaction.png", dpi=120)
plt.show()
print("Average satisfaction rating: {:.2f} / 5".format(fb["Rating"].mean()))

# %% [markdown]
# ## 3. Statistical Analysis
# ### 3.1 Descriptive statistics

# %%
desc = df[["Age", "Amount", "DoctorYearsExperience", "DepartmentWorkload"]].describe().T
desc["variance"] = df[["Age", "Amount", "DoctorYearsExperience", "DepartmentWorkload"]].var()
desc

# %% [markdown]
# ### 3.2 Correlation analysis

# %%
corr_cols = ["Age", "Amount", "SeverityScore", "DepartmentWorkload", "DoctorWorkload",
             "PreviousAppointmentsCount", "PriorAdmissionsCount", "WasAdmitted"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap — Key Numeric Features")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_correlation_heatmap.png", dpi=120)
plt.show()

# %% [markdown]
# ### 3.3 Hypothesis testing
#
# **Test 1 — Billing amount: Severe vs. Mild diagnosis severity**
# (All 20,000 patients carry an `InsuranceID` — insurance coverage is 100% in this
# dataset, so the brief's "insured vs. uninsured" comparison has no uninsured group
# to compare against; this is substituted with a comparably structured test.)
#
# H0: mean billing amount is equal between Severe and Mild severity appointments.
# H1: mean billing amount differs.

# %%
severe = df.loc[df["Severity"] == "Severe", "Amount"]
mild = df.loc[df["Severity"] == "Mild", "Amount"]
t_stat, p_val = stats.ttest_ind(severe, mild, equal_var=False)
print(f"Severe mean={severe.mean():.2f}, Mild mean={mild.mean():.2f}")
print(f"t-statistic={t_stat:.3f}, p-value={p_val:.4f}")
print("Conclusion:", "Reject H0 — significant difference (alpha=0.05)" if p_val < 0.05
      else "Fail to reject H0 — no significant difference (alpha=0.05)")
print("(Insurance coverage rate in this dataset: {:.1f}% — effectively universal, so no insured/uninsured split exists.)".format(df["HasInsurance"].mean() * 100))

# %% [markdown]
# **Test 2 — Patient satisfaction across facilities**
# H0: mean satisfaction rating is equal across facilities.
# H1: at least one facility differs.

# %%
fb_by_facility = [g["Rating"].values for _, g in fb.groupby("DeptFacilityID") if len(g) >= 5]
f_stat, p_val2 = stats.f_oneway(*fb_by_facility)
print(f"ANOVA F-statistic={f_stat:.3f}, p-value={p_val2:.4f}")
print("Conclusion:", "Reject H0 — satisfaction differs significantly across facilities" if p_val2 < 0.05
      else "Fail to reject H0 — no significant difference across facilities")

# %% [markdown]
# **Test 3 — Admission rate across age groups**
# H0: admission rate is independent of age group.
# H1: admission rate depends on age group.

# %%
ct = pd.crosstab(df["AgeGroup"], df["WasAdmitted"])
chi2, p_val3, dof, expected = stats.chi2_contingency(ct)
print(ct)
print(f"\nChi2={chi2:.3f}, dof={dof}, p-value={p_val3:.4f}")
print("Conclusion:", "Reject H0 — admission rate depends on age group" if p_val3 < 0.05
      else "Fail to reject H0 — admission rate is independent of age group")

# %% [markdown]
# ## 4. Part B — Classification Models
# Shared helper for train/test split + multi-model comparison.

# %%
results_summary = []

def evaluate_classifier(name, business_problem, y_true, y_pred, y_proba=None, average="binary"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    auc = None
    if y_proba is not None and average == "binary":
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = None
    majority_baseline = pd.Series(y_true).value_counts(normalize=True).max()
    print(f"--- {name} ({business_problem}) ---")
    print(f"Majority-class baseline accuracy: {majority_baseline:.3f}  |  Model accuracy: {acc:.3f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    results_summary.append({
        "algorithm": name, "business_problem": business_problem,
        "accuracy": round(acc, 3), "baseline_accuracy": round(majority_baseline, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3), "f1": round(f1, 3),
        "roc_auc": round(auc, 3) if auc is not None else None,
    })
    return cm

# %% [markdown]
# ### 4.1 Model 1 — Hospital Admission Prediction

# %%
m1_features = ["Age", "Gender", "HasInsurance", "HasChronicCondition", "VisitType",
               "Severity", "DoctorYearsExperience", "DepartmentWorkload", "DoctorWorkload",
               "PreviousAppointmentsCount", "PriorAdmissionsCount"]
m1 = df[m1_features + ["WasAdmitted"]].dropna()
X1 = pd.get_dummies(m1[m1_features], drop_first=True)
y1 = m1["WasAdmitted"]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=42, stratify=y1)

log_reg1 = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X1_train, y1_train)
evaluate_classifier("Logistic Regression", "Hospital Admission Prediction",
                     y1_test, log_reg1.predict(X1_test), log_reg1.predict_proba(X1_test)[:, 1])

rf1 = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42).fit(X1_train, y1_train)
evaluate_classifier("Random Forest", "Hospital Admission Prediction",
                     y1_test, rf1.predict(X1_test), rf1.predict_proba(X1_test)[:, 1])

if HAS_XGB:
    xgb1 = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              eval_metric="logloss", scale_pos_weight=(y1_train == 0).sum() / max((y1_train == 1).sum(), 1),
                              random_state=42).fit(X1_train, y1_train)
    evaluate_classifier("XGBoost", "Hospital Admission Prediction",
                         y1_test, xgb1.predict(X1_test), xgb1.predict_proba(X1_test)[:, 1])
    best_admission_model, best_admission_X_test = xgb1, X1_test
else:
    best_admission_model, best_admission_X_test = rf1, X1_test

# %% [markdown]
# ### 4.2 Model 2 — Patient Satisfaction Prediction (Low / Medium / High)

# %%
m2 = df.loc[df["GaveFeedback"] == 1].copy()
m2["SatisfactionClass"] = pd.cut(m2["Rating"], bins=[0, 2, 3, 5], labels=["Low", "Medium", "High"])
m2_features = ["Age", "Gender", "VisitType", "Severity", "WasAdmitted", "DoctorYearsExperience",
               "DepartmentWorkload", "BillingStatus", "HasInsurance"]
m2d = m2[m2_features + ["SatisfactionClass"]].dropna()
X2 = pd.get_dummies(m2d[m2_features], drop_first=True)
le2 = LabelEncoder()
y2 = le2.fit_transform(m2d["SatisfactionClass"])
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=42, stratify=y2)

rf2 = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42).fit(X2_train, y2_train)
evaluate_classifier("Random Forest", "Patient Satisfaction Prediction", y2_test, rf2.predict(X2_test), average="macro")
print("Classes:", list(le2.classes_))

# %% [markdown]
# ### 4.3 Model 3 — Insurance Claim Approval Prediction
# Note: the source data contains only `Approved` / `Pending` claim statuses (no
# `Rejected` claims), so this is a binary classification task rather than the
# three-class problem described in the brief — documented in
# `reports/data_quality_report.md`.

# %%
m3 = df.loc[df["HasInsuranceClaim"] == 1].copy()
m3_features = ["ClaimAmount", "Amount", "Provider", "PaymentMethod", "VisitType", "Severity"]
m3d = m3[m3_features + ["ClaimStatus"]].dropna()
X3 = pd.get_dummies(m3d[m3_features], drop_first=True)
y3 = (m3d["ClaimStatus"] == "Approved").astype(int)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.25, random_state=42, stratify=y3)

rf3 = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42).fit(X3_train, y3_train)
evaluate_classifier("Random Forest", "Insurance Claim Approval Prediction",
                     y3_test, rf3.predict(X3_test), rf3.predict_proba(X3_test)[:, 1])
importances3 = pd.Series(rf3.feature_importances_, index=X3.columns).sort_values(ascending=False)
print("\nTop features:\n", importances3.head(8))

# %% [markdown]
# ### 4.4 Model 4 — High-Risk Patient Classification
# `HighRiskPatient` is a constructed clinical proxy label (age, chronic condition,
# severity, admission history) — see `reports/feature_dictionary.md`.

# %%
m4_features = ["Age", "Gender", "HasInsurance", "VisitType", "DoctorYearsExperience",
               "PreviousAppointmentsCount", "DepartmentWorkload"]
m4 = df[m4_features + ["HighRiskPatient"]].dropna()
X4 = pd.get_dummies(m4[m4_features], drop_first=True)
y4 = m4["HighRiskPatient"]
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.25, random_state=42, stratify=y4)

rf4 = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42).fit(X4_train, y4_train)
evaluate_classifier("Random Forest", "High-Risk Patient Classification",
                     y4_test, rf4.predict(X4_test), rf4.predict_proba(X4_test)[:, 1])

# %% [markdown]
# ## 5. Part C — Regression Models

# %%
reg_results = []

def evaluate_regressor(name, business_problem, y_true, y_pred, y_train_for_baseline=None):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
    baseline_rmse = None
    if y_train_for_baseline is not None:
        baseline_pred = np.full_like(y_true, fill_value=y_train_for_baseline.mean(), dtype=float)
        baseline_rmse = mean_squared_error(y_true, baseline_pred) ** 0.5
    print(f"--- {name} ({business_problem}) ---")
    if baseline_rmse is not None:
        print(f"Mean-prediction baseline RMSE: {baseline_rmse:.2f}  |  Model RMSE: {rmse:.2f}")
    print(f"RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.3f} MAPE={mape:.1f}%")
    reg_results.append({"algorithm": name, "business_problem": business_problem,
                         "rmse": round(rmse, 2), "baseline_rmse": round(baseline_rmse, 2) if baseline_rmse else None,
                         "mae": round(mae, 2),
                         "r2": round(r2, 3), "mape_pct": round(mape, 1)})

# %% [markdown]
# ### 5.1 Model 5 — Revenue (Billing Amount) Prediction

# %%
m5_features = ["Age", "Gender", "HasInsurance", "VisitType", "Severity",
               "DoctorYearsExperience", "DepartmentWorkload", "WasAdmitted"]
m5 = df[m5_features + ["Amount"]].dropna()
X5 = pd.get_dummies(m5[m5_features], drop_first=True)
y5 = m5["Amount"]
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.25, random_state=42)

lin5 = LinearRegression().fit(X5_train, y5_train)
evaluate_regressor("Linear Regression", "Revenue Prediction", y5_test, lin5.predict(X5_test), y5_train)

rf5 = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42).fit(X5_train, y5_train)
evaluate_regressor("Random Forest Regressor", "Revenue Prediction", y5_test, rf5.predict(X5_test), y5_train)

gb5 = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42).fit(X5_train, y5_train)
evaluate_regressor("Gradient Boosting Regressor", "Revenue Prediction", y5_test, gb5.predict(X5_test), y5_train)

# %% [markdown]
# ### 5.2 Model 6 — Length of Stay Prediction

# %%
m6_features = ["Age", "Gender", "HasChronicCondition", "Severity", "VisitType", "DoctorYearsExperience"]
m6 = df.loc[df["WasAdmitted"] == 1, m6_features + ["LengthOfStayDays"]].dropna()
print("Admitted-patient sample size for LOS model:", len(m6))
X6 = pd.get_dummies(m6[m6_features], drop_first=True)
y6 = m6["LengthOfStayDays"]
X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size=0.25, random_state=42)

rf6 = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42).fit(X6_train, y6_train)
evaluate_regressor("Random Forest Regressor", "Length of Stay Prediction", y6_test, rf6.predict(X6_test), y6_train)

# %% [markdown]
# ## 6. Part D — Clustering
# ### 6.1 Patient Segmentation

# %%
patient_level = (
    df.groupby("PatientID")
    .agg(Age=("Age", "first"), HasChronicCondition=("HasChronicCondition", "first"),
         VisitCount=("AppointmentID", "count"), TotalBilling=("Amount", "sum"),
         HasInsurance=("HasInsurance", "first"), AvgSeverity=("SeverityScore", "mean"))
    .dropna()
)
scaler_p = StandardScaler()
Xp = scaler_p.fit_transform(patient_level)
kmeans_p = KMeans(n_clusters=4, random_state=42, n_init=10).fit(Xp)
patient_level["Cluster"] = kmeans_p.labels_
print("Patient segmentation silhouette score:", round(silhouette_score(Xp, kmeans_p.labels_), 3))
print(patient_level.groupby("Cluster").mean(numeric_only=True))

# %% [markdown]
# ### 6.2 Facility Segmentation

# %%
facility_agg = pd.read_csv("../data/processed/facility_aggregates.csv").dropna()
fac_features = ["appointment_count", "admission_rate", "avg_satisfaction", "total_revenue",
                 "unique_doctors", "staff_count", "bed_count"]
Xf = StandardScaler().fit_transform(facility_agg[fac_features])
kmeans_f = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xf)
facility_agg["Cluster"] = kmeans_f.labels_
print("Facility segmentation silhouette score:", round(silhouette_score(Xf, kmeans_f.labels_), 3))
facility_agg.groupby("Cluster")[fac_features].mean()

# %% [markdown]
# **Interpretation.** Patient segmentation (silhouette ≈ 0.27) separates four groups
# primarily along age and chronic-condition lines: a younger chronic-condition cluster
# (Cluster 1, avg age ~37, 100% chronic), an older chronic-condition cluster
# (Cluster 2, avg age ~53), a non-chronic cluster (Cluster 0), and a smaller
# frequent-visitor/high-billing cluster (Cluster 3, ~2 visits and markedly higher total
# billing). **Recommendation**: route Clusters 1 and 2 into a chronic-disease
# management programme (proactive follow-up, medication adherence outreach), and flag
# Cluster 3 for case management given its disproportionate utilization and spend.
#
# Facility segmentation (silhouette ≈ 0.20, 3 clusters over only 20 facilities — a
# small-N result to treat directionally) groups facilities mainly by appointment
# volume and revenue rather than by admission rate or satisfaction, which are nearly
# flat across clusters (consistent with the "no signal" finding in the Data Quality
# Report). **Recommendation**: use this segmentation for *capacity* planning (which
# facilities are high-volume/high-revenue and may need additional staffing or beds),
# not for quality benchmarking — the data does not support distinguishing facilities
# on quality in its current form.

# %% [markdown]
# ## 7. Part E — Anomaly Detection

# %%
iso = IsolationForest(contamination=0.03, random_state=42)
billing_feats = df[["Amount"]].dropna()
iso_labels = iso.fit_predict(billing_feats)
n_anomalous_billing = int((iso_labels == -1).sum())
print(f"Flagged {n_anomalous_billing} unusually high/low billing amounts out of {len(billing_feats)} ({n_anomalous_billing/len(billing_feats)*100:.1f}%)")

claims_feats = df.loc[df["HasInsuranceClaim"] == 1, ["ClaimAmount"]].dropna()
iso_c = IsolationForest(contamination=0.03, random_state=42)
iso_c_labels = iso_c.fit_predict(claims_feats)
n_anomalous_claims = int((iso_c_labels == -1).sum())
print(f"Flagged {n_anomalous_claims} suspicious insurance claims out of {len(claims_feats)} ({n_anomalous_claims/len(claims_feats)*100:.1f}%)")

dup_phones = int((20000 - df["PatientID"].nunique()))  # placeholder; see data quality report for duplicate phone finding
print("Duplicate patient records: see reports/data_quality_report.md (duplicate phone numbers = 19,990/20,000)")

fac_z = (facility_agg["appointment_count"] - facility_agg["appointment_count"].mean()) / facility_agg["appointment_count"].std()
unusual_facilities = facility_agg.loc[fac_z.abs() > 1.5, ["FacilityID", "appointment_count"]]
print("\nFacilities with unusually low/high patient volume (|z|>1.5):")
print(unusual_facilities)

# %% [markdown]
# **Discussion.** Isolation Forest flags the top ~3% most statistically unusual
# billing amounts and insurance claim amounts — a first-pass triage layer, not a
# fraud verdict. In production this would route flagged transactions to a billing
# auditor rather than auto-rejecting them (consistent with the Responsible AI
# principle of human oversight in `docs/responsible_ai.md`). Combined with the
# duplicate-phone-number finding (19,990/20,000 patients share a phone number with
# another patient — see Data Quality Report) and the facility-volume outliers above,
# anomaly detection here serves two purposes: (1) a fraud/billing-integrity triage
# signal, and (2) a continuous data-quality monitor that catches record-linkage
# problems (like the phone-number collisions) before they corrupt downstream
# reporting.

# %% [markdown]
# ## 8. Part F — Model Explainability

# %%
importances1 = pd.Series(rf1.feature_importances_, index=X1.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 5))
importances1.head(10).plot.barh(ax=ax, color="steelblue")
ax.invert_yaxis()
ax.set_title("Top 10 Feature Importances — Admission Prediction (Random Forest)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_feature_importance_admission.png", dpi=120)
plt.show()

# %%
if HAS_SHAP:
    explainer = shap.TreeExplainer(rf1)
    shap_values = explainer.shap_values(X1_test)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(sv, X1_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/07_shap_summary_admission.png", dpi=120)
    plt.show()
else:
    print("SHAP not installed in this environment — feature importance above substitutes.")

# %% [markdown]
# **Why this matters for trust.** In a clinical setting, a bare risk score ("82%
# admission risk") is not actionable and is hard for a clinician to trust or
# challenge. Surfacing *which* features drove a given prediction (age, prior
# admissions, department workload, etc. — via feature importance globally and SHAP
# per-prediction) lets a clinician or administrator sanity-check the model against
# their own judgment, challenge a prediction that rests on an implausible driver, and
# meets the Responsible AI commitment in `docs/responsible_ai.md` that predictions
# are decision-support, not a black-box verdict. It also gives the model-governance
# process (Part H) something concrete to audit when performance drifts.

# %% [markdown]
# ## 9. Part G — Model Comparison

# %%
summary_df = pd.DataFrame(results_summary)
summary_df

# %%
reg_summary_df = pd.DataFrame(reg_results)
reg_summary_df

# %%
summary_df.to_csv("../models/classification_model_comparison.csv", index=False)
reg_summary_df.to_csv("../models/regression_model_comparison.csv", index=False)
print("Saved model comparison tables to models/")

# %% [markdown]
# **Final model selection.** Every classification model here performs at or only
# modestly above its majority-class baseline, and every regression model performs at
# or below a naive mean-prediction baseline (negative R² throughout) — see the
# "Signal Check" section of `reports/data_quality_report.md` for why: the outcome
# fields do not vary meaningfully with the clinical/operational fields in this
# synthetic dataset. The one exception is **High-Risk Patient Classification**
# (Random Forest, 74.6% accuracy vs. 57.6% baseline, ROC-AUC 0.73) — expected, since
# its target is a composite built directly from the same clinical fields used as
# model inputs, so the model is recovering a learnable rule rather than discovering
# a hidden pattern in noise.
#
# **Recommended for deployment (of these candidates): the High-Risk Patient
# classifier**, as the only model with evidence of real predictive lift, with the
# explicit caveat that it formalizes an existing clinical rule rather than
# discovering new risk factors. The other five models are recommended to be
# **re-trained once real (non-synthetic) HealthConnect Rwanda data is available**,
# since the pipeline, feature engineering and evaluation harness built here are
# reusable as-is — only the data's actual signal content needs to improve, not the
# approach. The full comparison table above and the Data Quality Report together
# make this a founded, evidence-based recommendation rather than an inflated claim.

# %% [markdown]
# ## 10. Summary of Findings (feeds the Executive Summary / Business Recommendations)

# %%
findings = {
    "admission_rate_pct": round(float(df["WasAdmitted"].mean()) * 100, 1),
    "avg_satisfaction": round(float(fb["Rating"].mean()), 2),
    "avg_billing_amount": round(float(df["Amount"].mean()), 2),
    "pct_uninsured": round(float((1 - df["HasInsurance"].mean()) * 100), 1),
    "high_risk_patient_pct": round(float(df["HighRiskPatient"].mean()) * 100, 1),
    "anomalous_billing_pct": round(n_anomalous_billing / len(billing_feats) * 100, 1),
    "anomalous_claims_pct": round(n_anomalous_claims / len(claims_feats) * 100, 1),
}
print(json.dumps(findings, indent=2))
with open("../reports/key_findings.json", "w") as f:
    json.dump(findings, f, indent=2)
