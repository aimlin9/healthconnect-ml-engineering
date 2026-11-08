"""
Week 8 — Apply the Data Science release gate to the High-Risk Patient model.

Gate agreed with Data Science (E. E. Arhin, 8 Nov 2026), on a locked hold-out set:
  G1  ROC-AUC >= 0.60
  G2  lower bound of the 95% bootstrap CI for AUC > 0.50
  G3  label-shuffle test p < 0.05 (retrain on shuffled training labels, score the hold-out)
  G4  lift > 1.3 at the flag budget (20% reference; lift also reported at 10% and 30%)

Locked hold-out = the 25% stratified test split (seed 42). The committed model was trained
on the other 75% only, and its threshold was tuned on a split carved from that 75%, so the
hold-out has never been used for fitting or tuning.

Also answers the DS question "does the model only restate the rule that builds its label?":
the label is (Age>=65) + HasChronicCondition + (SeverityScore>=3) + (PriorAdmissionsCount>=1) >= 2,
and Age is also a model input. The script compares the model with an Age-only model.

Usage: python scripts/week8_release_gate.py [--shuffles 100]
Writes reports/week8/evidence/release_gate_high_risk.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hc_pipeline import config, load  # noqa: E402
from hc_pipeline.features import add_prior_appointments  # noqa: E402
from hc_pipeline.pipeline import build_pipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shuffles", type=int, default=100)
ap.add_argument("--bootstrap", type=int, default=2000)
args = ap.parse_args()

df = add_prior_appointments(pd.read_csv(config.FEATURES_CSV))
X, y = df[config.FEATURES], df[config.TARGET].to_numpy()
idx_train, idx_test = train_test_split(df.index, test_size=config.TEST_SIZE,
                                       random_state=config.RANDOM_STATE, stratify=y)
y_tr, y_te = y[idx_train], y[idx_test]
model, meta = load()
p_te = model.predict_proba(X.loc[idx_test])[:, 1]
rng = np.random.default_rng(config.RANDOM_STATE)

# G1/G2: AUC with bootstrap CI
auc = roc_auc_score(y_te, p_te)
boots = []
for _ in range(args.bootstrap):
    i = rng.integers(0, len(y_te), len(y_te))
    if y_te[i].min() != y_te[i].max():
        boots.append(roc_auc_score(y_te[i], p_te[i]))
ci = np.percentile(boots, [2.5, 97.5])

# G3: label-shuffle test (refit on permuted training labels)
null = []
for k in range(args.shuffles):
    m = clone(build_pipeline()).set_params(model__n_jobs=-1, model__random_state=k)
    m.fit(X.loc[idx_train], rng.permutation(y_tr))
    null.append(roc_auc_score(y_te, m.predict_proba(X.loc[idx_test])[:, 1]))
null = np.array(null)
p_shuffle = (1 + int((null >= auc).sum())) / (len(null) + 1)


# G4: lift at flag budgets
def lift_at(scores, frac):
    k = max(1, int(round(frac * len(scores))))
    top = np.argsort(-scores, kind="stable")[:k]
    return float(y_te[top].mean() / y_te.mean()), float(y_te[top].mean())


lifts = {f"{int(f*100)}%": dict(zip(("lift", "precision"), map(lambda v: round(v, 3), lift_at(p_te, f))))
         for f in (0.10, 0.20, 0.30)}

# Rule-restatement check
age_rf = RandomForestClassifier(**config.RF_PARAMS).fit(df.loc[idx_train, ["Age"]], y_tr)
p_age = age_rf.predict_proba(df.loc[idx_test, ["Age"]])[:, 1]
age65 = (df.loc[idx_test, "Age"] >= 65).astype(int).to_numpy()
components = {
    "Age>=65": (df["Age"] >= 65),
    "HasChronicCondition": df["HasChronicCondition"] == 1,
    "SeverityScore>=3": df["SeverityScore"] >= 3,
    "PriorAdmissionsCount>=1": df["PriorAdmissionsCount"] >= 1,
}
restate = {
    "label_definition": "HighRiskPatient = [(Age>=65) + HasChronicCondition + (SeverityScore>=3) + (PriorAdmissionsCount>=1)] >= 2",
    "model_inputs_in_label_rule": [c for c in config.FEATURES if c == "Age"],
    "label_components_not_model_inputs": ["HasChronicCondition", "SeverityScore", "PriorAdmissionsCount"],
    "share_of_rows_with_component": {k: round(float(v.mean()), 3) for k, v in components.items()},
    "holdout_auc": {"model_all_6_inputs": round(auc, 3),
                    "random_forest_on_Age_only": round(roc_auc_score(y_te, p_age), 3),
                    "rule_component_Age>=65_alone": round(roc_auc_score(y_te, age65), 3)},
    "lift_at_20pct": {"model_all_6_inputs": lifts["20%"]["lift"],
                      "random_forest_on_Age_only": round(lift_at(p_age, 0.20)[0], 3)},
}

gate = {
    "G1_auc>=0.60": bool(auc >= 0.60),
    "G2_ci_lower>0.50": bool(ci[0] > 0.50),
    "G3_shuffle_p<0.05": bool(p_shuffle < 0.05),
    "G4_lift@20%>1.3": bool(lifts["20%"]["lift"] > 1.3),
}
result = {
    "model_version": meta["model_version"],
    "target": config.TARGET,
    "locked_holdout": {"n": int(len(y_te)), "positive_rate": round(float(y_te.mean()), 3),
                       "definition": "25% stratified split, seed 42; never used for training or threshold tuning"},
    "metrics": {"roc_auc": round(auc, 3), "auc_95ci": [round(ci[0], 3), round(ci[1], 3)],
                "shuffle_test": {"n_shuffles": len(null), "null_auc_mean": round(float(null.mean()), 3),
                                 "null_auc_max": round(float(null.max()), 3), "p_value": round(p_shuffle, 4)},
                "lift": lifts},
    "gate": gate,
    "gate_passed": all(gate.values()),
    "rule_restatement_check": restate,
}
out = ROOT / "reports" / "week8" / "evidence" / "release_gate_high_risk.json"
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
