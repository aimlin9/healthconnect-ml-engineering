"""
Week 7 — Train, evaluate and persist the refined High-Risk Patient pipeline.

Compares, on the same held-out test split the Week 6 notebook used:
  1. Week 6 model (get_dummies + RF, original features, 0.5 threshold)
  2. Refined pipeline, 0.5 threshold   (feature fixes only)
  3. Refined pipeline, tuned threshold (feature fixes + recall-targeted threshold)
and repeats 1 and 3 with leave-one-facility-out validation. Saves the model to
models/high_risk_pipeline.joblib (+ _meta.json) and results to
reports/week7/evidence/model_refinement_results.json.

Usage: python scripts/03_train_pipeline.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hc_pipeline import config  # noqa: E402
from hc_pipeline.features import add_prior_appointments  # noqa: E402
from hc_pipeline.pipeline import metrics, save, train  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
OUT = ROOT / "reports" / "week7" / "evidence"
OUT.mkdir(parents=True, exist_ok=True)

df = add_prior_appointments(pd.read_csv(config.FEATURES_CSV))
y = df[config.TARGET]
idx_train, idx_test = train_test_split(df.index, test_size=config.TEST_SIZE,
                                       random_state=config.RANDOM_STATE, stratify=y)


def week6_fit_score(train_idx, score_idx):
    X = pd.get_dummies(df[config.WEEK6_FEATURES], drop_first=True)
    rf = RandomForestClassifier(**config.RF_PARAMS).fit(X.loc[train_idx], y.loc[train_idx])
    return rf.predict_proba(X.loc[score_idx])[:, 1]


X = df[config.FEATURES]
results = {"test_split": {}, "leave_one_facility_out": {}}

results["test_split"]["week6_model"] = metrics(y[idx_test], week6_fit_score(idx_train, idx_test), 0.5)
model, threshold = train(X.loc[idx_train], y.loc[idx_train])
proba = model.predict_proba(X.loc[idx_test])[:, 1]
results["test_split"]["refined_threshold_0.5"] = metrics(y[idx_test], proba, 0.5)
results["test_split"]["refined_tuned_threshold"] = metrics(y[idx_test], proba, threshold)

lofo = []
for fac in sorted(df["FacilityID"].unique()):
    held = df["FacilityID"] == fac
    tr, te = df.index[~held], df.index[held]
    w6 = metrics(y[te], week6_fit_score(tr, te), 0.5)
    m, t = train(X.loc[tr], y.loc[tr])
    ref = metrics(y[te], m.predict_proba(X.loc[te])[:, 1], t)
    lofo.append({"FacilityID": fac, "n": int(held.sum()),
                 **{f"week6_{k}": v for k, v in w6.items() if k in ("recall", "precision", "roc_auc")},
                 **{f"refined_{k}": v for k, v in ref.items() if k in ("recall", "precision", "roc_auc", "threshold")}})
lofo = pd.DataFrame(lofo)
results["leave_one_facility_out"]["per_facility"] = lofo.to_dict(orient="records")
results["leave_one_facility_out"]["summary"] = (
    lofo.drop(columns=["FacilityID", "n"]).agg(["min", "median", "max"]).round(3).to_dict())

meta = save(model, threshold, X.loc[idx_train], results["test_split"]["refined_tuned_threshold"])
results["artifact"] = {k: meta[k] for k in ("model_version", "threshold", "sklearn_version",
                                            "train_data_fingerprint", "n_train")}
(OUT / "model_refinement_results.json").write_text(json.dumps(results, indent=2))

print("\nHeld-out test split (n=%d):" % len(idx_test))
print(pd.DataFrame(results["test_split"]).T.to_string())
print("\nLeave-one-facility-out (min / median / max over 20 facilities):")
print(lofo.drop(columns=["FacilityID", "n"]).agg(["min", "median", "max"]).round(3).T.to_string())
