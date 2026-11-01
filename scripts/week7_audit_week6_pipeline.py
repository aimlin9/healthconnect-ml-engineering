"""
Week 7 — Baseline audit of the Week 6 High-Risk Patient inference path.

Reproduces exactly how the Week 6 notebook (section 4.4) prepared features and
scored patients (`pd.get_dummies(..., drop_first=True)` + RandomForest), then
runs a fixed set of probes against it: realistic inference inputs, invalid
inputs, persistence, reproducibility and feature-timing checks.

Nothing here is fixed — this is the "before" evidence. Results are written to
reports/week7/evidence/week6_audit_results.json and printed as a table.

Usage: python scripts/week7_audit_week6_pipeline.py
"""
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "week7" / "evidence"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "feature_engineered" / "healthconnect_features.csv",
                 parse_dates=["Date"])

# --- Week 6 notebook, section 4.4, verbatim ---------------------------------
m4_features = ["Age", "Gender", "HasInsurance", "VisitType", "DoctorYearsExperience",
               "PreviousAppointmentsCount", "DepartmentWorkload"]
m4 = df[m4_features + ["HighRiskPatient"]].dropna()
X4 = pd.get_dummies(m4[m4_features], drop_first=True)
y4 = m4["HighRiskPatient"]
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.25, random_state=42, stratify=y4)
rf4 = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42).fit(X4_train, y4_train)


def week6_score(rows):
    """Score new rows the only way the Week 6 code allows: re-run get_dummies."""
    X = pd.get_dummies(rows[m4_features], drop_first=True)
    return rf4.predict_proba(X)[:, 1]


raw_test = m4.loc[X4_test.index, m4_features]
results = []


def probe(test_id, area, scenario, expected, fn):
    """Run one probe. fn returns (passed: bool, actual: str)."""
    try:
        passed, actual = fn()
    except Exception as exc:  # the failure itself is the evidence
        passed, actual = False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    results.append({"id": test_id, "area": area, "scenario": scenario,
                    "expected": expected, "actual": actual, "result": "PASS" if passed else "FAIL"})


def p_artifact():
    found = sorted(p.name for p in (ROOT / "models").glob("*") if p.suffix in {".joblib", ".pkl", ".pickle"})
    return bool(found), f"model files found: {found or 'none'} — model exists only in notebook memory"
probe("W6-01", "Persistence", "A trained model artifact exists that a service could load",
      "Serialized model + metadata in models/", p_artifact)


def p_batch():
    p = week6_score(raw_test)
    return len(p) == len(raw_test), f"{len(p)} probabilities for {len(raw_test)} mixed-category rows"
probe("W6-02", "Model integration", "Score a large batch containing every category",
      "One probability per row", p_batch)


def p_single():
    p = week6_score(raw_test.iloc[[0]])
    return True, f"probability {p[0]:.3f}"
probe("W6-03", "Preprocessing", "Score a single patient (the real-time use case)",
      "One probability", p_single)


def p_one_category():
    rows = raw_test[raw_test["VisitType"] == "Emergency"].head(20)
    p = week6_score(rows)
    return True, f"{len(p)} probabilities"
probe("W6-04", "Preprocessing", "Score a batch where every patient is an Emergency visit",
      "One probability per row", p_one_category)


def p_unseen():
    rows = raw_test.head(5).copy()
    rows["VisitType"] = "Telehealth"
    p = week6_score(rows)
    return False, f"returned {len(p)} probabilities with no warning"
probe("W6-05", "Input validation", "Unseen category (VisitType='Telehealth')",
      "Clear rejection naming the bad value", p_unseen)


def p_missing_col():
    week6_score(raw_test.head(5).drop(columns=["DepartmentWorkload"]))
    return False, "scored without the column"
probe("W6-06", "Input validation", "Required column missing (DepartmentWorkload)",
      "Clear error naming the missing column", p_missing_col)


def p_string_age():
    rows = raw_test.head(5).copy()
    rows["Age"] = rows["Age"].astype(str)
    week6_score(rows)
    return False, "scored with Age as text"
probe("W6-07", "Input validation", "Age supplied as text (e.g. '45' from a CSV/form)",
      "Coerced to number, or clear type error", p_string_age)


def p_out_of_range():
    rows = raw_test.copy()  # full batch so get_dummies sees every category (isolates W6-03)
    rows.iloc[:2, rows.columns.get_loc("Age")] = [-5, 250]
    p = week6_score(rows)[:2]
    return False, f"silently scored impossible ages: {np.round(p, 3).tolist()}"
probe("W6-08", "Input validation", "Impossible values (Age = -5 and 250)",
      "Rejected with the offending rows", p_out_of_range)


def p_nan():
    rows = raw_test.copy().astype({"Age": float})
    rows.iloc[:2, rows.columns.get_loc("Age")] = np.nan
    p = week6_score(rows)[:2]
    return False, f"silently scored missing Age: {np.round(p, 3).tolist()}"
probe("W6-09", "Input validation", "Missing value (Age = NaN)",
      "Rejected with the offending rows", p_nan)


def p_empty():
    p = week6_score(raw_test.head(0))
    return False, f"returned {len(p)} rows"
probe("W6-10", "Input validation", "Empty input (0 rows)",
      "Clear 'no rows' error", p_empty)


def p_determinism():
    rf_b = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced",
                                  random_state=42).fit(X4_train, y4_train)
    same = np.array_equal(rf4.predict_proba(X4_test), rf_b.predict_proba(X4_test))
    return same, "identical probabilities across two training runs" if same else "probabilities differ"
probe("W6-11", "Reproducibility", "Train twice with the same seed",
      "Identical predictions", p_determinism)


def p_constant():
    const = [c for c in m4_features if df[c].nunique() <= 1]
    return not const, f"constant features: {const or 'none'}"
probe("W6-12", "Data processing", "Every model feature carries information (non-constant)",
      "No constant features", p_constant)


def p_timing():
    d = df.sort_values(["PatientID", "Date", "AppointmentID"])
    point_in_time = d.groupby("PatientID").cumcount()
    leaked = (d["PreviousAppointmentsCount"] != point_in_time)
    return not leaked.any(), (f"{leaked.mean():.1%} of rows count visits that happen AFTER the scored "
                              f"appointment (feature = total visits - 1, not prior visits)")
probe("W6-13", "Data processing", "PreviousAppointmentsCount uses only information available at prediction time",
      "Equals count of the patient's earlier appointments", p_timing)


def p_logging():
    src = (ROOT / "notebooks" / "healthconnect_ml_track.py").read_text()
    has = "import logging" in src
    return has, "structured logging present" if has else "no logging: print() only, no record of inputs rejected or predictions served"
probe("W6-14", "Observability", "Inference produces an execution log",
      "Logged run info and errors", p_logging)


(OUT / "week6_audit_results.json").write_text(json.dumps(results, indent=2))
table = pd.DataFrame(results)[["id", "area", "result", "actual"]]
pd.set_option("display.max_colwidth", 110)
pd.set_option("display.width", 200)
print(table.to_string(index=False))
print(f"\n{(table.result == 'PASS').sum()} / {len(table)} passed")
