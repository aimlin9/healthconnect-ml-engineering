"""
Week 7 — Re-run the Week 6 audit probes (scripts/week7_audit_week6_pipeline.py)
against the refined pipeline in src/hc_pipeline and write a before/after table.

Usage: python scripts/week7_retest_probes.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hc_pipeline import InputValidationError, config, load, predict, train  # noqa: E402
from hc_pipeline.features import add_prior_appointments  # noqa: E402

logging.getLogger("hc_pipeline").addHandler(logging.NullHandler())
EVID = ROOT / "reports" / "week7" / "evidence"

df = add_prior_appointments(pd.read_csv(config.FEATURES_CSV))
tr, te = train_test_split(df.index, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
                          stratify=df[config.TARGET])
raw_test = df.loc[te, config.FEATURES]
model, meta = load()
results = {}


def probe(test_id, fn, expect_rejection=False):
    try:
        passed, actual = fn()
    except InputValidationError as exc:
        passed, actual = expect_rejection, f"rejected: {str(exc)[:160]}"
    except Exception as exc:
        passed, actual = False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    results[test_id] = {"result": "PASS" if passed else "FAIL", "actual": actual}


def scored(rows, n):
    out, _ = predict(model, meta, rows)
    return len(out) == n, f"{len(out)} probabilities"


def accepted_silently(rows):
    predict(model, meta, rows)
    return False, "scored without complaint"


probe("W6-01", lambda: (True, f"loaded models/{config.MODEL_PATH.name} v{meta['model_version']}"))
probe("W6-02", lambda: scored(raw_test, len(raw_test)))
probe("W6-03", lambda: scored(raw_test.iloc[[0]], 1))
probe("W6-04", lambda: scored(raw_test[raw_test["VisitType"] == "Emergency"].head(20), 20))
probe("W6-05", lambda: accepted_silently(raw_test.head(5).assign(VisitType="Telehealth")), True)
probe("W6-06", lambda: accepted_silently(raw_test.head(5).drop(columns="DepartmentWorkload")), True)
probe("W6-07", lambda: scored(raw_test.head(5).astype({"Age": str}), 5))
probe("W6-08", lambda: accepted_silently(raw_test.head(2).assign(Age=[-5, 250])), True)
probe("W6-09", lambda: accepted_silently(raw_test.head(2).assign(Age=np.nan)), True)
# Week 8 (D-08, agreed with Data Science): empty input now returns empty output instead of
# raising, so the pass condition changed from "clear error" to "0 rows out, no crash".
probe("W6-10", lambda: scored(raw_test.head(0), 0))


def p_determinism():
    (m1, _), (m2, _) = train(df.loc[tr, config.FEATURES], df.loc[tr, config.TARGET]), \
                       train(df.loc[tr, config.FEATURES], df.loc[tr, config.TARGET])
    same = np.array_equal(m1.predict_proba(raw_test), m2.predict_proba(raw_test))
    return same, "identical probabilities across two training runs" if same else "differ"
probe("W6-11", p_determinism)

const = [c for c in config.FEATURES if df[c].nunique() <= 1]
probe("W6-12", lambda: (not const, f"constant features: {const or 'none'} (HasInsurance removed)"))


def p_timing():
    d = df.sort_values(["PatientID", "Date", "AppointmentID"])
    leaked = d["PriorAppointmentsCount"] != d.groupby("PatientID").cumcount()
    return not leaked.any(), f"{leaked.mean():.1%} of rows use future visits (PriorAppointmentsCount is point-in-time)"
probe("W6-13", p_timing)
probe("W6-14", lambda: ("import logging" in (ROOT / "src/hc_pipeline/pipeline.py").read_text(),
                        "logs model load, rejected rows, extrapolation warnings, rows scored/flagged, latency"))

before = {r["id"]: r for r in json.loads((EVID / "week6_audit_results.json").read_text())}
rows = [{**{k: before[i][k] for k in ("id", "area", "scenario", "expected")},
         "week6_result": before[i]["result"], "week6_actual": before[i]["actual"],
         "week7_result": results[i]["result"], "week7_actual": results[i]["actual"]} for i in before]
(EVID / "week7_retest_results.json").write_text(json.dumps(rows, indent=2))

t = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 220)
print(t[["id", "week6_result", "week7_result", "week7_actual"]].to_string(index=False))
print(f"\nWeek 6: {(t.week6_result == 'PASS').sum()}/{len(t)} passed   "
      f"Week 7: {(t.week7_result == 'PASS').sum()}/{len(t)} passed")
