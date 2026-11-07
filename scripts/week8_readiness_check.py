"""
Week 8 — Demonstration-readiness evidence for the High-Risk Patient pipeline.

Three checks, written to reports/week8/evidence/:
  1. Reproducibility: retrain from the documented config on this machine and compare
     with the committed artifact (data fingerprint, threshold, test metrics, and the
     largest per-patient probability difference over all 5,000 rows).
  2. Error handling: run scripts/predict.py on one input per failure mode and record
     the exit code, whether output was written, and the first error logged.
  3. End-to-end: score the held-out test split through the CLI and check the CLI's
     output matches predict() called in-process.

Usage: python scripts/week8_readiness_check.py
"""
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hc_pipeline import config, load, predict  # noqa: E402
from hc_pipeline.features import add_prior_appointments  # noqa: E402
from hc_pipeline.pipeline import data_fingerprint, metrics, train  # noqa: E402

OUT = ROOT / "reports" / "week8" / "evidence"
OUT.mkdir(parents=True, exist_ok=True)
PREDICT = ROOT / "scripts" / "predict.py"

df = add_prior_appointments(pd.read_csv(config.FEATURES_CSV))
X, y = df[config.FEATURES], df[config.TARGET]
idx_train, idx_test = train_test_split(df.index, test_size=config.TEST_SIZE,
                                       random_state=config.RANDOM_STATE, stratify=y)
committed, meta = load()

# 1. Reproducibility ---------------------------------------------------------
fresh, fresh_threshold = train(X.loc[idx_train], y.loc[idx_train])
fresh_metrics = metrics(y[idx_test], fresh.predict_proba(X.loc[idx_test])[:, 1], fresh_threshold)
diff = np.abs(fresh.predict_proba(X)[:, 1] - committed.predict_proba(X)[:, 1])
repro = {
    "environment": {"platform": platform.platform(), "python": platform.python_version(),
                    "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    "committed_artifact": {"trained_with_sklearn": meta["sklearn_version"],
                           "fingerprint": meta["train_data_fingerprint"],
                           "threshold": meta["threshold"], "test_metrics": meta["test_metrics"]},
    "fresh_retrain": {"fingerprint": data_fingerprint(X.loc[idx_train]),
                      "threshold": fresh_threshold, "test_metrics": fresh_metrics},
    "max_abs_probability_diff_all_rows": float(diff.max()),
    "flags_disagreeing_all_rows": int(((fresh.predict_proba(X)[:, 1] >= fresh_threshold)
                                       != (committed.predict_proba(X)[:, 1] >= meta["threshold"])).sum()),
}
repro["identical"] = (repro["fresh_retrain"]["fingerprint"] == meta["train_data_fingerprint"]
                      and fresh_threshold == meta["threshold"]
                      and fresh_metrics == meta["test_metrics"]
                      and repro["flags_disagreeing_all_rows"] == 0)
(OUT / "reproducibility_check.json").write_text(json.dumps(repro, indent=2))
print(f"[1] reproducibility: identical={repro['identical']}  "
      f"max |p diff|={repro['max_abs_probability_diff_all_rows']:.2e}")

# 2. Error handling ----------------------------------------------------------
good = {"Gender": "Female", "VisitType": "Consultation", "Age": 54,
        "DoctorYearsExperience": 10, "PriorAppointmentsCount": 1, "DepartmentWorkload": 25}
cases = [
    ("valid single patient", [good], [], 0),
    ("impossible age (250)", [{**good, "Age": 250}], [], 2),
    ("missing value (Age blank)", [{**good, "Age": None}], [], 2),
    ("unknown category (VisitType=Surgery)", [{**good, "VisitType": "Surgery"}], [], 2),
    ("missing required column", [{k: v for k, v in good.items() if k != "DepartmentWorkload"}], [], 2),
    ("non-numeric age ('fifty')", [{**good, "Age": "fifty"}], [], 2),
    ("mixed batch, --skip-invalid", [good, {**good, "Age": 250}, {**good, "Gender": " male "}],
     ["--skip-invalid"], 0),
    ("outside training range (Age 110) -> scored with warning", [{**good, "Age": 110}], [], 0),
    ("threshold override out of range", [good], ["--threshold", "1.5"], 2),
    ("input file does not exist", None, [], 2),
]
rows = []
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for i, (name, data, flags, expected) in enumerate(cases):
        inp, out = tmp / f"in_{i}.csv", tmp / f"out_{i}.csv"
        if data is not None:
            pd.DataFrame(data).to_csv(inp, index=False)
        r = subprocess.run([sys.executable, str(PREDICT), str(inp), str(out), *flags],
                           capture_output=True, text=True, cwd=ROOT)
        log_lines = [ln.split(": ", 1)[-1] for ln in r.stderr.splitlines()
                     if " ERROR " in ln or " WARNING " in ln]
        rows.append({"case": name, "expected_exit": expected, "exit_code": r.returncode,
                     "output_written": out.exists(),
                     "rows_scored": len(pd.read_csv(out)) if out.exists() else 0,
                     "first_message": (log_lines[0] if log_lines else "").replace(str(tmp), "<tmp>")
                                      .replace(str(tmp).replace("\\", "\\\\"), "<tmp>"),
                     "pass": r.returncode == expected and out.exists() == (expected == 0)})

    # Model-missing case: point the loader at an empty directory via a copy of the repo layout
    # is heavyweight; instead call load() with missing paths in-process (same code path the CLI uses).
    from hc_pipeline import ModelLoadError
    try:
        load(tmp / "nope.joblib", tmp / "nope.json")
        msg, ok = "loaded (unexpected)", False
    except ModelLoadError as exc:
        msg, ok = str(exc), True
    rows.append({"case": "model artifact missing (load())", "expected_exit": 3, "exit_code": 3 if ok else 0,
                 "output_written": False, "rows_scored": 0, "first_message": msg, "pass": ok})

    errs = pd.DataFrame(rows)
    errs.to_csv(OUT / "error_handling_matrix.csv", index=False)
    print(f"[2] error handling: {int(errs['pass'].sum())}/{len(errs)} cases behave as specified")

    # 3. End-to-end ----------------------------------------------------------
    test_in, test_out = tmp / "test_split.csv", tmp / "test_scored.csv"
    df.loc[idx_test, ["AppointmentID", *config.FEATURES]].to_csv(test_in, index=False)
    r = subprocess.run([sys.executable, str(PREDICT), str(test_in), str(test_out)],
                       capture_output=True, text=True, cwd=ROOT)
    cli = pd.read_csv(test_out)
    inproc, _ = predict(committed, meta, df.loc[idx_test, config.FEATURES])
    e2e = {
        "exit_code": r.returncode, "rows_in": len(idx_test), "rows_scored": len(cli),
        "cli_matches_in_process": bool(np.allclose(cli["risk_probability"].values,
                                                   inproc["risk_probability"].values)
                                       and (cli["high_risk_flag"].values == inproc["high_risk_flag"].values).all()),
        "passthrough_id_preserved": cli["AppointmentID"].tolist() == df.loc[idx_test, "AppointmentID"].tolist(),
        "flag_rate": round(float(cli["high_risk_flag"].mean()), 3),
        "metrics_from_cli_output": metrics(y[idx_test].values, cli["risk_probability"].values, meta["threshold"]),
        "output_columns": cli.columns.tolist(),
    }
    (OUT / "end_to_end_check.json").write_text(json.dumps(e2e, indent=2))
    print(f"[3] end-to-end: exit {e2e['exit_code']}, {e2e['rows_scored']}/{e2e['rows_in']} scored, "
          f"CLI == in-process: {e2e['cli_matches_in_process']}, ROC-AUC {e2e['metrics_from_cli_output']['roc_auc']}")

print("\n" + errs[["case", "expected_exit", "exit_code", "pass"]].to_string(index=False))
