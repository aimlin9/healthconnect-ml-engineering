"""
Week 8 — Data Science ↔ ML Engineering contract cross-check.

Data Science (E. E. Arhin) handed over an admission-proxy scorer package with a
15-test interface contract (hc_contract_tests.py, C1–C15). This script runs:

  A. The DS package's own tests, unchanged, in this environment
     (DS request 1: "run python hc_contract_tests.py, report versions and pass count").
  B. The same 15 tests against the ML Engineering pipeline's scoring function
     (DS request 2: "run_contract_tests(your_score_fn, sample_df) with your own
     preprocessing in front"). The adapter only translates output format —
     risk_probability -> risk_score, rejected rows -> status "REJECTED: <reason>" —
     and maps the DS column name PriorVisits to PriorAppointmentsCount (same concept).
     Validation behaviour is the pipeline's own; nothing is relaxed to pass a test.
  C. A schema comparison: DS schema.json vs. src/hc_pipeline/config.py.

The DS package is external code: keep it in its own folder and pass the path.
Usage: python scripts/week8_ds_contract_crosscheck.py PATH/TO/handoff_ml_engineering
Writes reports/week8/evidence/ds_contract/.
"""
import json
import logging
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hc_pipeline import InputValidationError, config, load, predict  # noqa: E402
from hc_pipeline.features import add_prior_appointments  # noqa: E402

if len(sys.argv) != 2:
    sys.exit(__doc__)
DS = Path(sys.argv[1]).resolve()
sys.path.insert(1, str(DS))
from hc_contract_tests import run_contract_tests  # noqa: E402
from hc_scorer import AdmissionProxyScorer  # noqa: E402

logging.getLogger("hc_pipeline").setLevel(logging.ERROR)
OUT = ROOT / "reports" / "week8" / "evidence" / "ds_contract"
OUT.mkdir(parents=True, exist_ok=True)
env = {"platform": platform.platform(), "python": platform.python_version(),
       "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
ds_schema = json.loads((DS / "schema.json").read_text())

# A. DS package, unchanged ---------------------------------------------------
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    scorer = AdmissionProxyScorer(DS)
ds_sample = pd.read_csv(DS / "sample_input.csv", keep_default_na=False, na_values=[""])
res_a = run_contract_tests(scorer.score, ds_sample)
res_a.to_csv(OUT / "A_ds_package_in_mle_env.csv", index=False)
version_warnings = sorted({str(w.message).split(" This might")[0] for w in caught})

# B. ML Engineering pipeline behind the same contract ------------------------
model, meta = load()


def mle_score_fn(df):
    """The ML Engineering pipeline, with output shaped to the DS contract."""
    data = df.rename(columns={"PriorVisits": "PriorAppointmentsCount"}).reset_index(drop=True)
    try:
        scores, rejected = predict(model, meta, data, errors="skip")
    except InputValidationError as exc:
        if any(row == "all" for row, _, _ in exc.problems) or data.empty:
            raise  # structural problem (missing column / no rows): the pipeline refuses the batch
        scores = pd.DataFrame(columns=["risk_probability"], index=data.index[:0])
        rejected = pd.DataFrame(exc.problems, columns=["row", "column", "problem"])
    reasons = rejected.groupby("row").apply(
        lambda g: "; ".join(f"{c}: {p}" for c, p in zip(g["column"], g["problem"]))) if len(rejected) else {}
    out = pd.DataFrame({
        "risk_score": scores["risk_probability"].reindex(data.index),
        "status": ["REJECTED: " + reasons[i] if i in reasons else "SCORED" for i in data.index],
    })
    out.index = df.index
    return out


df = add_prior_appointments(pd.read_csv(config.FEATURES_CSV))
mle_sample = (df[config.FEATURES].sample(20, random_state=config.RANDOM_STATE)
              .rename(columns={"PriorAppointmentsCount": "PriorVisits"}).reset_index(drop=True))
mle_sample.to_csv(OUT / "B_mle_sample_input.csv", index=False)
res_b = run_contract_tests(mle_score_fn, mle_sample)
res_b.to_csv(OUT / "B_mle_pipeline_vs_ds_contract.csv", index=False)

# C. Schema comparison --------------------------------------------------------
ds_cols = {**{c: "number" for c in ds_schema["numeric"]}, **{c: "category" for c in ds_schema["categorical"]}}
mle_cols = {**{c: "integer" for c in config.NUMERIC}, **{c: "category" for c in config.CATEGORICAL}}
alias = {"PriorVisits": "PriorAppointmentsCount"}
rows = []
for c in sorted(set(ds_cols) | {k for k in mle_cols if k not in alias.values()} | set(alias.values())):
    if c in alias.values():
        continue
    m = alias.get(c, c)
    ds_rng = (ds_schema["numeric"].get(c, {}).get("min"), ds_schema["numeric"].get(c, {}).get("max")) \
        if c in ds_schema["numeric"] else ds_schema["categorical"].get(c, {}).get("categories")
    mle_rng = config.NUMERIC[m]["valid"] if m in config.NUMERIC else config.CATEGORICAL.get(m)
    rows.append({"ds_column": c if c in ds_cols else "", "mle_column": m if m in mle_cols else "",
                 "ds_type": ds_cols.get(c, ""), "mle_type": mle_cols.get(m, ""),
                 "ds_valid": json.dumps(ds_rng), "mle_valid": json.dumps(mle_rng)})
schema_cmp = pd.DataFrame(rows)
schema_cmp.to_csv(OUT / "C_schema_comparison.csv", index=False)

# D. Integration: the MLE data pipeline feeding the DS scorer ------------------
# The MLE feature set already carries every DS input; only two names differ.
feed = df[["Age", "PriorAppointmentsCount", "HasChronicCondition", "Gender", "Diagnosis"]].rename(
    columns={"PriorAppointmentsCount": "PriorVisits", "HasChronicCondition": "HasChronic"})
fed = scorer.score(feed)
integration = {
    "rows_sent": len(feed),
    "status_counts": fed["status"].str.split(":").str[0].value_counts().to_dict(),
    "flag_top20_rate": round(float(fed["flag_top20"].astype(float).mean()), 4),
    "risk_score_range": [round(float(fed["risk_score"].min()), 4), round(float(fed["risk_score"].max()), 4)],
    "column_mapping": {"PriorAppointmentsCount": "PriorVisits", "HasChronicCondition": "HasChronic"},
}

summary = {
    "environment": env,
    "D_mle_data_into_ds_scorer": integration,
    "ds_package": {"model": ds_schema["model_name"], "version": ds_schema["version"],
                   "trained_with": {"python": ds_schema["python_version"], "sklearn": ds_schema["sklearn_version"]},
                   "release_gate_passed": ds_schema["release_gate"]["passed"]},
    "A_ds_package_unchanged": {"passed": int((res_a.Result == "PASS").sum()), "of": len(res_a),
                               "failed": res_a.loc[res_a.Result == "FAIL", "ID"].tolist(),
                               "load_warnings": version_warnings},
    "B_mle_pipeline_vs_ds_contract": {"passed": int((res_b.Result == "PASS").sum()), "of": len(res_b),
                                      "failed": res_b.loc[res_b.Result == "FAIL", "ID"].tolist()},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))

pd.set_option("display.width", 200, "display.max_colwidth", 95)
print(json.dumps(env))
print(f"\nA. DS package unchanged: {summary['A_ds_package_unchanged']['passed']}/15  "
      f"failed={summary['A_ds_package_unchanged']['failed']}")
print(f"B. MLE pipeline vs DS contract: {summary['B_mle_pipeline_vs_ds_contract']['passed']}/15  "
      f"failed={summary['B_mle_pipeline_vs_ds_contract']['failed']}\n")
print(res_b[["ID", "Scenario", "Result", "Detail"]].to_string(index=False))
print("\nC. Schema comparison\n" + schema_cmp.to_string(index=False))
print(f"\nD. MLE data -> DS scorer: {integration}")
