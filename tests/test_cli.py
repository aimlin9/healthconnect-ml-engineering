"""End-to-end: CSV in -> scored CSV out through scripts/predict.py."""
import subprocess
import sys

import pandas as pd

from hc_pipeline import config

SCRIPT = config.ROOT / "scripts" / "predict.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_cli_scores_csv(tmp_path, sample_rows):
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    sample_rows.assign(PatientRef=range(10)).to_csv(src, index=False)
    r = run(src, dst)
    assert r.returncode == 0, r.stderr
    out = pd.read_csv(dst)
    assert len(out) == 10 and {"PatientRef", "risk_probability", "high_risk_flag"} <= set(out.columns)


def test_cli_rejects_invalid_input_with_exit_code_2(tmp_path, sample_rows):
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    sample_rows.assign(Age=250).to_csv(src, index=False)
    r = run(src, dst)
    assert r.returncode == 2
    assert "outside valid range" in r.stderr
    assert not dst.exists()


def test_cli_skip_invalid_writes_rejects_file(tmp_path, sample_rows):
    rows = sample_rows.copy()
    rows.iloc[0, rows.columns.get_loc("VisitType")] = "Telehealth"
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    rows.to_csv(src, index=False)
    r = run(src, dst, "--skip-invalid")
    assert r.returncode == 0, r.stderr
    out = pd.read_csv(dst)
    assert len(out) == 10  # one output row per input row, in input order
    assert out["status"].iloc[0].startswith("REJECTED: VisitType: unknown category 'Telehealth'")
    assert pd.isna(out["risk_probability"].iloc[0])
    assert (out["status"].iloc[1:] == "SCORED").all()
    assert len(pd.read_csv(tmp_path / "out.rejected.csv")) == 1


def test_cli_empty_input_writes_header_only(tmp_path, sample_rows):
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    sample_rows.head(0).to_csv(src, index=False)
    r = run(src, dst)
    assert r.returncode == 0, r.stderr
    out = pd.read_csv(dst)
    assert out.empty and {"risk_probability", "status"} <= set(out.columns)


def test_cli_reports_csv_line_numbers(tmp_path, sample_rows):
    rows = sample_rows.reset_index(drop=True).astype(object)
    rows.iloc[3, rows.columns.get_loc("Age")] = 250  # 4th data row = CSV line 5
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    rows.to_csv(src, index=False)
    assert "CSV line 5, Age" in run(src, dst).stderr
    run(src, dst, "--skip-invalid")
    assert pd.read_csv(tmp_path / "out.rejected.csv")["csv_line"].tolist() == [5]


def test_cli_output_uses_normalised_values(tmp_path, sample_rows):
    rows = sample_rows.reset_index(drop=True).astype(object)
    rows.iloc[0, rows.columns.get_loc("Gender")] = " female "
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    rows.to_csv(src, index=False)
    assert run(src, dst).returncode == 0
    assert pd.read_csv(dst).loc[0, "Gender"] == "Female"


def test_cli_missing_file(tmp_path):
    r = run(tmp_path / "nope.csv", tmp_path / "out.csv")
    assert r.returncode == 2 and "cannot read" in r.stderr


def test_cli_threshold_override(tmp_path, sample_rows):
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    sample_rows.to_csv(src, index=False)
    assert run(src, dst, "--threshold", "0.99").returncode == 0
    assert pd.read_csv(dst)["high_risk_flag"].sum() == 0
    assert run(src, dst, "--threshold", "1.5").returncode == 2


def test_cli_writes_log_file(tmp_path, sample_rows):
    src, dst, logf = tmp_path / "in.csv", tmp_path / "out.csv", tmp_path / "run.log"
    sample_rows.to_csv(src, index=False)
    run(src, dst, "--log-file", logf)
    text = logf.read_text()
    assert "loaded model" in text and "scored 10 row(s)" in text
