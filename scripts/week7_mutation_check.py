"""
Week 7 — Check that the test suite actually catches defects (mutation testing).

Each mutation re-introduces a realistic bug into a temporary copy of src/hc_pipeline,
then runs the test suite against that copy. A mutation is "caught" if at least one
test fails. Results go to reports/week7/evidence/mutation_check_results.json.

Usage: python scripts/week7_mutation_check.py
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUTATIONS = [
    ("M1", "Range check removed (impossible ages accepted)", "validation.py",
     "ok & ((numeric < lo) | (numeric > hi))", "ok & (numeric != numeric)"),
    ("M2", "Unknown categories passed through to the model", "validation.py",
     'problems.append((idx, col, f"unknown category', 'pass  # problems.append((idx, col, f"unknown category'),
    ("M3", "Point-in-time feature ignores visit date (Week 6-style leakage)", "features.py",
     "[patient_col, date_col, id_col]", "[patient_col, id_col]"),
    ("M4", "Model seed removed (non-reproducible training)", "config.py",
     '"class_weight": "balanced", "random_state": 42', '"class_weight": "balanced"'),
    ("M5", "Decision flag ignores the stored threshold", "pipeline.py",
     '(proba >= meta["threshold"])', "(proba >= 0.5)"),
]

def run_suite(mutation=None):
    """Run the suite against a temp copy of src/ (optionally mutated). The copy sits in a
    mirror of the repo layout so config.ROOT still resolves to real data/ and models/."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(ROOT / "src", tmp / "src")
        (tmp / "data").symlink_to(ROOT / "data")
        shutil.copytree(ROOT / "models", tmp / "models")
        if mutation:
            fname, old, new = mutation
            target = tmp / "src" / "hc_pipeline" / fname
            text = target.read_text()
            assert text.count(old) == 1, f"mutation anchor not found exactly once in {fname}"
            target.write_text(text.replace(old, new))
        # test_cli runs scripts/predict.py, which imports the real src/, so it is excluded here.
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "-o", f"pythonpath={tmp / 'src'}", "--ignore=tests/test_cli.py"],
                           cwd=ROOT, capture_output=True, text=True)
        failed = sorted(set(re.findall(r"^FAILED (tests/\S+)", r.stdout, re.M)))
        errored = "error" in r.stdout.strip().splitlines()[-1]
        return r.returncode, r.stdout.strip().splitlines()[-1], failed, errored


# Control run: the unmutated copy must pass cleanly, or the harness itself is broken.
code, summary, _, _ = run_suite()
print(f"control (no mutation): {summary}")
if code != 0:
    sys.exit("control run failed — harness is broken, mutation results would be meaningless")

results = []
for mid, desc, fname, old, new in MUTATIONS:
    code, summary, failed, errored = run_suite((fname, old, new))
    caught = code != 0 and not errored  # setup errors would be a harness fault, not a catch
    results.append({"id": mid, "mutation": desc, "caught": caught,
                    "summary": summary, "failing_tests": failed})
    print(f"{mid} {'CAUGHT' if caught else 'MISSED'}  {desc}\n    {summary}")
    for t in failed:
        print(f"      {t}")

out = ROOT / "reports" / "week7" / "evidence" / "mutation_check_results.json"
out.write_text(json.dumps(results, indent=2))
print(f"\n{sum(r['caught'] for r in results)} / {len(results)} mutations caught")
