"""
Build the Week 8 individual-presentation slides (1920x1080 PNGs + one PDF).

Numbers come from the evidence files, not typed in, wherever an evidence file exists.
Rendering uses headless Microsoft Edge (no extra Python packages).

Usage: python presentation/week8/build_week8_slides.py
Output: presentation/week8/slides/slideNN.png, presentation/week8/HealthConnect_Week8_MLE_Slides_Ramsey.pdf
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EV7 = ROOT / "reports/week7/evidence"
EV8 = ROOT / "reports/week8/evidence"
OUT = HERE / "slides"
PDF = HERE / "HealthConnect_Week8_MLE_Slides_Ramsey.pdf"

EDGE = next((p for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                         r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                         shutil.which("msedge") or "", shutil.which("chromium") or "",
                         shutil.which("google-chrome") or ""] if p and Path(p).exists()), None)
if not EDGE:
    sys.exit("Needs Microsoft Edge or Chrome for headless rendering")

repro = json.loads((EV8 / "reproducibility_check.json").read_text())
e2e = json.loads((EV8 / "end_to_end_check.json").read_text())
ds = json.loads((EV8 / "ds_contract/summary.json").read_text())
w7 = json.loads((EV7 / "week7_retest_results.json").read_text())
w6 = json.loads((EV7 / "week6_audit_results.json").read_text())
meta = json.loads((ROOT / "models/high_risk_pipeline_meta.json").read_text())
gate = json.loads((EV8 / "release_gate_high_risk.json").read_text())
gm, rr = gate["metrics"], gate["rule_restatement_check"]["holdout_auc"]
errors = (EV8 / "error_handling_matrix.csv").read_text().splitlines()
n_err, n_err_pass = len(errors) - 1, sum(1 for l in errors[1:] if l.rstrip().endswith("True"))


def count_pass(rows, key):
    return sum(1 for r in rows if r[key] == "PASS"), len(rows)


w6_pass, w6_n = count_pass(w6, "result")
w7_pass, w7_n = count_pass(w7, "week7_result")
tm = meta["test_metrics"]
dsA, dsB, dsD = ds["A_ds_package_unchanged"], ds["B_mle_pipeline_vs_ds_contract"], ds["D_mle_data_into_ds_scorer"]

CSS = """
* { box-sizing: border-box; }
body { margin: 0; font-family: Calibri, 'Segoe UI', Arial, sans-serif; color: #000; background: #fff; }
.s { width: 1920px; height: 1080px; padding: 90px 120px 70px; position: relative; page-break-after: always; overflow: hidden; }
.k { font-size: 26px; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 18px; }
h1 { font-size: 66px; margin: 0 0 34px; line-height: 1.1; border-bottom: 3px solid #000; padding-bottom: 22px; }
h2 { font-size: 40px; margin: 0 0 16px; }
p, li { font-size: 34px; line-height: 1.45; } ul { margin: 0; padding-left: 40px; } li { margin-bottom: 12px; }
table { border-collapse: collapse; width: 100%; font-size: 30px; }
th, td { border: 2px solid #000; padding: 12px 18px; text-align: left; vertical-align: top; }
th { font-weight: bold; }
.big { display: flex; gap: 28px; margin: 10px 0 34px; }
.big div { flex: 1; border: 2px solid #000; padding: 22px 26px; }
.big b { display: block; font-size: 72px; line-height: 1; margin-bottom: 10px; }
.big span { font-size: 27px; }
.foot { position: absolute; left: 120px; right: 120px; bottom: 34px; font-size: 22px; display: flex; justify-content: space-between; border-top: 1px solid #000; padding-top: 10px; }
.flow { display: flex; align-items: stretch; gap: 0; margin: 20px 0 30px; }
.flow div { flex: 1; border: 2px solid #000; padding: 22px; font-size: 29px; text-align: center; }
.flow .arr { flex: 0 0 70px; border: none; font-size: 50px; display: flex; align-items: center; justify-content: center; padding: 0; }
.two { display: flex; gap: 60px; } .two > div { flex: 1; }
code { font-family: Consolas, monospace; font-size: 0.9em; }
.title { display: flex; flex-direction: column; justify-content: center; height: 100%; }
.title h1 { font-size: 86px; border: none; padding: 0; margin: 0 0 20px; }
.title .rule { border-top: 3px solid #000; width: 340px; margin: 26px 0 30px; }
"""

slides = []


def slide(kicker, title, body, n=[1]):  # slide 1 is the title slide
    n[0] += 1
    slides.append(f"""<section class='s'><div class='k'>{kicker}</div><h1>{title}</h1>{body}
<div class='foot'><span>HealthConnect · ML Engineering · Gyimah Ramsey Opoku</span><span>{n[0]}</span></div></section>""")


# 1 Title
slides.append(f"""<section class='s'><div class='title'>
<div class='k'>AnalystLab Africa Experience Lab · Week 8 · Batch E</div>
<h1>HealthConnect: Final ML Pipeline<br>Integration &amp; Readiness</h1><div class='rule'></div>
<p style='font-size:40px;margin:0'>Gyimah Ramsey Opoku</p>
<p style='font-size:34px;margin:6px 0 0'>Machine Learning Engineering track</p></div>
<div class='foot'><span>HealthConnect · ML Engineering · Gyimah Ramsey Opoku</span><span>1</span></div></section>""")

# 2 Problem & role
slide("Introduction", "The problem and my role", """
<p><b>Project question:</b> How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?</p>
<div class='two' style='margin-top:30px'>
<div><h2>The gap I own</h2><ul>
<li>A model in a notebook can't be used by anyone else</li>
<li>Other tracks and a clinic system need to score patients the same way every time</li>
<li>Bad input must be refused, not silently scored</li></ul></div>
<div><h2>My responsibility</h2><ul>
<li>Turn models into a tested, reproducible scoring pipeline</li>
<li>Integrate the Data Science model</li>
<li>Prove it works, and document what it can't do</li></ul></div></div>""")

# 3 Journey
slide("Project journey", "From notebook to integrated pipeline", f"""
<table><tr><th style='width:16%'>Week</th><th style='width:30%'>Stage</th><th>What I delivered</th></tr>
<tr><td>Week 6</td><td>Development</td><td>Data pipeline over 27 tables, 75 features, 6 models, each compared against a baseline. Only High-Risk Patient showed lift</td></tr>
<tr><td>Week 7</td><td>Testing → Refinement</td><td>Week 6 code passed <b>{w6_pass}/{w6_n}</b> reliability checks → refined pipeline <b>{w7_pass}/{w7_n}</b>; 61 automated tests</td></tr>
<tr><td>Week 8</td><td>Final integration</td><td>Reproduced on a second OS, error-handling matrix, end-to-end check, Data Science contract tests</td></tr></table>""")

# 4 What I built
slide("My track contribution", "What I built: the scoring pipeline", f"""
<div class='flow'><div><b>Input</b><br>CSV, table or one patient</div><div class='arr'>→</div>
<div><b>Validate</b><br>columns, categories, ranges, missing values</div><div class='arr'>→</div>
<div><b>Saved pipeline</b><br>fixed encoding + Random Forest</div><div class='arr'>→</div>
<div><b>Output + log</b><br>risk_probability, flag, per-row status, version</div></div>
<div class='two'><div><h2>Methods &amp; tools</h2><ul>
<li>Python, pandas, scikit-learn Pipeline, joblib</li><li>pytest + mutation testing</li><li>Command-line tool with exit codes and logging</li></ul></div>
<div><h2>Design decisions</h2><ul>
<li>Preprocessing saved inside the model, so it can't drift</li>
<li>Reject impossible or missing input with the line and reason</li>
<li>Threshold kept in metadata, adjustable without retraining</li></ul></div></div>""")

# 5 Results
slide("Results", "Evidence that it is ready to demonstrate", f"""
<div class='big'>
<div><b>{w7_pass}/{w7_n}</b><span>reliability checks<br>(Week 6 code: {w6_pass}/{w6_n})</span></div>
<div><b>65/65</b><span>automated tests pass<br>5/5 injected bugs caught</span></div>
<div><b>{n_err_pass}/{n_err}</b><span>error cases handled<br>as specified</span></div>
<div><b>{'Identical' if repro['identical'] else 'Differs'}</b><span>retrain on Windows +<br>Python {repro['environment']['python']}</span></div></div>
<table><tr><th>High-Risk model, held-out test set (n = {e2e['rows_in']:,})</th><th>Value</th></tr>
<tr><td>ROC-AUC (Week 6 → Week 7, after removing leaked features)</td><td>0.727 → {tm['roc_auc']}</td></tr>
<tr><td>Lift when the clinic flags its top 10% / 20% / 30% (replaces the 0.33 threshold)</td><td>{gm['lift']['10%']['lift']:.2f} / {gm['lift']['20%']['lift']:.2f} / {gm['lift']['30%']['lift']:.2f}</td></tr>
<tr><td>Data Science release gate (AUC, 95% CI, shuffle test, lift)</td><td>Passed, but Age ≥ 65 alone: {rr['rule_component_Age>=65_alone']}</td></tr></table>""")

# 6 Error handling
slide("Testing", "Bad input is refused, with a reason", """
<table><tr><th>Input</th><th style='width:12%'>Exit code</th><th>What the pipeline does</th></tr>
<tr><td>Valid patient</td><td>0</td><td>Scored</td></tr>
<tr><td>Age 250</td><td>2</td><td>Rejected: “CSV line 2, Age: 250 outside valid range [0, 120]”</td></tr>
<tr><td>Age blank / “fifty”</td><td>2</td><td>Rejected: missing value / not a number</td></tr>
<tr><td>VisitType = Surgery</td><td>2</td><td>Rejected, lists the allowed categories</td></tr>
<tr><td>Mixed batch with --skip-invalid</td><td>0</td><td>Good rows scored; bad rows marked REJECTED with the reason, and written to a rejects file</td></tr>
<tr><td>Age 110 (valid, outside training data)</td><td>0</td><td>Scored, with an extrapolation warning in the log</td></tr>
<tr><td>Model file missing</td><td>3</td><td>Clear error: run the training script first</td></tr></table>""")

# 7 Cross-track
slide("Cross-track collaboration", "Data Science ↔ ML Engineering", f"""
<div class='two'><div><h2>Received from Data Science</h2><ul>
<li>Final model package (admission-proxy scorer)</li><li>Requirements note + 15 interface contract tests</li>
<li>Status: <b>validated null result</b>, not for decisions</li></ul></div>
<div><h2>What I did with it</h2><ul>
<li>Reviewed the code and the model file before running</li><li>Ran the tests in two environments</li>
<li>Ran the same tests against my pipeline</li><li>Fed my data into their scorer</li></ul></div></div>
<div class='big' style='margin-top:30px'>
<div><b>{dsA['passed']}/{dsA['of']}</b><span>DS package in my environment<br>(Py 3.12 / sklearn 1.9.1 and 3.14 / 1.9.0)</span></div>
<div><b>11 → {dsB['passed']}/{dsB['of']}</b><span>DS tests against my pipeline<br>(7 Nov → shared contract, 8 Nov)</span></div>
<div><b>{dsD['rows_sent']:,}</b><span>of my rows scored by the DS model,<br>0 rejected</span></div></div>""")

# 8 What changed
slide("Cross-track collaboration", "From two contracts to one", f"""
<table style='font-size:28px'><tr><th style='width:33%'>What the tests revealed</th><th>What we agreed and did</th></tr>
<tr><td>Contracts disagreed on missing values, unknown categories and empty input</td><td>One shared contract: reject with a reason, per-row status, empty in → empty out. Both sides changed: <b>{dsB['passed']}/{dsB['of']}</b></td></tr>
<tr><td>Age 110 vs 120; decimals vs whole numbers</td><td>Shared limits: Age 0–120, whole numbers</td></tr>
<tr><td>PriorVisits might count future visits</td><td>Confirmed point-in-time, same as mine</td></tr>
<tr><td>DS release gate applied to my High-Risk model</td><td>Passes (AUC {gm['roc_auc']}), but Age ≥ 65 alone scores {rr['rule_component_Age>=65_alone']}: it restates its own label → <b>demo only</b></td></tr>
<tr><td>Threshold 0.33 flags two-thirds of patients</td><td>Flag budget: report lift at 10/20/30%; the clinic picks the percentage</td></tr></table>""")

# 9 Overall solution
slide("Overall HealthConnect solution", "How my work connects", """
<div class='flow'>
<div><b>Data Analytics</b><br>validated patterns and valid input ranges</div><div class='arr'>→</div>
<div><b>Data Science</b><br>final model, target and operating rule</div><div class='arr'>→</div>
<div><b>ML Engineering</b><br>validated, reproducible scoring workflow</div><div class='arr'>→</div>
<div><b>Project Management</b><br>issues, decisions, readiness, demo</div></div>
<ul><li><b>Business value:</b> a safe, repeatable way to score patients, so the clinic can test prioritising reminders and follow-up for missed appointments once a model has real signal</li>
<li><b>My part:</b> any model Data Science produces can be scored the same way, with bad input refused and every run logged</li>
<li>The GenAI assistant answers patient admin questions from the Knowledge Base; it does not call the model</li></ul>""")

# 10 Limitations
slide("Limitations", "What the solution cannot do yet", """
<ul>
<li><b>No usable predictor:</b> the DS model has no signal (hold-out AUC 0.467); my High-Risk model passes the release gate only by restating the Age ≥ 65 part of its constructed label</li>
<li><b>Synthetic data:</b> only 5,000 of 60,000 appointments have full context</li>
<li><b>Flag budget not set:</b> the clinic must choose 10, 20 or 30% from its follow-up capacity</li>
<li><b>Not production:</b> command-line tool and Python API only; no web service, authentication or monitoring</li>
<li><b>Fairness:</b> province and facility-type checks not possible with the current data</li></ul>
<p style='margin-top:24px'><b>Ready to demonstrate, not to make decisions about real patients.</b></p>""")

# 11 Closing
slide("Closing", "Key takeaway and next steps", """
<div class='two'><div><h2>Takeaway</h2><p>A model is only useful when others can run it safely and get the same answer. Testing showed the Week 6 code wasn't usable; it now is, and the evidence proves it.</p>
<h2 style='margin-top:30px'>What I learned</h2><ul><li>Test the pipeline, not just the model</li><li>Collaboration needs agreed contracts</li><li>A release gate must be paired with a check of how the target is built</li></ul></div>
<div><h2>Future version</h2><ul>
<li>Retrain on real outcomes (attendance, admissions)</li><li>Clinic sets the flag budget</li>
<li>Web API, monitoring and fairness checks</li></ul></div></div>
<p style='margin-top:40px;font-size:38px'><b>Thank you.</b></p>""")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("slide*.png"):
        old.unlink()
    for i, s in enumerate(slides, 1):
        page = tmp / f"slide{i:02d}.html"
        page.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{s}</body></html>",
                        encoding="utf-8")
        subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                        "--window-size=1920,1080", f"--screenshot={OUT / f'slide{i:02d}.png'}", page.as_uri()],
                       check=True, capture_output=True, timeout=120)
    deck = tmp / "deck.html"
    deck.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}"
                    "@page { size: 1920px 1080px; margin: 0; }</style></head><body>" + "".join(slides) + "</body></html>",
                    encoding="utf-8")
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", deck.as_uri()], check=True, capture_output=True, timeout=120)
print(f"{len(slides)} slides -> {OUT}\nPDF -> {PDF}")
