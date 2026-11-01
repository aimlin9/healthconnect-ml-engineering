"""
Week 7 — Render the LinkedIn/X post images from the evidence files.

Writes presentation/week7/:
  01_before_after_probes.png   14 reliability probes, Week 6 vs Week 7
  02_terminal_tests_cli.png    pytest summary + CLI rejecting bad input
  03_threshold_tradeoff.png    recall vs share of patients flagged, by threshold

Usage: python scripts/week7_post_images.py
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "reports" / "week7" / "evidence"
OUT = ROOT / "presentation" / "week7"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
SERIES1, SERIES2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})

# --- 1. Before / after probe table ----------------------------------------------
rows = json.loads((EVID / "week7_retest_results.json").read_text())
fig, ax = plt.subplots(figsize=(12, 7.4), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_axis_off()
fig.text(0.04, 0.94, "Week 6 pipeline vs Week 7 pipeline: 14 reliability checks", fontsize=18, weight="bold")
w6 = sum(r["week6_result"] == "PASS" for r in rows)
w7 = sum(r["week7_result"] == "PASS" for r in rows)
fig.text(0.04, 0.895, f"Same probes, same data.  Week 6: {w6}/14 passed   →   Week 7: {w7}/14 passed",
         fontsize=12.5, color=INK2)
cols = [0.04, 0.19, 0.66, 0.81]
y = 0.84
for x, h in zip(cols, ["Area", "Check", "Week 6", "Week 7"]):
    fig.text(x, y, h, fontsize=11, weight="bold", color=INK2)
for r in rows:
    y -= 0.054
    fig.add_artist(plt.Line2D([0.04, 0.96], [y + 0.033, y + 0.033], color=GRID, lw=0.8))
    fig.text(cols[0], y, r["area"], fontsize=10.5, color=INK2)
    scenario = r["scenario"] if len(r["scenario"]) <= 58 else r["scenario"][:56] + "…"
    fig.text(cols[1], y, scenario, fontsize=10.5)
    for x, res in ((cols[2], r["week6_result"]), (cols[3], r["week7_result"])):
        ok = res == "PASS"
        fig.text(x, y, ("✓ " if ok else "✗ ") + res, fontsize=10.5, weight="bold",
                 color=GOOD if ok else CRITICAL)
fig.text(0.04, 0.025, "Source: reports/week7/evidence/week6_audit_results.json, week7_retest_results.json",
         fontsize=8.5, color=INK2)
fig.savefig(OUT / "01_before_after_probes.png", facecolor=SURFACE)
plt.close(fig)

# --- 2. Terminal: tests + CLI ---------------------------------------------------
pytest_tail = [l for l in (EVID / "pytest_output.txt").read_text().splitlines() if "passed" in l][-1].strip("= ")
counts = {}
for line in (EVID / "pytest_output.txt").read_text().splitlines():
    m = re.match(r"tests/(test_\w+)\.py::.* PASSED", line)
    if m:
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
cli = [re.sub(r"^\d{4}-\d\d-\d\d [\d:,]+ ", "", l)
       for l in (EVID / "cli_demo" / "cli_session.txt").read_text().splitlines()]
cli = [l.replace(" hc_pipeline:", "") for l in cli if l.strip()]
lines = ["$ pytest"] + [f"  {k:<24} {v:>3} passed" for k, v in counts.items()] + [f"  {pytest_tail}", ""] + cli
term_bg, term_fg = "#1a1a19", "#f0efec"
fig = plt.figure(figsize=(12, 0.32 * len(lines) + 1.1), dpi=150)
fig.patch.set_facecolor(term_bg)
for i, c in enumerate(["#e34948", "#eda100", "#1baf7a"]):
    fig.add_artist(plt.Circle((0.025 + i * 0.022, 1 - 0.28 / fig.get_figheight()), 0.006,
                              color=c, transform=fig.transFigure))
for i, l in enumerate(lines):
    color = term_fg
    if l.startswith("$"):
        color = "#86b6ef"
    elif "ERROR" in l:
        color = "#e66767"
    elif "WARNING" in l:
        color = "#eda100"
    elif "passed" in l:
        color = "#5fd38f"
    fig.text(0.025, 1 - (0.75 + 0.32 * i) / fig.get_figheight(), l[:118], fontsize=10.5,
             family="DejaVu Sans Mono", color=color, va="top")
fig.savefig(OUT / "02_terminal_tests_cli.png", facecolor=term_bg)
plt.close(fig)

# --- 3. Threshold trade-off ---------------------------------------------------------
pts = json.loads((EVID / "threshold_operating_points.json").read_text())
t = [p["threshold"] for p in pts]
recall = [p["recall"] * 100 for p in pts]
flagged = [p["flagged_pct"] for p in pts]
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.plot(t, recall, color=SERIES1, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, label="Recall (high-risk patients caught)")
ax.plot(t, flagged, color=SERIES2, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, label="Patients flagged")
ax.axhline(42.4, color=INK2, lw=1, ls=(0, (4, 3)))
ax.text(0.505, 43.5, "42.4% actually high-risk", fontsize=9.5, color=INK2, ha="right", va="bottom")
for x, yv, dy in ((0.33, 78.5, 6), (0.33, 67.6, -16), (0.50, 55.3, 6), (0.50, 30.0, 6)):
    ax.annotate(f"{yv:.1f}%", (x, yv), textcoords="offset points", xytext=(8, dy), fontsize=10, color=INK, weight="bold")
ax.axvline(0.33, color=GRID, lw=6, zorder=0)
ax.text(0.333, 101, "Week 7 tuned threshold (0.33)", fontsize=9.5, color=INK2)
ax.set_xlim(0.28, 0.52)
ax.set_ylim(0, 108)
ax.set_xlabel("Decision threshold")
ax.set_ylabel("% of test-set patients")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.93), fontsize=10)
fig.suptitle("Catching more high-risk patients means flagging far more patients", x=0.06, ha="left",
             fontsize=15, weight="bold")
fig.text(0.06, 0.885, "Recall vs. share of patients flagged at each threshold (held-out test set, n = 1,250)",
         fontsize=10.5, color=INK2)
fig.text(0.06, 0.015, "Source: reports/week7/evidence/threshold_operating_points.json", fontsize=8.5, color=INK2)
fig.subplots_adjust(top=0.84, bottom=0.12, left=0.09, right=0.97)
fig.savefig(OUT / "03_threshold_tradeoff.png", facecolor=SURFACE)
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
