"""
Score patients with the saved High-Risk Patient pipeline.

Usage:
  python scripts/predict.py INPUT.csv OUTPUT.csv [--skip-invalid] [--threshold 0.5] [--log-file run.log]

INPUT.csv needs the columns in src/hc_pipeline/config.py (FEATURES); extra columns
are carried through to the output. Exit codes: 0 ok, 2 invalid input, 3 model missing.
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hc_pipeline import InputValidationError, ModelLoadError, load, predict, validate_input  # noqa: E402

log = logging.getLogger("hc_pipeline")


def csv_line(row):
    """DataFrame row label -> line number in the CSV (header is line 1)."""
    return row + 2 if isinstance(row, int) else row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--skip-invalid", action="store_true",
                    help="score valid rows and write rejected rows to OUTPUT.rejected.csv instead of failing")
    ap.add_argument("--threshold", type=float, help="override the model's stored decision threshold")
    ap.add_argument("--log-file")
    args = ap.parse_args(argv)

    handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        model, meta = load()
    except ModelLoadError as exc:
        log.error("%s", exc)
        return 3
    if args.threshold is not None:
        if not 0 < args.threshold < 1:
            log.error("--threshold must be between 0 and 1")
            return 2
        log.info("threshold overridden: %.2f -> %.2f", meta["threshold"], args.threshold)
        meta = {**meta, "threshold": args.threshold}

    try:
        data = pd.read_csv(args.input)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        log.error("cannot read %s: %s", args.input, exc)
        return 2
    log.info("read %d row(s) from %s", len(data), args.input)

    try:
        scores, rejected = predict(model, meta, data, errors="skip" if args.skip_invalid else "raise")
    except InputValidationError as exc:
        for row, col, msg in exc.problems:
            log.error("CSV line %s, %s: %s", csv_line(row), col, msg)
        log.error("no output written (%d problem(s)); fix the input or use --skip-invalid", len(exc.problems))
        return 2

    # Write the normalised values the model actually scored (e.g. ' female ' -> 'Female').
    accepted = data.loc[scores.index]
    clean, _ = validate_input(accepted)
    accepted.assign(**{c: clean[c] for c in clean.columns}).join(scores).to_csv(args.output, index=False)
    log.info("wrote %d scored row(s) to %s", len(scores), args.output)
    if len(rejected):
        rej_path = Path(args.output).with_suffix(".rejected.csv")
        rejected.insert(0, "csv_line", rejected["row"].map(csv_line))
        rejected.drop(columns="row").to_csv(rej_path, index=False)
        log.warning("wrote %d problem(s) to %s", len(rejected), rej_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
