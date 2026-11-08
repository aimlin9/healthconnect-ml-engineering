"""Input validation: every row is checked against the contract in config before it reaches the model."""
import logging

import numpy as np
import pandas as pd

from .config import CATEGORICAL, FEATURES, NUMERIC

log = logging.getLogger("hc_pipeline")


class InputValidationError(ValueError):
    """Raised when input rows break the contract. `problems` lists (row, column, message)."""

    def __init__(self, problems):
        self.problems = problems
        preview = "; ".join(f"row {r}, {c}: {m}" for r, c, m in problems[:5])
        more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        super().__init__(f"{len(problems)} input problem(s): {preview}{more}")


def validate_input(data, errors="raise"):
    """Check and normalise model input.

    Returns (clean_df, rejected) where clean_df holds only FEATURES in canonical
    form and rejected is a DataFrame of (row, column, problem). With
    errors="raise" any problem raises InputValidationError; with errors="skip"
    invalid rows are dropped and reported in `rejected`.
    """
    if errors not in {"raise", "skip"}:
        raise ValueError("errors must be 'raise' or 'skip'")
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame or dict, got {type(data).__name__}")
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        # Structural problem: no row can be scored, so this always raises.
        raise InputValidationError([("all", c, "required column missing") for c in missing])
    if data.empty:
        # Week 8 (D-08, agreed with Data Science): empty input gives empty output, not an error.
        log.info("input has no rows; nothing to score")
        return data[FEATURES].copy(), pd.DataFrame(columns=["row", "column", "problem"])
    extra = sorted(set(data.columns) - set(FEATURES))
    if extra:
        log.debug("ignoring %d non-model column(s): %s", len(extra), extra)

    df = data[FEATURES].copy()
    problems = []

    for col, allowed in CATEGORICAL.items():
        lookup = {a.lower(): a for a in allowed}
        normalised = df[col].astype("string").str.strip().str.lower().map(lookup)
        for idx in df.index[df[col].isna()]:
            problems.append((idx, col, "missing value"))
        for idx in df.index[df[col].notna() & normalised.isna()]:
            problems.append((idx, col, f"unknown category {df.at[idx, col]!r}; allowed {allowed}"))
        df[col] = normalised

    for col, spec in NUMERIC.items():
        numeric = pd.to_numeric(df[col], errors="coerce")
        lo, hi = spec["valid"]
        for idx in df.index[df[col].isna()]:
            problems.append((idx, col, "missing value"))
        for idx in df.index[df[col].notna() & numeric.isna()]:
            problems.append((idx, col, f"not a number: {df.at[idx, col]!r}"))
        ok = numeric.notna()
        for idx in df.index[ok & ((numeric < lo) | (numeric > hi))]:
            problems.append((idx, col, f"{numeric[idx]} outside valid range [{lo}, {hi}]"))
        if spec["integer"]:
            for idx in df.index[ok & (numeric != np.floor(numeric))]:
                problems.append((idx, col, f"{numeric[idx]} is not a whole number"))
        df[col] = numeric

    rejected = pd.DataFrame(problems, columns=["row", "column", "problem"])
    if problems:
        if errors == "raise":
            raise InputValidationError(problems)
        bad_rows = set(rejected["row"])
        df = df.loc[~df.index.isin(bad_rows)]
        log.warning("rejected %d of %d row(s) failing validation", len(bad_rows), len(data))
        if df.empty:
            raise InputValidationError(problems)

    for col in NUMERIC:
        df[col] = df[col].astype("int64")
    return df, rejected
