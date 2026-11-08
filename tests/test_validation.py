"""Input validation: expected inputs pass, invalid and unexpected inputs are rejected clearly."""
import numpy as np
import pandas as pd
import pytest

from hc_pipeline import InputValidationError, config, validate_input


def test_valid_rows_pass_unchanged(sample_rows):
    clean, rejected = validate_input(sample_rows)
    assert rejected.empty
    assert list(clean.columns) == config.FEATURES
    pd.testing.assert_frame_equal(clean, sample_rows[config.FEATURES].astype(clean.dtypes.to_dict()))


def test_single_dict_accepted(sample_rows):
    clean, _ = validate_input(sample_rows.iloc[0].to_dict())
    assert len(clean) == 1


def test_extra_columns_ignored(sample_rows):
    clean, _ = validate_input(sample_rows.assign(PatientName="x"))
    assert "PatientName" not in clean.columns


def test_column_order_irrelevant(sample_rows):
    a, _ = validate_input(sample_rows)
    b, _ = validate_input(sample_rows[sample_rows.columns[::-1]])
    pd.testing.assert_frame_equal(a, b)


def test_category_case_and_whitespace_normalised(sample_rows):
    rows = sample_rows.head(2).copy()
    rows["Gender"] = [" male", "FEMALE "]
    rows["VisitType"] = ["follow-up", "EMERGENCY"]
    clean, _ = validate_input(rows)
    assert clean["Gender"].tolist() == ["Male", "Female"]
    assert clean["VisitType"].tolist() == ["Follow-up", "Emergency"]


def test_numeric_strings_coerced(sample_rows):
    clean, _ = validate_input(sample_rows.astype({"Age": str}))
    assert clean["Age"].dtype == "int64"


@pytest.mark.parametrize("bad_input", [[], "a string", 42])
def test_non_tabular_input_rejected(bad_input):
    with pytest.raises(TypeError):
        validate_input(bad_input)


def test_empty_input_returns_empty(sample_rows):
    # Week 8 (D-08, agreed with Data Science): empty in, empty out.
    clean, rejected = validate_input(sample_rows.head(0))
    assert clean.empty and rejected.empty
    assert list(clean.columns) == config.FEATURES


def test_empty_input_still_needs_required_columns(sample_rows):
    with pytest.raises(InputValidationError, match="Age.*required column missing"):
        validate_input(sample_rows.head(0).drop(columns="Age"))


def test_missing_column_named(sample_rows):
    with pytest.raises(InputValidationError, match="DepartmentWorkload.*required column missing"):
        validate_input(sample_rows.drop(columns="DepartmentWorkload"))


def test_missing_column_is_fatal_even_when_skipping(sample_rows):
    with pytest.raises(InputValidationError):
        validate_input(sample_rows.drop(columns="Age"), errors="skip")


@pytest.mark.parametrize("col,value,message", [
    ("VisitType", "Telehealth", "unknown category"),
    ("Gender", "Other", "unknown category"),
    ("Age", -5, "outside valid range"),
    ("Age", 250, "outside valid range"),
    ("Age", 45.5, "not a whole number"),
    ("Age", "forty", "not a number"),
    ("Age", np.nan, "missing value"),
    ("Gender", None, "missing value"),
    ("DoctorYearsExperience", 99, "outside valid range"),
    ("PriorAppointmentsCount", -1, "outside valid range"),
])
def test_invalid_value_rejected_with_reason(sample_rows, col, value, message):
    rows = sample_rows.head(3).astype(object)
    rows.iloc[1, rows.columns.get_loc(col)] = value
    with pytest.raises(InputValidationError, match=message) as exc:
        validate_input(rows)
    assert exc.value.problems[0][0] == rows.index[1]
    assert exc.value.problems[0][1] == col


def test_all_problems_reported_at_once(sample_rows):
    rows = sample_rows.head(3).astype(object)
    rows.iloc[0, rows.columns.get_loc("Age")] = -1
    rows.iloc[2, rows.columns.get_loc("VisitType")] = "Telehealth"
    with pytest.raises(InputValidationError) as exc:
        validate_input(rows)
    assert len(exc.value.problems) == 2


def test_skip_mode_keeps_valid_rows(sample_rows):
    rows = sample_rows.head(4).astype(object)
    rows.iloc[1, rows.columns.get_loc("Age")] = 250
    clean, rejected = validate_input(rows, errors="skip")
    assert len(clean) == 3 and rows.index[1] not in clean.index
    assert rejected["row"].tolist() == [rows.index[1]]


def test_skip_mode_with_no_valid_rows_raises(sample_rows):
    rows = sample_rows.head(2).assign(Age=-1)
    with pytest.raises(InputValidationError):
        validate_input(rows, errors="skip")


def test_unknown_errors_mode_rejected(sample_rows):
    with pytest.raises(ValueError, match="errors must be"):
        validate_input(sample_rows, errors="ignore")
