"""Data-processing workflow: the feature file and the training-time feature logic."""
import pandas as pd

from hc_pipeline import config
from hc_pipeline.features import add_prior_appointments


def test_feature_file_has_required_columns(features_df):
    needed = set(config.FEATURES) | {config.TARGET, "PatientID", "Date", "AppointmentID", "FacilityID"}
    assert needed <= set(features_df.columns)


def test_appointment_ids_unique(features_df):
    assert features_df["AppointmentID"].is_unique


def test_no_missing_values_in_model_inputs(features_df):
    assert features_df[config.FEATURES + [config.TARGET]].notna().all().all()


def test_categories_match_contract(features_df):
    for col, allowed in config.CATEGORICAL.items():
        assert set(features_df[col].unique()) == set(allowed), col


def test_numeric_values_within_valid_limits(features_df):
    for col, spec in config.NUMERIC.items():
        lo, hi = spec["valid"]
        assert features_df[col].between(lo, hi).all(), col


def test_target_matches_documented_rule(features_df):
    rule = ((features_df["Age"] >= 65).astype(int) + features_df["HasChronicCondition"]
            + (features_df["SeverityScore"] >= 3).astype(int)
            + (features_df["PriorAdmissionsCount"] >= 1).astype(int))
    assert ((rule >= 2).astype(int) == features_df[config.TARGET]).all()


def test_model_features_are_not_constant(features_df):
    assert all(features_df[c].nunique() > 1 for c in config.FEATURES)


def test_prior_appointments_counts_only_earlier_visits():
    # ID order deliberately disagrees with date order, so sorting by ID alone would fail.
    visits = pd.DataFrame({
        "PatientID": ["P1", "P1", "P1", "P2"],
        "Date": ["2025-03-01", "2025-01-01", "2025-02-01", "2025-01-01"],
        "AppointmentID": ["A1", "A2", "A3", "A4"],
    })
    out = add_prior_appointments(visits).set_index("AppointmentID")["PriorAppointmentsCount"]
    assert out.to_dict() == {"A1": 2, "A2": 0, "A3": 1, "A4": 0}


def test_prior_appointments_same_day_ordered_by_id():
    visits = pd.DataFrame({"PatientID": ["P1", "P1"], "Date": ["2025-01-01"] * 2,
                           "AppointmentID": ["A2", "A1"]})
    out = add_prior_appointments(visits).set_index("AppointmentID")["PriorAppointmentsCount"]
    assert out.to_dict() == {"A1": 0, "A2": 1}


def test_point_in_time_feature_never_exceeds_week6_feature(features_df):
    assert (features_df["PriorAppointmentsCount"] <= features_df["PreviousAppointmentsCount"]).all()


def test_feature_preparation_does_not_mutate_input():
    visits = pd.DataFrame({"PatientID": ["P1"], "Date": ["2025-01-01"], "AppointmentID": ["A1"]})
    before = visits.copy()
    add_prior_appointments(visits)
    pd.testing.assert_frame_equal(visits, before)
