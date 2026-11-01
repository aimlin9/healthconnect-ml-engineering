"""Single source of truth for the High-Risk Patient pipeline's input contract and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = ROOT / "data" / "feature_engineered" / "healthconnect_features.csv"
MODEL_PATH = ROOT / "models" / "high_risk_pipeline.joblib"
META_PATH = ROOT / "models" / "high_risk_pipeline_meta.json"

MODEL_VERSION = "1.1.0-week7"
TARGET = "HighRiskPatient"

# Canonical category values. Matching at inference is case/whitespace-insensitive.
CATEGORICAL = {
    "Gender": ["Female", "Male"],
    "VisitType": ["Consultation", "Emergency", "Follow-up"],
}

# Hard validity limits (a value outside these is rejected as impossible) and the
# range seen in training (a value outside it is scored, but logged as a warning).
NUMERIC = {
    "Age": {"valid": (0, 120), "integer": True},
    "DoctorYearsExperience": {"valid": (0, 60), "integer": True},
    "PriorAppointmentsCount": {"valid": (0, 1000), "integer": True},
    "DepartmentWorkload": {"valid": (0, 100_000), "integer": True},
}

FEATURES = list(CATEGORICAL) + list(NUMERIC)

# Week 6 used `HasInsurance` (constant 1 in all 5,000 rows -> no information) and
# `PreviousAppointmentsCount` (total visits - 1, which counts future visits).
# Week 7 drops the first and replaces the second with a point-in-time count.
WEEK6_FEATURES = ["Age", "Gender", "HasInsurance", "VisitType", "DoctorYearsExperience",
                  "PreviousAppointmentsCount", "DepartmentWorkload"]

RF_PARAMS = {"n_estimators": 300, "max_depth": 6, "class_weight": "balanced", "random_state": 42}
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Threshold is tuned on a validation split carved from the training data (never
# the test set): the highest threshold whose validation recall meets this target.
TARGET_RECALL = 0.80
