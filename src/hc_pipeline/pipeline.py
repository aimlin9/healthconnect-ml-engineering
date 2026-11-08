"""Build, train, persist, load and serve the High-Risk Patient pipeline.

Preprocessing (one-hot encoding with a fixed category list) lives inside the
saved sklearn Pipeline, so training and inference cannot drift apart the way
re-running pd.get_dummies at inference did in Week 6.
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (CATEGORICAL, FEATURES, META_PATH, MODEL_PATH, MODEL_STATUS, MODEL_VERSION,
                     NUMERIC, RANDOM_STATE, RF_PARAMS, TARGET, TARGET_RECALL)
from .validation import validate_input

log = logging.getLogger("hc_pipeline")


class ModelLoadError(RuntimeError):
    pass


def build_pipeline():
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(categories=[CATEGORICAL[c] for c in CATEGORICAL],
                               handle_unknown="error"), list(CATEGORICAL)),
         ("num", "passthrough", list(NUMERIC))],
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocess", pre), ("model", RandomForestClassifier(**RF_PARAMS))])


def metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, pred), 3),
        "precision": round(precision_score(y_true, pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, pred), 3),
        "f1": round(f1_score(y_true, pred), 3),
        "roc_auc": round(roc_auc_score(y_true, proba), 3),
        "baseline_accuracy": round(max(np.mean(y_true), 1 - np.mean(y_true)), 3),
        "threshold": round(float(threshold), 3),
    }


def tune_threshold(y_true, proba, target_recall=TARGET_RECALL):
    """Highest threshold whose recall still meets target_recall (keeps precision as high as possible)."""
    for t in np.round(np.arange(0.95, 0.04, -0.01), 2):
        if recall_score(y_true, (proba >= t).astype(int)) >= target_recall:
            return float(t)
    return 0.05


def train(X, y, target_recall=TARGET_RECALL):
    """Tune the threshold on an inner validation split, then refit on all of X."""
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    threshold = tune_threshold(y_val, build_pipeline().fit(X_fit, y_fit).predict_proba(X_val)[:, 1], target_recall)
    model = build_pipeline().fit(X, y)
    log.info("trained on %d rows; tuned threshold %.2f (validation recall target %.2f)",
             len(X), threshold, target_recall)
    return model, threshold


def data_fingerprint(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def save(model, threshold, X_train, test_metrics, model_path=MODEL_PATH, meta_path=META_PATH):
    meta = {
        "model_version": MODEL_VERSION,
        "model_status": MODEL_STATUS,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "categories": CATEGORICAL,
        "training_range": {c: [int(X_train[c].min()), int(X_train[c].max())] for c in NUMERIC},
        "threshold": threshold,
        "target": TARGET,
        "n_train": int(len(X_train)),
        "train_data_fingerprint": data_fingerprint(X_train),
        "test_metrics": test_metrics,
    }
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("saved model %s to %s", MODEL_VERSION, model_path)
    return meta


def load(model_path=MODEL_PATH, meta_path=META_PATH):
    if not model_path.exists() or not meta_path.exists():
        raise ModelLoadError(f"model artifact not found ({model_path.name} / {meta_path.name}); "
                             "run scripts/03_train_pipeline.py first")
    meta = json.loads(meta_path.read_text())
    if meta.get("features") != FEATURES:
        raise ModelLoadError(f"artifact expects features {meta.get('features')}, code defines {FEATURES}")
    if meta.get("sklearn_version", "").split(".")[:2] != sklearn.__version__.split(".")[:2]:
        log.warning("model trained with scikit-learn %s, running %s — reinstall from requirements.txt",
                    meta.get("sklearn_version"), sklearn.__version__)
    model = joblib.load(model_path)
    log.info("loaded model %s (threshold %.2f)", meta["model_version"], meta["threshold"])
    return model, meta


def predict(model, meta, data, errors="raise"):
    """Validate input and return one row per accepted input row (original index kept):
    risk_probability, high_risk_flag, threshold, model_version, status, model_status.
    status is SCORED, or SCORED_WITH_WARNINGS: <reason> when a value lies outside the training
    range. Rejected rows (errors="skip") are returned as a second DataFrame."""
    start = time.perf_counter()
    X, rejected = validate_input(data, errors=errors)
    columns = ["risk_probability", "high_risk_flag", "threshold", "model_version", "status", "model_status"]
    if X.empty:
        return pd.DataFrame(columns=columns, index=X.index), rejected
    warnings = pd.Series("", index=X.index)
    for col, (lo, hi) in meta.get("training_range", {}).items():
        mask = (X[col] < lo) | (X[col] > hi)
        if mask.any():
            log.warning("%d row(s) have %s outside the training range [%d, %d]; "
                        "prediction is an extrapolation", int(mask.sum()), col, lo, hi)
            warnings[mask] += f"{col} outside training range [{lo}, {hi}]; "
    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "risk_probability": proba.round(4),
        "high_risk_flag": (proba >= meta["threshold"]).astype(int),
        "threshold": meta["threshold"],
        "model_version": meta["model_version"],
        "status": np.where(warnings == "", "SCORED", "SCORED_WITH_WARNINGS: " + warnings.str.rstrip("; ")),
        "model_status": meta.get("model_status", MODEL_STATUS),
    }, index=X.index)
    log.info("scored %d row(s), flagged %d high-risk, rejected %d, %.1f ms",
             len(out), int(out["high_risk_flag"].sum()),
             rejected["row"].nunique() if len(rejected) else 0, (time.perf_counter() - start) * 1000)
    return out, rejected
