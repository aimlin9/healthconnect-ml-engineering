"""Preprocessing <-> model interaction, outputs, persistence and reproducibility."""
import json
import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from hc_pipeline import InputValidationError, ModelLoadError, config, load, predict, save, train


def test_single_patient_scored(artifact, sample_rows):
    """Week 6 defect W6-03: one row could not be scored."""
    out, _ = predict(*artifact, sample_rows.head(1))
    assert len(out) == 1


def test_single_category_batch_scored(artifact, features_df):
    """Week 6 defect W6-04: a batch missing some categories could not be scored."""
    rows = features_df.loc[features_df["VisitType"] == "Emergency", config.FEATURES].head(20)
    out, _ = predict(*artifact, rows)
    assert len(out) == 20


def test_row_scores_independent_of_batch(artifact, sample_rows):
    batch, _ = predict(*artifact, sample_rows)
    singles = pd.concat([predict(*artifact, sample_rows.loc[[i]])[0] for i in sample_rows.index])
    pd.testing.assert_series_equal(batch["risk_probability"], singles["risk_probability"])


def test_output_contract(artifact, sample_rows):
    model, meta = artifact
    out, rejected = predict(model, meta, sample_rows)
    assert list(out.columns) == ["risk_probability", "high_risk_flag", "threshold", "model_version",
                                 "status", "model_status"]
    assert out.index.equals(sample_rows.index)
    assert out["risk_probability"].between(0, 1).all()
    assert set(out["high_risk_flag"]) <= {0, 1}
    assert ((out["risk_probability"] >= meta["threshold"]).astype(int) == out["high_risk_flag"]).all()
    assert (out["model_version"] == config.MODEL_VERSION).all()
    assert (out["status"] == "SCORED").all()
    assert out["model_status"].str.startswith("DEMO ONLY").all()
    assert rejected.empty


def test_extrapolation_visible_in_row_status(artifact, sample_rows):
    model, meta = artifact
    rows = sample_rows.head(2).copy()
    rows.iloc[0, rows.columns.get_loc("Age")] = 110  # valid, but above the training max of 90
    out, _ = predict(model, meta, rows)
    assert out["status"].iloc[0] == "SCORED_WITH_WARNINGS: Age outside training range [0, 90]"
    assert out["status"].iloc[1] == "SCORED"


def test_empty_input_returns_empty_output(artifact, sample_rows):
    model, meta = artifact
    out, rejected = predict(model, meta, sample_rows.head(0))
    assert out.empty and rejected.empty
    assert "status" in out.columns


def test_invalid_input_never_reaches_model(artifact, sample_rows):
    rows = sample_rows.head(2).assign(VisitType="Telehealth")
    with pytest.raises(InputValidationError):
        predict(*artifact, rows)


def test_skip_mode_scores_only_valid_rows(artifact, sample_rows):
    rows = sample_rows.head(5).astype(object)
    rows.iloc[0, rows.columns.get_loc("Age")] = 250
    out, rejected = predict(*artifact, rows, errors="skip")
    assert len(out) == 4 and rejected["row"].tolist() == [rows.index[0]]


def test_extrapolation_logged(artifact, sample_rows, caplog):
    rows = sample_rows.head(1).assign(DepartmentWorkload=5000)
    with caplog.at_level(logging.WARNING, logger="hc_pipeline"):
        predict(*artifact, rows)
    assert "outside the training range" in caplog.text


def test_scoring_logged(artifact, sample_rows, caplog):
    with caplog.at_level(logging.INFO, logger="hc_pipeline"):
        predict(*artifact, sample_rows)
    assert "scored 10 row(s)" in caplog.text


def test_preprocessing_output_shape(artifact):
    model, _ = artifact
    names = model.named_steps["preprocess"].get_feature_names_out()
    n_cat = sum(len(v) for v in config.CATEGORICAL.values())
    assert len(names) == n_cat + len(config.NUMERIC)


def test_heldout_performance_beats_baseline(artifact, features_df, split):
    """Model meets the Week 6 bar on the same held-out split (AUC >= 0.70)."""
    model, _ = artifact
    _, te = split
    y = features_df.loc[te, config.TARGET]
    auc = roc_auc_score(y, model.predict_proba(features_df.loc[te, config.FEATURES])[:, 1])
    assert auc >= 0.70


def test_metadata_matches_artifact(artifact):
    _, meta = artifact
    assert meta["features"] == config.FEATURES
    assert 0 < meta["threshold"] < 1
    assert meta["test_metrics"]["recall"] >= config.TARGET_RECALL - 0.05


def test_training_is_deterministic(features_df, split):
    tr, te = split
    X, y = features_df.loc[tr, config.FEATURES], features_df.loc[tr, config.TARGET]
    (m1, t1), (m2, t2) = train(X, y), train(X, y)
    Xte = features_df.loc[te, config.FEATURES]
    assert t1 == t2
    assert np.array_equal(m1.predict_proba(Xte), m2.predict_proba(Xte))


def test_saved_artifact_reproduces_fresh_training(artifact, features_df, split):
    tr, te = split
    fresh, _ = train(features_df.loc[tr, config.FEATURES], features_df.loc[tr, config.TARGET])
    Xte = features_df.loc[te, config.FEATURES]
    assert np.array_equal(artifact[0].predict_proba(Xte), fresh.predict_proba(Xte))


def test_save_load_roundtrip(tmp_path, features_df, split, sample_rows):
    tr, _ = split
    X, y = features_df.loc[tr, config.FEATURES], features_df.loc[tr, config.TARGET]
    model, t = train(X, y)
    mp, mt = tmp_path / "m.joblib", tmp_path / "m.json"
    save(model, t, X, {}, mp, mt)
    loaded, meta = load(mp, mt)
    assert np.array_equal(model.predict_proba(sample_rows), loaded.predict_proba(sample_rows))
    assert meta["threshold"] == t


def test_missing_artifact_gives_clear_error(tmp_path):
    with pytest.raises(ModelLoadError, match="run scripts/03_train_pipeline.py"):
        load(tmp_path / "none.joblib", tmp_path / "none.json")


def test_feature_mismatch_refuses_to_load(tmp_path):
    from hc_pipeline.config import META_PATH, MODEL_PATH
    meta = json.loads(META_PATH.read_text())
    meta["features"] = config.WEEK6_FEATURES
    bad = tmp_path / "meta.json"
    bad.write_text(json.dumps(meta))
    with pytest.raises(ModelLoadError, match="expects features"):
        load(MODEL_PATH, bad)
