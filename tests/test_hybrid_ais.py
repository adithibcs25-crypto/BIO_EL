"""Unit tests for the HybridAISClassifier module."""

import numpy as np
import pytest

from src.hybrid_ais import HybridAISClassifier


def test_hybrid_ais_fit_predict():
    rng = np.random.default_rng(42)
    # 50 normal samples near 0.2, 50 attack samples near 0.8
    X_train = np.vstack([
        rng.normal(0.2, 0.05, size=(50, 6)),
        rng.normal(0.8, 0.05, size=(50, 6)),
    ])
    X_train = np.clip(X_train, 0.0, 1.0)
    y_train = np.array([0] * 50 + [1] * 50)

    ais = HybridAISClassifier(
        num_detectors=10,
        matching_radius=0.45,
        clonal_generations=3,
        val_ratio=0.2,
        seed=42
    )
    ais.fit(X_train, y_train)

    assert ais.is_fitted_
    assert ais.detectors_ is not None
    assert len(ais.detectors_) == 10

    # Predictions
    preds = ais.predict(X_train)
    assert len(preds) == 100
    assert set(np.unique(preds)).issubset({0, 1})

    # Probabilities
    proba = ais.predict_proba(X_train)
    assert proba.shape == (100, 2)
    # Probabilities should sum to 1.0
    np.testing.assert_allclose(proba[:, 0] + proba[:, 1], 1.0, atol=1e-5)


def test_hybrid_ais_persistence(tmp_path):
    rng = np.random.default_rng(42)
    X = np.clip(rng.normal(0.5, 0.2, size=(30, 4)), 0.0, 1.0)
    y = np.array([0] * 15 + [1] * 15)

    ais = HybridAISClassifier(num_detectors=5, matching_radius=0.4, clonal_generations=2, seed=42)
    ais.fit(X, y)

    save_file = tmp_path / "ais_model.joblib"
    ais.save(save_file)

    loaded = HybridAISClassifier.load(save_file)
    assert loaded.is_fitted_

    orig_preds = ais.predict(X)
    loaded_preds = loaded.predict(X)
    np.testing.assert_array_equal(orig_preds, loaded_preds)
