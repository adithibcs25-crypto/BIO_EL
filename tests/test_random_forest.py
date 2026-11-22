"""Unit tests for the Random Forest baseline wrapper."""

import numpy as np
import pytest

from src.random_forest import train_random_forest, save_random_forest, load_random_forest


def test_random_forest_training_and_persistence(tmp_path):
    rng = np.random.default_rng(42)
    X_train = rng.uniform(0.0, 1.0, size=(40, 5))
    y_train = np.array([0] * 20 + [1] * 20)

    model, train_time = train_random_forest(X_train, y_train, n_estimators=10, max_depth=5, random_state=42)

    assert model is not None
    assert train_time >= 0.0

    preds = model.predict(X_train)
    assert len(preds) == 40

    save_path = tmp_path / "test_rf.joblib"
    save_random_forest(model, save_path)
    assert save_path.exists()

    loaded = load_random_forest(save_path)
    loaded_preds = loaded.predict(X_train)
    np.testing.assert_array_equal(preds, loaded_preds)
