"""Unit tests for the evaluation metrics module."""

import numpy as np
import pytest

from src.evaluation import evaluate_model, save_metrics


class DummyModel:
    def __init__(self, predictions, probas=None):
        self.predictions = predictions
        self.probas = probas

    def predict(self, X):
        return self.predictions

    def predict_proba(self, X):
        if self.probas is not None:
            return self.probas
        return np.column_stack([1.0 - self.predictions, self.predictions])


def test_evaluate_model_metrics():
    # 5 samples: 3 normals, 2 attacks
    y_true = np.array([0, 0, 0, 1, 1])
    # Predictions: 1 true negative, 2 false positives, 0 false negatives, 2 true positives
    y_pred = np.array([0, 1, 1, 1, 1])

    dummy = DummyModel(y_pred)
    metrics = evaluate_model(dummy, np.zeros((5, 2)), y_true, model_name="TestDummy", train_time=1.5)

    assert metrics["model_name"] == "TestDummy"
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 2
    assert metrics["false_negatives"] == 0
    assert metrics["true_positives"] == 2

    # Accuracy: (1 + 2) / 5 = 0.6
    assert metrics["accuracy"] == pytest.approx(0.6)
    # Recall: 2 / 2 = 1.0
    assert metrics["recall"] == pytest.approx(1.0)
    # FPR: 2 / (2 + 1) = 2/3
    assert metrics["false_positive_rate"] == pytest.approx(2 / 3)
    # Specificity: 1 / (1 + 2) = 1/3
    assert metrics["specificity"] == pytest.approx(1 / 3)
    assert "mcc" in metrics


def test_save_and_load_metrics(tmp_path):
    metrics_list = [{
        "model_name": "TestModel",
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.98,
        "f1_score": 0.95,
        "false_positive_rate": 0.03,
        "false_negative_rate": 0.02,
        "roc_auc": 0.99,
        "true_negatives": 97,
        "false_positives": 3,
        "false_negatives": 2,
        "true_positives": 98,
        "confusion_matrix": [[97, 3], [2, 98]],
        "training_time_sec": 1.2,
        "prediction_time_sec": 0.05,
        "throughput_samples_per_sec": 4000.0,
        "total_test_samples": 200,
    }]

    df = save_metrics(metrics_list, tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert len(df) == 1
