"""Evaluation metrics and benchmark reporting module."""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    train_time: float = 0.0,
) -> Dict[str, Any]:
    """Compute comprehensive intrusion detection classification metrics.

    Parameters
    ----------
    model : Any
        Trained classifier implementing .predict() (and optionally .predict_proba()).
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True binary labels (0 = normal, 1 = attack).
    model_name : str
        Human-readable name of the model.
    train_time : float
        Elapsed training duration in seconds.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing accuracy, precision, recall, f1, fpr, fnr, roc_auc,
        confusion matrix, execution times, and throughput.
    """
    n_samples = len(y_test)

    # Measure prediction latency
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter() - start_pred

    throughput = float(n_samples / max(1e-6, pred_time))

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # True Negative Rate (Specificity)
    specificity = float(tn / max(1, (tn + fp)))
    # False Positive Rate: FP / (FP + TN)
    fpr = float(fp / max(1, (fp + tn)))
    # False Negative Rate: FN / (FN + TP)
    fnr = float(fn / max(1, (fn + tp)))
    # Matthews Correlation Coefficient
    mcc = float(matthews_corrcoef(y_test, y_pred))

    # Compute ROC AUC if model provides probabilistic scores
    roc_auc: Optional[float] = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, proba))
        except Exception:
            roc_auc = None
    elif hasattr(model, "score_samples"):
        try:
            scores = model.score_samples(X_test)
            roc_auc = float(roc_auc_score(y_test, scores))
        except Exception:
            roc_auc = None

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist(),
        "training_time_sec": float(train_time),
        "prediction_time_sec": float(pred_time),
        "throughput_samples_per_sec": throughput,
        "total_test_samples": int(n_samples),
    }

    return metrics


def save_metrics(
    metrics_list: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    json_filename: str = "metrics.json",
    csv_filename: str = "metrics.csv",
) -> pd.DataFrame:
    """Save metrics to JSON and CSV formats."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Write JSON
    json_path = out_path / json_filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=4)

    # 2. Format DataFrame for CSV
    records = []
    for m in metrics_list:
        rec = {k: v for k, v in m.items() if k != "confusion_matrix"}
        records.append(rec)

    df_metrics = pd.DataFrame(records)
    csv_path = out_path / csv_filename
    df_metrics.to_csv(csv_path, index=False)

    return df_metrics


def load_metrics(metrics_file: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load evaluation metrics from JSON file."""
    path = Path(metrics_file)
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
