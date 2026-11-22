"""End-to-end integration test verifying complete data loading, training, and evaluation pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_nsl_kdd
from src.preprocessing import NSLPreprocessor
from src.hybrid_ais import HybridAISClassifier
from src.random_forest import train_random_forest
from src.evaluation import evaluate_model, save_metrics


def test_full_pipeline_on_real_dataset_sample():
    # Test on a small slice of actual NSL-KDD raw data
    raw_train = Path("Bio_EL_Intrusion_Detection/data/raw/KDDTrain+.txt")
    raw_test = Path("Bio_EL_Intrusion_Detection/data/raw/KDDTest+.txt")

    if not raw_train.exists() or not raw_test.exists():
        pytest.skip("Raw NSL-KDD files not present in data/raw")

    df_train = load_nsl_kdd(str(raw_train), sample_size=300, random_state=42)
    df_test = load_nsl_kdd(str(raw_test), sample_size=150, random_state=42)

    preprocessor = NSLPreprocessor()
    X_train = preprocessor.fit_transform(df_train)
    y_train = preprocessor.extract_labels(df_train)

    X_test = preprocessor.transform(df_test)
    y_test = preprocessor.extract_labels(df_test)

    # AIS
    ais = HybridAISClassifier(num_detectors=10, matching_radius=0.5, clonal_generations=2, seed=42)
    ais.fit(X_train, y_train)

    # RF
    rf, rf_time = train_random_forest(X_train, y_train, n_estimators=10, random_state=42)

    # Eval
    metrics_ais = evaluate_model(ais, X_test, y_test, model_name="Hybrid AIS", train_time=ais.train_time_)
    metrics_rf = evaluate_model(rf, X_test, y_test, model_name="Random Forest", train_time=rf_time)

    assert 0.0 <= metrics_ais["accuracy"] <= 1.0
    assert 0.0 <= metrics_rf["accuracy"] <= 1.0
    assert 0.0 <= metrics_ais["false_positive_rate"] <= 1.0
    assert metrics_ais["total_test_samples"] == 150
