"""Unit tests for the NSLPreprocessor module."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import COLUMN_NAMES
from src.preprocessing import NSLPreprocessor


@pytest.fixture
def sample_raw_dataframe():
    """Create a synthetic mini-DataFrame matching NSL-KDD structure."""
    data = []
    for i in range(20):
        row = [
            0,  # duration
            "tcp" if i % 2 == 0 else "udp",  # protocol_type
            "http" if i % 3 == 0 else "smtp",  # service
            "SF" if i % 2 == 0 else "S0",  # flag
            100 + i * 10,  # src_bytes
            200 + i * 20,  # dst_bytes
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1 + i, 1 + i, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            100, 100, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            "normal" if i < 12 else "neptune",  # label
            20  # difficulty
        ]
        data.append(row)
    df = pd.DataFrame(data, columns=COLUMN_NAMES)
    df["binary_label"] = (df["label"] != "normal").astype(int)
    return df


def test_preprocessor_fit_transform(sample_raw_dataframe):
    preprocessor = NSLPreprocessor()
    X = preprocessor.fit_transform(sample_raw_dataframe)
    y = preprocessor.extract_labels(sample_raw_dataframe)

    assert preprocessor.is_fitted
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 20
    assert X.shape[1] > 38  # 38 numerical + one-hot columns
    # Verify values are normalized in [0, 1]
    assert np.all(X >= -1e-6)
    assert np.all(X <= 1.0 + 1e-6)
    assert len(y) == 20
    assert set(np.unique(y)).issubset({0, 1})


def test_preprocessor_no_data_leakage(sample_raw_dataframe):
    train_df = sample_raw_dataframe.iloc[:10]
    test_df = sample_raw_dataframe.iloc[10:]

    preprocessor = NSLPreprocessor()
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)

    assert X_train.shape[1] == X_test.shape[1]
    assert preprocessor.is_fitted


def test_preprocessor_persistence(tmp_path, sample_raw_dataframe):
    preprocessor = NSLPreprocessor()
    X_orig = preprocessor.fit_transform(sample_raw_dataframe)

    save_path = tmp_path / "test_prep.joblib"
    preprocessor.save(save_path)

    loaded = NSLPreprocessor.load(save_path)
    X_loaded = loaded.transform(sample_raw_dataframe)

    np.testing.assert_allclose(X_orig, X_loaded)
