"""Data preprocessing pipeline for the NSL-KDD dataset.

Handles categorical encoding, numerical scaling, and prevents data leakage
by fitting exclusively on the training partition.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.data_loader import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


class NSLPreprocessor:
    """Preprocessor for NSL-KDD network intrusion detection.

    Encodes categorical features via OneHotEncoder and normalizes numerical
    features via MinMaxScaler (scaling to [0, 1] for uniform distance-based AIS
    matching). All transformers are fitted solely on training data.
    """

    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        feature_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.categorical_cols = categorical_cols or CATEGORICAL_COLUMNS
        self.numerical_cols = numerical_cols or NUMERICAL_COLUMNS
        self.feature_range = feature_range

        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.is_fitted = False
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "NSLPreprocessor":
        """Fit encoder and scaler on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw or loaded NSL-KDD training DataFrame.
        """
        # Fit OneHotEncoder on categorical features
        cat_data = df[self.categorical_cols].astype(str)
        self.encoder.fit(cat_data)
        encoded_cat_names = list(self.encoder.get_feature_names_out(self.categorical_cols))

        # Fit Scaler on numerical features
        num_data = df[self.numerical_cols].values.astype(np.float64)
        self.scaler.fit(num_data)

        # Store complete feature names
        self.feature_names_ = self.numerical_cols + encoded_cat_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted transformers.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        np.ndarray
            Normalized float64 matrix with numerical and encoded categorical features.
        """
        if not self.is_fitted:
            raise RuntimeError("NSLPreprocessor must be fitted before calling transform().")

        num_scaled = self.scaler.transform(df[self.numerical_cols].values.astype(np.float64))
        cat_encoded = self.encoder.transform(df[self.categorical_cols].astype(str))

        return np.hstack([num_scaled, cat_encoded]).astype(np.float64)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on training data and return transformed feature matrix."""
        return self.fit(df).transform(df)

    @staticmethod
    def extract_labels(df: pd.DataFrame) -> np.ndarray:
        """Extract binary target labels (0 = normal, 1 = attack)."""
        if "binary_label" in df.columns:
            return df["binary_label"].values.astype(int)
        if "label" in df.columns:
            return (df["label"].astype(str).str.strip().str.lower() != "normal").astype(int).values
        raise KeyError("DataFrame must contain 'binary_label' or 'label' column.")

    def save(self, filepath: Union[str, Path]) -> None:
        """Save the fitted preprocessor object to disk using joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "NSLPreprocessor":
        """Load a saved preprocessor object from disk."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessor artifact not found at {path.resolve()}")
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is of type {type(obj)}, expected {cls}.")
        return obj
