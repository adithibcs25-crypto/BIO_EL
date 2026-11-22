"""Random Forest baseline classifier for comparative intrusion detection benchmarking."""

from pathlib import Path
import time
from typing import Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = 16,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, float]:
    """Train a Scikit-learn RandomForestClassifier on the same feature matrix.

    Returns
    -------
    Tuple[RandomForestClassifier, float]
        (trained_model, training_time_seconds)
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    return model, train_time


def save_random_forest(model: RandomForestClassifier, filepath: Union[str, Path]) -> None:
    """Save the Random Forest model to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_random_forest(filepath: Union[str, Path]) -> RandomForestClassifier:
    """Load a trained Random Forest model from disk."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Random Forest model not found at {path.resolve()}")
    return joblib.load(path)
