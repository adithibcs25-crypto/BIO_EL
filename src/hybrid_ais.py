"""Hybrid Artificial Immune System (AIS) Classifier.

Integrates Negative Selection (censoring against Self profile) with Clonal
Selection (affinity maturation and somatic hypermutation on validation traffic)
into a unified, Scikit-learn compatible binary intrusion detector.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.negative_selection import NegativeSelection
from src.clonal_selection import ClonalSelection


class HybridAISClassifier:
    """Hybrid Artificial Immune System Classifier for Network Intrusion Detection.

    Parameters
    ----------
    num_detectors : int, default=100
        Target detector repertoire size.
    matching_radius : float, default=0.5
        Affinity threshold (Euclidean radius in normalized feature space).
    clonal_generations : int, default=8
        Number of affinity maturation generations.
    clone_factor : float, default=5.0
        Clone multiplication scale factor.
    mutation_rate : float, default=0.1
        Base mutation standard deviation for somatic hypermutation.
    selection_ratio : float, default=0.5
        Proportion of top-performing detectors selected for cloning.
    fp_penalty : float, default=3.0
        Penalty weight for false positives during affinity calculation.
    val_ratio : float, default=0.2
        Fraction of training data used for internal validation / tuning (never using final test data).
    self_subsample : int, default=5000
        Maximum Self instances used during negative selection censoring for computational tractability.
    max_trials : int, default=10000
        Maximum candidate generation attempts for negative selection.
    seed : int, default=42
        Random seed for full reproducibility.
    """

    def __init__(
        self,
        num_detectors: int = 100,
        matching_radius: float = 0.5,
        clonal_generations: int = 8,
        clone_factor: float = 5.0,
        mutation_rate: float = 0.1,
        selection_ratio: float = 0.5,
        fp_penalty: float = 3.0,
        val_ratio: float = 0.2,
        self_subsample: int = 5000,
        max_trials: int = 10000,
        seed: int = 42,
    ):
        self.num_detectors = num_detectors
        self.matching_radius = matching_radius
        self.clonal_generations = clonal_generations
        self.clone_factor = clone_factor
        self.mutation_rate = mutation_rate
        self.selection_ratio = selection_ratio
        self.fp_penalty = fp_penalty
        self.val_ratio = val_ratio
        self.self_subsample = self_subsample
        self.max_trials = max_trials
        self.seed = seed

        # Learned components
        self.detectors_: Optional[np.ndarray] = None  # (K, D)
        self.detector_affinities_: Optional[np.ndarray] = None
        self.is_fitted_ = False
        self.train_time_: float = 0.0

        # Sub-modules
        self.ns_engine_: Optional[NegativeSelection] = None
        self.cs_engine_: Optional[ClonalSelection] = None
        self.history_: List[Dict[str, Any]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridAISClassifier":
        """Train the Hybrid AIS on training data.

        Splits data internally into train-self and validation-tuning partitions to
        prevent any leakage or overfitting to unseen evaluation data.

        Parameters
        ----------
        X : np.ndarray
            Normalized feature matrix, shape (N, D).
        y : np.ndarray
            Binary labels (0 = normal, 1 = attack), shape (N,).
        """
        start_time = time.perf_counter()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int32)

        # 1. Split into training (for Self profile & seeds) and validation (for Clonal affinity tuning)
        if self.val_ratio > 0.0 and len(np.unique(y_arr)) > 1:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_arr,
                y_arr,
                test_size=self.val_ratio,
                stratify=y_arr,
                random_state=self.seed
            )
        else:
            X_tr, X_val, y_tr, y_val = X_arr, X_arr, y_arr, y_arr

        # Isolate Self (normal) and Non-Self (attack) subsets
        self_samples = X_tr[y_tr == 0]
        attack_samples = X_tr[y_tr == 1]

        if len(self_samples) == 0:
            raise ValueError("Training data must contain at least one normal (Self) instance.")

        # 2. Phase 1: Negative Selection (Generate Self-Tolerant Detectors)
        self.ns_engine_ = NegativeSelection(
            num_detectors=self.num_detectors,
            matching_radius=self.matching_radius,
            max_trials=self.max_trials,
            self_subsample=self.self_subsample,
            seed=self.seed,
        )
        self.ns_engine_.fit(self_samples, non_self_candidates=attack_samples)
        raw_detectors = self.ns_engine_.detectors_

        # 3. Phase 2: Clonal Selection (Affinity Maturation & Hypermutation)
        if self.clonal_generations > 0 and len(X_val) > 0:
            self.cs_engine_ = ClonalSelection(
                generations=self.clonal_generations,
                clone_factor=self.clone_factor,
                mutation_rate=self.mutation_rate,
                selection_ratio=self.selection_ratio,
                fp_penalty=self.fp_penalty,
                matching_radius=self.matching_radius,
                seed=self.seed,
            )
            self.cs_engine_.optimize(
                initial_detectors=raw_detectors,
                X_val=X_val,
                y_val=y_val,
                self_samples=self_samples,
            )
            self.detectors_ = self.cs_engine_.detectors_
            self.detector_affinities_ = self.cs_engine_.affinities_
            self.history_ = self.cs_engine_.history_
        else:
            self.detectors_ = raw_detectors
            self.detector_affinities_ = np.ones(len(raw_detectors), dtype=np.float64)
            self.history_ = []

        self.train_time_ = time.perf_counter() - start_time
        self.is_fitted_ = True
        return self

    def _compute_min_sq_distances(self, X: np.ndarray, chunk_size: int = 5000) -> np.ndarray:
        """Compute the minimum squared Euclidean distance from each sample to any detector.

        Vectorized with BLAS matrix products in sample chunks to handle large test sets.
        """
        if not self.is_fitted_ or self.detectors_ is None:
            raise RuntimeError("Classifier must be fitted before computing distances.")

        X_mat = np.asarray(X, dtype=np.float64)
        n_samples = X_mat.shape[0]
        d_sq = np.sum(self.detectors_ ** 2, axis=1)  # (K,)
        min_sq = np.full(n_samples, np.inf, dtype=np.float64)

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            x_chunk = X_mat[start:end]
            x_sq = np.sum(x_chunk ** 2, axis=1)  # (m,)

            # (m, K) distance matrix
            dist_sq_chunk = x_sq[:, None] + d_sq[None, :] - 2.0 * np.dot(x_chunk, self.detectors_.T)
            dist_sq_chunk = np.maximum(dist_sq_chunk, 0.0)

            min_sq[start:end] = np.min(dist_sq_chunk, axis=1)

        return min_sq

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary traffic class for input samples.

        Returns 1 (attack) if the sample falls within the matching radius of any
        mature detector, and 0 (normal) otherwise.
        """
        min_sq_dists = self._compute_min_sq_distances(X)
        r_sq = float(self.matching_radius ** 2)
        return (min_sq_dists <= r_sq).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute raw continuous anomaly score in [0, 1].

        Score is proportional to proximity to the nearest detector:
        S(x) = exp(-dist^2 / (2 * r^2)).
        """
        min_sq = self._compute_min_sq_distances(X)
        scale = 2.0 * (self.matching_radius ** 2)
        return np.exp(-min_sq / max(scale, 1e-6))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities: [P(normal), P(attack)]."""
        anomaly_scores = self.score_samples(X)
        p_attack = np.clip(anomaly_scores, 0.0, 1.0)
        p_normal = 1.0 - p_attack
        return np.column_stack([p_normal, p_attack])

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize and save the trained Hybrid AIS model."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "HybridAISClassifier":
        """Load a saved Hybrid AIS model from disk."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found at {path.resolve()}")
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object of type {type(obj)}, expected {cls}.")
        return obj
