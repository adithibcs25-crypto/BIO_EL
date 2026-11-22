"""Negative Selection Algorithm (NSA) for Artificial Immune Systems.

Generates self-tolerant detectors by censoring candidate detectors that match
the normal traffic (Self) profile within a specified affinity matching radius.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class NegativeSelection:
    """Negative Selection Algorithm for network intrusion detection.

    Implements candidate detector generation and self-tolerance censoring
    using vectorized matrix distance calculations.

    Parameters
    ----------
    num_detectors : int, default=100
        Target number of mature, self-tolerant detectors to generate.
    matching_radius : float, default=0.5
        Affinity threshold (Euclidean distance radius). Candidates within this
        distance to any Self sample are eliminated to prevent autoimmune reactions.
    max_trials : int, default=10000
        Maximum candidate generation attempts before terminating.
    batch_size : int, default=250
        Number of candidate detectors generated and evaluated per batch.
    self_subsample : Optional[int], default=5000
        Maximum number of Self instances used during censoring to ensure
        high responsiveness on standard CPU hardware. If None, uses all Self data.
    seed : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_detectors: int = 100,
        matching_radius: float = 0.5,
        max_trials: int = 10000,
        batch_size: int = 250,
        self_subsample: Optional[int] = 5000,
        seed: int = 42,
    ):
        self.num_detectors = num_detectors
        self.matching_radius = matching_radius
        self.max_trials = max_trials
        self.batch_size = batch_size
        self.self_subsample = self_subsample
        self.seed = seed

        self.detectors_: Optional[np.ndarray] = None  # Shape: (K, D)
        self.stats_: Dict[str, Union[int, float]] = {}

    @staticmethod
    def _compute_min_sq_distances(
        candidates: np.ndarray,
        self_samples: np.ndarray,
        chunk_size: int = 1000
    ) -> np.ndarray:
        """Compute the minimum squared Euclidean distance from each candidate to Self.

        Utilizes the expansion: ||c - s||^2 = ||c||^2 + ||s||^2 - 2 * c . s^T
        computed in Self chunks via BLAS GEMM for maximum vectorization and memory efficiency.
        """
        n_candidates = candidates.shape[0]
        c_sq = np.sum(candidates ** 2, axis=1)  # (n_candidates,)
        min_sq = np.full(n_candidates, np.inf, dtype=np.float64)

        n_self = self_samples.shape[0]
        for start in range(0, n_self, chunk_size):
            end = min(start + chunk_size, n_self)
            s_chunk = self_samples[start:end]
            s_sq = np.sum(s_chunk ** 2, axis=1)  # (m,)

            # (n_candidates, m)
            dist_sq_chunk = c_sq[:, None] + s_sq[None, :] - 2.0 * np.dot(candidates, s_chunk.T)
            dist_sq_chunk = np.maximum(dist_sq_chunk, 0.0)

            chunk_min = np.min(dist_sq_chunk, axis=1)
            min_sq = np.minimum(min_sq, chunk_min)

        return min_sq

    def fit(
        self,
        self_samples: np.ndarray,
        non_self_candidates: Optional[np.ndarray] = None
    ) -> "NegativeSelection":
        """Generate and censor detectors against Self profile.

        Parameters
        ----------
        self_samples : np.ndarray
            Matrix of normal network connections, shape (N, D).
        non_self_candidates : Optional[np.ndarray], default=None
            Optional known non-self instances or anomaly seeds to inform candidate
            proposal distribution alongside uniform stochastic exploration.
        """
        rng = np.random.default_rng(self.seed)

        # Ensure 2D float array
        self_data = np.asarray(self_samples, dtype=np.float64)
        if self_data.ndim != 2:
            raise ValueError(f"self_samples must be 2D, got shape {self_data.shape}")

        n_self, n_features = self_data.shape

        # Subsample Self if configured to avoid latency on large datasets
        if self.self_subsample is not None and n_self > self.self_subsample:
            idx = rng.choice(n_self, size=self.self_subsample, replace=False)
            censoring_self = self_data[idx]
        else:
            censoring_self = self_data

        accepted_detectors: List[np.ndarray] = []
        trials_count = 0
        total_candidates_generated = 0
        total_rejected = 0

        # Determine feature space boundaries for candidate proposal
        if non_self_candidates is not None and len(non_self_candidates) > 0:
            all_pts = np.vstack([self_data, non_self_candidates])
            low_bound = np.min(all_pts, axis=0)
            high_bound = np.max(all_pts, axis=0)
        else:
            # For normalized data, candidate space spans the [0, 1] domain
            low_bound = np.minimum(0.0, np.min(self_data, axis=0))
            high_bound = np.maximum(1.0, np.max(self_data, axis=0))

        # Ensure valid range
        span = np.where(high_bound - low_bound > 1e-6, high_bound - low_bound, 1.0)
        high_bound = low_bound + span

        r_sq = float(self.matching_radius ** 2)

        while len(accepted_detectors) < self.num_detectors and trials_count < self.max_trials:
            batch_count = min(self.batch_size, self.max_trials - trials_count)

            # Generate candidate pool
            if non_self_candidates is not None and len(non_self_candidates) > 0 and rng.random() < 0.4:
                # Guided proposal: slight perturbation around non-self examples
                seed_idx = rng.choice(len(non_self_candidates), size=batch_count, replace=True)
                noise = rng.normal(0.0, 0.05, size=(batch_count, n_features))
                candidates = np.clip(non_self_candidates[seed_idx] + noise, low_bound, high_bound)
            else:
                # Stochastic exploration across the entire normalized feature domain
                candidates = rng.uniform(
                    low=low_bound,
                    high=high_bound,
                    size=(batch_count, n_features)
                )

            # Censoring: check minimum distance to Self
            min_sq_dists = self._compute_min_sq_distances(candidates, censoring_self)

            # Candidate survives if it does NOT match any Self sample within radius
            survived_mask = min_sq_dists > r_sq
            survivors = candidates[survived_mask]

            total_rejected += int(np.sum(~survived_mask))
            total_candidates_generated += batch_count
            trials_count += batch_count

            for det in survivors:
                accepted_detectors.append(det)
                if len(accepted_detectors) >= self.num_detectors:
                    break

        if len(accepted_detectors) == 0:
            # Fallback: propose points far from self centroid
            self_mean = np.mean(self_data, axis=0)
            # Opposite corner in feature space
            opposite = np.where(self_mean < 0.5, high_bound, low_bound)
            accepted_detectors.append(opposite)

        self.detectors_ = np.array(accepted_detectors[:self.num_detectors], dtype=np.float64)

        self.stats_ = {
            "num_detectors": int(len(self.detectors_)),
            "candidates_generated": total_candidates_generated,
            "candidates_rejected": total_rejected,
            "acceptance_rate": float(len(self.detectors_) / max(1, total_candidates_generated)),
            "matching_radius": float(self.matching_radius),
        }

        return self

    def matches(self, samples: np.ndarray, radius: Optional[float] = None) -> np.ndarray:
        """Determine if input samples are matched by any mature detector.

        Parameters
        ----------
        samples : np.ndarray
            Network connection feature vectors, shape (N, D).
        radius : Optional[float]
            Detection radius. Defaults to self.matching_radius if not provided.

        Returns
        -------
        np.ndarray
            Boolean 1D array of length N (True if matched by at least one detector).
        """
        if self.detectors_ is None or len(self.detectors_) == 0:
            raise RuntimeError("NegativeSelection has not been fitted with detectors.")

        r = self.matching_radius if radius is None else radius
        r_sq = float(r ** 2)

        # Min sq distance from each sample to any detector
        min_sq = self._compute_min_sq_distances(samples, self.detectors_, chunk_size=len(self.detectors_))
        return min_sq <= r_sq
