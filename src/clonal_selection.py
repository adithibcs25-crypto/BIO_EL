"""Clonal Selection Algorithm (CSA) for Artificial Immune Systems.

Adapts and matures the detector repertoire through affinity maturation:
cloning high-affinity detectors, applying affinity-proportional hypermutation,
censoring self-reactive mutants, and retaining improved detectors over generations.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ClonalSelection:
    """Clonal Selection Algorithm for refining AIS detectors.

    Parameters
    ----------
    generations : int, default=10
        Number of clonal selection iterations.
    clone_factor : float, default=5.0
        Multiplier determining the number of clones generated per selected detector.
    mutation_rate : float, default=0.1
        Base mutation standard deviation for somatic hypermutation.
    selection_ratio : float, default=0.5
        Proportion of highest-affinity detectors selected for cloning.
    fp_penalty : float, default=3.0
        Penalty weight assigned to false positive detections.
    matching_radius : float, default=0.5
        Detection distance threshold.
    seed : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        generations: int = 10,
        clone_factor: float = 5.0,
        mutation_rate: float = 0.1,
        selection_ratio: float = 0.5,
        fp_penalty: float = 3.0,
        matching_radius: float = 0.5,
        seed: int = 42,
    ):
        self.generations = generations
        self.clone_factor = clone_factor
        self.mutation_rate = mutation_rate
        self.selection_ratio = selection_ratio
        self.fp_penalty = fp_penalty
        self.matching_radius = matching_radius
        self.seed = seed

        self.detectors_: Optional[np.ndarray] = None
        self.affinities_: Optional[np.ndarray] = None
        self.history_: List[Dict[str, Any]] = []

    def compute_affinity(
        self,
        detector: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        radius: float
    ) -> Tuple[float, int, int]:
        """Compute the affinity (fitness) of an individual detector.

        Affinity rewards true attack detections (TP) while heavily penalizing
        false alarms on normal traffic (FP).

        Returns
        -------
        Tuple[float, int, int]
            (affinity_score, true_positives, false_positives)
        """
        # Euclidean distances to all validation samples: ||X_val - detector||
        dists = np.linalg.norm(X_val - detector, axis=1)
        matches = dists <= radius

        tp = int(np.sum(matches & (y_val == 1)))
        fp = int(np.sum(matches & (y_val == 0)))

        n_attacks = max(1, int(np.sum(y_val == 1)))
        n_normals = max(1, int(np.sum(y_val == 0)))

        tpr = tp / n_attacks
        fpr = fp / n_normals

        # Penalize false positives: affinity = max(0, TPR - penalty * FPR)
        # Add a smooth detection bonus to break ties between zero-detection detectors
        raw_affinity = tpr - (self.fp_penalty * fpr)
        score = max(0.0001, float(raw_affinity) if raw_affinity > 0 else 0.0001 * (tp + 0.1))

        return score, tp, fp

    def _evaluate_repertoire(
        self,
        detectors: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate affinities, TPs, and FPs for all detectors in the repertoire."""
        n_det = len(detectors)
        affinities = np.zeros(n_det, dtype=np.float64)
        tps = np.zeros(n_det, dtype=np.int32)
        fps = np.zeros(n_det, dtype=np.int32)

        for i in range(n_det):
            aff, tp, fp = self.compute_affinity(detectors[i], X_val, y_val, radius)
            affinities[i] = aff
            tps[i] = tp
            fps[i] = fp

        return affinities, tps, fps

    def optimize(
        self,
        initial_detectors: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        self_samples: Optional[np.ndarray] = None,
    ) -> "ClonalSelection":
        """Run iterative clonal selection on the detector repertoire.

        Parameters
        ----------
        initial_detectors : np.ndarray
            Detectors produced by Negative Selection or initialized randomly, shape (K, D).
        X_val : np.ndarray
            Validation features used strictly for affinity evaluation, shape (N_val, D).
        y_val : np.ndarray
            Binary labels for validation instances (0=normal, 1=attack).
        self_samples : Optional[np.ndarray]
            Reference normal samples to maintain self-tolerance during hypermutation.
        """
        rng = np.random.default_rng(self.seed)
        detectors = np.copy(initial_detectors).astype(np.float64)
        k_target, n_features = detectors.shape
        r = float(self.matching_radius)
        r_sq = r ** 2

        self.history_ = []

        # Reference self data for censoring mutants (subsample for efficiency if large)
        censoring_self: Optional[np.ndarray] = None
        if self_samples is not None and len(self_samples) > 0:
            if len(self_samples) > 2000:
                sub_idx = rng.choice(len(self_samples), size=2000, replace=False)
                censoring_self = self_samples[sub_idx]
            else:
                censoring_self = self_samples

        for gen in range(self.generations):
            # 1. Evaluate current repertoire affinity
            affinities, tps, fps = self._evaluate_repertoire(detectors, X_val, y_val, r)

            # Record generation telemetry
            gen_stats = {
                "generation": gen + 1,
                "best_affinity": float(np.max(affinities)),
                "mean_affinity": float(np.mean(affinities)),
                "median_affinity": float(np.median(affinities)),
                "total_tp": int(np.sum(tps)),
                "total_fp": int(np.sum(fps)),
                "mean_tp": float(np.mean(tps)),
                "mean_fp": float(np.mean(fps)),
                "repertoire_size": int(len(detectors))
            }
            self.history_.append(gen_stats)

            # 2. Select top performers
            sort_indices = np.argsort(affinities)[::-1]
            n_select = max(2, int(k_target * self.selection_ratio))
            selected_indices = sort_indices[:n_select]
            selected_detectors = detectors[selected_indices]
            selected_affinities = affinities[selected_indices]

            # 3. Proportional Cloning & Somatic Hypermutation
            clones_list: List[np.ndarray] = []
            max_aff = max(1e-6, float(np.max(selected_affinities)))

            for i in range(n_select):
                det = selected_detectors[i]
                aff = selected_affinities[i]

                # Proportional number of clones
                rank_ratio = (n_select - i) / n_select
                n_clones = max(1, int(np.round(self.clone_factor * rank_ratio)))

                # Somatic hypermutation: lower affinity -> higher mutation rate
                norm_aff = aff / max_aff
                sigma = self.mutation_rate * (1.0 - norm_aff + 0.1)

                for _ in range(n_clones):
                    mutation_noise = rng.normal(0.0, sigma, size=n_features)
                    mutant = np.clip(det + mutation_noise, 0.0, 1.0)

                    # Self-tolerance check: discard mutant if it invades Self space
                    if censoring_self is not None:
                        c_sq = np.sum(mutant ** 2)
                        s_sq = np.sum(censoring_self ** 2, axis=1)
                        dist_sq = c_sq + s_sq - 2.0 * np.dot(censoring_self, mutant)
                        if np.min(dist_sq) <= r_sq:
                            continue  # Reject auto-reactive mutant

                    clones_list.append(mutant)

            # 4. Evaluate Clones
            if len(clones_list) > 0:
                clones_arr = np.array(clones_list, dtype=np.float64)
                clone_affs, _, _ = self._evaluate_repertoire(clones_arr, X_val, y_val, r)

                # 5. Selection & Repertoire Update
                # Pool original top detectors + high-performing clones
                all_candidates = np.vstack([selected_detectors, clones_arr])
                all_affinities = np.concatenate([selected_affinities, clone_affs])

                # Sort pooled population and keep top k_target
                best_indices = np.argsort(all_affinities)[::-1][:k_target]
                detectors = all_candidates[best_indices]
            else:
                detectors = detectors[sort_indices[:k_target]]

        # Final evaluation
        final_affinities, _, _ = self._evaluate_repertoire(detectors, X_val, y_val, r)
        final_sort = np.argsort(final_affinities)[::-1]
        self.detectors_ = detectors[final_sort]
        self.affinities_ = final_affinities[final_sort]

        return self
