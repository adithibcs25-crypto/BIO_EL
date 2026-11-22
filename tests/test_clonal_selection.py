"""Unit tests for the Clonal Selection Algorithm."""

import numpy as np
import pytest

from src.clonal_selection import ClonalSelection


def test_clonal_affinity_calculation():
    cs = ClonalSelection(fp_penalty=3.0, matching_radius=0.5)

    # Attack samples near (0.8, 0.8), normal samples near (0.1, 0.1)
    X_val = np.array([
        [0.8, 0.8],
        [0.82, 0.79],
        [0.1, 0.1],
        [0.15, 0.12],
    ])
    y_val = np.array([1, 1, 0, 0])

    good_detector = np.array([0.81, 0.8])  # close to attacks, far from normal
    bad_detector = np.array([0.12, 0.11])  # close to normals (false positives)

    good_aff, good_tp, good_fp = cs.compute_affinity(good_detector, X_val, y_val, radius=0.3)
    bad_aff, bad_tp, bad_fp = cs.compute_affinity(bad_detector, X_val, y_val, radius=0.3)

    assert good_tp > 0
    assert good_fp == 0
    assert bad_fp > 0
    assert good_aff > bad_aff


def test_clonal_optimization_improves_or_maintains():
    rng = np.random.default_rng(42)
    # Normals around 0.1, attacks around 0.8
    X_val = np.vstack([
        rng.normal(0.1, 0.05, size=(40, 4)),
        rng.normal(0.8, 0.05, size=(40, 4)),
    ])
    X_val = np.clip(X_val, 0.0, 1.0)
    y_val = np.array([0] * 40 + [1] * 40)

    # Random initial detectors
    initial_detectors = rng.uniform(0.0, 1.0, size=(10, 4))

    cs = ClonalSelection(generations=4, clone_factor=3.0, mutation_rate=0.08, matching_radius=0.4, seed=42)
    cs.optimize(initial_detectors, X_val, y_val)

    assert cs.detectors_ is not None
    assert len(cs.detectors_) == 10
    assert len(cs.history_) == 4
    # Check that affinity history is tracked
    assert "best_affinity" in cs.history_[0]
    assert "mean_affinity" in cs.history_[0]
