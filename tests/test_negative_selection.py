"""Unit tests for the Negative Selection Algorithm."""

import numpy as np
import pytest

from src.negative_selection import NegativeSelection


def test_negative_selection_self_tolerance():
    # Define Self as a cluster near (0.2, 0.2) in 5D space
    rng = np.random.default_rng(42)
    self_samples = rng.normal(loc=0.2, scale=0.05, size=(100, 5))
    self_samples = np.clip(self_samples, 0.0, 1.0)

    radius = 0.3
    ns = NegativeSelection(num_detectors=15, matching_radius=radius, max_trials=5000, seed=42)
    ns.fit(self_samples)

    assert ns.detectors_ is not None
    assert len(ns.detectors_) > 0
    assert ns.detectors_.shape[1] == 5

    # Critical check: None of the mature detectors must fall within 'radius' of any Self sample
    for det in ns.detectors_:
        min_dist_to_self = np.min(np.linalg.norm(self_samples - det, axis=1))
        assert min_dist_to_self > radius, f"Detector matches Self: {min_dist_to_self} <= {radius}"


def test_negative_selection_matching():
    # Known detector at (0.9, 0.9)
    ns = NegativeSelection(num_detectors=1, matching_radius=0.4, seed=42)
    ns.detectors_ = np.array([[0.9, 0.9]], dtype=np.float64)

    test_points = np.array([
        [0.85, 0.85],  # Very close -> should match
        [0.1, 0.1],    # Far away -> should not match
    ])

    matches = ns.matches(test_points)
    assert matches[0] == True
    assert matches[1] == False


def test_negative_selection_empty_error():
    ns = NegativeSelection()
    with pytest.raises(RuntimeError):
        ns.matches(np.array([[0.5, 0.5]]))
