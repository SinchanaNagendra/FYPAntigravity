"""
tests/unit/test_balancing.py
============================
Unit tests for data balancing and resampling methods in fl/balancing/methods.py.
"""

import numpy as np
import pytest

from fl.balancing.methods import (
    BalancingMethod,
    get_balancer,
    apply_balancing,
    compute_class_weights,
)


@pytest.fixture
def imbalanced_data():
    # 100 samples total: 90 class 0, 10 class 1 (10% minority)
    np.random.seed(42)
    X0 = np.random.randn(90, 5)
    y0 = np.zeros(90, dtype=int)
    X1 = np.random.randn(10, 5) + 2.0
    y1 = np.ones(10, dtype=int)
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    return X, y


def test_balancing_none(imbalanced_data):
    X, y = imbalanced_data
    X_res, y_res = apply_balancing(X, y, BalancingMethod.NONE)
    assert len(X_res) == len(X)
    assert np.array_equal(y_res, y)


def test_balancing_weighted(imbalanced_data):
    X, y = imbalanced_data
    X_res, y_res = apply_balancing(X, y, BalancingMethod.WEIGHTED)
    assert len(X_res) == len(X)
    weights = compute_class_weights(y)
    assert 0 in weights and 1 in weights
    assert weights[1] > weights[0]  # Minority gets higher weight


def test_balancing_ros(imbalanced_data):
    X, y = imbalanced_data
    X_res, y_res = apply_balancing(X, y, BalancingMethod.ROS, random_state=42)
    assert len(X_res) > len(X)
    unique, counts = np.unique(y_res, return_counts=True)
    assert counts[0] == counts[1] == 90


def test_balancing_rus(imbalanced_data):
    X, y = imbalanced_data
    X_res, y_res = apply_balancing(X, y, BalancingMethod.RUS, random_state=42)
    assert len(X_res) < len(X)
    unique, counts = np.unique(y_res, return_counts=True)
    assert counts[0] == counts[1] == 10


def test_balancing_smote(imbalanced_data):
    X, y = imbalanced_data
    X_res, y_res = apply_balancing(X, y, BalancingMethod.SMOTE, random_state=42)
    assert len(X_res) > len(X)
    unique, counts = np.unique(y_res, return_counts=True)
    assert counts[0] == counts[1] == 90


def test_balancing_adasyn(imbalanced_data):
    X, y = imbalanced_data
    X_res, y_res = apply_balancing(X, y, BalancingMethod.ADASYN, random_state=42)
    assert len(X_res) > len(X)
    unique, counts = np.unique(y_res, return_counts=True)
    assert counts[1] >= 80  # ADASYN adapts count closely to majority


def test_balancing_hybrid(imbalanced_data):
    X, y = imbalanced_data
    X_res, y_res = apply_balancing(X, y, BalancingMethod.HYBRID, random_state=42)
    # SMOTEENN removes noisy samples from both classes
    unique, counts = np.unique(y_res, return_counts=True)
    assert len(unique) == 2
    assert counts[1] > 10


def test_balancing_min_minority_fallback():
    # Only 3 minority samples -> should trigger fallback
    X = np.random.randn(50, 5)
    y = np.array([0] * 47 + [1] * 3)
    X_res, y_res = apply_balancing(X, y, BalancingMethod.SMOTE, min_minority_samples=10)
    # Falls back to no resampling
    assert len(X_res) == 50
