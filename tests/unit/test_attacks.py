"""
tests/unit/test_attacks.py
==========================
Unit tests for attack and defence implementations:
  - Poisoning: label flipping, model weight poisoning, free-rider attack
  - Inference: membership inference (loss threshold, shadow models)
  - Defences: coordinate-wise median, trimmed mean, Krum
"""

import numpy as np
import pytest
import torch

from attacks.defences.robust_agg import coordinate_median, krum, trimmed_mean
from attacks.inference.membership_inference import loss_threshold_attack, shadow_model_attack
from attacks.poisoning.free_rider import FreeRiderBank
from attacks.poisoning.label_flip import label_flip_attack
from attacks.poisoning.model_poison import backdoor_attack, scaled_update_attack, sign_flip_attack
from fl.models.logistic_regression import LogisticRegressionModel


def test_label_flip_attack():
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.zeros(100, dtype=np.int64)
    y[:20] = 1  # 20 fraud instances

    # 100% flip
    _, y_flipped = label_flip_attack(X, y, flip_rate=1.0, target_label=1, flip_to=0)
    assert np.sum(y_flipped == 1) == 0

    # 50% flip
    _, y_half = label_flip_attack(X, y, flip_rate=0.5, target_label=1, flip_to=0)
    assert np.sum(y_half == 1) == 10


def test_free_rider_bank():
    p = [np.ones((5, 5), dtype=np.float32)]
    # Zeros strategy
    fr_zeros = FreeRiderBank(strategy="zeros")
    fake_z, n_z = fr_zeros.get_update(p, n_samples=500)
    assert np.all(fake_z[0] == 0.0)
    assert n_z == 500

    # Random strategy
    fr_rand = FreeRiderBank(strategy="random")
    fake_r, n_r = fr_rand.get_update(p, n_samples=300)
    assert fake_r[0].shape == (5, 5)
    assert not np.all(fake_r[0] == 0.0)

    # Stale strategy
    fr_stale = FreeRiderBank(strategy="stale")
    fake_s1, _ = fr_stale.get_update(p)
    # Subsequent update returns previous
    p_new = [np.full((5, 5), 2.0, dtype=np.float32)]
    fake_s2, _ = fr_stale.get_update(p_new)
    np.testing.assert_array_equal(fake_s2[0], p[0])


def test_model_poisoning():
    weights = [np.array([1.0, 2.0, -3.0], dtype=np.float32)]

    # Scaled
    scaled = scaled_update_attack(weights, scale=10.0)
    np.testing.assert_allclose(scaled[0], [10.0, 20.0, -30.0])

    # Sign flip
    sign_flipped = sign_flip_attack(weights)
    np.testing.assert_allclose(sign_flipped[0], [-1.0, -2.0, 3.0])

    # Backdoor
    bd = backdoor_attack(weights, trigger_layer_idx=0, trigger_scale=5.0)
    assert not np.array_equal(bd[0], weights[0])


def test_robust_aggregation_median():
    # 5 banks with 1D parameter
    updates = [
        [np.array([1.0], dtype=np.float32)],
        [np.array([1.1], dtype=np.float32)],
        [np.array([0.9], dtype=np.float32)],
        [np.array([100.0], dtype=np.float32)],  # Outlier
        [np.array([-50.0], dtype=np.float32)],  # Outlier
    ]
    med = coordinate_median(updates)
    # Median of [-50, 0.9, 1.0, 1.1, 100] is 1.0
    assert pytest.approx(med[0][0], abs=1e-5) == 1.0


def test_robust_aggregation_trimmed_mean():
    updates = [
        [np.array([1.0], dtype=np.float32)],
        [np.array([1.2], dtype=np.float32)],
        [np.array([0.8], dtype=np.float32)],
        [np.array([100.0], dtype=np.float32)],  # Outlier
        [np.array([-100.0], dtype=np.float32)], # Outlier
    ]
    # Trim 20% on each side (trims top and bottom 1 sample out of 5)
    tm = trimmed_mean(updates, trim_fraction=0.2)
    # Remaining: [0.8, 1.0, 1.2], mean = 1.0
    assert pytest.approx(tm[0][0], abs=1e-4) == 1.0


def test_robust_aggregation_krum():
    updates = [
        [np.array([1.0, 1.0], dtype=np.float32)],
        [np.array([1.1, 0.9], dtype=np.float32)],
        [np.array([0.9, 1.1], dtype=np.float32)],
        [np.array([1.05, 0.95], dtype=np.float32)],
        [np.array([100.0, 100.0], dtype=np.float32)], # 1 adversary out of 5
    ]
    # N=5, m=1: n_banks > 2*m + 2 -> 5 > 4 (satisfied)
    selected = krum(updates, n_malicious=1)
    # Selected should be one of the clean updates around [1.0, 1.0]
    assert selected[0][0] < 5.0 and selected[0][1] < 5.0


def test_membership_inference():
    n_features = 10
    model = LogisticRegressionModel(n_features)
    np.random.seed(42)
    X_members = np.random.randn(100, n_features).astype(np.float32)
    y_members = np.zeros(100, dtype=np.int64)
    y_members[:20] = 1

    X_nonmembers = np.random.randn(100, n_features).astype(np.float32)
    y_nonmembers = np.zeros(100, dtype=np.int64)
    y_nonmembers[:20] = 1

    # Loss threshold attack
    res_lt = loss_threshold_attack(model, X_members, y_members, X_nonmembers, y_nonmembers, model_type="torch")
    assert "auc" in res_lt
    assert "advantage" in res_lt
    assert 0.0 <= res_lt["auc"] <= 1.0

    # Shadow model attack
    res_sm = shadow_model_attack(model, X_members, y_members, X_nonmembers, y_nonmembers, model_type="torch")
    assert "auc" in res_sm
    assert "advantage" in res_sm
    assert 0.0 <= res_sm["auc"] <= 1.0
