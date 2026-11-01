"""
tests/unit/test_strategies.py
=============================
Unit tests for FL strategies:
  - FabricFedAvg (weighted averaging and secure aggregation)
  - FabricFedProx (proximal loss computation and configuration)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from fl.models.logistic_regression import LogisticRegressionModel
from fl.strategies.fabric_fedavg import FabricFedAvg
from fl.strategies.fabric_fedprox import FabricFedProx


def test_fabric_fedavg_aggregate_fit():
    strategy = FabricFedAvg(n_banks=2)
    # Bank 0: 100 samples, weights [[1.0, 2.0]], bias [0.5]
    p0 = [np.array([[1.0, 2.0]], dtype=np.float32), np.array([0.5], dtype=np.float32)]
    # Bank 1: 300 samples, weights [[3.0, 4.0]], bias [1.5]
    p1 = [np.array([[3.0, 4.0]], dtype=np.float32), np.array([1.5], dtype=np.float32)]

    results = [(p0, 100), (p1, 300)]
    agg = strategy.aggregate_fit(results)

    # Expected: 0.25 * p0 + 0.75 * p1
    expected_w = 0.25 * p0[0] + 0.75 * p1[0]
    expected_b = 0.25 * p0[1] + 0.75 * p1[1]

    np.testing.assert_allclose(agg[0], expected_w, rtol=1e-5)
    np.testing.assert_allclose(agg[1], expected_b, rtol=1e-5)


def test_fabric_fedavg_empty_results_raises():
    strategy = FabricFedAvg(n_banks=2)
    with pytest.raises(ValueError, match="No results"):
        strategy.aggregate_fit([])


def test_fabric_fedprox_proximal_loss():
    strategy = FabricFedProx(n_banks=2, mu=0.2)
    model = LogisticRegressionModel(5)
    global_params = model.get_parameters()

    # Zero distance -> zero proximal loss
    loss_zero = strategy.proximal_loss(model, global_params)
    assert pytest.approx(loss_zero.item(), abs=1e-7) == 0.0

    # Perturb weights by 1.0 everywhere
    perturbed_params = [p + 1.0 for p in global_params]
    total_elements = sum(p.size for p in global_params)
    loss_perturbed = strategy.proximal_loss(model, perturbed_params)

    # Expected: (mu / 2) * total_elements * (1.0)^2 = (0.2 / 2) * total_elements
    expected_loss = 0.1 * total_elements
    assert pytest.approx(loss_perturbed.item(), rel=1e-5) == expected_loss


def test_fabric_fedprox_configure_fit():
    strategy = FabricFedProx(n_banks=3, mu=0.05)
    cfg = strategy.configure_fit(round_num=1, parameters=[])
    assert cfg["round_num"] == 1
    assert cfg["use_fedprox"] is True
    assert cfg["mu"] == 0.05
