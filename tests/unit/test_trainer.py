"""
tests/unit/test_trainer.py
==========================
Unit tests for LocalTrainer:
  - Verifies class balancing is applied to training set
  - Tests training routines across PyTorch, Scikit-Learn, and KNN models
  - Tests FedProx proximal regularization during local training
"""

import numpy as np
import pytest
import torch

from fl.balancing.methods import BalancingMethod
from fl.bank_node.trainer import LocalTrainer
from fl.models.cnn import CNNModel
from fl.models.decision_tree import DecisionTreeModel
from fl.models.gaussian_nb import GaussianNBModel
from fl.models.knn import KNNModel
from fl.models.logistic_regression import LogisticRegressionModel
from fl.models.random_forest import RandomForestModel


@pytest.fixture
def dummy_train_data():
    np.random.seed(42)
    n_samples = 200
    n_features = 30
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # 90% class 0, 10% class 1
    y = np.zeros(n_samples, dtype=np.int64)
    y[:20] = 1
    return X, y


def test_trainer_balancing_rus(dummy_train_data):
    X, y = dummy_train_data
    model = LogisticRegressionModel(30)
    trainer = LocalTrainer(
        model_type="logistic_regression",
        model=model,
        balancing=BalancingMethod.RUS,
        n_epochs=1,
        batch_size=16,
        random_state=42,
    )
    result = trainer.train(X, y)
    assert "train_time" in result
    assert "losses" in result
    # RUS balances 20 minority samples with 20 majority samples -> 40 samples
    assert result["n_samples"] == 40


def test_trainer_torch_lr(dummy_train_data):
    X, y = dummy_train_data
    model = LogisticRegressionModel(30)
    trainer = LocalTrainer(
        model_type="logistic_regression",
        model=model,
        balancing=BalancingMethod.NONE,
        n_epochs=2,
        batch_size=32,
    )
    result = trainer.train(X, y)
    assert len(result["losses"]) == 2
    assert result["n_samples"] == len(X)


def test_trainer_torch_cnn(dummy_train_data):
    X, y = dummy_train_data
    model = CNNModel(30)
    trainer = LocalTrainer(
        model_type="cnn",
        model=model,
        balancing=BalancingMethod.NONE,
        n_epochs=2,
        batch_size=32,
    )
    result = trainer.train(X, y)
    assert len(result["losses"]) == 2


def test_trainer_sklearn_models(dummy_train_data):
    X, y = dummy_train_data
    for m_type, m_cls in [
        ("decision_tree", DecisionTreeModel),
        ("random_forest", RandomForestModel),
        ("gaussian_nb", GaussianNBModel),
    ]:
        m = m_cls(30)
        trainer = LocalTrainer(
            model_type=m_type,
            model=m,
            balancing=BalancingMethod.NONE,
        )
        res = trainer.train(X, y)
        assert res["n_samples"] == len(X)
        assert res["train_time"] >= 0


def test_trainer_knn_local(dummy_train_data):
    X, y = dummy_train_data
    model = KNNModel(30, n_neighbors=3)
    trainer = LocalTrainer(
        model_type="knn",
        model=model,
        balancing=BalancingMethod.NONE,
    )
    res = trainer.train(X, y)
    assert res["n_samples"] == len(X)


def test_trainer_fedprox(dummy_train_data):
    X, y = dummy_train_data
    model = LogisticRegressionModel(30)
    global_params = model.get_parameters()
    trainer = LocalTrainer(
        model_type="logistic_regression",
        model=model,
        balancing=BalancingMethod.NONE,
        n_epochs=1,
        batch_size=32,
        fedprox_mu=0.1,
    )
    res = trainer.train(X, y, global_params=global_params)
    assert len(res["losses"]) == 1
