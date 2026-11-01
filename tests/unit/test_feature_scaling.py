"""
tests/unit/test_feature_scaling.py
==================================
Unit tests for federated feature scaling via secure sum of sufficient statistics.
"""

import numpy as np
import pandas as pd
import pytest

from fl.preprocessing import (
    compute_local_sufficient_stats,
    aggregate_sufficient_stats,
)


@pytest.fixture
def bank_dfs():
    # Create 3 bank dataframes with known numeric distributions
    np.random.seed(42)
    df0 = pd.DataFrame({"feat_a": [10.0, 20.0, 30.0], "feat_b": [1.0, 2.0, 3.0]})
    df1 = pd.DataFrame({"feat_a": [40.0, 50.0], "feat_b": [4.0, 5.0]})
    df2 = pd.DataFrame({"feat_a": [60.0, 70.0, 80.0, 90.0], "feat_b": [6.0, 7.0, 8.0, 9.0]})
    return [df0, df1, df2]


def test_compute_local_sufficient_stats(bank_dfs):
    stats = compute_local_sufficient_stats(bank_dfs[0], ["feat_a", "feat_b"])
    assert stats["feat_a"]["count"] == 3
    assert stats["feat_a"]["sum"] == pytest.approx(60.0)
    assert stats["feat_a"]["sum_sq"] == pytest.approx(100.0 + 400.0 + 900.0)
    assert stats["feat_b"]["count"] == 3
    assert stats["feat_b"]["sum"] == pytest.approx(6.0)


def test_aggregate_sufficient_stats_matches_pooled(bank_dfs):
    numeric_cols = ["feat_a", "feat_b"]
    local_stats = [compute_local_sufficient_stats(df, numeric_cols) for df in bank_dfs]
    
    # Aggregate via federated sufficient stats
    global_stats = aggregate_sufficient_stats(local_stats)
    
    # Calculate directly from pooled dataframe
    pooled_df = pd.concat(bank_dfs, ignore_index=True)
    
    for col in numeric_cols:
        expected_mean = float(pooled_df[col].mean())
        expected_std = float(pooled_df[col].std(ddof=0))  # population std
        
        assert global_stats[col]["count"] == len(pooled_df)
        assert global_stats[col]["mean"] == pytest.approx(expected_mean, rel=1e-5)
        assert global_stats[col]["std"] == pytest.approx(expected_std, rel=1e-5)


def test_aggregate_sufficient_stats_zero_variance():
    df_const = pd.DataFrame({"feat_const": [5.0, 5.0, 5.0]})
    local_stats = [compute_local_sufficient_stats(df_const, ["feat_const"])]
    global_stats = aggregate_sufficient_stats(local_stats)
    
    assert global_stats["feat_const"]["mean"] == pytest.approx(5.0)
    # std should fall back to 1.0 to avoid division by zero
    assert global_stats["feat_const"]["std"] == 1.0


def test_empty_sufficient_stats():
    res = aggregate_sufficient_stats([])
    assert res == {}
