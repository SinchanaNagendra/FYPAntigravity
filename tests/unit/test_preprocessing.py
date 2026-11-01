"""
tests/unit/test_preprocessing.py
================================
Unit tests for fixed, public schema preprocessing in fl/preprocessing.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from fl.preprocessing import FraudPreprocessor, DEFAULT_VOCAB_DIR


@pytest.fixture
def preprocessor():
    return FraudPreprocessor(vocab_dir=DEFAULT_VOCAB_DIR)


@pytest.fixture
def sample_df():
    # Construct a minimal dataframe containing some of the schema columns
    return pd.DataFrame({
        "TransactionID": [1001, 1002, 1003],
        "isFraud": [0, 1, 0],
        "TransactionDT": [86400, 86450, 86500],
        "TransactionAmt": [50.0, 150.25, 29.99],
        "ProductCD": ["W", "H", "C"],
        "card1": [13926, 2755, 4461],
        "card2": [None, 404.0, 325.0],
        "card4": ["discover", "mastercard", "visa"],
        "card6": ["credit", "debit", "debit"],
    })


def test_preprocessor_init(preprocessor):
    assert preprocessor.output_dim > 0
    assert len(preprocessor.feature_names) == preprocessor.output_dim
    assert "TransactionAmt" in preprocessor.num_stats


def test_preprocessor_transform_shape(preprocessor, sample_df):
    X, y = preprocessor.transform(sample_df)
    assert X.shape == (3, preprocessor.output_dim)
    assert y.shape == (3,)
    assert list(y) == [0, 1, 0]
    assert not np.isnan(X).any(), "Preprocessor output must not contain NaNs"
    assert X.dtype == np.float32


def test_preprocessor_deterministic(preprocessor, sample_df):
    X1, y1 = preprocessor.transform(sample_df)
    X2, y2 = preprocessor.transform(sample_df)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_preprocessor_handles_missing_columns(preprocessor):
    minimal_df = pd.DataFrame({
        "TransactionID": [9999],
        "isFraud": [0],
        "TransactionAmt": [100.0],
    })
    X, y = preprocessor.transform(minimal_df)
    assert X.shape == (1, preprocessor.output_dim)
    assert not np.isnan(X).any()


def test_preprocessor_no_label_raises_error(preprocessor):
    df_no_label = pd.DataFrame({
        "TransactionID": [8888],
        "TransactionAmt": [75.0],
    })
    with pytest.raises(ValueError, match="Label column 'isFraud' not found"):
        preprocessor.transform(df_no_label)
