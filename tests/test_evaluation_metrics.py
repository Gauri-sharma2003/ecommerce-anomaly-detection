"""
Tests for Evaluation Metrics module.
Run: pytest tests/test_evaluation_metrics.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation_metrics import (
    evaluate_detector,
    evaluate_all_methods,
    compute_threshold_analysis,
)


class TestEvaluateDetector:
    """Tests for single detector evaluation."""

    def test_perfect_detector(self):
        y_true = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0])
        result = evaluate_detector(y_true, y_pred, "perfect")
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1_score"] == 1.0
        assert result["false_positives"] == 0

    def test_all_zeros_prediction(self):
        y_true = np.array([0, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 0, 0])
        result = evaluate_detector(y_true, y_pred, "conservative")
        assert result["recall"] == 0.0
        assert result["false_negatives"] == 2

    def test_all_ones_prediction(self):
        """Flagging everything = recall of 1.0 but terrible precision."""
        y_true = np.array([0, 0, 0, 0, 1])
        y_pred = np.array([1, 1, 1, 1, 1])
        result = evaluate_detector(y_true, y_pred, "aggressive")
        assert result["recall"] == 1.0
        assert result["precision"] == 0.2
        assert result["false_positives"] == 4

    def test_output_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        result = evaluate_detector(y_true, y_pred, "test")
        expected_keys = {"method", "precision", "recall", "f1_score", "specificity",
                        "false_positive_rate", "true_positives", "false_positives",
                        "true_negatives", "false_negatives", "total_flagged", "total_anomalies"}
        assert set(result.keys()) == expected_keys

    def test_confusion_matrix_sums(self):
        """TP + FP + TN + FN should equal total samples."""
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        result = evaluate_detector(y_true, y_pred, "random")
        total = result["true_positives"] + result["false_positives"] + \
                result["true_negatives"] + result["false_negatives"]
        assert total == 100


class TestEvaluateAllMethods:
    """Tests for multi-method evaluation."""

    @pytest.fixture
    def daily_data(self):
        np.random.seed(42)
        n = 365
        df = pd.DataFrame({
            "order_date": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
            "is_true_anomaly": ([0] * 350 + [1] * 15),
            "zscore_pred": ([0] * 355 + [1] * 10),
            "iforest_pred": ([0] * 352 + [1] * 13),
        })
        return df

    def test_returns_dataframe(self, daily_data):
        cols = {"Z-Score": "zscore_pred", "Isolation Forest": "iforest_pred"}
        result = evaluate_all_methods(daily_data, cols)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_sorted_by_f1(self, daily_data):
        cols = {"Z-Score": "zscore_pred", "Isolation Forest": "iforest_pred"}
        result = evaluate_all_methods(daily_data, cols)
        assert result["f1_score"].is_monotonic_decreasing


class TestThresholdAnalysis:
    """Tests for threshold optimization."""

    def test_returns_multiple_thresholds(self):
        df = pd.DataFrame({
            "z_score": np.random.normal(0, 1, 365),
            "is_true_anomaly": [0] * 350 + [1] * 15,
        })
        result = compute_threshold_analysis(df)
        assert len(result) > 5

    def test_higher_threshold_fewer_flags(self):
        """Higher threshold should flag fewer data points."""
        df = pd.DataFrame({
            "z_score": np.random.normal(0, 1, 365),
            "is_true_anomaly": [0] * 350 + [1] * 15,
        })
        result = compute_threshold_analysis(df)
        assert result.iloc[0]["total_flagged"] >= result.iloc[-1]["total_flagged"]
