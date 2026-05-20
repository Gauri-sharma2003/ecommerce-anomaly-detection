"""
Comprehensive tests for the RCA analysis engine.
Run: pytest tests/test_rca_analysis.py -v
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rca_model_analysis import (
    classify_anomaly_rate,
    rca_method_performance,
    rca_feature_set_impact,
    rca_scaler_impact,
)


class TestClassifyAnomalyRate:
    """Unit tests for anomaly rate classification logic."""

    def test_too_conservative(self):
        assert classify_anomaly_rate(0.5) == "too_conservative"
        assert classify_anomaly_rate(0) == "too_conservative"

    def test_conservative(self):
        assert classify_anomaly_rate(1) == "conservative"
        assert classify_anomaly_rate(2.9) == "conservative"

    def test_reasonable(self):
        assert classify_anomaly_rate(3) == "reasonable"
        assert classify_anomaly_rate(5) == "reasonable"
        assert classify_anomaly_rate(10) == "reasonable"

    def test_aggressive(self):
        assert classify_anomaly_rate(11) == "aggressive"
        assert classify_anomaly_rate(20) == "aggressive"

    def test_broken(self):
        assert classify_anomaly_rate(21) == "broken"
        assert classify_anomaly_rate(80) == "broken"
        assert classify_anomaly_rate(100) == "broken"

    def test_boundary_values(self):
        """Test exact boundary conditions."""
        assert classify_anomaly_rate(1) == "conservative"
        assert classify_anomaly_rate(3) == "reasonable"
        assert classify_anomaly_rate(10) == "reasonable"
        assert classify_anomaly_rate(10.01) == "aggressive"
        assert classify_anomaly_rate(20.01) == "broken"


class TestRCAMethodPerformance:
    """Tests for method-level RCA analysis."""

    @pytest.fixture
    def sample_model_data(self):
        """Create sample model comparison data."""
        np.random.seed(42)
        methods = ["Isolation Forest", "DBSCAN", "LOF", "Statistical"]
        data = []
        for method in methods:
            for i in range(50):
                if method == "Isolation Forest":
                    pct = np.random.uniform(2, 8)
                elif method == "DBSCAN":
                    pct = np.random.uniform(0, 80)
                elif method == "LOF":
                    pct = np.random.uniform(1, 25)
                else:
                    pct = np.random.uniform(3, 7)
                data.append({
                    "model": f"{method}_{i}",
                    "method_type": method,
                    "anomaly_pct": round(pct, 1),
                    "feature_set": "basic_revenue",
                    "scaler": "standard",
                    "n_features": 2,
                })
        return pd.DataFrame(data)

    def test_returns_dataframe(self, sample_model_data):
        result = rca_method_performance(sample_model_data)
        assert isinstance(result, pd.DataFrame)

    def test_all_methods_present(self, sample_model_data):
        result = rca_method_performance(sample_model_data)
        assert len(result) == 4

    def test_success_rate_bounded(self, sample_model_data):
        result = rca_method_performance(sample_model_data)
        assert (result["success_rate_pct"] >= 0).all()
        assert (result["success_rate_pct"] <= 100).all()

    def test_isolation_forest_outperforms_dbscan(self, sample_model_data):
        result = rca_method_performance(sample_model_data)
        if_rate = result.loc["Isolation Forest", "success_rate_pct"]
        db_rate = result.loc["DBSCAN", "success_rate_pct"]
        assert if_rate > db_rate


class TestRCAFeatureSetImpact:
    """Tests for feature engineering impact analysis."""

    @pytest.fixture
    def feature_data(self):
        np.random.seed(42)
        features = ["basic_revenue", "multi_feature", "full_feature"]
        data = []
        for feat in features:
            for i in range(30):
                pct = np.random.uniform(2, 15) if feat == "basic_revenue" else np.random.uniform(3, 8)
                data.append({
                    "model": f"IF_{feat}_{i}",
                    "method_type": "Isolation Forest",
                    "anomaly_pct": round(pct, 1),
                    "feature_set": feat,
                    "scaler": "standard",
                    "n_features": {"basic_revenue": 2, "multi_feature": 5, "full_feature": 21}[feat],
                })
        return pd.DataFrame(data)

    def test_returns_all_feature_sets(self, feature_data):
        result = rca_feature_set_impact(feature_data)
        assert len(result) == 3

    def test_reasonable_pct_bounded(self, feature_data):
        result = rca_feature_set_impact(feature_data)
        assert (result["reasonable_pct"] >= 0).all()
        assert (result["reasonable_pct"] <= 100).all()
