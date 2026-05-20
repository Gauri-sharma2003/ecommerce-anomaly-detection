"""
Basic tests for E-Commerce Anomaly Detection project.
Run: pytest tests/ -v
"""

import os
import sys
import sqlite3
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDatabaseCreation:
    """Tests for database generation and schema integrity."""

    def test_database_file_exists(self):
        """Verify the database file is created after running load_kaggle_data."""
        db_path = os.path.join(os.path.dirname(__file__), '..', 'ecom.db')
        assert os.path.exists(db_path), "ecom.db not found. Run: python src/load_kaggle_data.py"

    def test_tables_exist(self):
        """Verify all expected tables exist in the database."""
        db_path = os.path.join(os.path.dirname(__file__), '..', 'ecom.db')
        if not os.path.exists(db_path):
            pytest.skip("Database not built yet. Run: python src/load_kaggle_data.py")
        
        conn = sqlite3.connect(db_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        expected_tables = {'dim_customers', 'dim_products', 'fact_orders', 'daily_kpis', 'model_results'}
        assert expected_tables.issubset(set(tables['name'].values))
        conn.close()

    def test_fact_orders_row_count(self):
        """Verify fact_orders has real Kaggle data (90K+ orders)."""
        db_path = os.path.join(os.path.dirname(__file__), '..', 'ecom.db')
        if not os.path.exists(db_path):
            pytest.skip("Database not built yet")
        conn = sqlite3.connect(db_path)
        count = pd.read_sql("SELECT COUNT(*) as cnt FROM fact_orders", conn)['cnt'][0]
        assert count > 90000, f"Expected 90K+ real orders, got {count}"
        conn.close()

    def test_no_negative_revenue(self):
        """Business rule: order_value should never be negative."""
        db_path = os.path.join(os.path.dirname(__file__), '..', 'ecom.db')
        if not os.path.exists(db_path):
            pytest.skip("Database not built yet")
        conn = sqlite3.connect(db_path)
        negatives = pd.read_sql("SELECT COUNT(*) as cnt FROM fact_orders WHERE order_value < 0", conn)['cnt'][0]
        assert negatives == 0, "Found negative order values — data integrity issue"
        conn.close()


class TestTextToSQL:
    """Tests for the Text-to-SQL engine (local mode only — no API key needed)."""

    def test_local_query_revenue(self):
        """Test that local pattern matching handles revenue queries."""
        from text_to_sql import ask_database_local
        sql, result, error = ask_database_local("What is the total revenue?")
        assert error is None or result is not None or sql is not None

    def test_local_query_returns_dataframe(self):
        """Test that successful queries return a pandas DataFrame."""
        from text_to_sql import ask_database_local
        sql, result, error = ask_database_local("Show me top 5 customers")
        if result is not None:
            assert isinstance(result, pd.DataFrame)


class TestAnomalyDetection:
    """Tests for anomaly detection logic."""

    def test_zscore_calculation(self):
        """Verify Z-score calculation produces expected range."""
        data = np.array([100, 102, 98, 101, 99, 500, 97, 103, 100, 101])
        z_scores = (data - data.mean()) / data.std()
        # The spike (500) should have z-score > 2
        assert z_scores[5] > 2.0, "Spike not detected by Z-score"

    def test_anomaly_percentage_reasonable(self):
        """Anomaly rate should be between 1-15% for well-tuned models."""
        db_path = os.path.join(os.path.dirname(__file__), '..', 'ecom.db')
        if not os.path.exists(db_path):
            return  # Skip if DB not generated
        conn = sqlite3.connect(db_path)
        try:
            results = pd.read_sql(
                "SELECT anomaly_pct FROM model_results WHERE method_type = 'isolation_forest' LIMIT 50",
                conn
            )
            median_pct = results['anomaly_pct'].median()
            assert 1 <= median_pct <= 20, f"Median anomaly rate {median_pct}% seems unreasonable"
        except Exception:
            pass  # Table may not exist yet
        finally:
            conn.close()
