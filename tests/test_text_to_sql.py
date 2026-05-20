"""
Integration tests for Text-to-SQL engine.
Run: pytest tests/test_text_to_sql.py -v
"""

import os
import sys
import sqlite3
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="module")
def db_exists():
    """Skip tests if database doesn't exist."""
    db_path = os.path.join(os.path.dirname(__file__), '..', 'ecom.db')
    if not os.path.exists(db_path):
        pytest.skip("ecom.db not found — run create_demo_data.py first")
    return db_path


class TestLocalPatternMatching:
    """Tests for the local (no-LLM) text-to-SQL engine."""

    def test_revenue_query(self, db_exists):
        from text_to_sql import ask_database_local
        sql, result, error = ask_database_local("What is the total revenue?")
        assert error is None
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_top_customers_query(self, db_exists):
        from text_to_sql import ask_database_local
        sql, result, error = ask_database_local("Show top 10 customers by spending")
        assert sql is not None

    def test_category_query(self, db_exists):
        from text_to_sql import ask_database_local
        sql, result, error = ask_database_local("Revenue by category")
        assert sql is not None

    def test_empty_query_handled(self, db_exists):
        from text_to_sql import ask_database_local
        sql, result, error = ask_database_local("")
        # Should not crash — graceful handling
        assert True

    def test_nonsense_query_handled(self, db_exists):
        from text_to_sql import ask_database_local
        sql, result, error = ask_database_local("xyzzy foobar quantum entanglement")
        # Should return error or None gracefully, not crash
        assert True

    def test_sql_injection_safe(self, db_exists):
        from text_to_sql import ask_database_local
        sql, result, error = ask_database_local("'; DROP TABLE fact_orders; --")
        # Should not drop any table
        db_path = os.path.join(os.path.dirname(__file__), '..', 'ecom.db')
        conn = sqlite3.connect(db_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        conn.close()
        assert "fact_orders" in tables["name"].values


class TestDatabaseConnection:
    """Tests for database connectivity."""

    def test_connection_succeeds(self, db_exists):
        from text_to_sql import get_db_connection
        conn = get_db_connection()
        assert conn is not None
        conn.close()

    def test_schema_queryable(self, db_exists):
        from text_to_sql import get_db_connection
        conn = get_db_connection()
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )
        assert len(tables) > 0
        conn.close()
