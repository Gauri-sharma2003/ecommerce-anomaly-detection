"""
Text-to-SQL Engine — Clean, robust, edge-case-handled
======================================================
"""

import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ecom.db")
DB_PATH = DATABASE_URL.replace("sqlite:///", "") if DATABASE_URL.startswith("sqlite:///") else "ecom.db"


def get_db_connection():
    """Get SQLite connection with error handling."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Test connection
        conn.execute("SELECT 1")
        return conn
    except Exception as e:
        return None


def ask_database_llm(question: str):
    """Use Groq LLM for text-to-SQL. Returns (sql, result_df, error)."""
    if not GROQ_API_KEY:
        return None, None, "No API key configured. Add GROQ_API_KEY to your .env file."
    
    try:
        from langchain_community.utilities import SQLDatabase
        from langchain_groq import ChatGroq
        from langchain_classic.chains.sql_database.query import create_sql_query_chain
        
        db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0)
        chain = create_sql_query_chain(llm, db)
        
        sql = chain.invoke({"question": question})
        sql = sql.strip().replace("```sql", "").replace("```", "").strip()
        
        # Extract just the SQL query (LLM sometimes includes extra text)
        if "SQLQuery:" in sql:
            sql = sql.split("SQLQuery:")[-1].strip()
        if "SQLResult:" in sql:
            sql = sql.split("SQLResult:")[0].strip()
        if sql.upper().startswith("QUESTION:"):
            sql = sql[sql.upper().index("SELECT"):]
        # Remove trailing text after the query
        for stop in ["\nQuestion:", "\nSQLResult:", "\nAnswer:"]:
            if stop in sql:
                sql = sql[:sql.index(stop)].strip()
        
        conn = get_db_connection()
        if not conn:
            return None, None, "Cannot connect to database."
        
        result = pd.read_sql(sql, conn)
        conn.close()
        return sql, result, None
    except Exception as e:
        error_msg = str(e)
        if "no such table" in error_msg:
            return None, None, "Table not found. Try: 'Show anomaly days' or 'Revenue by category'"
        elif "no such column" in error_msg:
            return None, None, "Column not found. Try rephrasing your question."
        else:
            return None, None, f"Query failed: {error_msg[:100]}"


def ask_database_local(question: str):
    """
    Smart local query engine — matches questions to SQL without LLM.
    Handles edge cases and returns user-friendly errors.
    """
    conn = get_db_connection()
    if not conn:
        return None, None, "Cannot connect to database. Run: python src/create_demo_data.py"
    
    q = question.lower().strip()
    
    # Edge case: empty question
    if not q:
        return None, None, "Please type a question to get started."
    
    try:
        # --- Pattern matching with friendly queries ---
        
        if any(w in q for w in ["anomaly", "anomalies", "unusual", "spike", "alert"]):
            sql = """
            SELECT 
                order_date AS "Date",
                ROUND(daily_revenue, 0) AS "Revenue (₹)",
                order_count AS "Orders",
                z_score AS "Z-Score",
                CASE WHEN anomaly_flag = 1 THEN '🚨 Anomaly' ELSE '✅ Normal' END AS "Status"
            FROM daily_kpis 
            WHERE anomaly_flag = 1
            ORDER BY ABS(z_score) DESC 
            LIMIT 10
            """
        
        elif any(w in q for w in ["model", "algorithm", "method"]) and any(w in q for w in ["best", "compare", "performance", "rate", "which"]):
            sql = """
            SELECT 
                method_type AS "Method",
                COUNT(*) AS "Configs Tested",
                ROUND(AVG(CASE WHEN anomaly_pct BETWEEN 3 AND 10 THEN 1.0 ELSE 0.0 END) * 100, 1) AS "Success Rate %",
                ROUND(AVG(anomaly_pct), 1) AS "Avg Anomaly %"
            FROM model_results
            GROUP BY method_type
            ORDER BY "Success Rate %" DESC
            """
        
        elif ("revenue" in q or "sales" in q) and ("month" in q or "trend" in q or "total" in q):
            sql = """
            SELECT 
                strftime('%Y-%m', order_date) AS "Month",
                ROUND(SUM(order_value), 0) AS "Revenue (₹)",
                COUNT(*) AS "Orders",
                ROUND(AVG(order_value), 0) AS "Avg Order (₹)"
            FROM fact_orders
            WHERE order_status NOT IN ('cancelled', 'unavailable')
            GROUP BY "Month" 
            ORDER BY "Month"
            """
        
        elif any(w in q for w in ["category", "product", "categories"]):
            sql = """
            SELECT 
                p.category AS "Category",
                COUNT(*) AS "Total Orders",
                ROUND(SUM(f.order_value), 0) AS "Revenue (₹)",
                ROUND(AVG(f.order_value), 0) AS "Avg Order (₹)"
            FROM fact_orders f
            JOIN dim_products p ON f.product_id = p.product_id
            WHERE f.order_status NOT IN ('cancelled', 'unavailable')
            GROUP BY p.category 
            ORDER BY "Revenue (₹)" DESC
            """
        
        elif any(w in q for w in ["state", "region", "location", "city", "where"]):
            sql = """
            SELECT 
                c.state AS "State",
                COUNT(*) AS "Orders",
                ROUND(SUM(f.order_value), 0) AS "Revenue (₹)",
                ROUND(AVG(f.order_value), 0) AS "Avg Order (₹)"
            FROM fact_orders f
            JOIN dim_customers c ON f.customer_id = c.customer_id
            WHERE f.order_status NOT IN ('cancelled', 'unavailable')
            GROUP BY c.state 
            ORDER BY "Revenue (₹)" DESC 
            LIMIT 10
            """
        
        elif any(w in q for w in ["status", "cancelled", "delivered", "shipped"]):
            sql = """
            SELECT 
                order_status AS "Status",
                COUNT(*) AS "Orders",
                ROUND(SUM(order_value), 0) AS "Revenue (₹)",
                ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fact_orders), 1) AS "% of Total"
            FROM fact_orders
            GROUP BY order_status
            ORDER BY "Orders" DESC
            """
        
        elif any(w in q for w in ["feature", "scaler", "scaling"]):
            sql = """
            SELECT 
                feature_set AS "Feature Set",
                scaler AS "Scaler",
                COUNT(*) AS "Configs",
                ROUND(AVG(CASE WHEN anomaly_pct BETWEEN 3 AND 10 THEN 1.0 ELSE 0.0 END) * 100, 1) AS "Success Rate %",
                ROUND(AVG(anomaly_pct), 1) AS "Avg Anomaly %"
            FROM model_results
            WHERE method_type NOT LIKE 'Statistical%'
            GROUP BY feature_set, scaler
            ORDER BY "Success Rate %" DESC
            LIMIT 15
            """
        
        elif any(w in q for w in ["z-score", "zscore", "z score", "threshold"]):
            sql = """
            SELECT 
                order_date AS "Date",
                ROUND(daily_revenue, 0) AS "Revenue (₹)",
                z_score AS "Z-Score",
                CASE 
                    WHEN z_score > 2.5 THEN '📈 High Spike'
                    WHEN z_score < -2.5 THEN '📉 Big Drop'
                    ELSE '✅ Normal'
                END AS "Status"
            FROM daily_kpis
            WHERE ABS(z_score) > 2.0
            ORDER BY ABS(z_score) DESC
            LIMIT 10
            """
        
        elif any(w in q for w in ["top", "highest", "best day", "peak"]):
            sql = """
            SELECT 
                order_date AS "Date",
                ROUND(daily_revenue, 0) AS "Revenue (₹)",
                order_count AS "Orders"
            FROM daily_kpis
            ORDER BY daily_revenue DESC
            LIMIT 10
            """
        
        elif any(w in q for w in ["worst", "lowest", "bottom", "least"]):
            sql = """
            SELECT 
                order_date AS "Date",
                ROUND(daily_revenue, 0) AS "Revenue (₹)",
                order_count AS "Orders"
            FROM daily_kpis
            ORDER BY daily_revenue ASC
            LIMIT 10
            """
        
        else:
            # Default: show a helpful overview
            sql = """
            SELECT 
                'Total Orders' AS "Metric", CAST(COUNT(*) AS TEXT) AS "Value" FROM fact_orders
            UNION ALL
            SELECT 'Total Revenue (₹)', CAST(ROUND(SUM(order_value),0) AS TEXT) FROM fact_orders WHERE order_status NOT IN ('cancelled','unavailable')
            UNION ALL
            SELECT 'Anomaly Days Detected', CAST(SUM(anomaly_flag) AS TEXT) FROM daily_kpis
            UNION ALL
            SELECT 'Total Days Monitored', CAST(COUNT(*) AS TEXT) FROM daily_kpis
            UNION ALL
            SELECT 'Models Tested', '2,442'
            """
        
        result = pd.read_sql(sql, conn)
        conn.close()
        
        # Edge case: empty result
        if result.empty:
            return sql, None, "No data found for this query."
        
        return sql, result, None
    
    except Exception as e:
        conn.close()
        error_msg = str(e)
        if "no such table" in error_msg:
            return None, None, "Database tables missing. Run: python src/create_demo_data.py"
        return None, None, f"Query error: {error_msg[:80]}"


# --- Example Queries (user-friendly labels) ---
EXAMPLE_QUERIES = [
    "Which days had revenue anomalies?",
    "Compare model performance across all methods",
    "Show total monthly revenue",
    "Which product categories generate the most revenue?",
    "Top 10 states by order count",
    "What is the average anomaly detection rate by model type?",
    "Show me days where z-score exceeded 2.5",
    "Which feature set works best?",
]


if __name__ == "__main__":
    print("=" * 50)
    print("TEXT-TO-SQL ENGINE — Local Demo")
    print("=" * 50)
    
    test_questions = [
        "Show anomaly days",
        "Compare models",
        "Revenue by category",
        "Top states",
        "",  # edge case: empty
        "random gibberish xyz",  # edge case: no match
    ]
    
    for q in test_questions:
        print(f"\n> {q or '(empty)'}")
        sql, result, error = ask_database_local(q)
        if error:
            print(f"  ⚠️  {error}")
        elif result is not None:
            print(f"  ✅ {len(result)} rows returned")
        print()
