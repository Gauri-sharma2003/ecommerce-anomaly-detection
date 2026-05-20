"""
Streamlit App: Text-to-SQL + RCA Dashboard
===========================================
Clean, user-friendly interface for querying e-commerce data
and viewing model RCA analysis.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_to_sql import ask_database_local, ask_database_llm, EXAMPLE_QUERIES, GROQ_API_KEY

# --- Page Config ---
st.set_page_config(
    page_title="E-commerce Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for cleaner UI ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 18px; }
    .insight-box { background: #f0fdf4; border-left: 4px solid #16a34a; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; font-size: 14px; }
    .warning-box { background: #fef3c7; border-left: 4px solid #d97706; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; font-size: 14px; }
    .stButton > button { border-radius: 8px; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("🔍 Anomaly Detection")
    st.caption("E-commerce Analytics Platform")
    st.divider()
    
    page = st.radio(
        "Navigate to:",
        ["💬 Ask Your Data", "📊 Model RCA", "📈 Quick Insights"],
        index=0
    )
    
    st.divider()
    use_llm = st.toggle("🤖 Use AI (Groq LLM)", value=False)
    if use_llm:
        if GROQ_API_KEY:
            st.success("API key loaded ✓", icon="✅")
        else:
            st.warning("Add GROQ_API_KEY to .env")
            use_llm = False
    
    st.divider()
    st.caption("💡 **Quick Tips**")
    st.caption("• Click example buttons to try")
    st.caption("• Charts auto-generate from results")
    st.caption("• Toggle AI for natural language queries")

# ============================================================
# PAGE 1: ASK YOUR DATA
# ============================================================
if page == "💬 Ask Your Data":
    st.title("💬 Ask Your Data")
    st.markdown("Get instant answers from your e-commerce database. Click a question or type your own.")
    
    st.markdown("")
    
    # Example questions as clean buttons
    st.markdown("##### 💡 Quick Questions")
    
    col1, col2, col3 = st.columns(3)
    examples = [
        ("🚨 Anomaly Days", "Which days had revenue anomalies?"),
        ("📦 By Category", "Which product categories generate the most revenue?"),
        ("🏆 Best Models", "Compare model performance across all methods"),
        ("📍 By State", "Top 10 states by order count"),
        ("💰 Monthly Trend", "Show total monthly revenue"),
        ("📊 Model Rates", "What is the average anomaly detection rate by model type?"),
    ]
    
    for i, (label, query) in enumerate(examples):
        col = [col1, col2, col3][i % 3]
        if col.button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state["q"] = query
    
    st.markdown("---")
    
    # Input box
    question = st.text_input(
        "🔍 Type your question:",
        value=st.session_state.get("q", ""),
        placeholder="e.g., Which days had the highest revenue?"
    )
    
    if question:
        with st.spinner("Querying..."):
            if use_llm:
                sql, result, error = ask_database_llm(question)
            else:
                sql, result, error = ask_database_local(question)
        
        if error:
            st.error(f"❌ {error}")
            st.info("💡 Try clicking one of the quick questions above, or rephrase your question.")
        elif result is None or result.empty:
            st.warning("⚠️ No results found. Try a different question.")
        else:
            # Results header
            st.markdown(f"**📋 {len(result)} results found**")
            
            # Clean dataframe display
            st.dataframe(result, use_container_width=True, hide_index=True, height=min(400, 35 * len(result) + 38))
            
            # Smart auto-chart
            if len(result) > 1 and len(result.columns) >= 2:
                num_cols = result.select_dtypes(include="number").columns.tolist()
                cat_cols = result.select_dtypes(exclude="number").columns.tolist()
                
                if num_cols and cat_cols:
                    x_col = cat_cols[0]
                    y_col = num_cols[0]
                    
                    # Choose chart type intelligently
                    if any(d in x_col.lower() for d in ["date", "month", "day", "year"]):
                        fig = px.line(result, x=x_col, y=y_col, markers=True,
                                      title=f"📈 {y_col} over time")
                        fig.update_traces(line_color="#6366f1", marker_color="#4f46e5")
                    elif len(result) <= 12:
                        fig = px.bar(result, x=x_col, y=y_col,
                                     title=f"📊 {y_col} by {x_col}",
                                     color=y_col,
                                     color_continuous_scale=["#c7d2fe", "#4338ca"])
                    else:
                        fig = px.bar(result.head(12), x=x_col, y=y_col,
                                     title=f"📊 Top 12 — {y_col} by {x_col}",
                                     color=y_col,
                                     color_continuous_scale=["#c7d2fe", "#4338ca"])
                    
                    fig.update_layout(
                        plot_bgcolor="white",
                        font=dict(size=12),
                        margin=dict(t=50, b=30, l=30, r=30),
                        coloraxis_showscale=False,
                        xaxis=dict(gridcolor="#f1f5f9"),
                        yaxis=dict(gridcolor="#f1f5f9"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # SQL in collapsed section (not overwhelming)
            with st.expander("🔧 View SQL Query", expanded=False):
                st.code(sql.strip(), language="sql")
    else:
        # Show helpful empty state
        st.markdown("")
        st.info("👆 Click a quick question above or type your own to get started!")

# ============================================================
# PAGE 2: MODEL RCA
# ============================================================
elif page == "📊 Model RCA":
    st.title("📊 Model RCA — Why Models Fail")
    st.markdown("Root Cause Analysis across **2,442 model configurations** to find what works and what breaks.")
    
    df = pd.read_csv("results/model_comparison.csv")
    
    # Top metrics
    reasonable = df[(df["anomaly_pct"] >= 3) & (df["anomaly_pct"] <= 10)]
    broken = df[df["anomaly_pct"] > 20]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧪 Total Tested", f"{len(df):,}")
    c2.metric("✅ Working", f"{len(reasonable):,}", f"{len(reasonable)/len(df)*100:.0f}%")
    c3.metric("❌ Broken", f"{len(broken):,}", f"-{len(broken)/len(df)*100:.0f}%", delta_color="inverse")
    c4.metric("🔧 Methods", df["method_type"].nunique())
    
    st.markdown("---")
    
    # Key findings - simple cards
    st.markdown("### 🔍 Key Findings at a Glance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-box">
            <strong>✅ Isolation Forest — Best Overall</strong><br>
            91.7% success rate across all configurations.<br>
            Works with any feature set and scaler. Fast and reliable.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
            <strong>✅ LOF — Strong Runner-up</strong><br>
            91.7% success. Best with robust scaler.<br>
            Good at detecting local density anomalies.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <strong>❌ DBSCAN — Fails Completely</strong><br>
            Only 10.6% success. Flags everything as anomaly.<br>
            <em>Cause: Can't handle 21 features (curse of dimensionality)</em>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="warning-box">
            <strong>❌ SVM (Poly Kernel) — Unreliable</strong><br>
            59% failure rate. Avg 31% anomaly rate (should be ~5%).<br>
            <em>Cause: Polynomial kernel explodes in high dimensions</em>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart: Success rate by method
    st.markdown("### 📊 Method Success Rates")
    method_summary = df.groupby("method_type").agg(
        configs=("model", "count"),
        success=("anomaly_pct", lambda x: ((x >= 3) & (x <= 10)).sum()),
    ).reset_index()
    method_summary["success_rate"] = (method_summary["success"] / method_summary["configs"] * 100).round(1)
    method_summary = method_summary.sort_values("success_rate", ascending=True)
    
    fig1 = px.bar(method_summary, x="success_rate", y="method_type",
                  orientation="h", text="success_rate",
                  color="success_rate",
                  color_continuous_scale=["#fca5a5", "#4ade80"])
    fig1.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig1.update_layout(
        plot_bgcolor="white", coloraxis_showscale=False,
        xaxis_title="Success Rate %", yaxis_title="",
        height=350, margin=dict(l=0, t=30)
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Chart: Feature dimensions vs performance
    st.markdown("### 🧪 More Features ≠ Better")
    st.caption("Bubble size = number of broken configurations")
    
    feat_summary = df.groupby("feature_set").agg(
        avg_anomaly=("anomaly_pct", "mean"),
        broken=("anomaly_pct", lambda x: (x > 20).sum()),
        n_features=("n_features", "first"),
    ).round(1).reset_index()
    
    fig2 = px.scatter(feat_summary, x="n_features", y="avg_anomaly",
                      size="broken", text="feature_set",
                      color="avg_anomaly",
                      color_continuous_scale=["#16a34a", "#dc2626"],
                      size_max=50)
    fig2.update_traces(textposition="top center", textfont_size=11)
    fig2.update_layout(
        plot_bgcolor="white", coloraxis_showscale=False,
        xaxis_title="Number of Features", yaxis_title="Avg Anomaly %",
        height=350, margin=dict(t=30)
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Heatmap
    st.markdown("### 🗺️ Full Heatmap (Method × Feature Set)")
    pivot = df.pivot_table(values="anomaly_pct", index="feature_set",
                           columns="method_type", aggfunc="mean").round(1)
    fig3 = px.imshow(pivot, text_auto=True, aspect="auto",
                     color_continuous_scale="RdYlGn_r")
    fig3.update_layout(height=350, margin=dict(t=30, b=30))
    st.plotly_chart(fig3, use_container_width=True)
    
    # Final recommendation
    st.markdown("---")
    st.markdown("### 🏆 Production Recommendation")
    st.success("""
    **Use this in production:**  
    • **Primary:** `IsolationForest(n_estimators=200, contamination=0.05)` + StandardScaler  
    • **Backup:** Z-Score (threshold=2.5) for statistical confirmation  
    • **Rule:** Only alert when **both** methods agree → fewer false alarms
    """)

# ============================================================
# PAGE 3: QUICK INSIGHTS
# ============================================================
elif page == "📈 Quick Insights":
    st.title("📈 Business Overview")
    st.markdown("Live snapshot of your e-commerce data and anomaly detection results.")
    
    import sqlite3
    conn = sqlite3.connect("ecom.db")
    
    # Check tables exist
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    if tables.empty:
        st.error("⚠️ Database is empty. Run this command first:")
        st.code("python src/create_demo_data.py", language="bash")
        st.stop()
    
    # KPI cards
    try:
        total_orders = pd.read_sql("SELECT COUNT(*) as c FROM fact_orders", conn).iloc[0, 0]
        total_revenue = pd.read_sql("SELECT ROUND(SUM(order_value),0) FROM fact_orders WHERE order_status NOT IN ('cancelled','unavailable')", conn).iloc[0, 0]
        anomaly_days = pd.read_sql("SELECT COUNT(*) FROM daily_kpis WHERE anomaly_flag=1", conn).iloc[0, 0]
        total_days = pd.read_sql("SELECT COUNT(*) FROM daily_kpis", conn).iloc[0, 0]
    except:
        st.error("Error reading data. Run: python src/create_demo_data.py")
        st.stop()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Orders", f"{total_orders:,}")
    c2.metric("💰 Revenue", f"₹{total_revenue:,.0f}")
    c3.metric("🚨 Anomalies", f"{anomaly_days} days")
    c4.metric("📅 Monitored", f"{total_days} days")
    
    st.markdown("---")
    
    # Revenue timeline with anomalies
    st.markdown("### 📈 Daily Revenue — Anomalies Highlighted")
    daily = pd.read_sql("SELECT * FROM daily_kpis ORDER BY order_date", conn)
    daily["order_date"] = pd.to_datetime(daily["order_date"])
    
    fig = go.Figure()
    normal = daily[daily["anomaly_flag"] == 0]
    anomalies = daily[daily["anomaly_flag"] == 1]
    
    fig.add_trace(go.Scatter(
        x=normal["order_date"], y=normal["daily_revenue"],
        mode="lines", name="Normal Days",
        line=dict(color="#6366f1", width=1.5)
    ))
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["order_date"], y=anomalies["daily_revenue"],
            mode="markers", name="🚨 Anomaly",
            marker=dict(color="#dc2626", size=12, symbol="diamond")
        ))
    fig.update_layout(
        plot_bgcolor="white", height=350,
        margin=dict(t=30, b=30),
        xaxis=dict(gridcolor="#f1f5f9"),
        yaxis=dict(gridcolor="#f1f5f9", title="Revenue (₹)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Two column charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🛍️ Revenue by Category")
        cat_data = pd.read_sql("""
            SELECT p.category, ROUND(SUM(f.order_value),0) as revenue
            FROM fact_orders f JOIN dim_products p ON f.product_id = p.product_id
            WHERE f.order_status NOT IN ('cancelled','unavailable')
            GROUP BY p.category ORDER BY revenue DESC
        """, conn)
        fig_cat = px.pie(cat_data, names="category", values="revenue", hole=0.45)
        fig_cat.update_layout(margin=dict(t=20, b=20), height=300, showlegend=True,
                              legend=dict(font=dict(size=10)))
        fig_cat.update_traces(textinfo="percent+label", textfont_size=10)
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        st.markdown("##### 📍 Top States by Revenue")
        state_data = pd.read_sql("""
            SELECT c.state, ROUND(SUM(f.order_value),0) as revenue
            FROM fact_orders f JOIN dim_customers c ON f.customer_id = c.customer_id
            WHERE f.order_status NOT IN ('cancelled','unavailable')
            GROUP BY c.state ORDER BY revenue DESC LIMIT 8
        """, conn)
        fig_state = px.bar(state_data, x="state", y="revenue",
                           color="revenue", color_continuous_scale=["#c7d2fe", "#4338ca"])
        fig_state.update_layout(
            plot_bgcolor="white", height=300, coloraxis_showscale=False,
            margin=dict(t=20, b=20), xaxis_title="", yaxis_title="Revenue (₹)"
        )
        st.plotly_chart(fig_state, use_container_width=True)
    
    conn.close()
