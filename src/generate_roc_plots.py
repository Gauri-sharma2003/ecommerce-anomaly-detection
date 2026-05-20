"""
Generate ROC & PR-AUC curve plots for README and reports.
Saves publication-quality plots to results/ directory.

Run: python src/generate_roc_plots.py
"""

import sys
import os
import numpy as np
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_anomaly_scores(daily: pd.DataFrame) -> dict:
    """Run multiple detectors and return anomaly scores (not binary)."""
    features = daily[["daily_revenue", "order_count", "avg_order_value", "avg_freight"]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    scores = {}

    # Z-Score (absolute z-score as anomaly score)
    scores["Z-Score"] = daily["z_score"].abs().values

    # Isolation Forest (decision_function returns anomaly score — more negative = more anomalous)
    iforest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iforest.fit(X)
    # Negate so higher = more anomalous
    scores["Isolation Forest"] = -iforest.decision_function(X)

    # LOF (negative_outlier_factor_ — more negative = more anomalous)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=False)
    lof.fit_predict(X)
    scores["LOF"] = -lof.negative_outlier_factor_

    return scores


def plot_roc_curves(y_true: np.ndarray, scores: dict, save_path: str = "results/roc_curves.html"):
    """Generate ROC curves for all methods."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("ROC Curve (Higher = Better)", "Precision-Recall Curve"),
                        horizontal_spacing=0.12)

    colors = {"Z-Score": "#3B82F6", "Isolation Forest": "#10B981", "LOF": "#F59E0B"}

    for method, score in scores.items():
        color = colors.get(method, "#6B7280")

        # ROC
        fpr, tpr, _ = roc_curve(y_true, score)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{method} (AUC={roc_auc:.3f})",
                       line=dict(color=color, width=2.5)),
            row=1, col=1
        )

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, score)
        pr_auc = auc(recall, precision)
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode="lines", name=f"{method} (PR-AUC={pr_auc:.3f})",
                       line=dict(color=color, width=2.5, dash="dot"), showlegend=False),
            row=1, col=2
        )

    # Random baseline for ROC
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                   line=dict(color="gray", width=1, dash="dash")),
        row=1, col=1
    )

    fig.update_layout(
        title="Anomaly Detection Model Evaluation — ROC & PR-AUC Curves",
        height=450, width=1000,
        template="plotly_white",
        legend=dict(x=0.35, y=-0.15, orientation="h"),
        margin=dict(t=80, b=100),
    )
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)

    # Also save static image if kaleido available
    try:
        fig.write_image(save_path.replace(".html", ".png"), scale=2)
        print(f"✅ Saved: {save_path.replace('.html', '.png')}")
    except Exception:
        print("   (Install kaleido for PNG export: pip install kaleido)")

    print(f"✅ Saved: {save_path}")
    return fig


def main():
    print("=" * 60)
    print("📈 Generating ROC & PR-AUC Curves")
    print("=" * 60)

    db_path = "ecom.db"
    if not os.path.exists(db_path):
        print("⚠️ ecom.db not found. Run: python src/create_demo_data.py")
        return

    conn = sqlite3.connect(db_path)
    daily = pd.read_sql("SELECT * FROM daily_kpis", conn)
    conn.close()

    if "is_true_anomaly" not in daily.columns:
        print("⚠️ No ground truth labels. Run updated create_demo_data.py first.")
        return

    y_true = daily["is_true_anomaly"].values
    print(f"   Total days: {len(daily)}, Anomaly days: {y_true.sum()}")

    scores = generate_anomaly_scores(daily)
    fig = plot_roc_curves(y_true, scores)

    # Print summary
    print("\n📊 AUC Summary:")
    for method, score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, score)
        pr_auc_val = auc(recall, precision)
        print(f"   {method:20s} → ROC-AUC: {roc_auc:.3f} | PR-AUC: {pr_auc_val:.3f}")


if __name__ == "__main__":
    main()
