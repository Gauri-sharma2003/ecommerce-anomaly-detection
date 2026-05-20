"""
Evaluation Metrics for Anomaly Detection Models
================================================
Provides proper statistical evaluation using ground truth labels:
- Precision, Recall, F1-Score
- ROC-AUC & PR-AUC curves
- Confusion matrix analysis
- Per-method comparative evaluation

This replaces the heuristic "success rate" with rigorous ML evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
from typing import Dict, Tuple, Optional
import sqlite3


def load_ground_truth(db_path: str = "ecom.db") -> pd.DataFrame:
    """Load ground truth anomaly labels from database."""
    conn = sqlite3.connect(db_path)
    try:
        gt = pd.read_sql("SELECT * FROM ground_truth_anomalies", conn)
        daily = pd.read_sql("SELECT order_date, is_true_anomaly FROM daily_kpis", conn)
        return daily
    except Exception as e:
        raise ValueError(f"Ground truth not found in database. Run create_demo_data.py first. Error: {e}")
    finally:
        conn.close()


def evaluate_detector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method_name: str = "Unknown"
) -> Dict[str, float]:
    """
    Evaluate a single anomaly detector against ground truth.
    
    Parameters
    ----------
    y_true : array-like, binary ground truth (1=anomaly, 0=normal)
    y_pred : array-like, binary predictions (1=anomaly, 0=normal)
    method_name : str, name for reporting
    
    Returns
    -------
    dict with precision, recall, f1, specificity, false_positive_rate
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        "method": method_name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "false_positive_rate": round(fpr, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_flagged": int(tp + fp),
        "total_anomalies": int(tp + fn),
    }


def evaluate_all_methods(
    daily_kpis: pd.DataFrame,
    anomaly_columns: Dict[str, str],
) -> pd.DataFrame:
    """
    Evaluate multiple detection methods against ground truth.
    
    Parameters
    ----------
    daily_kpis : DataFrame with 'is_true_anomaly' column and prediction columns
    anomaly_columns : dict mapping method_name -> column_name with binary predictions
    
    Returns
    -------
    DataFrame with evaluation metrics for each method
    """
    results = []
    y_true = daily_kpis["is_true_anomaly"].values
    
    for method_name, col in anomaly_columns.items():
        if col in daily_kpis.columns:
            y_pred = daily_kpis[col].values
            metrics = evaluate_detector(y_true, y_pred, method_name)
            results.append(metrics)
    
    return pd.DataFrame(results).sort_values("f1_score", ascending=False)


def compute_threshold_analysis(
    daily_kpis: pd.DataFrame,
    score_column: str = "z_score",
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Analyze how different thresholds affect precision/recall tradeoff.
    Helps find the optimal operating point.
    """
    if thresholds is None:
        thresholds = np.arange(1.0, 4.1, 0.25)
    
    y_true = daily_kpis["is_true_anomaly"].values
    results = []
    
    for t in thresholds:
        y_pred = (daily_kpis[score_column].abs() > t).astype(int).values
        metrics = evaluate_detector(y_true, y_pred, f"threshold_{t:.2f}")
        metrics["threshold"] = t
        results.append(metrics)
    
    return pd.DataFrame(results)


def generate_evaluation_report(db_path: str = "ecom.db") -> str:
    """
    Generate a complete evaluation report comparing all methods.
    Returns formatted string report.
    """
    conn = sqlite3.connect(db_path)
    daily = pd.read_sql("SELECT * FROM daily_kpis", conn)
    conn.close()
    
    if "is_true_anomaly" not in daily.columns:
        return "⚠️ No ground truth labels found. Run updated create_demo_data.py first."
    
    # Z-score based detection at different thresholds
    report_lines = [
        "=" * 60,
        "📊 ANOMALY DETECTION EVALUATION REPORT",
        "   Using Precision / Recall / F1 on Labeled Anomalies",
        "=" * 60,
        "",
        f"Total days: {len(daily)}",
        f"True anomaly days: {daily['is_true_anomaly'].sum()}",
        f"Normal days: {(~daily['is_true_anomaly'].astype(bool)).sum()}",
        "",
        "--- Z-Score Threshold Analysis ---",
    ]
    
    threshold_results = compute_threshold_analysis(daily)
    for _, row in threshold_results.iterrows():
        report_lines.append(
            f"  Z > {row['threshold']:.2f}: "
            f"Precision={row['precision']:.3f}, "
            f"Recall={row['recall']:.3f}, "
            f"F1={row['f1_score']:.3f} "
            f"(flagged {row['total_flagged']} days)"
        )
    
    best = threshold_results.loc[threshold_results["f1_score"].idxmax()]
    report_lines.extend([
        "",
        f"✅ OPTIMAL THRESHOLD: Z > {best['threshold']:.2f}",
        f"   F1={best['f1_score']:.3f}, Precision={best['precision']:.3f}, Recall={best['recall']:.3f}",
        "",
    ])
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    report = generate_evaluation_report()
    print(report)
