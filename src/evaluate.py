"""
Evaluation and Metrics module for Telecom Customer Churn Prediction.
Calculates test metrics, generates confusion matrix and ROC curves,
and performs customer risk segmentation.
"""

import os
import json
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Calculate core classification metrics on test data."""
    cm = confusion_matrix(y_true, y_pred).tolist()
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "true_negatives": int(cm[0][0]),
            "false_positives": int(cm[0][1]),
            "false_negatives": int(cm[1][0]),
            "true_positives": int(cm[1][1]),
            "matrix": cm,
        },
    }
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str = os.path.join(REPORTS_DIR, "confusion_matrix.png")):
    """Generate publication-ready confusion matrix visualization with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    labels = np.array([
        f"{count}\n({pct:.1%})"
        for count, pct in zip(cm.flatten(), cm_norm.flatten())
    ]).reshape(2, 2)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=1.5,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 13, "weight": "bold"},
    )
    ax.set_title("Held-Out Test Set Confusion Matrix", pad=15, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Retained (0)", "Churned (1)"])
    ax.set_yticklabels(["Retained (0)", "Churned (1)"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str = os.path.join(REPORTS_DIR, "roc_curve.png")):
    """Generate ROC curve plot with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = float(roc_auc_score(y_true, y_prob))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC Curve (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#e74c3c", lw=1.5, linestyle="--", label="Chance Level (AUC = 0.5000)")
    
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", pad=15, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def perform_risk_segmentation(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    low_thresh: float = 0.30,
    high_thresh: float = 0.60,
    output_dir: str = REPORTS_DIR,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Segment customers into Low / Medium / High churn risk tiers based on predicted probability.
    Saves a summary table of counts and business impact to CSV and JSON.
    """
    df_eval = df.copy()
    df_eval["Churn_Probability"] = np.array(y_prob, dtype=float)
    
    def assign_tier(prob: float) -> str:
        if prob < low_thresh:
            return "Low Risk"
        elif prob < high_thresh:
            return "Medium Risk"
        else:
            return "High Risk"

    df_eval["Risk_Segment"] = df_eval["Churn_Probability"].apply(assign_tier)

    # Compute aggregation
    segment_order = ["Low Risk", "Medium Risk", "High Risk"]
    summary_rows = []
    
    total_customers = len(df_eval)
    for seg in segment_order:
        subset = df_eval[df_eval["Risk_Segment"] == seg]
        count = int(len(subset))
        pct_base = round(float((count / total_customers) * 100), 2)
        mean_prob = round(float(subset["Churn_Probability"].mean() * 100), 2) if count > 0 else 0.0
        
        # If ground truth Churn exists, compute actual churn rate
        if "Churn" in subset.columns:
            actual_churn_rate = round(float((subset["Churn"] == "Yes").mean() * 100), 2) if count > 0 else 0.0
        else:
            actual_churn_rate = None

        monthly_rev = round(float(subset["MonthlyCharges"].sum()), 2) if "MonthlyCharges" in subset.columns else 0.0
        avg_monthly = round(float(subset["MonthlyCharges"].mean()), 2) if "MonthlyCharges" in subset.columns else 0.0

        summary_rows.append({
            "Risk_Segment": seg,
            "Probability_Range": f"< {int(low_thresh*100)}%" if seg == "Low Risk" else (f"{int(low_thresh*100)}% - {int(high_thresh*100)}%" if seg == "Medium Risk" else f">= {int(high_thresh*100)}%"),
            "Customer_Count": count,
            "Percentage_of_Total": pct_base,
            "Average_Predicted_Prob": mean_prob,
            "Actual_Churn_Rate": actual_churn_rate,
            "Total_Monthly_Revenue_USD": monthly_rev,
            "Avg_Monthly_Charges_USD": avg_monthly,
        })

    summary_df = pd.DataFrame(summary_rows)
    
    # Export CSV and JSON
    csv_path = os.path.join(output_dir, "risk_segments.csv")
    summary_df.to_csv(csv_path, index=False)
    
    json_path = os.path.join(output_dir, "risk_segments.json")
    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"Risk segmentation saved to {csv_path} and {json_path}")
    print("\nCustomer Risk Segmentation Summary:")
    print(summary_df.to_string(index=False))

    return summary_df, summary_rows
