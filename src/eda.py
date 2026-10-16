"""
Exploratory Data Analysis (EDA) module for Telecom Customer Churn Prediction.
Generates polished visualizations saved directly to reports/.
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Apply clean professional styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["figure.dpi"] = 300

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Cohesive modern color palette
CHURN_PALETTE = {"No": "#2b5c8f", "Yes": "#e74c3c"}


def plot_churn_by_contract(df: pd.DataFrame, output_dir: str = REPORTS_DIR):
    """Plot churn rate and count by contract type."""
    contract_df = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack() * 100
    contract_df = contract_df.reindex(["Month-to-month", "One year", "Two year"])

    fig, ax = plt.subplots(figsize=(8, 5))
    contract_df.plot(
        kind="bar",
        stacked=False,
        color=[CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]],
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
        width=0.6,
    )
    ax.set_title("Customer Churn Rate by Contract Type", pad=15)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Contract Type")
    ax.set_xticks(range(len(contract_df.index)))
    ax.set_xticklabels(contract_df.index, rotation=0)
    ax.legend(["Retained (No)", "Churned (Yes)"], frameon=True, facecolor="white")
    ax.set_ylim(0, 100)

    # Annotate bars with percentages
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height + 1.5),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    filepath = os.path.join(output_dir, "churn_by_contract.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved: {filepath}")


def plot_churn_by_tenure(df: pd.DataFrame, output_dir: str = REPORTS_DIR):
    """Plot customer tenure distribution stratified by churn."""
    fig, ax = plt.subplots(figsize=(9, 5))
    
    sns.kdeplot(
        data=df[df["Churn"] == "Yes"],
        x="tenure",
        fill=True,
        color=CHURN_PALETTE["Yes"],
        alpha=0.45,
        label="Churned (Yes)",
        linewidth=2,
        ax=ax,
    )
    sns.kdeplot(
        data=df[df["Churn"] == "No"],
        x="tenure",
        fill=True,
        color=CHURN_PALETTE["No"],
        alpha=0.45,
        label="Retained (No)",
        linewidth=2,
        ax=ax,
    )
    
    ax.set_title("Customer Tenure Distribution by Churn Status", pad=15)
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, facecolor="white", title="Customer Status")
    ax.set_xlim(0, 72)
    
    # Add annotation for new customer cliff
    ax.annotate(
        "Critical Churn Zone:\nFirst 0-12 Months",
        xy=(6, 0.035),
        xytext=(18, 0.042),
        arrowprops=dict(facecolor="#c0392b", shrink=0.05, width=1.5, headwidth=8),
        fontsize=10,
        fontweight="bold",
        color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fadbd8", edgecolor="#c0392b", alpha=0.9),
    )

    plt.tight_layout()
    filepath = os.path.join(output_dir, "churn_by_tenure.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved: {filepath}")


def plot_churn_by_monthly_charges(df: pd.DataFrame, output_dir: str = REPORTS_DIR):
    """Plot monthly charges distribution comparison between churned and retained customers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.5, 1]})

    # KDE on ax1
    sns.kdeplot(
        data=df[df["Churn"] == "Yes"],
        x="MonthlyCharges",
        fill=True,
        color=CHURN_PALETTE["Yes"],
        alpha=0.45,
        label="Churned (Yes)",
        linewidth=2,
        ax=ax1,
    )
    sns.kdeplot(
        data=df[df["Churn"] == "No"],
        x="MonthlyCharges",
        fill=True,
        color=CHURN_PALETTE["No"],
        alpha=0.45,
        label="Retained (No)",
        linewidth=2,
        ax=ax1,
    )
    ax1.set_title("Monthly Charges Density", pad=12)
    ax1.set_xlabel("Monthly Charges ($)")
    ax1.set_ylabel("Density")
    ax1.legend(frameon=True, facecolor="white")

    # Boxplot on ax2
    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges",
        hue="Churn",
        palette=[CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]],
        legend=False,
        ax=ax2,
        width=0.45,
        boxprops=dict(edgecolor="black", linewidth=1),
    )
    ax2.set_title("Monthly Charges Comparison", pad=12)
    ax2.set_xlabel("Churn Status")
    ax2.set_ylabel("Monthly Charges ($)")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Retained (No)", "Churned (Yes)"])

    plt.suptitle("Impact of Monthly Charges on Telecom Churn", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "churn_by_monthly_charges.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = REPORTS_DIR):
    """Generate correlation heatmap including numerical features and one-hot encoded drivers."""
    eda_df = df.copy()
    eda_df["Churn_Flag"] = (eda_df["Churn"] == "Yes").astype(int)
    
    cols_to_encode = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Contract",
        "InternetService",
        "PaymentMethod",
    ]
    
    encoded_subset = pd.get_dummies(
        eda_df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn_Flag"] + cols_to_encode],
        drop_first=True,
        dtype=float,
    )
    
    corr = encoded_subset.corr()
    top_corr_features = corr["Churn_Flag"].abs().sort_values(ascending=False).head(16).index
    top_corr = corr.loc[top_corr_features, top_corr_features]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        top_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.6,
        linecolor="#f0f0f0",
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax,
    )
    ax.set_title("Top Feature Correlations with Customer Churn", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def run_eda(data_path: str = "data/cleaned_telco_churn.csv", output_dir: str = REPORTS_DIR):
    """Execute full EDA suite and export report plots."""
    if not os.path.exists(data_path):
        from src.data_loader import load_and_clean_data
        df = load_and_clean_data()
    else:
        df = pd.read_csv(data_path)

    print("Running Exploratory Data Analysis (EDA)...")
    plot_churn_by_contract(df, output_dir)
    plot_churn_by_tenure(df, output_dir)
    plot_churn_by_monthly_charges(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    
    summary = {
        "total_customers": int(len(df)),
        "overall_churn_rate": float(round((df["Churn"] == "Yes").mean() * 100, 2)),
        "churn_rate_by_contract": {
            k: float(round(v * 100, 2))
            for k, v in df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()).to_dict().items()
        },
        "median_tenure_months": {
            "retained": float(df[df["Churn"] == "No"]["tenure"].median()),
            "churned": float(df[df["Churn"] == "Yes"]["tenure"].median()),
        },
        "median_monthly_charges_usd": {
            "retained": float(df[df["Churn"] == "No"]["MonthlyCharges"].median()),
            "churned": float(df[df["Churn"] == "Yes"]["MonthlyCharges"].median()),
        },
    }
    
    summary_path = os.path.join(output_dir, "eda_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"EDA Summary saved to {summary_path}")


if __name__ == "__main__":
    run_eda()
