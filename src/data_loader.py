"""
Data loading and cleaning module for IBM Telco Customer Churn dataset.
"""

import os
import requests
import pandas as pd
import numpy as np

DATA_URL_PRIMARY = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_URL_FALLBACK = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"

RAW_DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
CLEAN_DATA_PATH = os.path.join("data", "cleaned_telco_churn.csv")


def download_dataset(destination_path: str = RAW_DATA_PATH) -> str:
    """Download the raw IBM Telco Customer Churn dataset if not already present."""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    if os.path.exists(destination_path) and os.path.getsize(destination_path) > 0:
        print(f"Dataset already exists at {destination_path} ({os.path.getsize(destination_path)} bytes).")
        return destination_path

    print(f"Downloading IBM Telco Customer Churn dataset to {destination_path}...")
    urls = [DATA_URL_PRIMARY, DATA_URL_FALLBACK]
    last_err = None
    for url in urls:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(destination_path, "wb") as f:
                    f.write(resp.content)
                print(f"Successfully downloaded {os.path.getsize(destination_path)} bytes from {url}")
                return destination_path
        except Exception as e:
            last_err = e
            print(f"Failed to download from {url}: {e}")

    raise RuntimeError(f"Could not download dataset from any source: {last_err}")


def load_and_clean_data(raw_path: str = RAW_DATA_PATH, save_clean: bool = True) -> pd.DataFrame:
    """
    Load dataset and resolve known data quality issues:
    - Convert TotalCharges from whitespace strings to float (fill tenure=0 blank TotalCharges with 0.0)
    - Validate types and churn binary target
    """
    download_dataset(raw_path)
    df = pd.read_csv(raw_path)
    print(f"Raw dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    # Fix known issue: TotalCharges has blank spaces (" ") for tenure=0 customers
    blanks_mask = df["TotalCharges"].astype(str).str.strip() == ""
    num_blanks = blanks_mask.sum()
    print(f"Found {num_blanks} rows with blank/whitespace TotalCharges (tenure=0).")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # For customers with tenure == 0, their total charges billed is naturally 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Ensure SeniorCitizen is integer/categorical representation
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Validate Churn column
    if "Churn" in df.columns:
        churn_counts = df["Churn"].value_counts(normalize=True).to_dict()
        print(f"Churn distribution: {churn_counts}")

    if save_clean:
        df.to_csv(CLEAN_DATA_PATH, index=False)
        print(f"Cleaned dataset saved to {CLEAN_DATA_PATH}.")

    return df


if __name__ == "__main__":
    df = load_and_clean_data()
    print("Sample cleaned records:")
    print(df[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]].head())
