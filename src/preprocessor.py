"""
Data preprocessing pipeline module for Telecom Customer Churn.
Builds a reusable scikit-learn ColumnTransformer.
"""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define canonical feature columns
NUMERICAL_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"


def get_feature_lists() -> Tuple[List[str], List[str], str]:
    """Return numerical, categorical, and target column definitions."""
    return NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that handles:
    1. Numerical features: Median imputation followed by standard scaling.
    2. Categorical features: Most frequent imputation followed by One-Hot Encoding (drop='first').
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits DataFrame into feature matrix X and binary target vector y.
    Maps 'Yes' -> 1, 'No' -> 0.
    """
    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[features].copy()
    
    # Cast SeniorCitizen explicitly to object or int for categorical encoding
    X["SeniorCitizen"] = X["SeniorCitizen"].astype(str)
    
    y = None
    if TARGET_COLUMN in df.columns:
        y = (df[TARGET_COLUMN].astype(str).str.strip().str.lower() == "yes").astype(int)

    return X, y


if __name__ == "__main__":
    from src.data_loader import load_and_clean_data
    df = load_and_clean_data()
    X, y = prepare_features_and_target(df)
    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    print(f"X shape: {X.shape} -> Transformed shape: {X_trans.shape}")
    print(f"Number of generated features: {len(feature_names)}")
    print("Features sample:", feature_names[:10])
