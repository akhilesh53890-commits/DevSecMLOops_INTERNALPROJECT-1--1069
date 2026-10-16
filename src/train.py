"""
Model training, hyperparameter tuning, cross-validation, SHAP explanations,
and artifact generation module for Telecom Customer Churn Prediction.
"""

import os
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import load_and_clean_data
from src.evaluate import calculate_metrics, perform_risk_segmentation, plot_confusion_matrix, plot_roc_curve
from src.preprocessor import build_preprocessor, prepare_features_and_target

MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def train_and_evaluate_models():
    print("=" * 70)
    print("STEP 1: Loading and Preparing Data")
    print("=" * 70)
    df = load_and_clean_data()
    X, y = prepare_features_and_target(df)

    # 80/20 Stratified train/test split
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train split: {X_train.shape[0]} samples (Churn: {y_train.sum()} / {len(y_train)} = {y_train.mean():.2%})")
    print(f"Test split:  {X_test.shape[0]} samples (Churn: {y_test.sum()} / {len(y_test)} = {y_test.mean():.2%})")

    # Calculate class imbalance ratio for XGBoost
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count / pos_count)
    print(f"Class distribution: 0={neg_count}, 1={pos_count}. scale_pos_weight = {scale_pos_weight:.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---------------------------------------------------------
    # MODEL 1: Logistic Regression Pipeline & Tuning
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Tuning Logistic Regression with 5-Fold Stratified CV")
    print("=" * 70)
    lr_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])

    lr_param_grid = {
        "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        "classifier__solver": ["lbfgs"],
    }

    lr_search = GridSearchCV(
        lr_pipeline,
        param_grid=lr_param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    lr_search.fit(X_train, y_train)
    best_lr = lr_search.best_estimator_
    print(f"Logistic Regression Best CV ROC-AUC: {lr_search.best_score_:.4f}")
    print(f"Logistic Regression Best Params: {lr_search.best_params_}")

    # ---------------------------------------------------------
    # MODEL 2: XGBoost Pipeline & Tuning
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Tuning XGBoost Classifier with 5-Fold Stratified CV")
    print("=" * 70)
    xgb_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    xgb_param_dist = {
        "classifier__n_estimators": [100, 150, 200, 250],
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__learning_rate": [0.02, 0.05, 0.08, 0.12],
        "classifier__subsample": [0.7, 0.85, 1.0],
        "classifier__colsample_bytree": [0.7, 0.85, 1.0],
        "classifier__min_child_weight": [1, 3, 5],
        "classifier__gamma": [0, 0.1, 0.3],
    }

    xgb_search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=xgb_param_dist,
        n_iter=25,
        cv=cv,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    print(f"XGBoost Best CV ROC-AUC: {xgb_search.best_score_:.4f}")
    print(f"XGBoost Best Params: {xgb_search.best_params_}")

    # ---------------------------------------------------------
    # STEP 4: Held-Out Test Set Evaluation & Comparison
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Held-Out Test Set Evaluation")
    print("=" * 70)

    # Evaluate Logistic Regression
    lr_test_probs = best_lr.predict_proba(X_test)[:, 1].astype(float)
    lr_test_preds = (lr_test_probs >= 0.5).astype(int)
    lr_metrics = calculate_metrics(y_test.values, lr_test_preds, lr_test_probs)

    # Evaluate XGBoost
    xgb_test_probs = best_xgb.predict_proba(X_test)[:, 1].astype(float)
    xgb_test_preds = (xgb_test_probs >= 0.5).astype(int)
    xgb_metrics = calculate_metrics(y_test.values, xgb_test_preds, xgb_test_probs)

    print("Test Set Metrics Summary:")
    comparison_df = pd.DataFrame({
        "Metric": ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"],
        "Logistic Regression": [
            lr_metrics["roc_auc"],
            lr_metrics["accuracy"],
            lr_metrics["precision"],
            lr_metrics["recall"],
            lr_metrics["f1_score"],
        ],
        "XGBoost": [
            xgb_metrics["roc_auc"],
            xgb_metrics["accuracy"],
            xgb_metrics["precision"],
            xgb_metrics["recall"],
            xgb_metrics["f1_score"],
        ],
    })
    print(comparison_df.to_string(index=False))

    # Champion model selection based on ROC-AUC
    if xgb_metrics["roc_auc"] >= lr_metrics["roc_auc"]:
        champion_name = "XGBoost"
        champion_pipeline = best_xgb
        champion_metrics = xgb_metrics
        champion_probs = xgb_test_probs
        champion_preds = xgb_test_preds
        champion_params = {k.replace("classifier__", ""): (int(v) if isinstance(v, (np.integer, int)) else (float(v) if isinstance(v, (np.floating, float)) else v)) for k, v in xgb_search.best_params_.items()}
    else:
        champion_name = "Logistic Regression"
        champion_pipeline = best_lr
        champion_metrics = lr_metrics
        champion_probs = lr_test_probs
        champion_preds = lr_test_preds
        champion_params = {k.replace("classifier__", ""): (int(v) if isinstance(v, (np.integer, int)) else (float(v) if isinstance(v, (np.floating, float)) else v)) for k, v in lr_search.best_params_.items()}

    print(f"\nChampion Model Selected: {champion_name} (ROC-AUC: {champion_metrics['roc_auc']:.4f})")

    # Save diagnostic test plots
    plot_confusion_matrix(y_test.values, champion_preds, os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_test.values, champion_probs, os.path.join(REPORTS_DIR, "roc_curve.png"))

    # Save metrics.json
    all_metrics_payload = {
        "champion_model": champion_name,
        "champion_test_metrics": champion_metrics,
        "champion_hyperparameters": champion_params,
        "model_comparison": {
            "logistic_regression": {
                "cv_roc_auc_mean": round(float(lr_search.best_score_), 4),
                "test_metrics": lr_metrics,
                "best_params": {k.replace("classifier__", ""): (int(v) if isinstance(v, (np.integer, int)) else (float(v) if isinstance(v, (np.floating, float)) else v)) for k, v in lr_search.best_params_.items()},
            },
            "xgboost": {
                "cv_roc_auc_mean": round(float(xgb_search.best_score_), 4),
                "test_metrics": xgb_metrics,
                "best_params": {k.replace("classifier__", ""): (int(v) if isinstance(v, (np.integer, int)) else (float(v) if isinstance(v, (np.floating, float)) else v)) for k, v in xgb_search.best_params_.items()},
            },
        },
    }
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics_payload, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # ---------------------------------------------------------
    # STEP 5: SHAP Explanations
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Generating SHAP Model Explanations")
    print("=" * 70)
    preprocessor = champion_pipeline.named_steps["preprocessor"]
    classifier = champion_pipeline.named_steps["classifier"]
    
    # Preprocess test set to match classifier input
    X_test_trans = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    plt.clf()
    plt.close("all")

    if champion_name == "XGBoost":
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(X_test_trans)
        shap_values.feature_names = list(feature_names)
        
        # 1. SHAP Beeswarm summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, features=X_test_trans, feature_names=feature_names, show=False, max_display=15)
        plt.title("SHAP Beeswarm Feature Impact on Churn", pad=15, fontsize=14, fontweight="bold")
        plt.tight_layout()
        shap_summary_path = os.path.join(REPORTS_DIR, "shap_summary.png")
        plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary plot: {shap_summary_path}")

        # 2. SHAP Bar importance plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.title("Top 15 Churn Drivers by Mean |SHAP Value|", pad=15, fontsize=14, fontweight="bold")
        plt.tight_layout()
        shap_bar_path = os.path.join(REPORTS_DIR, "shap_feature_importance.png")
        plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP feature importance plot: {shap_bar_path}")

    else:
        explainer = shap.LinearExplainer(classifier, X_test_trans)
        shap_vals = explainer.shap_values(X_test_trans)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_vals, features=X_test_trans, feature_names=feature_names, show=False, max_display=15)
        plt.title("SHAP Feature Impact on Churn (Logistic Regression)", pad=15, fontsize=14, fontweight="bold")
        plt.tight_layout()
        shap_summary_path = os.path.join(REPORTS_DIR, "shap_summary.png")
        plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary plot: {shap_summary_path}")

    # ---------------------------------------------------------
    # STEP 6: Customer Risk Segmentation
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: Customer Risk Segmentation (Low / Medium / High)")
    print("=" * 70)
    perform_risk_segmentation(df_test, champion_probs, low_thresh=0.30, high_thresh=0.60, output_dir=REPORTS_DIR)

    # ---------------------------------------------------------
    # STEP 7: Persisting Artifacts
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7: Persisting Models and Metadata")
    print("=" * 70)
    model_save_path = os.path.join(MODELS_DIR, "best_churn_model.joblib")
    preprocessor_save_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    
    # Save full pipeline and standalone preprocessor
    joblib.dump(champion_pipeline, model_save_path)
    joblib.dump(preprocessor, preprocessor_save_path)
    print(f"Full pipeline saved to: {model_save_path}")
    print(f"Preprocessor saved to: {preprocessor_save_path}")

    # Save metadata
    metadata = {
        "model_name": champion_name,
        "feature_count": len(feature_names),
        "raw_features": list(X.columns),
        "encoded_features": list(feature_names),
        "best_hyperparameters": champion_params,
        "test_metrics": champion_metrics,
        "class_balance_strategy": "Cost-sensitive class weighting (balanced / scale_pos_weight)",
    }
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_path}")

    return champion_pipeline, all_metrics_payload


if __name__ == "__main__":
    train_and_evaluate_models()
