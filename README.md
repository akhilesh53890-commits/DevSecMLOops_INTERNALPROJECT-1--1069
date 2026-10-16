# Telecom Customer Churn Prediction (Business)
### Internal Project 2 — End-to-End Machine Learning Pipeline & Executive Intelligence Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.51.0-brightgreen.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.0-ff4b4b.svg)](https://streamlit.io/)

---

## 📌 Project Overview & Domain Context

Customer churn poses a major challenge in telecommunications, directly impairing Monthly Recurring Revenue (MRR) and inflating Customer Acquisition Costs (CAC). Retaining existing subscribers is 5 to 7 times more cost-effective than acquiring new ones.

**Telecom Customer Churn Prediction (Internal Project 2)** delivers a production-grade machine learning system designed to:
1. **Identify attrition risks early** using calibrated probabilistic modeling on subscriber demographic, contract, service, and billing patterns.
2. **Stratify subscribers into actionable risk tiers** (Low, Medium, High Risk) with immediate portfolio financial exposure metrics.
3. **Provide Explainable AI (XAI) diagnostics via SHAP**, allowing commercial retention and marketing teams to implement targeted countermeasures rather than generic discount churn campaigns.
4. **Deploy an interactive executive intelligence dashboard** with single-subscriber inference and portfolio risk analytics.

---

## 💾 Dataset Information

- **Dataset Name**: IBM Telco Customer Churn Dataset
- **Public Source Link**: [IBM ICP4D Telco Customer Churn Repository](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) (Mirror: [Treselle Systems Customer Churn](https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv))
- **Records**: 7,043 customer accounts
- **Features**: 20 subscriber attributes (demographics, services, contracts, payments, charges)
- **Target Variable**: `Churn` (`Yes` / `No`, positive churn class: ~26.54%)

### Data Quality & Preprocessing Fixes
- **TotalCharges Whitespace Resolution**: In the raw dataset, 11 customer records had blank whitespace strings (`" "`) in `TotalCharges` corresponding to brand-new accounts with `tenure == 0`. These were converted to numeric type and imputed to `0.0` to preserve domain integrity without dropping new subscriber data.
- **Leakage Prevention**: All transformations, scalers, and encoders are fit strictly within stratified cross-validation folds and training sets.

---

## ⚖️ Class Imbalance Handling Strategy

The dataset exhibits an imbalance of **73.46% retained vs. 26.54% churned** (ratio ~1:2.77). 

### Chosen Strategy: Cost-Sensitive Class Weighting (`scale_pos_weight` & `class_weight='balanced'`)
We evaluated synthetic sampling (**SMOTE**) versus **Cost-Sensitive Class Weighting**. Class weighting was chosen for the following reasons:
1. **Preservation of Categorical Distribution**: SMOTE creates synthetic interpolations between categorical dummy features, generating artificial combinations of services that do not exist in reality (e.g., fractional service bundles).
2. **Probability Calibration**: Class weighting directly alters the loss function gradient:
   $$\text{scale\_pos\_weight} = \frac{\text{Negative Samples}}{\text{Positive Samples}} \approx 2.769$$
   This penalizes false negatives (missed churners) without distorting the underlying feature space geometry, yielding well-calibrated probabilities essential for risk tiering.
3. **Inference Latency & Production Simplicity**: No external resampling transformers need to be serialized in the final production inference pipeline.

---

## ⚙️ Model Tuning & Hyperparameters

Both **Logistic Regression** (linear baseline) and **XGBoost Classifier** (non-linear gradient boosting) were tuned using **5-Fold Stratified Cross-Validation** optimizing for **ROC-AUC**.

### Final Tuned Hyperparameters

| Model | Hyperparameter | Tuned Value | Description |
| :--- | :--- | :--- | :--- |
| **XGBoost (Champion)** | `max_depth` | `4` | Maximum tree depth preventing over-specialization |
| | `learning_rate` | `0.02` | Shrinkage step size ensuring gradual convergence |
| | `n_estimators` | `150` | Number of gradient boosted boosting rounds |
| | `subsample` | `0.70` | Subsample ratio of the training instances |
| | `colsample_bytree` | `0.85` | Subsample ratio of columns when constructing each tree |
| | `min_child_weight` | `1` | Minimum sum of instance weight (hessian) in a child |
| | `gamma` | `0.3` | Minimum loss reduction required to make a further partition |
| | `scale_pos_weight` | `2.769` | Imbalance compensation ratio |
| **Logistic Regression** | `C` | `10.0` | Inverse of regularization strength (L2 penalty) |
| | `solver` | `lbfgs` | Quasi-Newton optimization algorithm |
| | `class_weight` | `'balanced'` | Inversely proportional class frequency weights |

---

## 📊 Results & Model Comparison

Both models were evaluated on an identical **held-out test set (1,409 customers / 20% stratified sample)** that was unseen during training and tuning.

| Metric | Logistic Regression | XGBoost Classifier (Champion) | Business Impact / Advantage |
| :--- | :---: | :---: | :--- |
| **ROC-AUC (Test)** | `0.8408` | **`0.8454`** | Superior ranking ability across discrimination thresholds |
| **5-Fold CV ROC-AUC** | `0.8462` | **`0.8483`** | Highly stable generalization across all validation folds |
| **Accuracy** | `73.74%` | **`75.02%`** | Overall correct classification rate |
| **Precision** | `50.34%` | **`51.88%`** | Minimizes wasted marketing spend on non-churners |
| **Recall (Sensitivity)** | `78.34%` | **`81.02%`** | **Captures 81% of all churners** before they leave |
| **F1-Score** | `0.6130` | **`0.6326`** | Harmonic balance between Precision and Recall |

### Confusion Matrix on Held-Out Test Set (XGBoost)
- **True Negatives (Correctly Retained)**: 754
- **False Positives (Predicted Churn, Did Not Churn)**: 281
- **False Negatives (Missed Churners)**: 71
- **True Positives (Identified Churners)**: 303 (81.02% Detection Rate)

---

## 🎯 Customer Risk Segmentation

Subscribers are segmented into three operational tiers based on predicted churn probabilities:

| Risk Segment | Probability Range | Customer Count | % of Cohort | Avg Predicted Prob | Actual Churn Rate | Monthly Revenue at Stake | Avg Monthly Bill |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🟢 **Low Risk** | `< 30%` | 579 | 41.09% | 13.10% | **4.32%** | $29,748.30 | $51.38 |
| 🟡 **Medium Risk** | `30% - 60%` | 369 | 26.19% | 44.85% | **23.04%** | $25,290.25 | $68.54 |
| 🔴 **High Risk** | `>= 60%` | 461 | 32.72% | 74.92% | **57.27%** | **$35,262.65** | $76.49 |

> 💡 **Commercial Insight**: High-Risk accounts represent **$35,262.65 in monthly revenue** (39.0% of cohort revenue). Retention campaigns focused on this tier address over 57% actual churners while maximizing revenue defense.

---

## 🧠 Key Churn Drivers (SHAP Interpretability)

Analysis with **SHAP TreeExplainer** reveals the top drivers influencing churn:
1. **Contract Type (Month-to-Month)**: Single strongest positive driver of churn. Customers on monthly plans churn at **42.7%** vs **2.8%** on 2-year contracts.
2. **Customer Tenure**: Early tenure (first 12 months) has the largest positive contribution to churn probability.
3. **Internet Service (Fiber Optic)**: High monthly charge load without bundle support correlates strongly with customer dissatisfaction.
4. **Payment Method (Electronic Check)**: Higher churn rate due to payment friction and lack of automated billing commitment.
5. **Lack of Value-Added Services**: Customers lacking `OnlineSecurity` and `TechSupport` have markedly higher churn propensity.

---

## 📁 Project Structure

```
telecom-churn-prediction/
├── .gitignore                     # Git ignore rules for venv, cache, OS files
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Pinned project dependencies
├── app.py                         # Streamlit multi-page executive dashboard
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Downloaded raw dataset
│   └── cleaned_telco_churn.csv   # Cleaned dataset with numeric TotalCharges
├── models/
│   ├── best_churn_model.joblib   # Serialized production pipeline (XGBoost)
│   ├── preprocessor.joblib       # Standalone scikit-learn ColumnTransformer
│   └── model_metadata.json       # Hyperparameters, feature schema, test metrics
├── reports/
│   ├── churn_by_contract.png     # EDA: Churn rate by contract type
│   ├── churn_by_tenure.png       # EDA: Tenure density distribution
│   ├── churn_by_monthly_charges.png # EDA: Monthly charges distribution
│   ├── correlation_heatmap.png   # EDA: Feature correlation matrix
│   ├── confusion_matrix.png      # Test set confusion matrix
│   ├── roc_curve.png             # Test set ROC-AUC curve
│   ├── shap_summary.png          # SHAP beeswarm plot
│   ├── shap_feature_importance.png # SHAP mean absolute value bar plot
│   ├── risk_segments.csv         # Risk cohort summary table (CSV)
│   ├── risk_segments.json        # Risk cohort summary data (JSON)
│   ├── eda_summary.json          # Key descriptive statistics
│   └── metrics.json              # Full benchmark and validation metrics
└── src/
    ├── __init__.py               # Package initializer
    ├── data_loader.py            # Public data ingestion and validation
    ├── eda.py                    # EDA visualization generator
    ├── preprocessor.py           # ColumnTransformer preprocessing pipeline
    ├── train.py                  # Stratified CV, tuning, SHAP & export
    └── evaluate.py               # Metrics, confusion matrix, risk segmentation
```

---

## 🚀 Installation & Quick Start

### 1. Prerequisites & Virtual Environment
Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/krutharyan22/telecom-churn-prediction.git
cd telecom-churn-prediction

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis (EDA)
Downloads the raw dataset, resolves anomalies, and generates plots in `reports/`:
```bash
python -m src.eda
```

### 3. Run Training & Evaluation Pipeline
Runs 5-fold Stratified Cross-Validation, hyperparameter tuning, test set evaluation, SHAP analysis, and exports all model artifacts:
```bash
python -m src.train
```

### 4. Launch the Streamlit Intelligence Dashboard
Launch the interactive web application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Dashboard Features

1. **🔮 Customer Churn Predictor**:
   - Single-customer interactive form with demographic, service, and billing inputs.
   - Quick presets: "Load High-Risk Sample" & "Load Loyal Sample".
   - Real-time churn probability gauge and risk tier classification badge (🟢 Low / 🟡 Medium / 🔴 High).
   - Prescriptive retention playbooks tailored to the customer's specific risk drivers.
2. **📊 Risk Segments & Portfolio Analytics**:
   - Portfolio exposure overview, subscriber count distribution, and total monthly revenue at stake.
3. **🧠 Top Churn Drivers & SHAP**:
   - High-resolution SHAP summary beeswarm and feature importance charts.
   - Diagnostic confusion matrix and ROC-AUC curve.
4. **📈 Exploratory Insights (EDA)**:
   - Visual analytics on contract types, tenure dynamics, monthly charge distributions, and correlation heatmaps.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
