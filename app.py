"""
Telecom Customer Churn Prediction & Risk Analytics Dashboard
Internal Project 2 - Streamlit Web Application
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Telecom Churn Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern executive aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-med {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .recommendation-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    model_path = os.path.join("models", "best_churn_model.joblib")
    metrics_path = os.path.join("reports", "metrics.json")
    segments_path = os.path.join("reports", "risk_segments.json")
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
    segments = []
    if os.path.exists(segments_path):
        with open(segments_path, "r") as f:
            segments = json.load(f)

    return model, metrics, segments


model, metrics_data, segments_data = load_model_artifacts()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/satellite-sending-signal.png", width=64)
st.sidebar.title("Telecom Intelligence")
st.sidebar.caption("Machine Learning Churn Prediction Platform")

app_mode = st.sidebar.radio(
    "Navigation Menu",
    [
        "🔮 Customer Churn Predictor",
        "📊 Risk Segments & Portfolio",
        "🧠 Top Churn Drivers & SHAP",
        "📈 Exploratory Insights (EDA)",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "**Project Context**:\n"
    "• Dataset: IBM Telco Customer Churn\n"
    "• Model: Tuned XGBoost Classifier\n"
    "• Strategy: Cost-Sensitive Loss Weighting\n"
    "• Internal Project 2"
)


# ==============================================================================
# PAGE 1: Single Customer Churn Predictor
# ==============================================================================
if app_mode == "🔮 Customer Churn Predictor":
    st.markdown('<div class="main-header">🔮 Customer Churn Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate individual subscriber churn likelihood, risk classification, and prescriptive retention actions.</div>', unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model artifact not found at `models/best_churn_model.joblib`. Please run `python -m src.train` first.")
        st.stop()

    # Presets row
    preset_cols = st.columns([1, 1, 3])
    preset_chosen = preset_cols[0].button("⚡ Load High-Risk Sample")
    preset_loyal = preset_cols[1].button("🛡️ Load Loyal Sample")

    # Default values or presets
    if preset_chosen:
        default_gender = "Female"
        default_senior = 1
        default_partner = "No"
        default_dependents = "No"
        default_tenure = 2
        default_contract = "Month-to-month"
        default_paperless = "Yes"
        default_payment = "Electronic check"
        default_phone = "Yes"
        default_lines = "No"
        default_internet = "Fiber optic"
        default_security = "No"
        default_backup = "No"
        default_protection = "No"
        default_tech = "No"
        default_tv = "Yes"
        default_movies = "Yes"
        default_monthly = 89.85
        default_total = 179.70
    elif preset_loyal:
        default_gender = "Male"
        default_senior = 0
        default_partner = "Yes"
        default_dependents = "Yes"
        default_tenure = 60
        default_contract = "Two year"
        default_paperless = "No"
        default_payment = "Credit card (automatic)"
        default_phone = "Yes"
        default_lines = "Yes"
        default_internet = "DSL"
        default_security = "Yes"
        default_backup = "Yes"
        default_protection = "Yes"
        default_tech = "Yes"
        default_tv = "No"
        default_movies = "No"
        default_monthly = 55.20
        default_total = 3312.00
    else:
        default_gender = "Female"
        default_senior = 0
        default_partner = "No"
        default_dependents = "No"
        default_tenure = 9
        default_contract = "Month-to-month"
        default_paperless = "Yes"
        default_payment = "Electronic check"
        default_phone = "Yes"
        default_lines = "No"
        default_internet = "Fiber optic"
        default_security = "No"
        default_backup = "No"
        default_protection = "No"
        default_tech = "No"
        default_tv = "No"
        default_movies = "No"
        default_monthly = 70.35
        default_total = 633.15

    with st.form("churn_prediction_form"):
        tab1, tab2, tab3 = st.tabs(["👤 Customer Demographics & Account", "🌐 Subscribed Services", "💳 Billing & Financials"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            gender = c1.selectbox("Gender", ["Female", "Male"], index=0 if default_gender == "Female" else 1)
            senior = c2.selectbox("Senior Citizen", [0, 1], index=default_senior, format_func=lambda x: "Yes (Senior)" if x == 1 else "No")
            partner = c3.selectbox("Partner", ["Yes", "No"], index=0 if default_partner == "Yes" else 1)

            c4, c5, c6 = st.columns(3)
            dependents = c4.selectbox("Dependents", ["Yes", "No"], index=0 if default_dependents == "Yes" else 1)
            tenure = c5.slider("Customer Tenure (Months)", min_value=0, max_value=72, value=default_tenure, help="Number of months customer has been with the provider.")
            contract = c6.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(default_contract))

        with tab2:
            s1, s2, s3 = st.columns(3)
            phone = s1.selectbox("Phone Service", ["Yes", "No"], index=0 if default_phone == "Yes" else 1)
            lines_options = ["No phone service", "No", "Yes"] if phone == "No" else ["No", "Yes"]
            multiple_lines = s2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(default_lines))
            internet = s3.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(default_internet))

            s4, s5, s6 = st.columns(3)
            security = s4.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_security))
            backup = s5.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_backup))
            protection = s6.selectbox("Device Protection Plan", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_protection))

            s7, s8, _ = st.columns(3)
            tech = s7.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_tech))
            tv = s8.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_tv))
            movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_movies))

        with tab3:
            b1, b2, b3 = st.columns(3)
            paperless = b1.selectbox("Paperless Billing", ["Yes", "No"], index=0 if default_paperless == "Yes" else 1)
            payment = b2.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(default_payment),
            )
            monthly_charges = b3.number_input("Monthly Charges ($ USD)", min_value=15.0, max_value=150.0, value=float(default_monthly), step=0.5)

            # Auto calculate sensible total charges if desired
            est_total = float(round(tenure * monthly_charges, 2)) if tenure > 0 else 0.0
            total_charges = st.number_input("Total Cumulative Charges ($ USD)", min_value=0.0, max_value=10000.0, value=est_total if not preset_chosen else float(default_total), step=10.0)

        submit_btn = st.form_submit_button("⚡ Predict Churn Risk", use_container_width=True)

    # Perform Prediction
    if submit_btn or preset_chosen or preset_loyal:
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": str(senior),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }])

        prob = float(model.predict_proba(input_data)[0, 1])
        pct_prob = prob * 100

        st.subheader("Assessment Results")
        res_col1, res_col2, res_col3 = st.columns([1.2, 1.2, 1.6])

        with res_col1:
            st.metric("Predicted Churn Probability", f"{pct_prob:.1f}%")
            st.progress(prob)

        with res_col2:
            st.write("**Risk Classification:**")
            if prob < 0.30:
                st.markdown('<span class="badge-low">🟢 LOW CHURN RISK</span>', unsafe_allow_html=True)
                risk_tier = "Low"
            elif prob < 0.60:
                st.markdown('<span class="badge-med">🟡 MEDIUM CHURN RISK</span>', unsafe_allow_html=True)
                risk_tier = "Medium"
            else:
                st.markdown('<span class="badge-high">🔴 HIGH CHURN RISK</span>', unsafe_allow_html=True)
                risk_tier = "High"

        with res_col3:
            st.write("**Monthly Value at Stake:**")
            st.write(f"**${monthly_charges:.2f} / month** (${monthly_charges * 12:.2f} Annualized)")

        # Prescription logic based on SHAP feature driver rules
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.write("### 📋 Automated Retention Actions & Playbook:")
        actions = []
        if contract == "Month-to-month":
            actions.append("• **Contract Incentive**: Customer is on a Month-to-Month plan (top risk driver). Offer a 15% discount for upgrading to a 1-year or 2-year commitment.")
        if tenure < 12:
            actions.append(f"• **Early Lifecycle Onboarding**: Customer is in the first {tenure} months (critical attrition window). Assign proactive customer success manager check-in.")
        if internet == "Fiber optic" and tech == "No":
            actions.append("• **Support Package**: Customer has high-speed Fiber Optic without Tech Support. Offer a 3-month complimentary Tech Support & Device Security add-on.")
        if payment == "Electronic check":
            actions.append("• **Billing Migration**: Electronic check is associated with higher default and dissatisfaction. Offer a $10 one-time bill credit to set up automatic bank/card debit.")
        if len(actions) == 0:
            actions.append("• **Loyalty Engagement**: Customer profile is healthy and stable. Enroll in the VIP loyalty rewards tier to maintain lifetime engagement.")

        for act in actions:
            st.write(act)
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: Risk Segments & Portfolio Analytics
# ==============================================================================
elif app_mode == "📊 Risk Segments & Portfolio":
    st.markdown('<div class="main-header">📊 Customer Risk Segments & Portfolio Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Stratified breakdown of subscriber population into risk cohorts with revenue exposure metrics.</div>', unsafe_allow_html=True)

    if segments_data:
        seg_df = pd.DataFrame(segments_data)
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_eval = int(seg_df["Customer_Count"].sum())
        high_risk_row = seg_df[seg_df["Risk_Segment"] == "High Risk"].iloc[0]
        total_high_risk = int(high_risk_row["Customer_Count"])
        high_risk_rev = float(high_risk_row["Total_Monthly_Revenue_USD"])
        high_risk_churn_rate = float(high_risk_row["Actual_Churn_Rate"])

        kpi1.metric("Evaluated Test Cohort", f"{total_eval:,} users")
        kpi2.metric("High Risk Population", f"{total_high_risk:,} ({high_risk_row['Percentage_of_Total']}%)")
        kpi3.metric("High Risk Monthly Revenue", f"${high_risk_rev:,.2f}")
        kpi4.metric("High Risk Actual Churn Rate", f"{high_risk_churn_rate:.1f}%")

        st.divider()

        col_left, col_right = st.columns([1.3, 1])
        with col_left:
            st.subheader("Cohort Segmentation Breakdown")
            display_df = seg_df[[
                "Risk_Segment",
                "Probability_Range",
                "Customer_Count",
                "Percentage_of_Total",
                "Average_Predicted_Prob",
                "Actual_Churn_Rate",
                "Total_Monthly_Revenue_USD",
            ]].copy()
            display_df.columns = [
                "Segment",
                "Prob Range",
                "Customers",
                "% of Total",
                "Avg Pred Prob (%)",
                "Actual Churn (%)",
                "Monthly Revenue ($)",
            ]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        with col_right:
            st.subheader("Distribution by Risk Tier")
            chart_df = seg_df.set_index("Risk_Segment")[["Customer_Count"]]
            st.bar_chart(chart_df, color="#1e3a8a")

        st.info(
            "💡 **Key Portfolio Finding**: While High-Risk subscribers make up ~32.7% of the customer base, "
            "they generate over $35,260 in monthly revenue (~39% of evaluated revenue). Intervening early in this cohort yields the highest return on retention spend."
        )
    else:
        st.warning("Risk segmentation artifact not found. Please train the model first.")


# ==============================================================================
# PAGE 3: Top Churn Drivers & SHAP
# ==============================================================================
elif app_mode == "🧠 Top Churn Drivers & SHAP":
    st.markdown('<div class="main-header">🧠 Model Interpretability & Churn Drivers</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explainable AI (XAI) using SHAP (SHapley Additive exPlanations) values to decode decision boundaries.</div>', unsafe_allow_html=True)

    shap_col1, shap_col2 = st.columns(2)
    with shap_col1:
        st.subheader("Global Feature Importance (Mean |SHAP|)")
        if os.path.exists("reports/shap_feature_importance.png"):
            st.image("reports/shap_feature_importance.png", use_container_width=True)
        else:
            st.info("SHAP feature importance plot will appear once `python -m src.train` completes.")

    with shap_col2:
        st.subheader("SHAP Beeswarm Feature Impact")
        if os.path.exists("reports/shap_summary.png"):
            st.image("reports/shap_summary.png", use_container_width=True)
        else:
            st.info("SHAP summary plot will appear once `python -m src.train` completes.")

    st.divider()

    st.subheader("🏆 Model Comparison & Verification")
    if metrics_data:
        champ_name = metrics_data.get("champion_model", "XGBoost")
        champ_metrics = metrics_data.get("champion_test_metrics", {})
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ROC-AUC", f"{champ_metrics.get('roc_auc', 0):.4f}")
        m2.metric("Accuracy", f"{champ_metrics.get('accuracy', 0):.2%}")
        m3.metric("Precision", f"{champ_metrics.get('precision', 0):.2%}")
        m4.metric("Recall (Sensitivity)", f"{champ_metrics.get('recall', 0):.2%}")
        m5.metric("F1-Score", f"{champ_metrics.get('f1_score', 0):.4f}")

        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.write("#### Confusion Matrix")
            if os.path.exists("reports/confusion_matrix.png"):
                st.image("reports/confusion_matrix.png", use_container_width=True)
        with diag_col2:
            st.write("#### ROC-AUC Curve")
            if os.path.exists("reports/roc_curve.png"):
                st.image("reports/roc_curve.png", use_container_width=True)


# ==============================================================================
# PAGE 4: Exploratory Insights (EDA)
# ==============================================================================
elif app_mode == "📈 Exploratory Insights (EDA)":
    st.markdown('<div class="main-header">📈 Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical validation and key patterns discovered in the IBM Telco subscriber base.</div>', unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        st.subheader("1. Churn by Contract Type")
        if os.path.exists("reports/churn_by_contract.png"):
            st.image("reports/churn_by_contract.png", use_container_width=True)
            st.caption("Month-to-month contracts exhibit an alarming 42.7% churn rate, compared to 2.8% for two-year contracts.")

    with e2:
        st.subheader("2. Tenure Distribution by Churn Status")
        if os.path.exists("reports/churn_by_tenure.png"):
            st.image("reports/churn_by_tenure.png", use_container_width=True)
            st.caption("A sharp churn spike occurs within the first 0-12 months before stabilizing over extended tenure.")

    e3, e4 = st.columns(2)
    with e3:
        st.subheader("3. Impact of Monthly Charges")
        if os.path.exists("reports/churn_by_monthly_charges.png"):
            st.image("reports/churn_by_monthly_charges.png", use_container_width=True)
            st.caption("Churned subscribers have a substantially higher median monthly fee ($79.65 vs $64.42).")

    with e4:
        st.subheader("4. Feature Correlations Heatmap")
        if os.path.exists("reports/correlation_heatmap.png"):
            st.image("reports/correlation_heatmap.png", use_container_width=True)
            st.caption("Contract type, tenure length, online security, and electronic check methods show strongest correlation with attrition.")
