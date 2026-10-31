import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from utils.preprocessing import load_transaction_data, create_sample_data
from utils.risk_engine import (
    calculate_risk_score,
    get_risk_level,
    get_risk_factors,
    get_recommendations,
)


# Load trained fraud model
MODEL_PATH = "model/fraud_model.pkl"

if os.path.exists(MODEL_PATH):
    fraud_model = joblib.load(MODEL_PATH)
else:
    fraud_model = None


def predict_fraud(transaction_type, amount, avg_amount):
    """Predict fraud probability using the trained PaySim model."""

    if fraud_model is None:
        return None, None

    # Approximate PaySim-style transaction features
    old_balance = max(avg_amount * 2, amount * 1.5)
    new_balance = max(0, old_balance - amount)

    old_dest_balance = avg_amount * 5
    new_dest_balance = old_dest_balance + amount

    balance_change = old_balance - new_balance

    amount_ratio = amount / (old_balance + 1)

    destination_change = new_dest_balance - old_dest_balance

    transaction = pd.DataFrame([{
        "step": 1,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": old_balance,
        "newbalanceOrig": new_balance,
        "oldbalanceDest": old_dest_balance,
        "newbalanceDest": new_dest_balance,
        "balance_change": balance_change,
        "amount_to_balance_ratio": amount_ratio,
        "destination_balance_change": destination_change
    }])

    probability = fraud_model.predict_proba(transaction)[0][1]

    prediction = fraud_model.predict(transaction)[0]

    return probability, prediction

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Fraud Risk Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
}

.main-title {
    font-size: 38px;
    font-weight: 700;
}

.subtitle {
    color: #888;
    font-size: 17px;
}

.risk-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #444;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🛡️ Bank Fraud Risk Assessment Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze banking activity, estimate fraud risk and identify preventive actions.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

DATA_PATH = "data/paysim.csv"

if os.path.exists(DATA_PATH):
    transaction_data = load_transaction_data(DATA_PATH)
else:
    transaction_data = create_sample_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("Customer Information")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Student",
            "Employee",
            "Business Owner",
            "Self Employed",
            "Retired",
            "Other"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income (₹)",
        min_value=0.0,
        value=50000.0,
        step=5000.0
    )

    account_age = st.number_input(
        "Account Age (Years)",
        min_value=0.0,
        max_value=50.0,
        value=3.0,
        step=0.5
    )

    digital_banking = st.selectbox(
        "Digital Banking Usage",
        ["Low", "Medium", "High"]
    )

    st.divider()

    st.header("Security Information")

    failed_logins = st.number_input(
        "Failed Login Attempts",
        min_value=0,
        max_value=20,
        value=0
    )

    two_factor = st.selectbox(
        "Two-Factor Authentication",
        ["Enabled", "Disabled"]
    )

    password_changed = st.selectbox(
        "Recently Changed Password?",
        ["No", "Yes"]
    )

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Risk Assessment",
        "Transaction Analysis",
        "Risk Analysis",
        "Prevention"
    ]
)

# ==================================================
# TAB 1 - RISK ASSESSMENT
# ==================================================

with tab1:

    st.subheader("Transaction Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "UPI",
                "Card",
                "Bank Transfer",
                "Cash Withdrawal",
                "Online Payment"
            ]
        )


        #Paysim transaction type mapping 
        #as model and dashbaord takes different values name
        paysim_type_map = {
            "UPI": "PAYMENT",
        "Card": "PAYMENT",
        "Bank Transfer": "TRANSFER",
        "Cash Withdrawal": "CASH_OUT",
        "Online Payment": "PAYMENT"
        }

        paysim_type = paysim_type_map[transaction_type]
        

        unusual_time = st.selectbox(
            "Transaction at Unusual Time?",
            ["No", "Yes"]
        )

    with col2:

        new_device = st.selectbox(
            "New Device?",
            ["No", "Yes"]
        )

        new_location = st.selectbox(
            "New Location?",
            ["No", "Yes"]
        )

        new_beneficiary = st.selectbox(
            "New Beneficiary?",
            ["No", "Yes"]
        )

    with col3:

        ip_changed = st.selectbox(
            "IP Address Changed?",
            ["No", "Yes"]
        )

        transaction_frequency = st.selectbox(
            "Transaction Frequency",
            ["Low", "Normal", "High"]
        )

        normal_average_transaction = st.number_input(
            "Normal Average Transaction (₹)",
            min_value=0.0,
            value=3000.0,
            step=500.0
        )

        fraud_probability, fraud_prediction = predict_fraud(
            paysim_type,
            amount,
            normal_average_transaction
        )

    st.divider()

    # ------------------------------------------
    # CALCULATE RISK
    # ------------------------------------------

    user_inputs = {
        "age": age,
        "monthly_income": monthly_income,
        "account_age": account_age,
        "failed_logins": failed_logins,
        "two_factor": two_factor,
        "amount": amount,
        "normal_average_transaction": normal_average_transaction,
        "new_device": new_device,
        "new_location": new_location,
        "new_beneficiary": new_beneficiary,
        "ip_changed": ip_changed,
        "unusual_time": unusual_time,
        "transaction_frequency": transaction_frequency
    }

    risk_score = calculate_risk_score(user_inputs)

    risk_level = get_risk_level(risk_score)

    risk_factors = get_risk_factors(user_inputs)

    # ML model returns probability between 0 and 1.
    # Convert it to a percentage for the dashboard.
    if fraud_probability is not None:
        ml_risk = fraud_probability * 100
    else:
        ml_risk = 0

    # Combine transaction fraud probability with the
    # rule-based account/security risk.
    combined_risk = (
        ml_risk * 0.60 +
        risk_score * 0.40
    )

    combined_risk = min(100, combined_risk)

    if combined_risk <= 20:
        overall_level = "Very Low"
    elif combined_risk <= 40:
        overall_level = "Low"
    elif combined_risk <= 60:
        overall_level = "Moderate"
    elif combined_risk <= 80:
        overall_level = "High"
    else:
        overall_level = "Critical"

    st.subheader("Overall Risk Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "ML Fraud Risk",
            f"{ml_risk:.1f}%"
        )

    with col2:
        st.metric(
            "Security Risk",
            f"{risk_score}/100"
        )

    with col3:
        st.metric(
            "Overall Risk",
            f"{combined_risk:.1f}%"
        )

    st.warning(f"Overall Risk Level: **{overall_level}**")

    if fraud_probability is not None:

        st.subheader("ML Fraud Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Fraud Probability",
                f"{ml_risk:.2f}%"
            )

        with col2:
            if fraud_prediction == 1:
                st.error("Potential Fraud Detected")
            else:
                st.success("Transaction Appears Normal")

    # ------------------------------------------
    # TOP METRICS
    # ------------------------------------------

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Risk Score",
        f"{risk_score}/100"
    )

    m2.metric(
        "Risk Level",
        risk_level
    )

    m3.metric(
        "ML Fraud Risk",
        f"{ml_risk:.1f}%"
    )

    m4.metric(
        "Detected Risk Factors",
        len(risk_factors)
    )

    st.divider()

    # ------------------------------------------
    # GAUGE
    # ------------------------------------------

    left, right = st.columns([1.3, 1])

    with left:

        st.subheader("Overall Risk Score")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={"text": "Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ff4b4b"},
                    "steps": [
                        {"range": [0, 20], "color": "#d4edda"},
                        {"range": [20, 40], "color": "#fff3cd"},
                        {"range": [40, 60], "color": "#ffeeba"},
                        {"range": [60, 80], "color": "#f8d7da"},
                        {"range": [80, 100], "color": "#dc3545"}
                    ]
                }
            )
        )

        fig.update_layout(height=350)

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # ------------------------------------------
    # RISK FACTORS
    # ------------------------------------------

    with right:

        st.subheader("Detected Risk Factors")

        if risk_factors:

            for factor in risk_factors:
                st.warning(factor)

        else:
            st.success(
                "No major suspicious factors were detected."
            )

    # ------------------------------------------
    # RESULT MESSAGE
    # ------------------------------------------

    st.subheader("Assessment")

    if risk_level == "Very Low":
        st.success(
            "The transaction currently appears to have very low risk."
        )

    elif risk_level == "Low":
        st.info(
            "The transaction has a low level of suspicious activity."
        )

    elif risk_level == "Moderate":
        st.warning(
            "Moderate risk detected. Review the transaction before proceeding."
        )

    elif risk_level == "High":
        st.error(
            "High risk detected. Additional verification is recommended."
        )

    else:
        st.error(
            "Critical risk detected. The transaction should be reviewed immediately."
        )

# ==================================================
# TAB 2 - TRANSACTION ANALYSIS
# ==================================================

with tab2:

    st.subheader("Transaction Activity")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Transactions",
        len(transaction_data)
    )

    col2.metric(
        "Average Amount",
        f"₹{transaction_data['amount'].mean():,.2f}"
    )

    col3.metric(
        "Maximum Amount",
        f"₹{transaction_data['amount'].max():,.2f}"
    )

    st.divider()

    # ------------------------------------------
    # TRANSACTION TREND
    # ------------------------------------------

    st.subheader("Transaction Amount Trend")

    trend_fig = px.line(
        transaction_data,
        x="transaction_id",
        y="amount",
        markers=True,
        labels={
            "transaction_id": "Transaction",
            "amount": "Amount (₹)"
        }
    )

    st.plotly_chart(
        trend_fig,
        width="stretch"
    )

    # ------------------------------------------
    # CHARTS
    # ------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Transaction Types")

        type_counts = (
            transaction_data["type"]
            .value_counts()
            .reset_index()
        )

        type_counts.columns = [
            "Transaction Type",
            "Count"
        ]

        type_fig = px.pie(
            type_counts,
            names="Transaction Type",
            values="Count"
        )

        st.plotly_chart(
            type_fig,
            width="stretch"
        )

    with c2:

        st.subheader("Transaction Amount by Type")

        amount_fig = px.bar(
            transaction_data,
            x="type",
            y="amount",
            labels={
                "type": "Transaction Type",
                "amount": "Amount (₹)"
            }
        )

        st.plotly_chart(
            amount_fig,
            width="stretch"
        )

    # ------------------------------------------
    # DATA TABLE
    # ------------------------------------------

    st.subheader("Recent Transactions")

    st.dataframe(
        transaction_data,
        width="stretch",
        hide_index=True
    )

# ==================================================
# TAB 3 - RISK ANALYSIS
# ==================================================

with tab3:

    st.subheader("Risk Factor Analysis")

    if risk_factors:

        risk_dataframe = pd.DataFrame({
            "Risk Factor": risk_factors,
            "Contribution": [
                max(10, 35 - (i * 4))
                for i in range(len(risk_factors))
            ]
        })

        fig = px.bar(
            risk_dataframe,
            x="Contribution",
            y="Risk Factor",
            orientation="h",
            labels={
                "Contribution": "Risk Contribution",
                "Risk Factor": "Factor"
            }
        )

        fig.update_layout(
            yaxis={
                "categoryorder": "total ascending"
            }
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    else:

        st.success(
            "No major risk factors were found."
        )

    st.divider()

    # ------------------------------------------
    # POSSIBLE THREATS
    # ------------------------------------------

    st.subheader("Possible Threat Assessment")

    account_takeover = min(
        risk_score + failed_logins * 5,
        100
    )

    phishing = min(
        int(risk_score * 0.75),
        100
    )

    payment_fraud = min(
        int(risk_score * 0.65),
        100
    )

    threat_data = pd.DataFrame({
        "Threat": [
            "Account Takeover",
            "Phishing",
            "Payment Fraud"
        ],
        "Probability": [
            account_takeover,
            phishing,
            payment_fraud
        ]
    })

    threat_fig = px.bar(
        threat_data,
        x="Threat",
        y="Probability",
        range_y=[0, 100],
        labels={
            "Probability": "Risk Estimate (%)"
        }
    )

    st.plotly_chart(
        threat_fig,
        width="stretch"
    )

    st.caption(
        "These threat values are heuristic security estimates. "
        "The ML model separately predicts transaction fraud probability."
    )

# ==================================================
# TAB 4 - PREVENTION
# ==================================================

with tab4:

    st.subheader("Recommended Precautions")

    recommendations = get_recommendations(
        risk_score,
        user_inputs
    )

    for number, recommendation in enumerate(
        recommendations,
        start=1
    ):
        st.write(
            f"**{number}. {recommendation}**"
        )

    st.divider()

    st.subheader("Basic Banking Security Guidelines")

    st.markdown("""
- Never share your **OTP, PIN, CVV or banking password**.
- Verify a beneficiary before transferring money.
- Use two-factor authentication whenever available.
- Avoid banking through unknown or public Wi-Fi networks.
- Review bank transaction notifications regularly.
- Use a unique password for your banking account.
- Immediately contact your bank if an unauthorized transaction occurs.
- Keep your banking application and operating system updated.
- Avoid clicking banking links received from unknown senders.
- Set reasonable transaction limits.
""")

    st.info(
        "This dashboard is intended for academic risk analysis and "
        "should not be treated as a replacement for a bank's fraud detection system."
    )