# Core Libraries
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==============================
# Load Model and Scaler
# ==============================
BASE_DIR = Path(__file__).parent
model = tf.keras.models.load_model(BASE_DIR / 'models' / 'churn_model.h5')
scaler = joblib.load(BASE_DIR / 'models' / 'scaler.pkl')

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ==============================
# Custom CSS
# ==============================
st.markdown("""
    <style>
    .main-header {
        font-size: 48px !important;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 10px;
    }
    .sub-header {
        font-size: 28px !important;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .risk-high {
        background-color: #ffe0e0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid red;
    }
    .risk-low {
        background-color: #e0ffe0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid green;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Header
# ==============================
st.markdown('<p class="main-header">📊 Customer Churn Prediction</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict whether a telecom customer will churn using Artificial Neural Network</p>',
            unsafe_allow_html=True)

st.divider()

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.image("https://img.icons8.com/color/96/000000/customer.png",
                 width=80)
st.sidebar.header("📋 Customer Details")
st.sidebar.write("Fill in customer information:")

st.sidebar.subheader("📌 Account Information")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0, 120, 50)
total_charges = st.sidebar.slider("Total Charges ($)", 0, 9000, 500)
contract = st.sidebar.selectbox("Contract Type",
            ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)",
             "Credit card (automatic)"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])

st.sidebar.subheader("👤 Personal Information")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

st.sidebar.subheader("📡 Services")
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines",
            ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service",
            ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security",
            ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup",
            ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection",
            ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support",
            ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV",
            ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies",
            ["No", "Yes", "No internet service"])

# ==============================
# Predict Button
# ==============================
predict_btn = st.sidebar.button("🔍 Predict Churn", use_container_width=True)

# ==============================
# Model Performance Section
# ==============================
st.subheader("🏆 Model Performance")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="AUC ROC Score", value="0.857")
with col2:
    st.metric(label="Recall", value="76.4%")
with col3:
    st.metric(label="F1 Score", value="0.639")
with col4:
    st.metric(label="Accuracy", value="77%")

st.divider()

# ==============================
# Prediction Logic
# ==============================
if predict_btn:

    # Encode inputs
    gender_val = 1 if gender == "Male" else 0
    senior_val = 1 if senior_citizen == "Yes" else 0
    partner_val = 1 if partner == "Yes" else 0
    dependents_val = 1 if dependents == "Yes" else 0
    phone_val = 1 if phone_service == "Yes" else 0
    paperless_val = 1 if paperless_billing == "Yes" else 0

    multiple_lines_no_phone = 1 if multiple_lines == "No phone service" else 0
    multiple_lines_yes = 1 if multiple_lines == "Yes" else 0

    internet_fiber = 1 if internet_service == "Fiber optic" else 0
    internet_no = 1 if internet_service == "No" else 0

    online_sec_no_internet = 1 if online_security == "No internet service" else 0
    online_sec_yes = 1 if online_security == "Yes" else 0

    online_bac_no_internet = 1 if online_backup == "No internet service" else 0
    online_bac_yes = 1 if online_backup == "Yes" else 0

    device_no_internet = 1 if device_protection == "No internet service" else 0
    device_yes = 1 if device_protection == "Yes" else 0

    tech_no_internet = 1 if tech_support == "No internet service" else 0
    tech_yes = 1 if tech_support == "Yes" else 0

    streaming_tv_no_internet = 1 if streaming_tv == "No internet service" else 0
    streaming_tv_yes = 1 if streaming_tv == "Yes" else 0

    streaming_mov_no_internet = 1 if streaming_movies == "No internet service" else 0
    streaming_mov_yes = 1 if streaming_movies == "Yes" else 0

    contract_one_year = 1 if contract == "One year" else 0
    contract_two_year = 1 if contract == "Two year" else 0

    payment_mailed = 1 if payment_method == "Mailed check" else 0
    payment_bank = 1 if payment_method == "Bank transfer (automatic)" else 0
    payment_credit = 1 if payment_method == "Credit card (automatic)" else 0

    # Create input array
    input_data = np.array([[
        gender_val, senior_val, partner_val, dependents_val,
        tenure, phone_val, paperless_val, monthly_charges,
        total_charges,
        multiple_lines_no_phone, multiple_lines_yes,
        internet_fiber, internet_no,
        online_sec_no_internet, online_sec_yes,
        online_bac_no_internet, online_bac_yes,
        device_no_internet, device_yes,
        tech_no_internet, tech_yes,
        streaming_tv_no_internet, streaming_tv_yes,
        streaming_mov_no_internet, streaming_mov_yes,
        contract_one_year, contract_two_year,
        payment_mailed, payment_bank, payment_credit
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    probability = float(prediction[0][0])

    # ==============================
    # Results Section
    # ==============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Prediction Result")

        if probability >= 0.3:
            st.markdown(f"""
                <div class="risk-high">
                <h2>🔴 HIGH CHURN RISK</h2>
                <h3>Churn Probability: {probability*100:.1f}%</h3>
                <p>This customer is likely to cancel their subscription.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="risk-low">
                <h2>🟢 LOW CHURN RISK</h2>
                <h3>Churn Probability: {probability*100:.1f}%</h3>
                <p>This customer is likely to stay.</p>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Risk Probability Gauge")
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['green' if probability < 0.3 else 'red']
        ax.barh(['Churn Risk'], [probability * 100],
                color=colors, height=0.4)
        ax.barh(['Churn Risk'], [100 - probability * 100],
                left=[probability * 100],
                color='lightgrey', height=0.4)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.axvline(x=30, color='orange',
                   linestyle='--', label='Threshold 30%')
        ax.legend()
        ax.set_title(f"Churn Probability: {probability*100:.1f}%")
        st.pyplot(fig)

    st.divider()

    # ==============================
    # Customer Profile
    # ==============================
    st.subheader("👤 Customer Profile Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Account Details**")
        st.write(f"• Tenure: {tenure} months")
        st.write(f"• Contract: {contract}")
        st.write(f"• Monthly Charges: ${monthly_charges}")
        st.write(f"• Total Charges: ${total_charges}")

    with col2:
        st.write("**Personal Details**")
        st.write(f"• Gender: {gender}")
        st.write(f"• Senior Citizen: {senior_citizen}")
        st.write(f"• Partner: {partner}")
        st.write(f"• Dependents: {dependents}")

    with col3:
        st.write("**Services**")
        st.write(f"• Internet: {internet_service}")
        st.write(f"• Tech Support: {tech_support}")
        st.write(f"• Streaming TV: {streaming_tv}")
        st.write(f"• Payment: {payment_method}")

    st.divider()

    # ==============================
    # Retention Suggestions
    # ==============================
    st.subheader("💡 Retention Recommendations")

    if probability >= 0.3:
        recommendations = []

        if contract == "Month-to-month":
            recommendations.append(
                "📋 Offer discounted yearly or two-year contract")
        if monthly_charges > 70:
            recommendations.append(
                "💰 Review pricing — consider loyalty discount")
        if tenure < 12:
            recommendations.append(
                "🎁 Offer new customer loyalty bonus")
        if payment_method == "Electronic check":
            recommendations.append(
                "💳 Encourage switch to automatic payment")
        if tech_support == "No":
            recommendations.append(
                "🛠 Offer free tech support trial")
        if online_security == "No":
            recommendations.append(
                "🔒 Offer free online security package")

        if recommendations:
            for rec in recommendations:
                st.warning(rec)
        else:
            st.warning("⚠️ Monitor this customer closely")
    else:
        st.success("✅ Customer appears satisfied. Maintain service quality.")

    st.divider()

    # ==============================
    # Footer
    # ==============================
    st.markdown("""
        <p style='text-align: center; color: grey;'>
        Built with ANN Model | AUC ROC: 0.857 | 
        Dataset: IBM Telco Customer Churn
        </p>
    """, unsafe_allow_html=True)