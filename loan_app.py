import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model, scaler, encoders, and feature names
model = joblib.load("models/loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/label_encoders.pkl")
feature_names = joblib.load("models/feature_names.pkl")

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💳")

st.title("💳 Loan Approval Predictor")
st.write("Enter applicant details to check loan approval status.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150)
    loan_amount_term = st.number_input("Loan Term (months)", min_value=12, value=360, step=12)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])

# Predict
if st.button("Predict Loan Approval", type="primary"):
    # Build input as a DataFrame with proper column names
    input_dict = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": float(loan_amount_term),
        "Credit_History": float(credit_history),
        "Property_Area": property_area,
    }
    df = pd.DataFrame([input_dict])

    # Encode categorical fields using saved label encoders
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])

    # Ensure column order matches training
    df = df[feature_names]

    # Scale features
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]

    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")