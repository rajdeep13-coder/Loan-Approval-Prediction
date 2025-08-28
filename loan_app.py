import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("loan_model.pkl", "rb"))

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💳")

st.title("💳 Loan Approval Predictor")
st.write("Enter applicant details to check loan approval status.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=12, step=12)
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert categorical to numeric
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Predict
if st.button("Predict Loan Approval"):
    input_data = np.array([[gender, married, int(dependents.replace("3+", "3")), 
                            education, self_employed, applicant_income, coapplicant_income, 
                            loan_amount, loan_amount_term, credit_history, property_area]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")