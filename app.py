import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load trained model, scaler, and encoders
with open("models/loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Home page route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()

        # Convert inputs into a DataFrame (single row)
        df = pd.DataFrame([data])

        # Ensure numerical fields are cast properly
        numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Encode categorical fields using saved encoders
        for col, encoder in encoders.items():
            if col in df:
                df[col] = encoder.transform(df[col])

        # Scale numerical features
        df_scaled = scaler.transform(df)

        # Make prediction
        prediction = model.predict(df_scaled)[0]

        # Convert to readable output
        result = "Approved ✅" if prediction == 1 else "Rejected ❌"

        return render_template("index.html", prediction_text=f"Loan Status: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
