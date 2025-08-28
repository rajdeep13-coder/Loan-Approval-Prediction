import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data_path = os.path.join("data", "loan.csv")
df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns)

# Drop Loan_ID if present
if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)

# Handle missing values (simple fill)
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])  # fill categorical with mode
    else:
        df[col] = df[col].fillna(df[col].median())  # fill numerical with median

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

# Save model + encoders + scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/loan_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("Model, scaler, and encoders saved in models/ folder.")
