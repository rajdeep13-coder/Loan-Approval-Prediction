import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data_path = os.path.join("data", "loan.csv")
df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# Drop Loan_ID if present
if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)

# Handle missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    if col == "Loan_Status":
        continue  # encode target separately
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target variable
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Split features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

print(f"Features: {list(X.columns)}")
print(f"Target distribution:\n{y.value_counts()}\n")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare multiple models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

best_model = None
best_acc = 0
best_name = ""

for name, clf in models.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring="accuracy")
    print(f"--- {name} ---")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  CV Accuracy   : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Classification Report:\n{classification_report(y_test, y_pred)}\n")

    if acc > best_acc:
        best_acc = acc
        best_model = clf
        best_name = name

print(f"Best model: {best_name} (Accuracy: {best_acc:.4f})\n")

# Save best model + encoders + scaler + feature names
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/loan_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")

print("Model, scaler, encoders, and feature names saved in models/ folder.")
