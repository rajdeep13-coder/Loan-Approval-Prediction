# 💳 Loan Approval Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/github/license/rajdeep13-coder/Loan-Approval-Prediction-)](https://github.com/rajdeep13-coder/Loan-Approval-Prediction-/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/rajdeep13-coder/Loan-Approval-Prediction-?style=social)](https://github.com/rajdeep13-coder/Loan-Approval-Prediction-/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/rajdeep13-coder/Loan-Approval-Prediction-?style=social)](https://github.com/rajdeep13-coder/Loan-Approval-Prediction-/network/members)

A machine learning project that predicts whether a loan application will be **approved or rejected** based on applicant details. Built with both **Streamlit** and **Flask** frontends.

---

## 📌 Features

- **Multi-model comparison** — Logistic Regression, Random Forest, and Gradient Boosting are trained and the best model is auto-selected
- **Two web interfaces** — Streamlit app for quick demos, Flask app for deployment
- **Preprocessing pipeline** — Handles missing values, label encoding, and feature scaling
- **86% accuracy** on the test set with stratified train-test split and cross-validation

---

## 🗂️ Project Structure

```
├── train.py              # Model training & evaluation script
├── loan_app.py           # Streamlit web app
├── app.py                # Flask web app
├── requirements.txt      # Python dependencies
├── data/
│   └── loan.csv          # Training dataset (614 records)
├── models/
│   ├── loan_model.pkl    # Trained model
│   ├── scaler.pkl        # StandardScaler
│   ├── label_encoders.pkl# LabelEncoders for categorical features
│   └── feature_names.pkl # Feature column order
└── templates/
    └── index.html        # Flask HTML template
```

---

## 📊 Dataset

The dataset contains **614 loan applications** with the following features:

| Feature | Description |
|---------|-------------|
| Gender | Male / Female |
| Married | Yes / No |
| Dependents | 0, 1, 2, 3+ |
| Education | Graduate / Not Graduate |
| Self_Employed | Yes / No |
| ApplicantIncome | Applicant's monthly income |
| CoapplicantIncome | Co-applicant's monthly income |
| LoanAmount | Loan amount (in thousands) |
| Loan_Amount_Term | Term of loan (in months) |
| Credit_History | 1 = good, 0 = bad |
| Property_Area | Urban / Semiurban / Rural |
| **Loan_Status** | **Y (Approved) / N (Rejected)** |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
# Clone the repo
git clone https://github.com/rajdeep13-coder/Loan-Approval-Prediction-.git
cd Loan-Approval-Prediction-

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

This compares three models and saves the best one to `models/`.

### Run the Streamlit App

```bash
streamlit run loan_app.py
```

### Run the Flask App

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 🧠 Model Performance

| Model | Test Accuracy | CV Accuracy |
|-------|:------------:|:-----------:|
| **Logistic Regression** | **86.18%** | **79.63%** |
| Random Forest | 82.93% | 77.60% |
| Gradient Boosting | 79.67% | 75.16% |

---

## 🛠️ Tech Stack

- **Python** — Core language
- **pandas / NumPy** — Data processing
- **scikit-learn** — ML models & preprocessing
- **Streamlit** — Interactive web UI
- **Flask** — Lightweight web framework
- **joblib** — Model serialization

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

<p align="center">
  Made by <a href="https://github.com/rajdeep13-coder">rajdeep13-coder</a>
</p>
